from collections.abc import Mapping
from contextlib import nullcontext
from copy import deepcopy
from dataclasses import asdict
import math
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

from ..model.neural_network import ChineseChessNet
from ..model.model_config import ModelConfig
from ..utils.model_io import (
    atomic_torch_save, build_model_artifact, safe_torch_load, validate_model_artifact,
    validate_tensor,
)
from .replay_buffer import ReplayBuffer

# Training constants
DEFAULT_LR_STEP_SIZE = 10
DEFAULT_LR_GAMMA = 0.9
DEFAULT_NUM_WORKERS = 0


def _validate_integer(value: object, name: str, minimum: int = 0) -> None:
    if type(value) is not int or value < minimum:
        raise ValueError(f'{name} must be an integer >= {minimum}.')


def _validate_number(value: object, name: str, minimum: float = 0) -> None:
    if type(value) not in (int, float) or not math.isfinite(value) or value < minimum:
        raise ValueError(f'{name} must be finite and >= {minimum}.')


def _validate_adam_group(group: Mapping) -> None:
    for name in ('lr', 'initial_lr', 'weight_decay', 'eps'):
        _validate_number(group[name], f'Adam {name}')
    if group['eps'] == 0:
        raise ValueError('Adam eps must be positive.')
    betas = group['betas']
    if not isinstance(betas, (tuple, list)) or len(betas) != 2:
        raise ValueError('Invalid Adam betas.')
    for beta in betas:
        _validate_number(beta, 'Adam beta')
        if beta >= 1:
            raise ValueError('Adam beta must be less than one.')
    for name in ('amsgrad', 'maximize', 'capturable', 'differentiable', 'decoupled_weight_decay'):
        if name in group and type(group[name]) is not bool:
            raise ValueError(f'Invalid Adam flag {name}.')
    for name in ('foreach', 'fused'):
        if group[name] is not None and type(group[name]) is not bool:
            raise ValueError(f'Invalid Adam flag {name}.')
    if group['fused'] and group['foreach']:
        raise ValueError('Adam fused and foreach modes are mutually exclusive.')
    if group['differentiable']:
        raise ValueError('Differentiable Adam is unsupported by this Trainer.')


def _validate_scaler(state: dict) -> None:
    if not isinstance(state, Mapping):
        raise ValueError('Invalid scaler structure.')
    if not state:
        return
    fields = {'scale', 'growth_factor', 'backoff_factor', 'growth_interval', '_growth_tracker'}
    if set(state) != fields:
        raise ValueError('Incomplete/invalid scaler state.')
    for name in ('scale', 'growth_factor', 'backoff_factor'):
        _validate_number(state[name], f'scaler {name}')
    if state['scale'] <= 0 or state['growth_factor'] <= 1 or not 0 < state['backoff_factor'] < 1:
        raise ValueError('Invalid scaler scale/growth/backoff values.')
    _validate_integer(state['growth_interval'], 'scaler growth interval', minimum=1)
    _validate_integer(state['_growth_tracker'], 'scaler growth tracker')

class ChessDataset(Dataset):
    """PyTorch Dataset cho training data"""
    
    def __init__(self, states: np.ndarray, policies: np.ndarray, values: np.ndarray):
        self.states = torch.from_numpy(states)
        self.policies = torch.from_numpy(policies)
        self.values = torch.from_numpy(values)
    
    def __len__(self):
        return len(self.states)
    
    def __getitem__(self, idx):
        return self.states[idx], self.policies[idx], self.values[idx]

class Trainer:
    """Trainer cho Chinese Chess Neural Network"""
    
    def __init__(self, model: ChineseChessNet, config: ModelConfig,
                 checkpoint_dir: str = "models/checkpoints",
                 log_dir: str = "logs/tensorboard", *,
                 device: Optional[str] = None,
                 num_workers: int = DEFAULT_NUM_WORKERS,
                 gradient_accumulation_steps: int = 1,
                 max_batch_size: Optional[int] = None):
        """Configure device, explicit microbatch budgets and modern CUDA AMP.

        max_batch_size caps microbatches; accumulation steps multiply their
        effective batch size. Neither option is changed implicitly on OOM.
        """
        if model.config != config:
            raise ValueError('Trainer model config does not match the supplied config')
        if type(num_workers) is not int or num_workers < 0:
            raise ValueError('num_workers must be a nonnegative integer')
        if type(gradient_accumulation_steps) is not int or gradient_accumulation_steps <= 0:
            raise ValueError('gradient_accumulation_steps must be a positive integer')
        if max_batch_size is not None and (type(max_batch_size) is not int or max_batch_size <= 0):
            raise ValueError('max_batch_size must be a positive integer')
        self.num_workers = num_workers
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_batch_size = max_batch_size
        self.model = model
        self.config = config
        
        # Setup device
        self.device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
        self.model.to(self.device)
        
        # Optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=DEFAULT_LR_STEP_SIZE,
            gamma=DEFAULT_LR_GAMMA,
        )
        
        # Loss functions
        self.value_loss_fn = nn.MSELoss()
        
        # Mixed precision training
        self.use_amp = config.use_amp and self.device.type == 'cuda'
        self.scaler = torch.amp.GradScaler('cuda', enabled=self.use_amp)
        self._dormant_scaler_state: dict = {}
        
        # Logging
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.writer = SummaryWriter(log_dir)
        self.global_step = 0
        self.epoch = 0
        
        print(f"Trainer initialized on device: {self.device}")
        print(f"Mixed Precision: {self.use_amp}")
        print(f"Model: {model}")

    def _compute_losses(
        self,
        batch_states: torch.Tensor,
        batch_policies: torch.Tensor,
        batch_values: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute policy, value, and combined losses for one batch."""
        policy_logits, value_pred = self.model(batch_states)
        policy_loss = -(
            batch_policies * torch.log_softmax(policy_logits, dim=1)
        ).sum(dim=1).mean()
        value_loss = self.value_loss_fn(value_pred, batch_values)
        return policy_loss, value_loss, policy_loss + value_loss
    
    def _optimizer_step(self) -> None:
        # AMP handles overflows by skipping updates and reducing its scale.
        if not self.use_amp:
            gradients = [parameter.grad for parameter in self.model.parameters()
                         if parameter.grad is not None]
            if not all(torch.isfinite(gradient).all() for gradient in gradients):
                self.optimizer.zero_grad(set_to_none=True)
                raise ValueError('Training gradients are non-finite')
        self.scaler.step(self.optimizer)
        self.scaler.update()

    def train_epoch(self, replay_buffer: ReplayBuffer, 
                   num_samples: int = 50000,
                   batch_size: Optional[int] = None) -> Tuple[float, float, float]:
        """Train an epoch; return sample-weighted policy, value and total losses."""
        self.model.train()
        
        if len(replay_buffer) == 0:
            raise ValueError("Cannot train with an empty replay buffer")
        if type(num_samples) is not int or num_samples <= 0:
            raise ValueError("num_samples must be positive")

        if batch_size is None:
            batch_size = self.config.batch_size
        if type(batch_size) is not int or batch_size <= 0:
            raise ValueError("batch_size must be positive")
        
        if self.max_batch_size is not None:
            batch_size = min(batch_size, self.max_batch_size)

        # Sample data từ replay buffer
        num_samples = min(num_samples, len(replay_buffer))
        states, policies, values = replay_buffer.sample(num_samples)
        
        # Create dataset and dataloader
        dataset = ChessDataset(states, policies, values)
        dataloader = DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.device.type == 'cuda',
        )
        
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(dataloader, desc=f"Epoch {self.epoch}")
        
        self.optimizer.zero_grad(set_to_none=True)
        for batch_index, (batch_states, batch_policies, batch_values) in enumerate(pbar):
            # Move to device
            batch_states = batch_states.to(self.device)
            batch_policies = batch_policies.to(self.device)
            batch_values = batch_values.to(self.device).unsqueeze(1)
            
            autocast_context = torch.amp.autocast(device_type=self.device.type) if self.use_amp else nullcontext()
            with autocast_context:
                policy_loss, value_loss, loss = self._compute_losses(
                    batch_states,
                    batch_policies,
                    batch_values,
                )

            if not torch.isfinite(loss):
                self.optimizer.zero_grad(set_to_none=True)
                raise ValueError('Training loss is non-finite')
            group_start = (batch_index // self.gradient_accumulation_steps) * self.gradient_accumulation_steps * batch_size
            group_samples = min(self.gradient_accumulation_steps * batch_size, len(dataset) - group_start)
            scaled_loss = loss * len(batch_states) / group_samples
            self.scaler.scale(scaled_loss).backward()
            group_end = (batch_index + 1) % self.gradient_accumulation_steps == 0
            if group_end or batch_index + 1 == len(dataloader):
                self._optimizer_step()
                if batch_index + 1 < len(dataloader):
                    self.optimizer.zero_grad(set_to_none=True)
            
            # Accumulate losses
            total_policy_loss += policy_loss.item() * len(batch_states)
            total_value_loss += value_loss.item() * len(batch_states)
            total_loss += loss.item() * len(batch_states)
            num_batches += len(batch_states)
            
            # Update progress bar
            pbar.set_postfix({
                'p_loss': f'{policy_loss.item():.4f}',
                'v_loss': f'{value_loss.item():.4f}',
                'loss': f'{loss.item():.4f}'
            })
            
            # Log to tensorboard
            self.writer.add_scalar('Loss/Policy', policy_loss.item(), self.global_step)
            self.writer.add_scalar('Loss/Value', value_loss.item(), self.global_step)
            self.writer.add_scalar('Loss/Total', loss.item(), self.global_step)
            self.global_step += 1
        
        # Average losses
        avg_policy_loss = total_policy_loss / num_batches
        avg_value_loss = total_value_loss / num_batches
        avg_total_loss = total_loss / num_batches
        
        # Update learning rate
        self.scheduler.step()
        current_lr = self.scheduler.get_last_lr()[0]
        self.writer.add_scalar('Learning_Rate', current_lr, self.epoch)
        
        self.epoch += 1
        
        return avg_policy_loss, avg_value_loss, avg_total_loss
    
    def save_checkpoint(self, filepath: str, additional_info: Optional[dict] = None) -> None:
        """Atomically save training state using only tensors and primitive metadata."""
        artifact = build_model_artifact(self.model)
        checkpoint = {
            'checkpoint_format_version': 1,
            'format_version': artifact['format_version'],
            'metadata': artifact['metadata'],
            'epoch': self.epoch,
            'global_step': self.global_step,
            'model_state_dict': artifact['state_dict'],
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'scaler_state_dict': self.scaler.state_dict() or self._dormant_scaler_state,
            'config': asdict(self.config),
            'trainer_options': {
                'num_workers': self.num_workers,
                'gradient_accumulation_steps': self.gradient_accumulation_steps,
                'max_batch_size': self.max_batch_size,
            },
        }
        if additional_info:
            overlap = checkpoint.keys() & additional_info.keys()
            if overlap:
                raise ValueError(f'Cannot overwrite reserved checkpoint fields: {sorted(overlap)}')
            checkpoint.update(additional_info)
        self.validate_checkpoint(checkpoint)
        atomic_torch_save(checkpoint, filepath)

    def validate_checkpoint(self, checkpoint: dict) -> None:
        """Validate trainer state without mutation; pipeline state is the caller's responsibility."""
        if (not isinstance(checkpoint, dict)
                or type(checkpoint.get('checkpoint_format_version')) is not int
                or checkpoint.get('checkpoint_format_version') != 1):
            raise ValueError('Legacy/unversioned checkpoint; retrain with current encoding.')
        required = {'model_state_dict', 'optimizer_state_dict', 'scheduler_state_dict',
                    'scaler_state_dict', 'config', 'trainer_options', 'epoch', 'global_step'}
        if not required.issubset(checkpoint):
            raise ValueError('Incomplete checkpoint: missing training state fields.')
        options = dict(num_workers=self.num_workers, max_batch_size=self.max_batch_size,
                       gradient_accumulation_steps=self.gradient_accumulation_steps)
        if checkpoint['trainer_options'] != options:
            raise ValueError('Checkpoint trainer options/config mismatch.')
        artifact = dict(checkpoint, state_dict=checkpoint['model_state_dict'])
        validate_model_artifact(artifact, self.config)
        if checkpoint.get('config') != asdict(self.config):
            raise ValueError('Checkpoint training config mismatch; use the original config to resume.')
        if checkpoint.get('metadata', {}).get('config') != checkpoint['config']:
            raise ValueError('Checkpoint config and metadata disagree.')
        for name in ('epoch', 'global_step'):
            _validate_integer(checkpoint[name], name)
        self._validate_optimizer(checkpoint['optimizer_state_dict'])
        self._validate_scheduler(checkpoint['scheduler_state_dict'])
        _validate_scaler(checkpoint['scaler_state_dict'])

    def _validate_optimizer(self, payload: dict) -> None:
        if not isinstance(payload, Mapping) or set(payload) != {'state', 'param_groups'}:
            raise ValueError('Invalid optimizer state structure.')
        groups, states = payload['param_groups'], payload['state']
        expected_groups = self.optimizer.state_dict()['param_groups']
        if not isinstance(groups, list) or len(groups) != len(expected_groups) or not isinstance(states, Mapping):
            raise ValueError('Invalid optimizer parameter groups/state.')
        parameters = {}
        for group, expected, live in zip(groups, expected_groups, self.optimizer.param_groups):
            if not isinstance(group, Mapping) or set(group) != set(expected):
                raise ValueError('Invalid Adam parameter group fields.')
            ids = group['params']
            if (not isinstance(ids, list) or any(type(index) is not int for index in ids)
                    or ids != expected['params']):
                raise ValueError('Adam parameter group associations do not match.')
            _validate_adam_group(group)
            if group['capturable'] and self.device.type == 'cpu':
                raise ValueError('Adam capturable state is not supported on CPU.')
            parameters.update(zip(ids, live['params']))
        if any(type(index) is not int or index not in parameters for index in states):
            raise ValueError('Orphan Adam state parameter association.')
        amsgrad = {index: group['amsgrad'] for group in groups for index in group['params']}
        for index, state in states.items():
            fields = {'step', 'exp_avg', 'exp_avg_sq'}
            if amsgrad[index]:
                fields.add('max_exp_avg_sq')
            if not isinstance(state, Mapping) or set(state) != fields:
                raise ValueError('Incomplete/invalid Adam moment state.')
            for name in fields - {'step'}:
                validate_tensor(state[name], f'Adam {index}.{name}', parameters[index])
                if name != 'exp_avg' and (state[name] < 0).any():
                    raise ValueError(f'Adam second moment {name} must be nonnegative.')
            step = state['step']
            if (not isinstance(step, torch.Tensor) or step.shape != torch.Size([])
                    or step.dtype not in (torch.float32, torch.float64)
                    or step.device.type == 'meta' or step.layout != torch.strided):
                raise ValueError('Invalid Adam step tensor.')
            _validate_number(step.item(), 'Adam step', minimum=0)
            if not step.item().is_integer():
                raise ValueError('Adam step must be an integer.')

    def _validate_scheduler(self, state: dict) -> None:
        expected = self.scheduler.state_dict()
        if not isinstance(state, Mapping) or set(state) != set(expected):
            raise ValueError('Invalid scheduler structure.')
        _validate_integer(state['step_size'], 'scheduler step_size', minimum=1)
        _validate_integer(state['last_epoch'], 'scheduler last_epoch', minimum=-1)
        _validate_integer(state['_step_count'], 'scheduler step count')
        _validate_number(state['gamma'], 'scheduler gamma', minimum=0)
        for name in ('base_lrs', '_last_lr'):
            values = state[name]
            if not isinstance(values, list) or len(values) != len(self.optimizer.param_groups):
                raise ValueError(f'Invalid scheduler {name}.')
            for value in values:
                _validate_number(value, f'scheduler {name}', minimum=0)
        for name, value in expected.items():
            if type(value) is bool and type(state[name]) is not bool:
                raise ValueError(f'Invalid scheduler flag {name}.')

    def load_checkpoint(self, filepath: str, *, checkpoint: dict | None = None) -> dict:
        """Restore pre-read or CPU-loaded trainer state transactionally."""
        checkpoint = self._safe_load(filepath) if checkpoint is None else checkpoint
        self.validate_checkpoint(checkpoint)
        weights = {name: tensor.detach().clone() for name, tensor in self.model.state_dict().items()}
        optimizer_before = self.optimizer.__dict__.copy()
        scheduler_before = self.scheduler.__dict__.copy()
        scaler_before = self.scaler.__dict__.copy()
        scaler_tensors = {name: value.detach().clone() for name, value in scaler_before.items()
                          if isinstance(value, torch.Tensor)}
        try:
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(deepcopy(checkpoint['optimizer_state_dict']))
            self.scheduler.load_state_dict(deepcopy(checkpoint['scheduler_state_dict']))
            if self.use_amp and checkpoint['scaler_state_dict']:
                self.scaler.load_state_dict(deepcopy(checkpoint['scaler_state_dict']))
        except (ValueError, RuntimeError, TypeError, KeyError, IndexError, AssertionError) as error:
            with torch.no_grad():
                for name, tensor in self.model.state_dict().items():
                    tensor.copy_(weights[name])
            for target, previous in ((self.optimizer, optimizer_before),
                                     (self.scheduler, scheduler_before),
                                     (self.scaler, scaler_before)):
                target.__dict__.clear()
                target.__dict__.update(previous)
            with torch.no_grad():
                for name, tensor in scaler_tensors.items():
                    scaler_before[name].copy_(tensor)
            raise ValueError('Checkpoint restoration failed; live state was preserved.') from error
        self._dormant_scaler_state = deepcopy(checkpoint['scaler_state_dict'])
        self.epoch, self.global_step = checkpoint['epoch'], checkpoint['global_step']
        return checkpoint

    def _safe_load(self, filepath: str) -> dict:
        # Session payloads include the entire replay buffer; keep those tensors
        # on CPU. load_state_dict transfers model/optimizer state as needed.
        return safe_torch_load(filepath, 'cpu')

    def save_model(self, filepath: str) -> None:
        """Save versioned model weights with a reconstructable architecture."""
        atomic_torch_save(build_model_artifact(self.model), filepath)

    def load_model(self, filepath: str) -> None:
        """Load weights only after architecture, encoding and ruleset verification."""
        artifact = self._safe_load(filepath)
        validate_model_artifact(artifact, self.config)
        weights = {name: tensor.detach().clone() for name, tensor in self.model.state_dict().items()}
        try:
            self.model.load_state_dict(artifact['state_dict'])
        except (ValueError, RuntimeError, TypeError, KeyError, IndexError, AssertionError) as error:
            with torch.no_grad():
                for name, tensor in self.model.state_dict().items():
                    tensor.copy_(weights[name])
            raise ValueError('Model restoration failed; live weights were preserved.') from error

    def close(self) -> None:
        """Close the TensorBoard writer."""
        self.writer.close()
