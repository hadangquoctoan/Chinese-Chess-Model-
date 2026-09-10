from contextlib import nullcontext
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
from .replay_buffer import ReplayBuffer

# Training constants
DEFAULT_LR_STEP_SIZE = 10
DEFAULT_LR_GAMMA = 0.9
DEFAULT_NUM_WORKERS = 0

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
                 log_dir: str = "logs/tensorboard"):
        """
        Args:
            model: Neural network
            config: Model config
            checkpoint_dir: Directory để lưu checkpoints
            log_dir: Directory cho TensorBoard logs
        """
        self.model = model
        self.config = config
        
        # Setup device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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
        self.use_amp = config.use_amp and torch.cuda.is_available()
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)
        
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
    
    def train_epoch(self, replay_buffer: ReplayBuffer, 
                   num_samples: int = 50000,
                   batch_size: Optional[int] = None) -> Tuple[float, float, float]:
        """
        Train một epoch
        
        Args:
            replay_buffer: Buffer chứa training data
            num_samples: Số samples để train
            batch_size: Batch size (mặc định dùng config)
        
        Returns:
            avg_policy_loss, avg_value_loss, avg_total_loss
        """
        self.model.train()
        
        if len(replay_buffer) == 0:
            raise ValueError("Cannot train with an empty replay buffer")
        if num_samples <= 0:
            raise ValueError("num_samples must be positive")

        if batch_size is None:
            batch_size = self.config.batch_size
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        
        # Sample data từ replay buffer
        num_samples = min(num_samples, len(replay_buffer))
        states, policies, values = replay_buffer.sample(num_samples)
        
        # Create dataset and dataloader
        dataset = ChessDataset(states, policies, values)
        dataloader = DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=True,
            num_workers=DEFAULT_NUM_WORKERS,
            pin_memory=torch.cuda.is_available(),
        )
        
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(dataloader, desc=f"Epoch {self.epoch}")
        
        for batch_states, batch_policies, batch_values in pbar:
            # Move to device
            batch_states = batch_states.to(self.device)
            batch_policies = batch_policies.to(self.device)
            batch_values = batch_values.to(self.device).unsqueeze(1)
            
            autocast_context = torch.cuda.amp.autocast() if self.use_amp else nullcontext()
            with autocast_context:
                policy_loss, value_loss, loss = self._compute_losses(
                    batch_states,
                    batch_policies,
                    batch_values,
                )

            self.optimizer.zero_grad()
            if self.use_amp:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self.optimizer.step()
            
            # Accumulate losses
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_loss += loss.item()
            num_batches += 1
            
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
    
    def save_checkpoint(self, filepath: str, additional_info: Optional[dict] = None):
        """Lưu checkpoint"""
        checkpoint = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': self.config,
        }
        
        if self.use_amp:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        if additional_info:
            checkpoint.update(additional_info)
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, filepath)
        print(f"Checkpoint saved: {filepath}")
    
    def load_checkpoint(self, filepath: str):
        """Load checkpoint"""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        if self.use_amp and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        self.epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        
        print(f"Checkpoint loaded: {filepath}")
        print(f"Resuming from epoch {self.epoch}")
    
    def save_model(self, filepath: str):
        """Lưu model (chỉ weights)"""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), filepath)
        print(f"Model saved: {filepath}")
    
    def load_model(self, filepath: str):
        """Load model (chỉ weights)"""
        self.model.load_state_dict(torch.load(filepath, map_location=self.device))
        print(f"Model loaded: {filepath}")
    
    def close(self):
        """Đóng tensorboard writer"""
        self.writer.close()
