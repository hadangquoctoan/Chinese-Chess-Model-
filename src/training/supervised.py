"""No-grad validation metrics and validation-driven early stopping."""

import math
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ..model.neural_network import ChineseChessNet
from .replay_buffer import ReplayBuffer
from .trainer import ChessDataset, Trainer


def _positive_integer(value: object, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f'{name} must be a positive integer')
    return value


def evaluate_supervised(model: ChineseChessNet, replay_buffer: ReplayBuffer,
                        batch_size: int, num_workers: int = 0) -> dict[str, float]:
    """Evaluate every replay sample once without updating model or optimizer state.

    Losses mirror ``Trainer._compute_losses`` and are averaged per sample, so
    validation metrics do not depend on the final batch size. The model keeps
    its previous train/eval mode after evaluation.
    """
    _positive_integer(batch_size, 'batch_size')
    if type(num_workers) is not int or num_workers < 0:
        raise ValueError('num_workers must be a nonnegative integer')
    if len(replay_buffer) == 0:
        raise ValueError('Cannot evaluate an empty replay buffer')

    states, policies, values = replay_buffer.sample(len(replay_buffer))
    dataset = ChessDataset(states, policies, values)
    device = next(model.parameters()).device
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=device.type == 'cuda')
    totals = {'policy_loss': 0.0, 'value_loss': 0.0, 'top1_accuracy': 0.0}
    sample_count = 0
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            for batch_states, batch_policies, batch_values in loader:
                batch_states = batch_states.to(device)
                batch_policies = batch_policies.to(device)
                batch_values = batch_values.to(device).unsqueeze(1)
                policy_logits, value_pred = model(batch_states)
                policy_loss = -(batch_policies * torch.log_softmax(
                    policy_logits, dim=1)).sum(dim=1).mean()
                value_loss = F.mse_loss(value_pred, batch_values)
                batch_count = len(batch_states)
                totals['policy_loss'] += policy_loss.item() * batch_count
                totals['value_loss'] += value_loss.item() * batch_count
                matches = policy_logits.argmax(dim=1) == batch_policies.argmax(dim=1)
                totals['top1_accuracy'] += matches.sum().item()
                sample_count += batch_count
    finally:
        model.train(was_training)

    policy_loss = totals['policy_loss'] / sample_count
    value_loss = totals['value_loss'] / sample_count
    return {
        'policy_loss': policy_loss,
        'value_loss': value_loss,
        'total_loss': policy_loss + value_loss,
        'top1_accuracy': totals['top1_accuracy'] / sample_count,
    }


def _validation_metrics(trainer: Trainer,
                        validation_replay: ReplayBuffer) -> dict[str, float]:
    """Evaluate validation data with the trainer's model and loader settings.

    ``Trainer`` guarantees ``model``, ``config.batch_size`` and ``num_workers``.
    Direct access fails loudly for an incompatible double instead of silently
    substituting defaults that would change the reported metrics.
    """
    return evaluate_supervised(
        trainer.model, validation_replay,
        batch_size=trainer.config.batch_size,
        num_workers=trainer.num_workers)


def fit_supervised(trainer: Trainer, train_replay: ReplayBuffer,
                   validation_replay: ReplayBuffer | None,
                   output_path: str | Path, epochs: int,
                   patience: int) -> list[dict]:
    """Train supervised epochs, saving only strict validation-loss improvements.

    Stops after ``patience`` consecutive epochs without a strictly lower
    validation total loss. Without validation data the final model is always
    saved, preserving the legacy pretraining behavior.
    """
    _positive_integer(epochs, 'epochs')
    _positive_integer(patience, 'patience')
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    best_loss = math.inf
    stale_epochs = 0
    history: list[dict] = []
    for _ in range(epochs):
        train_losses = trainer.train_epoch(train_replay, len(train_replay))
        if validation_replay is None:
            history.append({'epoch': trainer.epoch, 'train_losses': train_losses})
            continue
        metrics = _validation_metrics(trainer, validation_replay)
        history.append({'epoch': trainer.epoch, 'train_losses': train_losses,
                        'validation': metrics})
        if metrics['total_loss'] < best_loss:
            best_loss = metrics['total_loss']
            stale_epochs = 0
            trainer.save_model(str(output_path))
        else:
            stale_epochs += 1
        if stale_epochs >= patience:
            break
    if validation_replay is None:
        trainer.save_model(str(output_path))
    return history
