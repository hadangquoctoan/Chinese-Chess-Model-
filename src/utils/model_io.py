"""Safe, versioned neural-network artifacts and atomic file writes."""
from dataclasses import asdict
import os
import pickle
from pathlib import Path
import tempfile
from typing import Any, Mapping, Optional

import torch

from ..game.rules import ACTION_ENCODING_VERSION, RULESET_VERSION
from ..model.model_config import ModelConfig
from ..model.neural_network import ChineseChessNet

MODEL_FORMAT_VERSION = 1
ARCHITECTURE_FIELDS = (
    'num_residual_blocks', 'num_channels', 'policy_channels', 'value_channels',
    'input_channels', 'board_height', 'board_width', 'action_space',
    'action_encoding_version', 'ruleset_version', 'max_game_plies', 'no_capture_limit',
)


def atomic_torch_save(payload: Any, filepath: str | Path) -> None:
    """Replace a file atomically without leaving a partial checkpoint."""
    validate_safe_payload(payload)
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, prefix=f'.{path.name}.')
    try:
        with os.fdopen(descriptor, 'wb') as stream:
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def validate_safe_payload(payload: Any) -> None:
    """Reject custom objects that cannot be restored by weights_only=True."""
    if isinstance(payload, torch.Tensor) or type(payload) in (str, int, float, bool, bytes, type(None)):
        return
    if isinstance(payload, Mapping):
        for key, value in payload.items():
            validate_safe_payload(key)
            validate_safe_payload(value)
        return
    if type(payload) in (tuple, list):
        for value in payload:
            validate_safe_payload(value)
        return
    raise ValueError(f'Unsupported safe checkpoint payload type: {type(payload).__name__}')


def safe_torch_load(filepath: str | Path, device: str | torch.device = 'cpu') -> Any:
    """Use the restricted loader; never retry unsafe legacy pickle formats."""
    try:
        return torch.load(filepath, map_location=device, weights_only=True)
    except pickle.UnpicklingError as error:
        raise ValueError('Unsafe/legacy artifact rejected; retrain using versioned artifacts.') from error


def build_model_artifact(model: ChineseChessNet) -> dict:
    """Package weights with architecture and action/rule compatibility metadata."""
    artifact = {
        'format_version': MODEL_FORMAT_VERSION,
        'metadata': {
            'config': asdict(model.config),
            'action_encoding_version': getattr(model.config, 'action_encoding_version', ACTION_ENCODING_VERSION),
            'ruleset_version': getattr(model.config, 'ruleset_version', RULESET_VERSION),
        },
        'state_dict': model.state_dict(),
    }
    validate_model_artifact(artifact, model.config)
    return artifact


def validate_tensor(tensor: Any, name: str, expected: torch.Tensor) -> None:
    """Validate a dense persisted tensor without silently casting its dtype."""
    if (not isinstance(tensor, torch.Tensor) or tensor.layout != torch.strided
            or tensor.device.type == 'meta'):
        raise ValueError(f'Invalid tensor in state_dict: {name}')
    if tensor.shape != expected.shape or tensor.dtype != expected.dtype:
        raise ValueError(f'State tensor shape/dtype mismatch: {name}')
    if not torch.isfinite(tensor).all():
        raise ValueError(f'Non-finite state tensor: {name}')


def _validate_weights(state: Any, config: ModelConfig) -> None:
    if not isinstance(state, Mapping) or config.num_residual_blocks > len(state):
        raise ValueError('Invalid/incomplete model state_dict.')
    # Meta modules establish the actual architecture contract without allocating
    # potentially enormous weights from untrusted architecture metadata.
    try:
        with torch.device('meta'):
            expected = ChineseChessNet(config).float().state_dict()
    except (TypeError, ValueError, RuntimeError, OverflowError) as error:
        raise ValueError('Invalid model architecture/state tensor dimensions.') from error
    if state.keys() != expected.keys():
        raise ValueError('Model state_dict names do not match the architecture.')
    for name, tensor in state.items():
        validate_tensor(tensor, name, expected[name])


def validate_model_artifact(artifact: Mapping, config: Optional[ModelConfig] = None) -> ModelConfig:
    """Reject legacy encodings and incompatible architectures before loading weights."""
    if (not isinstance(artifact, Mapping) or type(artifact.get('format_version')) is not int
            or artifact.get('format_version') != MODEL_FORMAT_VERSION):
        raise ValueError('Legacy/unversioned model artifact: retrain with the current action encoding.')
    metadata = artifact.get('metadata', {})
    if not isinstance(metadata, Mapping):
        raise ValueError('Invalid model metadata; retrain or supply a current artifact.')
    for name, version in [('action_encoding_version', ACTION_ENCODING_VERSION), ('ruleset_version', RULESET_VERSION)]:
        if metadata.get(name) != version:
            raise ValueError(f'Incompatible {name}; retrain with {version}.')
    saved_config = metadata.get('config')
    if not isinstance(saved_config, dict) or 'state_dict' not in artifact:
        raise ValueError('Model artifact must include config and state_dict.')
    for name in ('action_encoding_version', 'ruleset_version'):
        if saved_config.get(name) != metadata[name]:
            raise ValueError(f'Missing/incompatible config {name}; retrain with a versioned artifact.')
    if not set(ARCHITECTURE_FIELDS).issubset(saved_config):
        raise ValueError('Incomplete model config/architecture metadata; retrain.')
    try:
        restored = ModelConfig(**saved_config)
        restored.validate()
    except (TypeError, ValueError) as error:
        raise ValueError('Invalid model artifact config.') from error
    if config is not None:
        differences = [name for name in ARCHITECTURE_FIELDS
                       if getattr(restored, name, None) != getattr(config, name, None)]
        if differences:
            raise ValueError(f'Model config/architecture mismatch: {", ".join(differences)}')
    _validate_weights(artifact['state_dict'], restored)
    return restored


def load_model_artifact(filepath: str | Path, device: str | torch.device = 'cpu') -> ChineseChessNet:
    """Instantiate an inference network from a safe versioned artifact."""
    artifact = safe_torch_load(filepath, device)
    config = validate_model_artifact(artifact)
    model = ChineseChessNet(config).to(device=device, dtype=torch.float32)
    model.load_state_dict(artifact['state_dict'])
    model.eval()
    return model
