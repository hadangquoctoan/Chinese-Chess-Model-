"""Reject corrupt persistence payloads before mutating live training state."""
from copy import deepcopy

import pytest
import torch

from src.model.model_config import ModelConfig
from src.model.neural_network import ChineseChessNet
from src.training.trainer import Trainer
from src.utils.model_io import build_model_artifact, validate_model_artifact


@pytest.fixture
def trainer(tmp_path):
    config = ModelConfig(num_residual_blocks=0, num_channels=2,
                         policy_channels=1, value_channels=1, use_amp=False)
    instance = Trainer(ChineseChessNet(config), config, device='cpu',
                       checkpoint_dir=str(tmp_path / 'cp'), log_dir=str(tmp_path / 'logs'))
    try:
        yield instance
    finally:
        instance.close()


@pytest.mark.parametrize('corruption', ['nan', 'inf', 'shape', 'dtype', 'missing', 'extra', 'not_tensor'])
def test_artifact_rejects_invalid_weights(trainer, corruption):
    artifact = deepcopy(build_model_artifact(trainer.model))
    state = artifact['state_dict']
    name = 'conv_input.weight'
    if corruption in ('nan', 'inf'):
        state[name].flatten()[0] = float(corruption)
    elif corruption == 'shape':
        state[name] = torch.zeros(1)
    elif corruption == 'dtype':
        state[name] = state[name].double()
    elif corruption == 'missing':
        del state[name]
    elif corruption == 'extra':
        state['unexpected.weight'] = torch.zeros(1)
    else:
        state[name] = []
    with pytest.raises(ValueError, match='(?i)state|weight|tensor|finite|dtype|shape'):
        validate_model_artifact(artifact, trainer.config)


def test_build_artifact_rejects_nonfinite_live_weights(trainer):
    with torch.no_grad():
        trainer.model.conv_input.weight.flatten()[0] = float('nan')
    with pytest.raises(ValueError, match='(?i)finite'):
        build_model_artifact(trainer.model)


def test_trainer_rejects_nan_model_load_without_mutation(trainer, tmp_path):
    artifact = deepcopy(build_model_artifact(trainer.model))
    artifact['state_dict']['conv_input.weight'].add_(5)
    artifact['state_dict']['value_head.fc2.bias'][0] = float('nan')
    path = tmp_path / 'bad-model.pt'
    torch.save(artifact, path)
    before = deepcopy(trainer.model.state_dict())
    with pytest.raises(ValueError, match='(?i)finite'):
        trainer.load_model(path)
    for name, tensor in before.items():
        torch.testing.assert_close(trainer.model.state_dict()[name], tensor, rtol=0, atol=0)


def assert_equal(left, right):
    if isinstance(left, torch.Tensor):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert type(left) is type(right) and len(left) == len(right)
        for a, b in zip(left, right):
            assert_equal(a, b)
    else:
        assert left == right


def snapshot(trainer):
    return deepcopy(dict(model=trainer.model.state_dict(),
                         optimizer=trainer.optimizer.state_dict(),
                         scheduler=trainer.scheduler.state_dict(),
                         scaler=trainer.scaler.state_dict(),
                         dormant=trainer._dormant_scaler_state,
                         epoch=trainer.epoch, global_step=trainer.global_step))


def checkpoint_payload(trainer, tmp_path):
    for parameter in trainer.model.parameters():
        parameter.grad = torch.ones_like(parameter)
    trainer.optimizer.step()
    trainer.optimizer.zero_grad(set_to_none=True)
    trainer.scheduler.step()
    trainer.epoch, trainer.global_step = 2, 3
    path = tmp_path / 'checkpoint.pt'
    trainer.save_checkpoint(path)
    return path, torch.load(path, weights_only=True)


@pytest.mark.parametrize('corruption', [
    'epoch_negative', 'epoch_bool', 'epoch_float', 'step_negative', 'step_bool',
    'step_float', 'adam_shape', 'adam_nan', 'adam_dtype', 'adam_step',
    'adam_negative_step', 'adam_missing', 'adam_orphan', 'group_duplicate',
    'group_order', 'group_lr', 'group_betas', 'group_params', 'scheduler_missing',
    'scheduler_step', 'scheduler_nan', 'scheduler_lr', 'scheduler_epoch',
    'scaler_missing', 'scaler_nan', 'scaler_growth',
])
def test_bad_checkpoint_rejected_without_mutation(trainer, tmp_path, corruption):
    path, payload = checkpoint_payload(trainer, tmp_path)
    before = snapshot(trainer)
    model, optimizer, scheduler, scaler = trainer.model, trainer.optimizer, trainer.scheduler, trainer.scaler
    payload['model_state_dict']['conv_input.weight'].add_(5)
    adam = payload['optimizer_state_dict']
    first = next(iter(adam['state'].values()))
    group = adam['param_groups'][0]
    schedule = payload['scheduler_state_dict']
    if corruption.startswith('epoch_') or corruption in ('step_negative', 'step_bool', 'step_float'):
        field = 'epoch' if corruption.startswith('epoch_') else 'global_step'
        payload[field] = {'negative': -1, 'bool': True, 'float': 1.5}[corruption.split('_')[1]]
    elif corruption == 'adam_shape':
        first['exp_avg'] = torch.zeros(1)
    elif corruption == 'adam_nan':
        first['exp_avg_sq'].flatten()[0] = float('nan')
    elif corruption == 'adam_dtype':
        first['exp_avg'] = first['exp_avg'].double()
    elif corruption == 'adam_step':
        first['step'] = torch.tensor(float('inf'))
    elif corruption == 'adam_negative_step':
        first['step'] = torch.tensor(-1.)
    elif corruption == 'adam_missing':
        del first['exp_avg_sq']
    elif corruption == 'adam_orphan':
        adam['state'][999999] = deepcopy(first)
    elif corruption == 'group_duplicate':
        group['params'][1] = group['params'][0]
    elif corruption == 'group_order':
        group['params'].reverse()
    elif corruption == 'group_lr':
        group['lr'] = float('nan')
    elif corruption == 'group_betas':
        group['betas'] = (1.5, 0.999)
    elif corruption == 'group_params':
        group['params'] = 'invalid'
    elif corruption == 'scheduler_missing':
        del schedule['step_size']
    elif corruption == 'scheduler_step':
        schedule['step_size'] = 0
    elif corruption == 'scheduler_nan':
        schedule['gamma'] = float('nan')
    elif corruption == 'scheduler_lr':
        schedule['_last_lr'] = []
    elif corruption == 'scheduler_epoch':
        schedule['last_epoch'] = True
    elif corruption == 'scaler_missing':
        payload['scaler_state_dict'] = {'scale': 1.}
    else:
        payload['scaler_state_dict'] = torch.amp.GradScaler('cpu').state_dict()
        field = 'scale' if corruption == 'scaler_nan' else 'growth_factor'
        payload['scaler_state_dict'][field] = float('nan') if field == 'scale' else 0.5
    torch.save(payload, path)
    with pytest.raises(ValueError):
        trainer.load_checkpoint(path)
    assert (trainer.model, trainer.optimizer, trainer.scheduler, trainer.scaler) == (model, optimizer, scheduler, scaler)
    assert_equal(before, snapshot(trainer))


def test_checkpoint_runtime_failure_rolls_back_all_live_state(trainer, tmp_path, monkeypatch):
    path, payload = checkpoint_payload(trainer, tmp_path)
    before = snapshot(trainer)
    old_optimizer_state = trainer.optimizer.state
    old_groups = trainer.optimizer.param_groups
    parameters = tuple(trainer.model.parameters())
    payload['model_state_dict']['conv_input.weight'].add_(5)
    payload['optimizer_state_dict']['param_groups'][0]['lr'] *= 2
    payload['scheduler_state_dict']['last_epoch'] += 1
    load = type(trainer.scheduler).load_state_dict

    def fail_after_loading(instance, state):
        load(instance, state)
        raise RuntimeError('Simulated scheduler restoration failure')

    monkeypatch.setattr(type(trainer.scheduler), 'load_state_dict', fail_after_loading)
    with pytest.raises(ValueError, match='(?i)preserved|restoration'):
        trainer.load_checkpoint(path, checkpoint=payload)
    assert_equal(before, snapshot(trainer))
    assert trainer.optimizer.state is old_optimizer_state
    assert trainer.optimizer.param_groups is old_groups
    assert tuple(trainer.model.parameters()) == parameters


def test_preread_checkpoint_validation_and_load_do_not_reread_or_alias(trainer, tmp_path, monkeypatch):
    path, payload = checkpoint_payload(trainer, tmp_path)
    before = snapshot(trainer)
    assert trainer.validate_checkpoint(payload) is None
    assert_equal(before, snapshot(trainer))
    monkeypatch.setattr(trainer, '_safe_load', lambda path: pytest.fail('Unexpected second disk read'))
    assert trainer.load_checkpoint(path, checkpoint=payload) is payload
    next(iter(payload['optimizer_state_dict']['state'].values()))['exp_avg'].add_(7)
    assert_equal(before, snapshot(trainer))


def test_checkpoint_save_rejects_corrupt_optimizer_before_replacing_file(trainer, tmp_path):
    path, payload = checkpoint_payload(trainer, tmp_path)
    previous = path.read_bytes()
    next(iter(trainer.optimizer.state.values()))['exp_avg'] = torch.zeros(1)
    with pytest.raises(ValueError, match='(?i)shape|Adam'):
        trainer.save_checkpoint(path)
    assert path.read_bytes() == previous


def test_adam_negative_second_moment_rejected(trainer, tmp_path):
    path, payload = checkpoint_payload(trainer, tmp_path)
    next(iter(payload['optimizer_state_dict']['state'].values()))['exp_avg_sq'].fill_(-1)
    with pytest.raises(ValueError, match='(?i)moment|negative'):
        trainer.load_checkpoint(path, checkpoint=payload)


def test_cpu_rejects_adam_capturable_checkpoint_before_training(trainer, tmp_path):
    path, payload = checkpoint_payload(trainer, tmp_path)
    payload['optimizer_state_dict']['param_groups'][0]['capturable'] = True
    with pytest.raises(ValueError, match='(?i)Adam|capturable'):
        trainer.load_checkpoint(path, checkpoint=payload)


def test_weight_load_runtime_failure_preserves_parameters(trainer, tmp_path, monkeypatch):
    artifact = deepcopy(build_model_artifact(trainer.model))
    artifact['state_dict']['conv_input.weight'].add_(5)
    path = tmp_path / 'weights.pt'
    torch.save(artifact, path)
    before = snapshot(trainer)
    load = trainer.model.load_state_dict

    def fail_after_loading(state):
        load(state)
        raise RuntimeError('Simulated model restore failure')

    monkeypatch.setattr(trainer.model, 'load_state_dict', fail_after_loading)
    with pytest.raises(ValueError, match='(?i)preserved|restoration'):
        trainer.load_model(path)
    assert_equal(before, snapshot(trainer))


def test_scaler_runtime_failure_preserves_existing_tensor_identity(trainer, tmp_path, monkeypatch):
    path, payload = checkpoint_payload(trainer, tmp_path)
    trainer.scaler = torch.amp.GradScaler('cpu')
    trainer.scaler.scale(torch.ones(1))
    trainer.use_amp = True
    before = snapshot(trainer)
    original_scale = trainer.scaler._scale
    original_tracker = trainer.scaler._growth_tracker
    payload['scaler_state_dict'] = trainer.scaler.state_dict()
    payload['scaler_state_dict']['scale'] *= 2
    load = type(trainer.scaler).load_state_dict

    def fail_after_loading(instance, state):
        load(instance, state)
        raise RuntimeError('Simulated scaler restore failure')

    monkeypatch.setattr(type(trainer.scaler), 'load_state_dict', fail_after_loading)
    with pytest.raises(ValueError):
        trainer.load_checkpoint(path, checkpoint=payload)
    assert_equal(before, snapshot(trainer))
    assert trainer.scaler._scale is original_scale
    assert trainer.scaler._growth_tracker is original_tracker


def test_artifact_validation_is_independent_of_global_default_dtype(trainer):
    artifact = build_model_artifact(trainer.model)
    previous = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float64)
        validate_model_artifact(artifact)
    finally:
        torch.set_default_dtype(previous)


def test_artifact_bool_version_rejected(trainer):
    artifact = build_model_artifact(trainer.model)
    artifact['format_version'] = True
    with pytest.raises(ValueError, match='(?i)format|version'):
        validate_model_artifact(artifact)


def test_huge_architecture_checked_on_meta_without_weight_allocations(trainer, monkeypatch):
    import src.utils.model_io as model_io

    artifact = build_model_artifact(trainer.model)
    artifact['metadata']['config']['num_channels'] = 10**9
    constructor = model_io.ChineseChessNet

    def meta_only_constructor(config):
        assert torch.get_default_device().type == 'meta'
        return constructor(config)

    monkeypatch.setattr(model_io, 'ChineseChessNet', meta_only_constructor)
    with pytest.raises(ValueError, match='(?i)shape|state|architecture'):
        validate_model_artifact(artifact)


@pytest.mark.parametrize('flags', [dict(foreach=True, fused=True), dict(differentiable=True)])
def test_adam_unsupported_execution_flags_rejected_before_epoch(trainer, tmp_path, flags):
    path, payload = checkpoint_payload(trainer, tmp_path)
    payload['optimizer_state_dict']['param_groups'][0].update(flags)
    with pytest.raises(ValueError, match='(?i)Adam|optimizer'):
        trainer.load_checkpoint(path, checkpoint=payload)


def test_artifact_oversized_metadata_raises_validation_error(trainer):
    artifact = build_model_artifact(trainer.model)
    artifact['metadata']['config']['num_channels'] = 10**30
    with pytest.raises(ValueError, match='(?i)architecture|state|shape'):
        validate_model_artifact(artifact)


def test_inference_load_preserves_artifact_dtype_not_global_default(trainer, tmp_path):
    from src.utils.model_io import load_model_artifact

    path = tmp_path / 'model.pt'
    torch.save(build_model_artifact(trainer.model), path)
    previous = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float64)
        restored = load_model_artifact(path)
        assert restored.conv_input.weight.dtype == torch.float32
    finally:
        torch.set_default_dtype(previous)
