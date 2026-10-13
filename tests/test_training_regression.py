"""Training and replay regressions using only tiny CPU batches."""
import numpy as np
import pytest
import torch

from src.training.replay_buffer import ReplayBuffer


def position():
    state = np.zeros((19, 10, 9), dtype=np.float32)
    state[0, 0, 0] = 1
    policy = np.zeros(1800, dtype=np.float32)
    policy[0] = 1
    return state, policy, 1.0


@pytest.mark.parametrize('invalid', ['state_shape', 'policy_shape', 'state_nan', 'policy_nan', 'negative', 'not_normalized', 'value_nan', 'value_range'])
def test_replay_rejects_invalid_examples(invalid):
    state, policy, value = position()
    if invalid == 'state_shape': state = state[0]
    if invalid == 'policy_shape': policy = policy[:3]
    if invalid == 'state_nan': state[0, 0, 0] = np.nan
    if invalid == 'policy_nan': policy[0] = np.nan
    if invalid == 'negative': policy[1] = -1
    if invalid == 'not_normalized': policy[0] = 0.5
    if invalid == 'value_nan': value = float('nan')
    if invalid == 'value_range': value = 2
    replay = ReplayBuffer(3)
    with pytest.raises(ValueError):
        replay.add(state, policy, value)
    assert len(replay) == 0


def test_replay_safe_roundtrip_preserves_order_capacity_and_owns_data(tmp_path):
    replay = ReplayBuffer(2)
    state, policy, value = position()
    replay.add(state, policy, value)
    state[:] = 9
    policy[:] = 0
    replay.add(*position())
    replay.add(position()[0], position()[1], -1)
    payload = replay.state_dict()
    path = tmp_path / 'replay.pt'
    torch.save(payload, path)
    restored = ReplayBuffer(7)
    restored.load_state_dict(torch.load(path, weights_only=True))
    assert restored.max_size == 2
    assert [v for _, _, v in restored.buffer] == [1, -1]
    states, policies, values = restored.sample(5)
    assert states.shape == (2, 19, 10, 9)
    assert policies.shape == (2, 1800)
    assert values.shape == (2,)
    assert states.max() == 1
    broken = dict(payload)
    broken['values'] = torch.tensor([1.0])
    with pytest.raises(ValueError):
        restored.load_state_dict(broken)
    assert len(restored) == 2
    broken = dict(payload)
    broken['action_encoding_version'] = 'old'
    with pytest.raises(ValueError, match='(?i)encoding|retrain'):
        restored.load_state_dict(broken)


def test_empty_replay_state_roundtrip():
    replay = ReplayBuffer(3)
    restored = ReplayBuffer(7)
    restored.load_state_dict(replay.state_dict())
    assert restored.max_size == 3 and restored.is_empty()


def test_replay_invalid_game_does_not_partially_append():
    replay = ReplayBuffer(2)
    state, policy, _ = position()
    with pytest.raises(ValueError):
        replay.add_game([state, state[0]], [policy, policy], 1)
    assert replay.is_empty()


@pytest.mark.parametrize('payload', [{}, {'format_version': 1}, {'format_version': 1,
    'action_encoding_version': 'piece-slots-v2', 'ruleset_version': 'xiangqi-training-v2',
    'states': 1, 'policies': 1, 'values': 1, 'max_size': 2}])
def test_replay_malformed_snapshot_raises_valueerror(payload):
    replay = ReplayBuffer(2)
    with pytest.raises(ValueError): replay.load_state_dict(payload)


@pytest.mark.parametrize('number', [True, 1.2])
def test_replay_rejects_noninteger_capacity_and_sample_size(number):
    with pytest.raises(ValueError): ReplayBuffer(number)
    replay = ReplayBuffer(2)
    replay.add(*position())
    with pytest.raises(ValueError): replay.sample(number)


def make_trainer(tmp_path, **options):
    from src.model.model_config import ModelConfig
    from src.model.neural_network import ChineseChessNet
    from src.training.trainer import Trainer

    config = ModelConfig(num_residual_blocks=1, num_channels=4,
                         policy_channels=2, value_channels=1,
                         batch_size=2, use_amp=False)
    return Trainer(ChineseChessNet(config), config,
                   checkpoint_dir=str(tmp_path / 'checkpoints'),
                   log_dir=str(tmp_path / 'logs'), **options)


def test_cpu_optimizer_update_finite_nonzero_gradients_modern_amp(tmp_path):
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter('error', FutureWarning)
        trainer = make_trainer(tmp_path, device='cpu', num_workers=0,
                               gradient_accumulation_steps=2, max_batch_size=2)
    try:
        replay = ReplayBuffer(4)
        for _ in range(3): replay.add(*position())
        before = trainer.model.policy_head.fc.weight.detach().clone()
        losses = trainer.train_epoch(replay, num_samples=3)
        assert np.isfinite(losses).all()
        gradients = [p.grad for p in trainer.model.parameters() if p.grad is not None]
        assert gradients and all(torch.isfinite(g).all() for g in gradients)
        assert any(torch.count_nonzero(g) > 0 for g in gradients)
        assert not torch.equal(before, trainer.model.policy_head.fc.weight)
        assert trainer.epoch == 1 and trainer.global_step == 2
        assert trainer.optimizer.state
    finally:
        trainer.close()


def test_trainer_rejects_nonfinite_gradients_before_optimizer_update(tmp_path):
    trainer = make_trainer(tmp_path, device='cpu')
    parameter = trainer.model.policy_head.fc.weight
    before = parameter.detach().clone()
    hook = parameter.register_hook(lambda gradient: torch.full_like(gradient, float('inf')))
    replay = ReplayBuffer(1)
    replay.add(*position())
    try:
        with pytest.raises(ValueError, match='(?i)gradient|finite'):
            trainer.train_epoch(replay, 1)
        torch.testing.assert_close(parameter, before)
        assert trainer.global_step == trainer.epoch == 0
    finally:
        hook.remove()
        trainer.close()


@pytest.mark.parametrize('options', [
    {'num_workers': True}, {'gradient_accumulation_steps': True}, {'max_batch_size': True}])
def test_trainer_rejects_boolean_budgets(tmp_path, options):
    with pytest.raises(ValueError): make_trainer(tmp_path, **options)


@pytest.mark.parametrize('argument', ['num_samples', 'batch_size'])
@pytest.mark.parametrize('number', [True, 1.2])
def test_trainer_rejects_noninteger_epoch_budgets(tmp_path, argument, number):
    trainer = make_trainer(tmp_path, device='cpu')
    replay = ReplayBuffer(1)
    replay.add(*position())
    try:
        with pytest.raises(ValueError): trainer.train_epoch(replay, **{argument: number})
    finally:
        trainer.close()


def test_trainer_rejects_invalid_options_and_config(tmp_path):
    from src.model.model_config import ModelConfig
    from src.model.neural_network import ChineseChessNet
    from src.training.trainer import Trainer

    for options in ({'num_workers': -1}, {'gradient_accumulation_steps': 0}, {'max_batch_size': 0}):
        with pytest.raises(ValueError): make_trainer(tmp_path, **options)
    config = ModelConfig(num_channels=4, num_residual_blocks=0)
    wrong = ModelConfig(num_channels=8, num_residual_blocks=0)
    with pytest.raises(ValueError, match='(?i)config'):
        Trainer(ChineseChessNet(config), wrong, log_dir=str(tmp_path / 'logs'),
                checkpoint_dir=str(tmp_path / 'checkpoints'))


@pytest.mark.parametrize('size', [0, 2])
def test_safe_dataset_roundtrip_empty_and_nonempty(tmp_path, size):
    from src.utils.dataset_saver import DatasetSaver

    saver = DatasetSaver(str(tmp_path))
    replay = ReplayBuffer(5)
    for _ in range(size): replay.add(*position())
    path = saver.export_to_numpy(replay)
    with np.load(path, allow_pickle=False) as archive:
        assert archive['states'].shape == (size, 19, 10, 9)
        assert archive['policies'].shape == (size, 1800)
    restored = saver.import_from_numpy()
    assert len(restored) == size
    assert restored.max_size == 5
    saved_path = saver.save_buffer(replay)
    with np.load(saved_path, allow_pickle=False) as archive:
        assert archive['values'].shape == (size,)
    assert len(saver.load_buffer()) == size
    games = [] if size == 0 else [([position()[0]] * size, [position()[1]] * size, 1.0)]
    saver.save_games(games, 'games.npz')
    loaded_games = saver.load_games('games.npz')
    assert len(loaded_games) == len(games)
    if games:
        assert len(loaded_games[0][0]) == size and loaded_games[0][2] == 1


def test_dataset_rejects_unversioned_and_pickle_files(tmp_path):
    from src.utils.dataset_saver import DatasetSaver
    import pickle

    saver = DatasetSaver(str(tmp_path))
    with (tmp_path / 'legacy.pkl').open('wb') as stream:
        pickle.dump([], stream)
    with pytest.raises(ValueError, match='(?i)legacy|safe|npz'):
        saver.load_games('legacy.pkl')
    np.savez(tmp_path / 'legacy.npz', states=np.empty((0, 19, 10, 9)),
             policies=np.empty((0, 1800)), values=np.empty(0))
    with pytest.raises(ValueError, match='(?i)legacy|version|retrain'):
        saver.import_from_numpy('legacy.npz')
