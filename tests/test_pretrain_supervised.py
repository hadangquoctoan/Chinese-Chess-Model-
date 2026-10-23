from types import SimpleNamespace

import pytest

from scripts.pretrain_supervised import check_init_architecture


def _fields():
    return {
        'num_residual_blocks': 1, 'num_channels': 4, 'policy_channels': 1,
        'value_channels': 1, 'input_channels': 19, 'board_height': 10,
        'board_width': 9, 'action_space': 1800,
        'action_encoding_version': 'piece-slots-v2',
        'ruleset_version': 'xiangqi-training-v2',
        'max_game_plies': 4, 'no_capture_limit': 2,
    }


def test_architecture_mismatch_is_rejected():
    config = SimpleNamespace(**_fields())
    loaded = SimpleNamespace(config=SimpleNamespace(**_fields()))
    check_init_architecture(loaded, config)
    loaded.config.max_game_plies = 200
    with pytest.raises(ValueError, match='max_game_plies'):
        check_init_architecture(loaded, config)
