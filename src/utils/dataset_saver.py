"""Versioned numeric NPZ datasets. Legacy pickle inputs are never executed."""
import os
from pathlib import Path
import tempfile
from typing import List, Tuple

import numpy as np

from ..training.replay_buffer import ReplayBuffer

GameData = Tuple[List[np.ndarray], List[np.ndarray], float]


class DatasetSaver:
    """Save/load validated training data using allow_pickle=False only."""

    def __init__(self, data_dir: str = 'data/training_data'):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def _filepath(self, filename: str) -> Path:
        if Path(filename).name != filename or Path(filename).suffix != '.npz':
            raise ValueError('Safe datasets require an .npz basename; legacy pickle is unsupported.')
        return self.data_dir / filename

    def _write(self, filename: str, payload: dict) -> str:
        path = self._filepath(filename)
        descriptor, temporary = tempfile.mkstemp(dir=path.parent, prefix=f'.{path.name}.')
        try:
            with os.fdopen(descriptor, 'wb') as stream:
                np.savez_compressed(stream, **payload)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
        return str(path)

    def _read(self, filename: str) -> dict:
        with np.load(self._filepath(filename), allow_pickle=False) as archive:
            if 'format_version' not in archive:
                raise ValueError('Legacy/unversioned dataset; retrain using versioned data.')
            payload = {name: archive[name] for name in archive.files}
        for name in ('format_version', 'max_size', 'action_encoding_version', 'ruleset_version'):
            if name in payload:
                payload[name] = payload[name].item()
        return payload

    @staticmethod
    def _numpy_payload(replay_buffer: ReplayBuffer) -> dict:
        payload = replay_buffer.state_dict()
        return {key: value.cpu().numpy() if hasattr(value, 'cpu') else value
                for key, value in payload.items()}

    def save_buffer(self, replay_buffer: ReplayBuffer, filename: str = 'replay_buffer.npz') -> str:
        """Persist an ordered replay snapshot and its capacity as safe numeric arrays."""
        return self._write(filename, self._numpy_payload(replay_buffer))

    def load_buffer(self, filename: str = 'replay_buffer.npz') -> ReplayBuffer:
        """Restore a fully validated versioned replay snapshot."""
        replay = ReplayBuffer(1)
        replay.load_state_dict(self._read(filename))
        return replay

    def save_games(self, games: List[GameData], filename: str) -> str:
        """Flatten games into numeric arrays with offsets, preserving empty games."""
        count = sum(len(states) for states, _, _ in games)
        replay = ReplayBuffer(max(1, count))
        offsets, winners = [0], []
        for states, policies, winner in games:
            replay.add_game(states, policies, winner)
            offsets.append(len(replay))
            winners.append(winner)
        payload = self._numpy_payload(replay)
        payload.update(offsets=np.asarray(offsets, dtype=np.int64),
                       winners=np.asarray(winners, dtype=np.float32))
        return self._write(filename, payload)

    def load_games(self, filename: str) -> List[GameData]:
        """Reconstruct validated games without object arrays or pickle deserialization."""
        payload = self._read(filename)
        replay = ReplayBuffer(1)
        replay.load_state_dict(payload)
        offsets, winners = payload['offsets'], payload['winners']
        if winners.ndim != 1 or not np.isin(winners, [-1, 0, 1]).all():
            raise ValueError('Invalid game winners')
        if offsets.dtype.kind not in 'iu' or offsets.shape != (len(winners) + 1,):
            raise ValueError('Invalid game offsets')
        if (offsets[0] != 0 or offsets[-1] != len(replay)
                or (offsets[1:] < offsets[:-1]).any()
                or (offsets > len(replay)).any()):
            raise ValueError('Invalid game boundaries')
        entries = list(replay.buffer)
        games = []
        for index, winner in enumerate(winners):
            game = entries[int(offsets[index]):int(offsets[index + 1])]
            expected_values = [float(winner) if ply % 2 == 0 else -float(winner)
                               for ply in range(len(game))]
            if any(value != expected for (_, _, value), expected
                   in zip(game, expected_values)):
                raise ValueError('Stored values disagree with game winner/parity')
            games.append(([state for state, _, _ in game],
                          [policy for _, policy, _ in game], float(winner)))
        return games

    def save_iteration_data(self, iteration: int, games: List[GameData], replay_buffer: ReplayBuffer) -> None:
        """Save safe game and replay snapshots for one iteration."""
        self.save_games(games, f'iter_{iteration:03d}_games.npz')
        self.save_buffer(replay_buffer, f'iter_{iteration:03d}_buffer.npz')

    def export_to_numpy(self, replay_buffer: ReplayBuffer, filename: str = 'dataset.npz') -> str:
        """Export shape-stable arrays, including a zero-position buffer."""
        return self.save_buffer(replay_buffer, filename)

    def import_from_numpy(self, filename: str = 'dataset.npz') -> ReplayBuffer:
        """Import numeric arrays and preserve original capacity, including when empty."""
        return self.load_buffer(filename)
