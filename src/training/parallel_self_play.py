"""Process-parallel self-play games with one model replica per worker."""
from __future__ import annotations

from concurrent.futures import Future, ProcessPoolExecutor
import multiprocessing
import random
from typing import Any

import numpy as np
import torch

from ..model.neural_network import ChineseChessNet
from .self_play import SelfPlayWorker

_WORKER: SelfPlayWorker | None = None


def _initialize_worker(model_config: Any, state_dict: dict[str, torch.Tensor],
                       device: str, options: dict[str, Any]) -> None:
    global _WORKER
    torch.set_num_threads(1)
    model = ChineseChessNet(model_config)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    _WORKER = SelfPlayWorker(model, **options)


def _play_game(seed: int) -> tuple[list[np.ndarray], list[np.ndarray], float]:
    if _WORKER is None:
        raise RuntimeError("parallel self-play worker was not initialized")
    random.seed(seed)
    np.random.seed(seed)
    return _WORKER.play_game(verbose=False)


class ParallelSelfPlayRunner:
    """Run independent self-play games in spawn-based worker processes.

    Every worker keeps its own model replica and CUDA context, so batch-1 MCTS
    inference from several games overlaps instead of running one at a time.
    """

    def __init__(self, model: ChineseChessNet, options: dict[str, Any], workers: int) -> None:
        if type(workers) is not int or workers < 2:
            raise ValueError("parallel self-play requires at least two workers")
        state_dict = {name: tensor.detach().cpu().clone()
                      for name, tensor in model.state_dict().items()}
        device = str(next(model.parameters()).device)
        self._executor = ProcessPoolExecutor(
            max_workers=workers,
            mp_context=multiprocessing.get_context("spawn"),
            initializer=_initialize_worker,
            initargs=(model.config, state_dict, device, options),
        )
        self._closed = False

    def submit(self, seed: int) -> Future:
        """Queue one self-play game and return its future."""
        if self._closed:
            raise RuntimeError("parallel self-play runner is closed")
        return self._executor.submit(_play_game, seed)

    def close(self) -> None:
        """Wait for running games, then stop all worker processes."""
        if not self._closed:
            self._closed = True
            self._executor.shutdown(wait=True, cancel_futures=True)
