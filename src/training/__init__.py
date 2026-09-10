"""Training modules for Chinese Chess AI.

The replay buffer remains importable without the optional PyTorch dependency;
self-play is loaded lazily because it depends on the model stack.
"""
from .replay_buffer import ReplayBuffer

__all__ = ['SelfPlayWorker', 'generate_self_play_data', 'ReplayBuffer']


def __getattr__(name):
    """Load PyTorch-backed training helpers only when requested."""
    if name in {'SelfPlayWorker', 'generate_self_play_data'}:
        from .self_play import SelfPlayWorker, generate_self_play_data

        globals().update(
            SelfPlayWorker=SelfPlayWorker,
            generate_self_play_data=generate_self_play_data,
        )
        return globals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
