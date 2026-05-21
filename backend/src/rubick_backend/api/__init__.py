"""HTTP API routers (mounted by ``main.py``).

Foreground search embeds run through ``embed.executor.ModelExecutor``
(HIGH priority); background ingest jobs use the FIFO ``JobQueue`` in
``worker/``. Both share one MLX model instance — see ARCHITECTURE.md
(Process model).
"""

from . import healthz, search

__all__ = ["healthz", "search"]
