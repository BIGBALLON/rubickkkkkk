"""Background worker subsystem.

Ingest jobs drain through a single FIFO ``asyncio.Queue`` (``JobQueue``).
Search/query embeds bypass this queue and go through
``embed.executor.ModelExecutor``, which already implements HIGH/LOW
priority on the shared MLX model.
"""

from .job_queue import (
    Job,
    JobQueue,
    JobStatus,
)

__all__ = ["Job", "JobQueue", "JobStatus"]
