"""Background ingest job queue.

The Swift main process discovers new files via FSEvents and POSTs
them to ``/index/job``. That route turns the request into an
in-memory :class:`Job`, enqueues it, and returns immediately — the
caller polls ``GET /index/job/{id}`` for status / stats. A single
worker coroutine drains the queue serially.

Why serial (vs N parallel workers):

- We only own one MLX model instance (see ARCHITECTURE.md — Process
  model). Two workers would either fight for the GIL or load a second
  1.8 GB model — both bad.
- Ingest is dominated by ``embed_*`` calls, which already run on the
  GPU. Running them in lock-step keeps memory pressure predictable
  on entry-level M-series Macs.

When ``/search`` arrives during a long ingest, the search route
embeds through ``ModelExecutor`` at HIGH priority while ingest
embeds queue at LOW — see ``embed/executor.py``.

Lifecycle:

    queue = JobQueue()
    await queue.start()
    ...
    job = await queue.enqueue(paths=[...])
    ...
    await queue.stop()
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import os
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

# How many job records we keep around in the in-memory dict. Older
# completed jobs are evicted FIFO. 256 is enough headroom for a
# dogfood session that re-indexes a few folders; v1 doesn't need
# disk persistence (a server restart loses pending jobs, which is
# fine — Swift re-fires them from its watch state).
MAX_JOB_HISTORY: int = 256


class JobStatus(StrEnum):
    """Linear state machine: ``queued → running → (succeeded | failed)``.

    ``StrEnum`` (Python 3.11+) gives us free string serialization
    (FastAPI / JSON encoders treat instances as their ``.value``)
    without the deprecated multi-inherit-from-``str``+``Enum`` pattern.
    """

    queued = "queued"
    running = "running"
    succeeded = "succeeded"
    failed = "failed"


@dataclass
class Job:
    """One ingest request. Mutable on purpose — the worker rewrites
    ``status`` / ``stats`` / ``progress`` / ``error`` in place to keep
    the polling endpoint simple (a single dict lookup, no merging).

    ``progress`` is ``{"total": int, "done": int, "current": str | None,
    "embedded": int}`` while a job is running and ``None`` while queued
    or after a final state. ``total`` is the file count enumerated up
    front by the pipeline's directory walk; ``done`` ticks per completed
    file (regardless of skip / add); ``embedded`` is the subset of
    ``done`` we actually pushed through the model (everything else was
    a fast-skip / sha-skip / reject); ``current`` is the absolute path
    of the file most recently processed. Across multi-path jobs (rare —
    the Swift main process splits FSEvents batches per folder) ``total``
    is the running grand total observed so far so the UI never has
    a denominator that decreases mid-run.
    """

    id: str
    paths: list[str]
    status: JobStatus = JobStatus.queued
    enqueued_at: int = field(default_factory=lambda: int(time.time()))
    started_at: int | None = None
    finished_at: int | None = None
    stats: dict[str, int] | None = None
    progress: dict[str, Any] | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable view. ``status`` becomes its string value
        thanks to the ``str``-mixin enum."""
        d = asdict(self)
        d["status"] = self.status.value
        return d


# Signature of the per-path ingest callable the queue invokes. The
# default is ``rubick_backend.ingest.ingest_path``; tests inject a
# fake to avoid touching LanceDB / the embed model. New deployments
# pass ``progress_cb`` and ``pause_event`` through (added v1.x); the
# worker uses ``inspect.signature`` to detect the legacy zero-kwarg
# form and falls back gracefully so existing test stubs keep working.
IngestCallable = Callable[..., dict[str, int]]


class JobQueue:
    """Async FIFO of ingest jobs + a single drain worker.

    Not safe to share across event loops (asyncio primitives bind to
    a loop on first use). One instance per FastAPI app lifespan is
    the intended pattern.
    """

    def __init__(
        self,
        *,
        ingest_fn: IngestCallable | None = None,
        max_history: int = MAX_JOB_HISTORY,
    ) -> None:
        self._queue: asyncio.Queue[Job] = asyncio.Queue()
        self._jobs: dict[str, Job] = {}
        self._order: list[str] = []  # insertion order for FIFO eviction
        self._worker: asyncio.Task[None] | None = None
        self._ingest_fn = ingest_fn  # lazy-resolved on first use
        self._max_history = max_history
        # Set whenever ``pending == 0``. Cleared on enqueue, set when
        # the worker finishes a job and the queue is empty.
        # ``wait_idle`` and the lifespan shutdown both await it.
        self._idle_event: asyncio.Event = asyncio.Event()
        self._idle_event.set()
        # Pause gate. ``set()`` means "go"; ``clear()`` means "block any
        # further dequeue". The worker awaits this *before* each
        # ``_queue.get()`` so pausing never interrupts in-flight work
        # at the queue level. ``_unpaused_thread`` is the threading
        # mirror that the *ingest pipeline* checks once per file —
        # pause() / resume() flip both events together so pause takes
        # effect within ~one file even mid-folder. Jobs enqueued while
        # paused accumulate in the queue (and stay in ``pending``); on
        # ``resume()`` they drain in submission order.
        self._unpaused: asyncio.Event = asyncio.Event()
        self._unpaused.set()
        self._unpaused_thread: threading.Event = threading.Event()
        self._unpaused_thread.set()

    # === Lifecycle =========================================================

    async def start(self) -> None:
        """Spawn the drain task. Idempotent."""
        if self._worker is not None and not self._worker.done():
            return
        self._worker = asyncio.create_task(self._run(), name="rubick-job-queue")

    async def stop(self) -> None:
        """Cancel the drain task and wait for it to unwind.

        In-flight jobs raise ``CancelledError`` inside ``ingest_fn``
        which propagates through ``asyncio.to_thread`` — the wrapped
        ``ingest_path`` may still be partway through writing rows when
        we cancel. That's acceptable: the rows already written are
        valid (LanceDB commits per ``add()`` call), and the job lands
        in ``failed`` state so the caller knows to retry.
        """
        if self._worker is None:
            return
        if not self._worker.done():
            self._worker.cancel()
        try:
            await self._worker
        except asyncio.CancelledError:
            pass
        self._worker = None

    # === Public API ========================================================

    async def enqueue(self, *, paths: list[str | Path]) -> Job:
        """Create a ``Job`` for ``paths`` and put it on the queue.

        Returns the newly-minted Job immediately; the worker will
        flip its status to ``running`` and then ``succeeded`` /
        ``failed`` over the next few seconds-to-minutes.
        """
        if not paths:
            raise ValueError("paths must be non-empty")
        # ``os.path.expanduser`` is pure string work — no syscalls —
        # so it's safe in an async function. (ruff ASYNC240 noqa.)
        normalized = [os.path.expanduser(str(p)) for p in paths]  # noqa: ASYNC240

        job = Job(
            id=uuid.uuid4().hex[:12],
            paths=normalized,
        )
        self._jobs[job.id] = job
        self._order.append(job.id)
        self._evict_old()
        self._idle_event.clear()
        await self._queue.put(job)
        log.info("job %s enqueued — %d path(s)", job.id, len(job.paths))
        return job

    def get(self, job_id: str) -> Job | None:
        return self._jobs.get(job_id)

    def list_recent(self, limit: int = 20) -> list[Job]:
        """Recent jobs, newest first. Used by ``GET /index/jobs`` for
        a small dashboard / debugging surface."""
        ids = list(reversed(self._order))[:limit]
        return [self._jobs[i] for i in ids if i in self._jobs]

    @property
    def pending(self) -> int:
        """Queued + running. Tests and the UI use this to wait on
        idle (poll ``pending == 0``)."""
        return self._queue.qsize() + sum(
            1 for j in self._jobs.values() if j.status is JobStatus.running
        )

    # === Pause / resume ====================================================

    @property
    def paused(self) -> bool:
        """``True`` when the worker is gated from picking the next job.

        Surfaced by ``GET /index/status`` so the Swift UI can render
        a "Paused" badge and toggle the Pause / Resume button correctly
        on Settings → Index appearance.
        """
        return not self._unpaused.is_set()

    def pause(self) -> None:
        """Stop dispatching new jobs to the worker, and block the
        in-flight ingest at the next per-file checkpoint.

        Already-running file work runs to completion (image embed
        ~2 s, video embed ~13 s on cold caches); the per-file
        ``threading.Event.wait()`` inside ``ingest_path`` blocks
        before the next file. Jobs enqueued while paused stay in
        the queue and resume in submission order.

        Idempotent — pausing an already-paused queue is a no-op.
        """
        if self._unpaused.is_set():
            self._unpaused.clear()
            self._unpaused_thread.clear()
            log.info("job-queue paused (pending=%d)", self.pending)

    def resume(self) -> None:
        """Allow the worker to fetch the next queued job AND unblock
        any per-file pause currently held inside ``ingest_path``.

        Idempotent — resuming a running queue is a no-op.
        """
        if not self._unpaused.is_set():
            self._unpaused.set()
            self._unpaused_thread.set()
            log.info("job-queue resumed (pending=%d)", self.pending)

    async def wait_idle(self, *, timeout: float | None = None) -> None:  # noqa: ASYNC109
        """Block until the queue is empty AND no job is running.

        ``timeout`` is here for ergonomic test code (``await
        queue.wait_idle(timeout=1)`` reads better than wrapping
        ``asyncio.timeout`` at every call site). We raise
        ``TimeoutError`` (the builtin alias for ``asyncio.TimeoutError``
        since Python 3.11) on overrun to keep test asserts portable.
        """
        if timeout is None:
            await self._idle_event.wait()
            return
        try:
            async with asyncio.timeout(timeout):
                await self._idle_event.wait()
        except TimeoutError as e:
            raise TimeoutError(f"queue still has {self.pending} pending after {timeout}s") from e

    # === Worker loop =======================================================

    async def _run(self) -> None:
        ingest = self._resolve_ingest_fn()
        log.info("job-queue worker started")
        try:
            while True:
                # Pause gate. Awaiting *before* ``_queue.get()`` is what
                # gives "pause = don't start new work" its precise
                # semantics: a job enqueued while paused stays in the
                # queue (and stays in ``pending``) until ``resume()``
                # flips the event, at which point dequeue resumes in
                # submission order. ``CancelledError`` from ``stop()``
                # propagates out of either wait.
                await self._unpaused.wait()
                job = await self._queue.get()
                try:
                    await self._process(job, ingest)
                finally:
                    self._queue.task_done()
                    # Worker just finished a job. If the queue is empty
                    # and no other job is somehow running, flip the
                    # idle event so wait_idle / shutdown can wake.
                    if self.pending == 0:
                        self._idle_event.set()
        except asyncio.CancelledError:
            log.info("job-queue worker cancelled (graceful shutdown)")
            self._idle_event.set()
            raise
        except Exception:  # noqa: BLE001 — never let the worker die silently
            log.exception("job-queue worker crashed; subsequent jobs will not run")
            self._idle_event.set()
            raise

    async def _process(self, job: Job, ingest: IngestCallable) -> None:
        job.status = JobStatus.running
        job.started_at = int(time.time())
        log.info("job %s starting (paths=%d)", job.id, len(job.paths))

        # Initialise progress so a poller hitting the endpoint between
        # ``running`` flip and the first ``progress_cb`` call sees a
        # well-defined zero state (rather than ``null``, which the
        # Swift decoder must already tolerate but the UI rendering
        # path treats as "unknown progress").
        job.progress = {"total": 0, "done": 0, "current": None, "embedded": 0}

        # Aggregate counter so multi-path jobs (rare) report a single
        # ``done / total`` that climbs monotonically. Each ``progress_cb``
        # invocation from a per-path ``ingest_path`` reports *that
        # path's* (done, total, embedded); we add them on top of the
        # previous paths' totals before writing to ``job.progress``.
        prior_total = 0
        prior_done = 0
        prior_embedded = 0

        # Tests inject ingest fns with the legacy ``ingest(path)`` signature.
        # Detect the new kwargs once and avoid passing them when unsupported.
        try:
            sig = inspect.signature(ingest)
            params = sig.parameters
            supports_kwargs = any(
                p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()
            )
            supports_progress_cb = supports_kwargs or "progress_cb" in params
            supports_pause_event = supports_kwargs or "pause_event" in params
        except (TypeError, ValueError):
            # Built-ins / C functions don't support introspection — assume
            # the modern signature; the call below would surface a TypeError
            # the same way it would have without this probe.
            supports_progress_cb = True
            supports_pause_event = True

        def _progress_cb(
            done: int, total: int, current: str | None, embedded: int = 0
        ) -> None:
            job.progress = {
                "total": prior_total + total,
                "done": prior_done + done,
                "current": current,
                "embedded": prior_embedded + embedded,
            }

        try:
            accumulated = {"files": 0, "chunks": 0, "skipped": 0}
            for path in job.paths:
                # Run the (synchronous, embed-heavy) ingest on a worker
                # thread so the event loop stays responsive — search
                # requests on the FastAPI side keep arriving even mid-
                # ingest, and the model singleton already serializes
                # access internally.
                kwargs: dict[str, Any] = {}
                if supports_progress_cb:
                    kwargs["progress_cb"] = _progress_cb
                if supports_pause_event:
                    kwargs["pause_event"] = self._unpaused_thread
                stats = await asyncio.to_thread(ingest, path, **kwargs)
                for k in accumulated:
                    accumulated[k] += int(stats.get(k, 0))
                # Roll the per-path progress into the prior baseline so
                # the next ``ingest_path`` call's progress_cb starts
                # at zero again without rewinding the published ``done``.
                if job.progress is not None:
                    prior_done = int(job.progress.get("done", prior_done))
                    prior_total = int(job.progress.get("total", prior_total))
                    prior_embedded = int(
                        job.progress.get("embedded", prior_embedded)
                    )
            job.stats = accumulated
            job.status = JobStatus.succeeded
            log.info(
                "job %s succeeded: %d file(s), %d chunk(s), %d skipped",
                job.id,
                accumulated["files"],
                accumulated["chunks"],
                accumulated["skipped"],
            )
        except asyncio.CancelledError:
            job.error = "cancelled"
            job.status = JobStatus.failed
            raise
        except Exception as e:  # noqa: BLE001 — surface the failure to the caller
            job.error = f"{type(e).__name__}: {e}"
            job.status = JobStatus.failed
            log.exception("job %s failed", job.id)
        finally:
            job.finished_at = int(time.time())

    # === Internals =========================================================

    def _resolve_ingest_fn(self) -> IngestCallable:
        """Lazy-import ``ingest_path`` so test queues that inject a
        fake ``ingest_fn`` don't pay the import cost of LanceDB +
        the embed model just to verify queue mechanics.
        """
        if self._ingest_fn is not None:
            return self._ingest_fn
        from ..ingest import ingest_path

        # Adapter: forward ``progress_cb`` / ``pause_event`` kwargs the
        # worker passes through to ``ingest_path``; the worker doesn't
        # pass ``table`` (let ``ingest_path`` open the default table
        # itself).
        def _wrap(
            p: str | Path,
            *,
            progress_cb: Callable[..., Any] | None = None,
            pause_event: threading.Event | None = None,
        ) -> dict[str, int]:
            return ingest_path(p, progress_cb=progress_cb, pause_event=pause_event)

        return _wrap

    def _evict_old(self) -> None:
        """Trim ``_jobs`` / ``_order`` to ``max_history`` entries."""
        excess = len(self._order) - self._max_history
        if excess <= 0:
            return
        for _ in range(excess):
            old_id = self._order.pop(0)
            self._jobs.pop(old_id, None)
