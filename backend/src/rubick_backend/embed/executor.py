"""Priority-aware single-threaded executor for the MLX model.

ARCHITECTURE.md (Process model) calls for a single MLX model
instance behind an ``asyncio.PriorityQueue`` so foreground search
preempts background indexing. We implement that with a plain
``queue.PriorityQueue`` + one dedicated worker thread:

- **One MLX model** lives in process memory (jina v5 omni nano,
  1.8 GB). Concurrent forward passes from multiple threads aren't
  worth chasing — Metal serializes them on the same command queue
  anyway. Funneling everything through *one* worker thread makes
  scheduling predictable and removes a class of thread-safety
  worries on the model.
- **Priorities** are an integer; lower = higher priority. We
  expose two named slots: ``HIGH`` for ``embed_query`` (foreground
  search) and ``LOW`` for the ingest pipelines' ``embed_document``
  / ``embed_image`` / ``embed_video``.
- **Preemption is non-cooperative**: once a LOW task starts running
  it finishes before HIGH can run. Worst case ingest is a single
  video chunk (~0.4 s) or image embed (~2 s); LOW tasks are
  bounded, so a query waits at most one in-flight chunk instead
  of the whole ingest backlog.

The executor is a process-wide singleton (``get_executor``) lazily
started on first submission. It runs as a daemon thread so process
exit doesn't block on it. A graceful ``shutdown`` exists for tests.

Public surface::

    executor = get_executor()
    future = executor.submit(Priority.HIGH, fn, *args, **kwargs)
    result = future.result()  # blocking; raises if fn did
"""

from __future__ import annotations

import itertools
import logging
import queue
import threading
from collections.abc import Callable
from concurrent.futures import Future
from enum import IntEnum
from typing import Any

log = logging.getLogger(__name__)


class Priority(IntEnum):
    """Numerical priority — lower runs first.

    The gap between HIGH and LOW is deliberately large so future
    intermediate levels (e.g. MEDIUM for user-triggered "Re-index
    this file now") slot in cleanly without churning enum values.
    """

    HIGH = 0
    LOW = 10


class ModelExecutor:
    """Submit work to a single background thread, prioritized.

    Multiple ``ModelExecutor`` instances are allowed (tests use
    short-lived ones with stub fns) but in production exactly one
    is constructed via ``get_executor()``.
    """

    def __init__(self, *, name: str = "rubick-model") -> None:
        # ``queue.PriorityQueue`` returns the *smallest* tuple. Each
        # entry is ``(priority, seq, kind)`` where seq enforces FIFO
        # ordering for equal priorities — without it, ties depend on
        # ``Future``/``Callable`` comparability, which is undefined.
        self._queue: queue.PriorityQueue[tuple[int, int, Any]] = queue.PriorityQueue()
        self._seq = itertools.count()
        self._shutdown = False
        self._thread = threading.Thread(target=self._run, daemon=True, name=name)
        self._thread.start()

    # === Public API ========================================================

    def submit(
        self,
        priority: Priority | int,
        fn: Callable[..., Any],
        /,
        *args: Any,
        **kwargs: Any,
    ) -> Future:
        """Enqueue ``fn(*args, **kwargs)`` for execution on the worker thread.

        Returns a ``Future`` whose ``.result()`` blocks until ``fn``
        either returns or raises. ``priority`` is one of ``Priority.HIGH``
        / ``Priority.LOW`` (or any plain ``int`` for forward-compat).
        Lower numbers preempt later in the queue but **not** a task
        that's already running.
        """
        if self._shutdown:
            raise RuntimeError("ModelExecutor is shut down")
        fut: Future = Future()
        seq = next(self._seq)
        self._queue.put((int(priority), seq, (fut, fn, args, kwargs)))
        return fut

    def call(
        self,
        priority: Priority | int,
        fn: Callable[..., Any],
        /,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """Submit and block on the result — convenience for sync callers.

        ``embed_query`` / ``embed_document`` use this internally so
        the existing sync API surface stays unchanged for the ingest
        pipelines and the FastAPI route.
        """
        return self.submit(priority, fn, *args, **kwargs).result()

    def shutdown(self, *, wait: bool = True, timeout: float | None = 5.0) -> None:
        """Stop the worker thread after draining in-flight work.

        Pending (not-yet-started) tasks have their futures cancelled.
        Designed for tests; production callers can rely on the daemon
        thread dying with the process.
        """
        if self._shutdown:
            return
        self._shutdown = True
        # Drain remaining entries *first* — cancel their futures so
        # callers don't deadlock on ``.result()``. We then put the
        # SENTINEL last so the worker either picks it up directly
        # (currently idle on queue.get) or picks it up after its
        # in-flight task completes. Doing this in the opposite order
        # would consume the SENTINEL during drain and the worker
        # would block on ``queue.get`` forever.
        while True:
            try:
                _, _, payload = self._queue.get_nowait()
            except queue.Empty:
                break
            if payload is _SENTINEL_SHUTDOWN:
                continue
            fut, _, _, _ = payload
            fut.cancel()
        self._queue.put((Priority.HIGH, -1, _SENTINEL_SHUTDOWN))
        if wait:
            self._thread.join(timeout=timeout)

    # === Worker loop =======================================================

    def _run(self) -> None:
        while True:
            _, _, payload = self._queue.get()
            if payload is _SENTINEL_SHUTDOWN:
                return
            fut, fn, args, kwargs = payload
            if not fut.set_running_or_notify_cancel():
                # Cancelled before we got to it — skip.
                continue
            try:
                result = fn(*args, **kwargs)
            except BaseException as e:  # noqa: BLE001 — propagate everything
                fut.set_exception(e)
            else:
                fut.set_result(result)


# Sentinel used by ``shutdown`` to nudge the worker out of ``queue.get``.
_SENTINEL_SHUTDOWN: Any = object()


# === Process-wide singleton ================================================

_executor_lock = threading.Lock()
_executor: ModelExecutor | None = None


def get_executor() -> ModelExecutor:
    """Lazy singleton accessor. Thread-safe."""
    global _executor
    if _executor is not None:
        return _executor
    with _executor_lock:
        if _executor is None:
            _executor = ModelExecutor()
        return _executor


def reset_executor_for_tests() -> None:
    """Tear down the singleton — only for tests that need a clean slate."""
    global _executor
    with _executor_lock:
        if _executor is not None:
            _executor.shutdown()
            _executor = None
