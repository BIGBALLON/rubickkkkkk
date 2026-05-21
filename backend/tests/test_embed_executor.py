"""Tests for :mod:`rubick_backend.embed.executor`.

Pure-Python unit coverage of the priority queue + single-worker
executor. No MLX, no model load — we substitute trivial Python
``fn`` callables and verify ordering / exception propagation /
shutdown invariants.

The integration story (HIGH ``embed_query`` actually preempting an
in-flight ingest LOW batch) is implicit in the wrapper code paths
of ``embed/loader.py``; covering it end-to-end would require a real
model load (slow) and timing assertions that flake under CI load.
The unit tests below pin the *mechanism* — that LOW tasks queued
behind a HIGH task release the HIGH task first.
"""

from __future__ import annotations

import threading
import time

import pytest

from rubick_backend.embed.executor import ModelExecutor, Priority


def test_priority_high_runs_before_pending_low() -> None:
    """A LOW task already in flight prevents preemption (non-cooperative),
    but any LOW task still **in the queue** when a HIGH lands gets
    leapfrogged.

    We park the worker on a single in-flight task by holding an
    ``Event``, queue 4 LOW tasks plus 1 HIGH, then release the
    worker. The first task is whichever was first dequeued (LOW #1,
    if FIFO at equal priority), and the next one out must be HIGH
    — *not* LOW #2.
    """
    ex = ModelExecutor(name="test-priority")
    try:
        # Block #1 (LOW, will be picked up immediately by an idle
        # worker) until we say go. While it's parked, we queue more.
        gate = threading.Event()
        order: list[str] = []
        lock = threading.Lock()

        def low(label: str) -> str:
            if label == "first":
                gate.wait(timeout=2.0)
            with lock:
                order.append(label)
            return label

        def high(label: str) -> str:
            with lock:
                order.append(label)
            return label

        first = ex.submit(Priority.LOW, low, "first")
        # Give the worker a moment to start the first task before
        # queueing more, so it's truly in flight.
        time.sleep(0.05)
        for i in range(3):
            ex.submit(Priority.LOW, low, f"low-{i}")
        h = ex.submit(Priority.HIGH, high, "HIGH")

        gate.set()
        first.result(timeout=2.0)
        h.result(timeout=2.0)

        # ``order`` until HIGH must contain ``first`` and possibly
        # *zero* LOWs — the worker should pick HIGH next because
        # it's lowest priority in the queue.
        idx_high = order.index("HIGH")
        assert order[0] == "first"
        assert idx_high == 1, f"HIGH should land right after the in-flight LOW, got order={order}"
    finally:
        ex.shutdown()


def test_equal_priority_runs_in_submission_order() -> None:
    """LOW vs LOW: FIFO. We pin this so future Priority enum
    refactors don't accidentally introduce instability."""
    ex = ModelExecutor(name="test-fifo")
    try:
        order: list[int] = []
        lock = threading.Lock()

        def add(n: int) -> int:
            with lock:
                order.append(n)
            return n

        futures = [ex.submit(Priority.LOW, add, i) for i in range(10)]
        for f in futures:
            f.result(timeout=2.0)
        assert order == list(range(10))
    finally:
        ex.shutdown()


def test_exception_is_propagated_via_future() -> None:
    ex = ModelExecutor(name="test-exc")
    try:

        def boom() -> None:
            raise ValueError("boom!")

        fut = ex.submit(Priority.HIGH, boom)
        with pytest.raises(ValueError, match="boom!"):
            fut.result(timeout=2.0)
    finally:
        ex.shutdown()


def test_call_blocks_and_returns_value() -> None:
    """``ModelExecutor.call`` is the sync ergonomic wrapper used by
    ``embed_query`` etc. Verify it round-trips a return value."""
    ex = ModelExecutor(name="test-call")
    try:
        out = ex.call(Priority.HIGH, lambda x, y: x + y, 7, 8)
        assert out == 15
    finally:
        ex.shutdown()


def test_submit_after_shutdown_raises() -> None:
    ex = ModelExecutor(name="test-shutdown")
    ex.shutdown()
    with pytest.raises(RuntimeError, match="shut down"):
        ex.submit(Priority.LOW, lambda: 1)


def test_shutdown_cancels_pending_futures() -> None:
    """Tasks queued but not yet started must come back cancelled,
    not deadlock-forever."""
    ex = ModelExecutor(name="test-cancel-pending")
    gate = threading.Event()

    def park() -> None:
        gate.wait(timeout=2.0)

    # Park the worker, queue 3 more, then shutdown.
    blocker = ex.submit(Priority.LOW, park)
    time.sleep(0.05)
    pending = [ex.submit(Priority.LOW, lambda: 1) for _ in range(3)]
    gate.set()
    ex.shutdown()

    blocker.result(timeout=2.0)
    for f in pending:
        # Cancelled futures raise ``CancelledError`` from .result().
        # Some may have already started running before shutdown saw
        # them; accept either cancelled or completed.
        if f.cancelled():
            continue
        # If not cancelled it must have completed normally.
        assert f.done()


def test_priority_enum_ordering() -> None:
    """Sanity-check: HIGH must be a smaller integer than LOW (lower =
    runs first), and Priority must be Enum-comparable as plain int."""
    assert int(Priority.HIGH) < int(Priority.LOW)
    assert (int(Priority.HIGH), 0) < (int(Priority.LOW), 0)
