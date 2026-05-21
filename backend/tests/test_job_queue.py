"""Unit + integration tests for :mod:`rubick_backend.worker.job_queue`
and the ``/index/...`` HTTP routes.

Fast tests stub the ``ingest_fn`` so we never load the embed model or
touch LanceDB. The single slow test (marked) exercises the full
end-to-end path: POST /index/job → backend ingests → /search returns
the new rows. This pair gives us:

- O(ms) coverage of queue mechanics for CI / pre-commit
- One belt-and-suspenders smoke that the wiring (FastAPI ↔ queue ↔
  ingest pipeline) is correct, gated behind ``RUBICK_RUN_SLOW=1``
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from rubick_backend.worker.job_queue import JobQueue, JobStatus

pytestmark = pytest.mark.asyncio


# === Fakes ==================================================================


def make_fake_ingest(*, files_per_path: int = 1, chunks_per_path: int = 1):
    """Return a fake ``ingest_fn`` that doesn't touch the disk / model.

    Tracks calls in a list so tests can assert what got dispatched.
    """
    calls: list[Path] = []

    def _fake(path):  # noqa: ANN001 — match IngestCallable signature
        calls.append(Path(path))
        return {
            "files": files_per_path,
            "chunks": chunks_per_path,
            "skipped": 0,
        }

    _fake.calls = calls  # type: ignore[attr-defined]
    return _fake


def make_slow_ingest(delay_s: float = 0.1):
    """Fake that blocks for ``delay_s`` seconds — lets tests time the
    queue serializing behaviour and verify ``pending`` transitions."""

    def _fake(path):  # noqa: ANN001
        time.sleep(delay_s)
        return {"files": 1, "chunks": 1, "skipped": 0}

    return _fake


# === Queue mechanics ========================================================


async def test_enqueue_returns_queued_job() -> None:
    queue = JobQueue(ingest_fn=make_fake_ingest())
    await queue.start()
    try:
        job = await queue.enqueue(paths=["/tmp/x"])
        # Note: status may already be running or succeeded by the time
        # we get here — the worker drains very quickly with our fake.
        # The only thing we can deterministically assert is that an
        # id was assigned and the job is registered.
        assert len(job.id) == 12
        assert queue.get(job.id) is job
    finally:
        await queue.stop()


async def test_enqueue_rejects_empty_paths() -> None:
    queue = JobQueue(ingest_fn=make_fake_ingest())
    await queue.start()
    try:
        with pytest.raises(ValueError, match="non-empty"):
            await queue.enqueue(paths=[])
    finally:
        await queue.stop()


async def test_worker_drains_and_marks_succeeded() -> None:
    fake = make_fake_ingest(files_per_path=3, chunks_per_path=5)
    queue = JobQueue(ingest_fn=fake)
    await queue.start()
    try:
        job = await queue.enqueue(paths=["/tmp/a", "/tmp/b"])
        await queue.wait_idle(timeout=1.0)

        final = queue.get(job.id)
        assert final is not None
        assert final.status is JobStatus.succeeded
        # accumulator over 2 paths
        assert final.stats == {"files": 6, "chunks": 10, "skipped": 0}
        assert final.started_at is not None
        assert final.finished_at is not None
        assert final.finished_at >= final.started_at
        # The fake was called once per path, in order.
        assert [str(p) for p in fake.calls] == ["/tmp/a", "/tmp/b"]
    finally:
        await queue.stop()


async def test_worker_marks_failed_on_ingest_exception() -> None:
    def boom(_path):  # noqa: ANN001
        raise RuntimeError("disk on fire")

    queue = JobQueue(ingest_fn=boom)
    await queue.start()
    try:
        job = await queue.enqueue(paths=["/tmp/x"])
        await queue.wait_idle(timeout=1.0)
        final = queue.get(job.id)
        assert final is not None
        assert final.status is JobStatus.failed
        assert final.error is not None
        assert "disk on fire" in final.error
        assert final.stats is None
    finally:
        await queue.stop()


async def test_jobs_are_processed_serially() -> None:
    """Two jobs enqueued back-to-back must run in submission order
    (we only have one MLX model). With ~0.1 s per job, 3 jobs
    should take >=0.3 s wall-clock."""
    queue = JobQueue(ingest_fn=make_slow_ingest(delay_s=0.05))
    await queue.start()
    try:
        t0 = time.perf_counter()
        ids = []
        for _ in range(3):
            j = await queue.enqueue(paths=["/tmp/x"])
            ids.append(j.id)
        await queue.wait_idle(timeout=2.0)
        elapsed = time.perf_counter() - t0
        assert elapsed >= 0.05 * 3 * 0.9, f"expected serial >=0.135s, got {elapsed:.3f}s"
        for jid in ids:
            assert queue.get(jid).status is JobStatus.succeeded
    finally:
        await queue.stop()


async def test_list_recent_returns_newest_first() -> None:
    queue = JobQueue(ingest_fn=make_fake_ingest())
    await queue.start()
    try:
        first = await queue.enqueue(paths=["/tmp/a"])
        second = await queue.enqueue(paths=["/tmp/b"])
        third = await queue.enqueue(paths=["/tmp/c"])
        await queue.wait_idle(timeout=1.0)
        recent = queue.list_recent(limit=10)
        ids = [j.id for j in recent]
        assert ids == [third.id, second.id, first.id]
    finally:
        await queue.stop()


async def test_history_is_lru_evicted_past_cap() -> None:
    queue = JobQueue(ingest_fn=make_fake_ingest(), max_history=3)
    await queue.start()
    try:
        keep_ids: list[str] = []
        for i in range(5):
            j = await queue.enqueue(paths=[f"/tmp/x{i}"])
            keep_ids.append(j.id)
        await queue.wait_idle(timeout=1.0)
        # First two should have been evicted; last three remain.
        assert queue.get(keep_ids[0]) is None
        assert queue.get(keep_ids[1]) is None
        for jid in keep_ids[2:]:
            assert queue.get(jid) is not None
    finally:
        await queue.stop()


# === Progress field =========================================================


async def test_job_progress_field_updates_during_run() -> None:
    """An ingest fn that accepts ``progress_cb`` must drive
    ``Job.progress`` in real time so a poll mid-run can render
    "N / total" with the in-flight current filename."""
    # Snapshots collected mid-run by polling from the test loop while
    # the worker is sleeping inside the fn. Using a ``threading.Event``
    # to coordinate so we never race the worker's progress writes.
    import threading

    file_2_done = threading.Event()
    test_observed = threading.Event()

    def fake_with_progress(path, *, progress_cb=None, pause_event=None):  # noqa: ANN001
        if progress_cb is not None:
            progress_cb(0, 3, None, 0)
            for i, name in enumerate(["a.md", "b.md", "c.md"], start=1):
                progress_cb(i, 3, f"{path}/{name}", i)
                if i == 2:
                    # Pause inside the worker thread so the event-loop
                    # poll below sees a stable mid-run snapshot.
                    file_2_done.set()
                    test_observed.wait(timeout=1.0)
        return {"files": 3, "chunks": 3, "skipped": 0}

    queue = JobQueue(ingest_fn=fake_with_progress)
    await queue.start()
    try:
        job = await queue.enqueue(paths=["/tmp/notes"])

        # Mid-run snapshot: wait for file 2 to publish, observe, release.
        await asyncio.get_running_loop().run_in_executor(None, file_2_done.wait, 1.0)
        mid = queue.get(job.id)
        assert mid is not None
        assert mid.progress is not None
        assert mid.progress["total"] == 3
        assert mid.progress["done"] == 2
        assert mid.progress["current"] == "/tmp/notes/b.md"
        assert mid.progress["embedded"] == 2
        test_observed.set()

        await queue.wait_idle(timeout=1.0)
        final = queue.get(job.id)
        assert final is not None
        assert final.status is JobStatus.succeeded
        # After completion the last published progress should hold.
        assert final.progress == {
            "total": 3,
            "done": 3,
            "current": "/tmp/notes/c.md",
            "embedded": 3,
        }
    finally:
        await queue.stop()


async def test_job_progress_aggregates_across_multi_path_jobs() -> None:
    """A job spanning multiple paths must report a monotone (done /
    total) — the per-path callback restarts at zero, but the published
    progress must accumulate so the UI never sees the denominator
    decrease mid-run."""

    def fake(path, *, progress_cb=None, pause_event=None):  # noqa: ANN001
        if progress_cb is not None:
            progress_cb(0, 2, None, 0)
            progress_cb(1, 2, f"{path}/a", 1)
            progress_cb(2, 2, f"{path}/b", 2)
        return {"files": 2, "chunks": 2, "skipped": 0}

    queue = JobQueue(ingest_fn=fake)
    await queue.start()
    try:
        job = await queue.enqueue(paths=["/tmp/p1", "/tmp/p2"])
        await queue.wait_idle(timeout=1.0)

        final = queue.get(job.id)
        assert final is not None
        # Both paths contribute 2 files each → grand total 4.
        assert final.progress == {
            "total": 4,
            "done": 4,
            "current": "/tmp/p2/b",
            "embedded": 4,
        }
        assert final.stats == {"files": 4, "chunks": 4, "skipped": 0}
    finally:
        await queue.stop()


async def test_job_progress_falls_back_for_legacy_ingest_fn() -> None:
    """A legacy ingest fn that doesn't accept ``progress_cb`` /
    ``pause_event`` must still work — the worker introspects the
    signature and skips the kwargs when they're not supported.
    Existing test stubs across the codebase rely on this."""

    def legacy(path):  # noqa: ANN001 — explicitly the old shape
        return {"files": 1, "chunks": 1, "skipped": 0}

    queue = JobQueue(ingest_fn=legacy)
    await queue.start()
    try:
        job = await queue.enqueue(paths=["/tmp/x"])
        await queue.wait_idle(timeout=1.0)
        final = queue.get(job.id)
        assert final is not None
        assert final.status is JobStatus.succeeded
        # Progress was initialised to a zero state but never advanced
        # because the legacy fn didn't fire ``progress_cb``.
        assert final.progress == {
            "total": 0, "done": 0, "current": None, "embedded": 0,
        }
    finally:
        await queue.stop()


# === Pause / resume =========================================================


async def test_pause_blocks_new_jobs_until_resume() -> None:
    """Enqueuing while paused must not drain — the worker is blocked
    on the pause gate and ``pending`` stays > 0 until ``resume()``
    flips the event. Re-running on resume must drain in submission
    order (FIFO is preserved across the gate).
    """
    fake = make_fake_ingest()
    queue = JobQueue(ingest_fn=fake)
    await queue.start()
    try:
        queue.pause()
        assert queue.paused is True

        await queue.enqueue(paths=["/tmp/a"])
        await queue.enqueue(paths=["/tmp/b"])

        # Give the worker plenty of opportunity to (incorrectly) drain.
        # Two enqueues × ~instant fake means a 100 ms grace is huge
        # headroom on any laptop.
        await asyncio.sleep(0.1)
        assert queue.pending == 2, "paused worker drained the queue"
        assert [str(p) for p in fake.calls] == []

        queue.resume()
        assert queue.paused is False

        await queue.wait_idle(timeout=1.0)
        assert [str(p) for p in fake.calls] == ["/tmp/a", "/tmp/b"]
    finally:
        await queue.stop()


async def test_pause_does_not_cancel_in_flight_job() -> None:
    """Calling ``pause()`` mid-ingest must let the current job
    finish — the gate only governs the *next* dequeue. A second
    job enqueued before resume must stay queued.
    """
    started = asyncio.Event()

    def slow(_path):  # noqa: ANN001
        started.set()
        time.sleep(0.05)
        return {"files": 1, "chunks": 1, "skipped": 0}

    queue = JobQueue(ingest_fn=slow)
    await queue.start()
    try:
        first = await queue.enqueue(paths=["/tmp/a"])
        await started.wait()  # worker is inside ``slow`` now
        queue.pause()

        second = await queue.enqueue(paths=["/tmp/b"])

        # Give the in-flight job a moment to finish.
        await asyncio.sleep(0.2)

        assert queue.get(first.id).status is JobStatus.succeeded
        # ``second`` is still ``queued`` because the pause gate
        # short-circuited the next ``_queue.get()``.
        assert queue.get(second.id).status is JobStatus.queued

        queue.resume()
        await queue.wait_idle(timeout=1.0)
        assert queue.get(second.id).status is JobStatus.succeeded
    finally:
        await queue.stop()


async def test_pause_blocks_inside_per_file_loop() -> None:
    """Mid-folder pause must take effect within ~one file. The worker
    threads its ``_unpaused_thread`` event into the ingest fn; when
    cleared, a per-file ``Event.wait()`` blocks the loop. ``resume()``
    must release every blocked file in the same call."""
    file_done = asyncio.Event()
    loop = asyncio.get_running_loop()

    def ingest_with_pause(path, *, progress_cb=None, pause_event=None):  # noqa: ANN001
        # Process two synthetic files; check the pause event before each.
        for i, name in enumerate(["a.md", "b.md"], start=1):
            if pause_event is not None:
                pause_event.wait()
            if i == 1:
                # Signal back to the test that file 1 has cleared the
                # gate so the test can pause *before* file 2.
                loop.call_soon_threadsafe(file_done.set)
                # Tiny sleep so the test's ``pause()`` call lands before
                # we look at the event again on the next iteration.
                time.sleep(0.05)
            if progress_cb is not None:
                progress_cb(i, 2, f"{path}/{name}", i)
        return {"files": 2, "chunks": 2, "skipped": 0}

    queue = JobQueue(ingest_fn=ingest_with_pause)
    await queue.start()
    try:
        job = await queue.enqueue(paths=["/tmp/folder"])

        # Wait until file 1 finished, then immediately pause. The
        # worker is now inside the ingest fn but blocked on the
        # ``Event.wait()`` before file 2.
        await asyncio.wait_for(file_done.wait(), timeout=1.0)
        queue.pause()

        # Generous grace period — if the gate were broken, file 2
        # would race through here and ``done`` would land at 2.
        await asyncio.sleep(0.2)
        in_flight = queue.get(job.id)
        assert in_flight is not None
        # Could be 1 (paused before 2nd progress_cb fires) — never 2.
        assert (in_flight.progress or {}).get("done", 0) < 2

        queue.resume()
        await queue.wait_idle(timeout=1.0)
        final = queue.get(job.id)
        assert final is not None
        assert final.status is JobStatus.succeeded
        assert final.progress == {
            "total": 2,
            "done": 2,
            "current": "/tmp/folder/b.md",
            "embedded": 2,
        }
    finally:
        await queue.stop()


async def test_pause_resume_are_idempotent() -> None:
    queue = JobQueue(ingest_fn=make_fake_ingest())
    await queue.start()
    try:
        assert queue.paused is False
        queue.pause()
        queue.pause()  # second call is a no-op
        assert queue.paused is True
        queue.resume()
        queue.resume()
        assert queue.paused is False
    finally:
        await queue.stop()


async def test_stop_unblocks_paused_worker() -> None:
    """A paused worker waiting on the gate must still respond to
    ``stop()`` (``CancelledError`` propagates out of
    ``_unpaused.wait()``). Without this the lifespan shutdown would
    hang forever if the user quit Rubick mid-pause.
    """
    queue = JobQueue(ingest_fn=make_fake_ingest())
    await queue.start()
    queue.pause()
    await queue.enqueue(paths=["/tmp/x"])
    await asyncio.sleep(0.05)
    # Should return promptly even though the worker is parked on the gate.
    await asyncio.wait_for(queue.stop(), timeout=1.0)


async def test_stop_is_idempotent() -> None:
    queue = JobQueue(ingest_fn=make_fake_ingest())
    await queue.start()
    await queue.stop()
    await queue.stop()  # should be a no-op, not raise


async def test_stop_cancels_in_flight_job() -> None:
    """A job that's mid-sleep when we call stop() lands in failed
    state with ``error="cancelled"`` — *not* succeeded. This is how
    SIGTERM during ingest looks from the outside."""

    started = asyncio.Event()

    def really_slow(_path):  # noqa: ANN001
        started.set()
        time.sleep(5.0)
        return {"files": 1, "chunks": 1, "skipped": 0}

    queue = JobQueue(ingest_fn=really_slow)
    await queue.start()
    job = await queue.enqueue(paths=["/tmp/x"])
    await started.wait()  # in-flight
    await queue.stop()
    final = queue.get(job.id)
    assert final is not None
    assert final.status is JobStatus.failed
    assert final.error == "cancelled"


# === HTTP layer (fast — uses fake ingest via app.state override) ============


@pytest.fixture()
def app_with_fake_queue():
    """Build a FastAPI app whose lifespan installs a queue with a
    fake ``ingest_fn``. We do this by importing the app's lifespan
    *contents* manually, because patching ``JobQueue`` at the
    ``main`` import boundary would also affect other tests.
    """
    from fastapi import FastAPI

    from rubick_backend.api import index as index_api

    fake = make_fake_ingest(files_per_path=2, chunks_per_path=4)

    app = FastAPI()
    app.include_router(index_api.router)
    return app, fake


async def test_post_index_job_enqueues_and_returns_202(
    tmp_path: Path,
    app_with_fake_queue,
) -> None:
    app, fake = app_with_fake_queue
    queue = JobQueue(ingest_fn=fake)
    app.state.job_queue = queue
    await queue.start()
    try:
        target = tmp_path / "notes"
        target.mkdir()

        with TestClient(app) as client:
            r = client.post("/index/job", json={"paths": [str(target)]})
            assert r.status_code == 202, r.text
            body = r.json()
            assert body["status"] == "queued"
            assert len(body["id"]) == 12
            assert body["paths"] == [str(target)]
    finally:
        await queue.stop()


async def test_post_index_job_rejects_missing_path(
    tmp_path: Path,
    app_with_fake_queue,
) -> None:
    app, fake = app_with_fake_queue
    queue = JobQueue(ingest_fn=fake)
    app.state.job_queue = queue
    await queue.start()
    try:
        nonexistent = tmp_path / "does-not-exist"
        with TestClient(app) as client:
            r = client.post("/index/job", json={"paths": [str(nonexistent)]})
            assert r.status_code == 422
            assert "not found" in r.json()["detail"]
    finally:
        await queue.stop()


async def test_post_index_job_503_when_queue_missing(
    tmp_path: Path,
    app_with_fake_queue,
) -> None:
    app, _fake = app_with_fake_queue
    # Don't attach a queue.
    target = tmp_path / "notes"
    target.mkdir()
    with TestClient(app) as client:
        r = client.post("/index/job", json={"paths": [str(target)]})
        assert r.status_code == 503


async def test_get_index_job_404_for_unknown_id(
    app_with_fake_queue,
) -> None:
    app, fake = app_with_fake_queue
    queue = JobQueue(ingest_fn=fake)
    app.state.job_queue = queue
    await queue.start()
    try:
        with TestClient(app) as client:
            r = client.get("/index/job/abcdef123456")
            assert r.status_code == 404
    finally:
        await queue.stop()


async def test_post_then_poll_until_succeeded(
    tmp_path: Path,
    app_with_fake_queue,
) -> None:
    app, fake = app_with_fake_queue
    queue = JobQueue(ingest_fn=fake)
    app.state.job_queue = queue
    await queue.start()
    try:
        target = tmp_path / "notes"
        target.mkdir()
        with TestClient(app) as client:
            r = client.post("/index/job", json={"paths": [str(target)]})
            job_id = r.json()["id"]
            await queue.wait_idle(timeout=1.0)
            r2 = client.get(f"/index/job/{job_id}")
            assert r2.status_code == 200
            body = r2.json()
            assert body["status"] == "succeeded"
            assert body["stats"] == {"files": 2, "chunks": 4, "skipped": 0}
            assert body["finished_at"] is not None
    finally:
        await queue.stop()


async def test_pause_and_resume_endpoints(
    tmp_path: Path,
    app_with_fake_queue,
) -> None:
    """End-to-end of the new ``POST /index/{pause,resume}`` +
    ``GET /index/status`` trio: pause stops drain, status reflects
    pause + pending count, resume drains the backlog.
    """
    app, fake = app_with_fake_queue
    queue = JobQueue(ingest_fn=fake)
    app.state.job_queue = queue
    await queue.start()
    try:
        target = tmp_path / "notes"
        target.mkdir()

        with TestClient(app) as client:
            r0 = client.get("/index/status")
            assert r0.status_code == 200
            assert r0.json() == {"paused": False, "pending": 0}

            r1 = client.post("/index/pause")
            assert r1.status_code == 200
            assert r1.json()["paused"] is True

            client.post("/index/job", json={"paths": [str(target)]})
            client.post("/index/job", json={"paths": [str(target)]})

            await asyncio.sleep(0.1)
            r2 = client.get("/index/status")
            body = r2.json()
            assert body["paused"] is True
            assert body["pending"] == 2

            r3 = client.post("/index/resume")
            assert r3.status_code == 200
            assert r3.json()["paused"] is False

            await queue.wait_idle(timeout=1.0)
            r4 = client.get("/index/status")
            assert r4.json() == {"paused": False, "pending": 0}
    finally:
        await queue.stop()


async def test_pause_503_when_queue_missing(
    app_with_fake_queue,
) -> None:
    """The pause / resume / status trio shares ``_require_queue`` with
    the rest of the ``/index/...`` routes — verify it surfaces 503
    before the lifespan attaches a queue so the Swift client can
    differentiate "backend booting" from "backend rejected the call".
    """
    app, _fake = app_with_fake_queue
    with TestClient(app) as client:
        for path in ("/index/pause", "/index/resume"):
            r = client.post(path)
            assert r.status_code == 503
        assert client.get("/index/status").status_code == 503


async def test_list_jobs_returns_newest_first_and_pending_count(
    tmp_path: Path,
    app_with_fake_queue,
) -> None:
    app, fake = app_with_fake_queue
    queue = JobQueue(ingest_fn=fake)
    app.state.job_queue = queue
    await queue.start()
    try:
        target = tmp_path / "notes"
        target.mkdir()
        with TestClient(app) as client:
            client.post("/index/job", json={"paths": [str(target)]})
            client.post("/index/job", json={"paths": [str(target)]})
            await queue.wait_idle(timeout=1.0)
            r = client.get("/index/jobs?limit=10")
            body = r.json()
            assert body["count"] == 2
            assert body["pending"] == 0
            assert body["jobs"][0]["enqueued_at"] >= body["jobs"][1]["enqueued_at"]
    finally:
        await queue.stop()
