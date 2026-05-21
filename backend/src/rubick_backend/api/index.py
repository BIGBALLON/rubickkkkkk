"""``/index/...`` routes — enqueue and inspect ingest jobs.

The Swift main process posts paths here when its FSEvents monitor
notices new / changed files, and again on user "Add folder…" UI
actions. The backend turns each request into a :class:`Job` on the
shared :class:`JobQueue` and returns its id immediately. Polling
``GET /index/job/{id}`` yields the live status / stats.

Why not stream progress via SSE for v1: ingest progress callbacks
would require threading a callback through every per-modality
pipeline. We defer that work and let the client poll on a sane
interval (every 500 ms during an active ingest; back off when no
job is running). Adding SSE later is purely additive — existing
clients keep working.
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any

from fastapi import APIRouter, HTTPException, Query, Request, status
from pydantic import BaseModel, Field

from ..store.schema import delete_by_path_prefix, drop_all_index_data, index_stats

log = logging.getLogger(__name__)

router = APIRouter(prefix="/index", tags=["index"])


class IndexJobRequest(BaseModel):
    """Body schema for ``POST /index/job``.

    A list of absolute paths (file or directory). The backend
    expands ``~`` and resolves before walking; relative paths are
    interpreted relative to the backend process's cwd, which is
    typically the ``backend/`` folder when launched by the Swift
    main process — so always send absolute from the Swift side.
    """

    paths: list[str] = Field(..., min_length=1, max_length=64)


@router.post(
    "/job",
    status_code=status.HTTP_202_ACCEPTED,
    summary="Enqueue an ingest job",
)
async def enqueue_job(payload: IndexJobRequest, request: Request) -> dict[str, Any]:
    """Accept paths for asynchronous indexing.

    Returns immediately with ``{"job_id", "status", "paths"}``. The
    caller should poll ``GET /index/job/{id}`` for progress.

    422 if ``paths`` is empty or > 64 entries; 503 if the queue
    isn't running (server still booting).
    """
    queue = _require_queue(request)

    # Validate that every supplied path resolves to something on disk
    # *now*. The worker re-validates later (file could disappear
    # between enqueue and run), but failing fast here gives the
    # client an actionable 422 instead of a "succeeded with 0 files"
    # job ten seconds later.
    #
    # ``os.path.exists`` does a stat() syscall — technically blocking
    # in an asyncio handler, but a single stat is sub-millisecond on
    # any modern filesystem so the event-loop hit is negligible. The
    # alternative (``run_in_executor``) would be overkill for v1.
    missing: list[str] = []
    for raw in payload.paths:
        expanded = os.path.expanduser(raw)  # noqa: ASYNC240 — pure str
        if not os.path.exists(expanded):  # noqa: ASYNC240 — single stat, sub-ms
            missing.append(raw)
    if missing:
        # Starlette < 0.40 spelled it ``_ENTITY``; ≥ 0.40 renamed to
        # ``_CONTENT`` and deprecated the old name. Use the
        # constant the installed version exposes, fallback to 422.
        raise HTTPException(
            status_code=getattr(
                status,
                "HTTP_422_UNPROCESSABLE_CONTENT",
                getattr(status, "HTTP_422_UNPROCESSABLE_ENTITY", 422),
            ),
            detail=f"path(s) not found: {missing}",
        )

    job = await queue.enqueue(paths=payload.paths)
    return job.to_dict()


@router.get(
    "/job/{job_id}",
    summary="Get one job's status / stats",
)
async def get_job(job_id: str, request: Request) -> dict[str, Any]:
    queue = _require_queue(request)
    job = queue.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"unknown job_id {job_id!r}")
    return job.to_dict()


@router.get(
    "/jobs",
    summary="Recent jobs (newest first)",
)
async def list_jobs(request: Request, limit: int = 20) -> dict[str, Any]:
    """Lightweight dashboard endpoint. Mostly used for the dev UI
    and tests; the Swift main process polls a specific id."""
    queue = _require_queue(request)
    jobs = queue.list_recent(limit=limit)
    return {
        "count": len(jobs),
        "pending": queue.pending,
        "jobs": [j.to_dict() for j in jobs],
    }


@router.delete(
    "/by-path-prefix",
    summary="Delete every chunk whose path starts with the given prefix",
)
async def delete_prefix(
    prefix: str = Query(
        ...,
        description=(
            "Absolute path prefix; rows whose ``file_paths[1]`` starts with "
            "this string are removed. Used when clearing chunks under a "
            "watched-folder path (e.g. re-index). Refused for "
            "``/`` or single-character values — see helper for the rule. "
            "(We don't ``min_length=2`` at the FastAPI layer because the "
            "helper's error message is more actionable than a generic 422.)"
        ),
    ),
) -> dict[str, Any]:
    """Bulk-delete by path prefix. Returns counts of what was removed.

    The Swift side typically passes a watched-folder URL when re-indexing.
    The endpoint itself is generic; the helper raises
    ``ValueError`` (→ HTTP 400) for obviously dangerous prefixes
    like ``/``.
    """
    try:
        return await asyncio.to_thread(delete_by_path_prefix, prefix)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@router.delete(
    "/all",
    summary="Drop the entire index (all chunks, all modalities)",
)
async def delete_all() -> dict[str, Any]:
    """Clear all indexed data — drops the LanceDB table and recreates it empty.

    Used by the Settings → Index "Clear All Index Data" button.
    Also clears the path-mtime cache and nebula map file so
    re-indexing starts fully fresh.
    """
    return await asyncio.to_thread(drop_all_index_data)


@router.post(
    "/pause",
    summary="Pause ingest-job dispatch",
)
async def pause_index(request: Request) -> dict[str, Any]:
    """Stop the worker from picking up new ingest jobs.

    In-flight ingest is **not** cancelled — current work completes; only
    the next ``queue.get()`` blocks until ``/index/resume`` flips it
    back. Jobs enqueued while paused (FSEvents callbacks, manual
    "Re-scan" clicks) accumulate in the queue and drain in submission
    order once the user resumes.

    Idempotent — pausing an already-paused queue returns the same
    snapshot. Returns the live counts so the Swift UI can render the
    "N jobs waiting" hint next to the Resume button without a second
    round-trip.
    """
    queue = _require_queue(request)
    queue.pause()
    return {"paused": queue.paused, "pending": queue.pending}


@router.post(
    "/resume",
    summary="Resume ingest-job dispatch",
)
async def resume_index(request: Request) -> dict[str, Any]:
    """Allow the worker to fetch the next queued job.

    Idempotent — resuming a running queue is a no-op. Pending jobs
    (enqueued while paused) drain in submission order.
    """
    queue = _require_queue(request)
    queue.resume()
    return {"paused": queue.paused, "pending": queue.pending}


@router.get(
    "/status",
    summary="Queue lifecycle status (paused + pending)",
)
async def queue_status(request: Request) -> dict[str, Any]:
    """Light-weight polling endpoint for "is indexing paused?".

    Separate from ``/index/stats`` (which counts LanceDB rows and
    requires a ``count_rows`` / ``to_pandas`` round-trip) because
    Pause / Resume can flip independently of any row-level change —
    callers want a cheap answer to "should I show the Pause or the
    Resume button right now?". This endpoint is two attribute lookups.
    """
    queue = _require_queue(request)
    return {"paused": queue.paused, "pending": queue.pending}


@router.get(
    "/stats",
    summary="Aggregate index counts (for the Settings → Index tab)",
)
async def get_stats(
    path_prefix: str | None = Query(
        None,
        description=(
            "Optional path-prefix filter (v1.x). When set, counts only "
            "docs whose canonical first path starts with this string. "
            "The Watched-folders sidebar uses this to re-derive each "
            "folder's items / chunks line after an app restart wipes "
            "the in-memory per-folder stats."
        ),
    ),
) -> dict[str, Any]:
    """Return current LanceDB row counts grouped by modality, plus
    distinct-doc count. Cheap on small tables; see
    ``store.schema.index_stats`` for the O(N) caveat.

    Wrapped in ``asyncio.to_thread`` because the underlying
    ``count_rows`` / ``to_pandas`` calls are sync — blocking the
    event loop on an inflight ``/search`` would stall concurrent queries.
    """
    return await asyncio.to_thread(index_stats, path_prefix=path_prefix)


def _require_queue(request: Request):
    """Pull the shared :class:`JobQueue` off ``app.state``.

    The lifespan in ``main.py`` attaches it at startup. If we hit
    this code path before lifespan ran (or after shutdown), surface
    503 — never crash with a missing-attribute error.
    """
    queue = getattr(request.app.state, "job_queue", None)
    if queue is None:
        raise HTTPException(
            status_code=503,
            detail="index queue not running (server still booting?)",
        )
    return queue
