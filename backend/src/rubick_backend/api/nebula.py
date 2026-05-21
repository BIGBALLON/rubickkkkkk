"""Nebula M3 API — 3-D semantic map endpoints.

Three routes:
- GET  /nebula/map       → precomputed UMAP 3-D coordinates
- POST /nebula/recompute → trigger async UMAP recompute
- GET  /nebula/status    → compute state + staleness
"""

from __future__ import annotations

import asyncio
import logging
import time

from fastapi import APIRouter

from ..nebula import compute, state

log = logging.getLogger(__name__)
router = APIRouter(prefix="/nebula", tags=["nebula"])

# Debounce: track last auto-recompute timestamp (epoch seconds).
# Manual recompute (POST) bypasses this.
_last_auto_recompute: int = 0
_AUTO_RECOMPUTE_COOLDOWN: int = 3600  # 1 hour


@router.get("/map")
async def get_map():
    """Return the precomputed UMAP 3-D map."""
    return compute.load_map()


@router.post("/recompute")
async def trigger_recompute():
    """Trigger an async UMAP recompute. Returns immediately."""
    if state.is_computing():
        return {"job_id": None, "status": "already_computing"}

    job_id = f"nebula-recompute-{int(time.time())}"
    asyncio.get_event_loop().run_in_executor(None, compute.run_nebula_compute)
    return {"job_id": job_id, "status": "started"}


@router.get("/status")
async def get_status():
    """Return compute state + staleness indicator."""
    status = state.get_status()
    status["stale"] = compute.is_stale()
    return status


def maybe_auto_recompute() -> None:
    """Called after ingest completes. Queues recompute if stale + debounce OK.

    This is NOT an endpoint — it's called by the job queue worker after
    each ingest job finishes. Runs the staleness check synchronously
    (cheap: one count_rows) and schedules the compute in a thread if needed.
    """
    global _last_auto_recompute

    if state.is_computing():
        return

    now = int(time.time())
    if now - _last_auto_recompute < _AUTO_RECOMPUTE_COOLDOWN:
        return

    if not compute.is_stale():
        return

    _last_auto_recompute = now
    log.info("nebula: auto-recompute triggered (stale map detected)")
    import concurrent.futures
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    executor.submit(compute.run_nebula_compute)
