"""``/model/...`` routes — destructive operations on the HF cache.

Read-only state lives at ``GET /healthz/model`` (see ``api/healthz.py``).
Mutating state lives here because the ``/healthz/*`` family is for
liveness / readiness probes; deletes belong on their own surface so we
can attach auth / confirmation flows later without disturbing the
boot poll loop.

Currently exposes one route:

- ``DELETE /model/cache?id=<embedding>`` — wipe the on-disk
  HuggingFace cache subtree for the named model. The in-process
  singleton (if hydrated) is left alone; the user re-launches Rubick
  to trigger a fresh ``snapshot_download``. See
  ``model_status.delete_model_cache`` for the exact semantics.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from fastapi import APIRouter, HTTPException, Query

from .. import model_status, settings
from ..embed.loader import get_download_state

log = logging.getLogger(__name__)

router = APIRouter(prefix="/model", tags=["model"])


# Stable id → HF repo map. Currently a one-entry dict; structured this
# way so a future second model lands as a one-line addition without
# the route having to learn about its name. The set of valid ids is
# what we accept on ``?id=...``; anything else returns 404.
_KNOWN_MODELS: dict[str, str] = {
    "embedding": settings.MAIN_MODEL_REPO,
}


@router.delete(
    "/cache",
    summary="Wipe the on-disk cache for one model",
)
async def delete_model_cache(
    id: str = Query(  # noqa: A002 — ``id`` is the HTTP query param name
        ...,
        description=(
            "Stable model id. Currently only ``embedding`` is recognised; "
            "other values return 404. Required (no implicit "
            "\"delete-them-all\" default — we want every destructive call "
            "to be explicit about which model it targets)."
        ),
    ),
) -> dict[str, Any]:
    """Wipe the on-disk HuggingFace cache subtree for the named model.

    Returns ``{deleted_bytes, path, was_present, id, repo}`` so the
    client can echo "freed 1.8 GB at /Users/.../models--..." without
    a second round-trip. Idempotent — calling against an already-
    absent cache returns ``deleted_bytes=0`` + ``was_present=False``
    rather than 404 (the user said "make sure it's gone"; we did).

    Wrapped in ``asyncio.to_thread`` because ``shutil.rmtree`` on a
    1.8 GB directory blocks for ~50 ms on an APFS-backed Mac and
    we don't want to stall the event loop's ``/search`` traffic.
    """
    repo = _KNOWN_MODELS.get(id)
    if repo is None:
        raise HTTPException(
            status_code=404,
            detail=(
                f"unknown model id {id!r}; "
                f"known: {sorted(_KNOWN_MODELS)}"
            ),
        )
    try:
        result = await asyncio.to_thread(
            model_status.delete_model_cache, repo
        )
    except OSError as e:
        # Permission / IO failure under the cache root surfaces as
        # 500 — the user can't fix it themselves but they shouldn't
        # see "succeeded" either. ``shutil.rmtree`` keeps the partial
        # state, so a retry after the user resolves permissions will
        # finish the job.
        log.exception("delete_model_cache failed for %s", repo)
        raise HTTPException(
            status_code=500,
            detail=f"could not delete cache for {repo!r}: {e}",
        ) from e

    return {"id": id, "repo": repo, **result}


@router.get(
    "/download-progress",
    summary="Current model download progress",
)
async def download_progress() -> dict[str, Any]:
    """Return download state: status, downloaded_bytes, total_bytes, error.

    Polled by the Swift UI during model download to show a progress bar.
    """
    state = get_download_state()
    # Also enrich with model_status on-disk info for byte-level progress
    try:
        info = await asyncio.to_thread(
            model_status.get_model_info, settings.MAIN_MODEL_REPO
        )
        state["downloaded_bytes"] = info.get("cache_bytes", 0)
        state["total_bytes"] = settings.MAIN_MODEL_DECLARED_BYTES
    except Exception:
        pass
    return state


@router.post(
    "/download",
    summary="Trigger model download (with mirror fallback)",
)
async def trigger_download(
    endpoint: str = Query(
        default="",
        description="HuggingFace endpoint URL. Empty=auto (try official then mirror).",
    ),
) -> dict[str, Any]:
    """Start model download in background. Poll /model/download-progress for status.

    If endpoint is provided, sets HF_ENDPOINT for this download attempt.
    """
    import os
    import threading

    from ..embed.loader import _download_state

    if _download_state["status"] == "downloading":
        return {"status": "already_downloading"}

    def _bg_download():
        from huggingface_hub import snapshot_download

        from ..embed.loader import _download_with_fallback

        if endpoint:
            os.environ["HF_ENDPOINT"] = endpoint
            settings.HF_ENDPOINT = endpoint

        try:
            _download_with_fallback(snapshot_download)
        except Exception as e:
            log.error("background download failed: %s", e)

    threading.Thread(target=_bg_download, daemon=True).start()
    return {"status": "started", "endpoint": endpoint or "auto"}
