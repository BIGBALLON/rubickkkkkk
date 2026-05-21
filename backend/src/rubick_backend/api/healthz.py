"""Health + readiness routes.

Three endpoints:

- ``GET /healthz`` — liveness probe, used by ``BackendController`` for
  the boot poll loop. Returns immediately and never depends on heavy
  imports (no MLX, no LanceDB) so the Swift parent can tell "uvicorn
  finished its app-startup hooks" within hundreds of ms.
- ``GET /healthz/model`` — readiness/state probe describing the
  HuggingFace embedding model the backend depends on (download
  status + on-disk size + in-memory load state). Powers the
  Settings → Model UI and the Onboarding "model setup" step.
- ``GET /healthz/permissions`` — macOS TCC probe (v1.x #2). Today
  only reports Full Disk Access; see ``permissions.py`` for the
  rationale on why notifications etc. aren't here yet.

Why a router instead of inlining in ``main.py``: ``/healthz`` is the
right place to add liveness/readiness probes generally, and the
``/model`` route keeps growing. Splitting the file now means the next
change is a one-line route addition rather than another bloat to
``main.py``.
"""

from __future__ import annotations

import asyncio

from fastapi import APIRouter

from .. import __version__, settings
from ..embed import is_loaded as embed_is_loaded
from ..model_status import snapshot
from ..permissions import probe_full_disk_access

router = APIRouter()


@router.get("/healthz")
async def healthz() -> dict[str, str]:
    """Liveness probe — small, fast, no heavy imports."""
    return {"status": "ok", "version": __version__}


@router.get("/healthz/model")
async def healthz_model() -> dict[str, object]:
    """Readiness/state probe for the embedding model.

    Returned shape::

        {
          "models": [
            {
              "id": "embedding",
              "repo": "jinaai/jina-embeddings-v5-omni-nano-retrieval-mlx",
              "purpose": "...",
              "declared_bytes": 1900000000,
              "cache_path": "/Users/x/.cache/huggingface/hub/models--...",
              "cache_bytes": 1822334455,
              "download_status": "complete" | "partial" | "absent",
              "loaded_in_memory": true | false
            }
          ]
        }

    The payload is wrapped in a ``models`` list so adding a future
    second model (e.g. a re-introduced transcription engine) is a
    one-line backend change without breaking the Swift consumers.

    Cheap to call (only stat()s the cache root); no MLX work, no HF
    network round-trip. Intended to be polled at, say, 1 s while a
    download is in progress.
    """
    embedding = snapshot(
        model_id="embedding",
        repo=settings.MAIN_MODEL_REPO,
        purpose=settings.MAIN_MODEL_PURPOSE,
        declared_bytes=settings.MAIN_MODEL_DECLARED_BYTES,
        loaded_in_memory=embed_is_loaded(),
    )
    return {"models": [embedding.to_dict()]}


@router.get("/healthz/permissions")
async def healthz_permissions() -> dict[str, object]:
    """Probe macOS TCC permission state (v1.x #2).

    Returned shape::

        {
          "full_disk_access": {
            "granted": false,
            "probe_path": "/Library/Application Support/.../TCC.db",
            "probe_error": "Permission denied",
            "platform": "Darwin"
          }
        }

    The top-level dict is keyed by permission name so a future
    addition (Accessibility, Screen Recording, …) lands as a
    sibling key without breaking older clients.

    Cheap (one ``open()`` syscall); wrapped in ``asyncio.to_thread``
    only because a hung NFS-mounted ``/Library`` root could
    otherwise stall the event loop.
    """
    probe = await asyncio.to_thread(probe_full_disk_access)
    return {"full_disk_access": probe.to_dict()}
