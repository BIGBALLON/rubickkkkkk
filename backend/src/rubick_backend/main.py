"""Rubick backend entry point.

Mounts the API routers and owns the per-process lifecycle of the
background ingest worker (``JobQueue``). The Swift main process
spawns us with ``--port 0`` and discovers the actual port either
from a JSON line on our stdout (legacy) or by tailing uvicorn's
own banner.

Routers currently wired up:

- ``healthz``  — liveness (``/healthz``) + per-model readiness
                 (``/healthz/model``) for the boot poll loop and the
                 Settings → Model UI (``api/healthz.py``).
- ``search``   — hybrid vector + BM25 retrieval (``api/search.py``).
- ``index``    — ingest job enqueue / status (``api/index.py``).
- ``settings`` — user-editable chunking params (``api/settings.py``,
                 chunking params).
- ``model``    — destructive HF cache operations
                 (``DELETE /model/cache``; ``api/model.py``, v1.x #5).
- ``nebula``   — 3-D UMAP semantic map endpoints
                 (``/nebula/map|recompute|status``; ``api/nebula.py``, M3).
"""

from __future__ import annotations

import json
import logging
import sys
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from . import __version__
from .api import healthz as healthz_api
from .api import index as index_api
from .api import model as model_api
from .api import nebula as nebula_api
from .api import search as search_api
from .api import settings as settings_api
from .worker import JobQueue

log = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Per-app-instance lifecycle.

    Boot:
      1. Emit a JSON event on stdout so the Swift parent knows we're up.
      2. Stand up the background ``JobQueue`` and stash it on
         ``app.state`` so the ``/index/job`` routes can find it.

    Shutdown:
      3. Stop the queue worker (graceful cancel + await). In-flight
         ingest is interrupted; the rows already committed survive
         because LanceDB persists per ``add()`` call.
    """
    sys.stdout.write(json.dumps({"event": "startup", "version": __version__}) + "\n")
    sys.stdout.flush()

    queue = JobQueue()
    await queue.start()
    app.state.job_queue = queue
    log.info("job queue started")

    try:
        yield
    finally:
        log.info("stopping job queue (pending=%d)", queue.pending)
        await queue.stop()
        app.state.job_queue = None
        # Best-effort flush of the path+mtime fast-skip cache on
        # graceful shutdown.  ``ingest_path`` already flushes every
        # 50 files mid-run; this catches the tail (last < 50 files)
        # so a clean Cmd-Q saves them too.  SIGKILL still loses up
        # to the last 50, which is the existing per-batch bound.
        try:
            from .store import path_cache

            path_cache.flush()
        except Exception as e:  # noqa: BLE001 — best-effort cleanup
            log.debug("path_cache flush on shutdown failed: %s", e)


app = FastAPI(
    title="Rubick Backend",
    version=__version__,
    docs_url=None,  # no public docs in production
    redoc_url=None,
    lifespan=lifespan,
)

app.include_router(healthz_api.router)
app.include_router(search_api.router)
app.include_router(index_api.router)
app.include_router(settings_api.router)
app.include_router(model_api.router)
app.include_router(nebula_api.router)
