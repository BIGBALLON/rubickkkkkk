"""Per-modality ingestion pipelines + the cross-modality dispatcher.

The actual ingest logic lives in modality-specific modules (``text``,
``image``, ``video``) — each exports its own ``SUPPORTED_EXTENSIONS`` and
``ingest_file(path) -> list[row]``. This file ties them together with
two facade functions used by the CLI and the API:

- :func:`ingest_file` — process one file, picking the right pipeline
  from its extension.
- :func:`ingest_path` — walk a single file *or* a directory tree and
  feed every supported file through the right pipeline.
"""

from __future__ import annotations

import fnmatch
import logging
import os
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .. import settings as _settings
from . import image as _image
from . import text as _text
from . import video as _video

if TYPE_CHECKING:
    import lancedb

log = logging.getLogger(__name__)

# === Pipeline registry ======================================================

_PIPELINES = (_text, _image, _video)

# === Memory hygiene =========================================================
#
# How often (in files) we ask the embedding loader to drop its Metal cache
# + run a Python GC pass. Picked empirically so the per-batch cleanup cost
# is negligible against the embed cost (text ~10 ms, image ~2 s) yet
# small enough that a multi-thousand-image folder never accumulates more
# than ~50 forwards' worth of unreleased GPU buffers. Production crashes
# we shipped this to fix were dominated by image-heavy folders so the
# default targets the image path; text-only folders effectively no-op
# because Metal has nothing to release after each ``encode_text``.
_CLEAR_CACHE_EVERY_N_FILES: int = 50

# Directory names we never recurse into during ingest. Hidden dirs
# (anything starting with ``.``) are also pruned. This keeps ``.venv``,
# ``node_modules``, ``.git``, etc. out — matters a lot when users point
# Rubick at e.g. ``~/code/``. Pulled into the dispatcher so every
# modality's walk benefits.
EXCLUDED_DIR_NAMES: frozenset[str] = frozenset(
    {
        "node_modules",
        "venv",
        "__pycache__",
        "build",
        "dist",
        "DerivedData",
        "Pods",
        "target",
    }
)


def _all_supported_extensions() -> frozenset[str]:
    """Union of every registered pipeline's ``SUPPORTED_EXTENSIONS``."""
    out: set[str] = set()
    for pipeline in _PIPELINES:
        out.update(pipeline.SUPPORTED_EXTENSIONS)
    return frozenset(out)


SUPPORTED_EXTENSIONS: frozenset[str] = _all_supported_extensions()


def _pipeline_for(path: Path):
    """Return the pipeline module that should handle ``path``, or None.

    Lookup is by lowercase suffix. We accept the first hit; extension
    sets are disjoint by construction (text uses ``.md`` / ``.txt`` /
    ``.org`` / ``.markdown``, image uses photo formats), so order
    doesn't matter for correctness — but ``_text`` is first to keep
    short text smokes from importing PIL.
    """
    ext = path.suffix.lower()
    for pipeline in _PIPELINES:
        if ext in pipeline.SUPPORTED_EXTENSIONS:
            return pipeline
    return None


# === Public API =============================================================


def ingest_file(path: Path | str) -> list[dict[str, Any]]:
    """Process one file by routing on its extension; return ready rows."""
    p = Path(path)
    pipeline = _pipeline_for(p)
    if pipeline is None:
        log.warning("skip %s — no pipeline for extension", p)
        return []
    return pipeline.ingest_file(p)


ProgressCallback = Callable[[int, int, str | None, int], None]
"""``progress_cb(done, total, current_file, embedded)``.

``done`` is the number of files processed so far (regardless of skip /
add outcome) and starts at 0; ``total`` is the count enumerated up
front by ``_walk_supported_files``. ``current_file`` is the absolute
string path of the file just processed (or ``None`` for the initial
``progress_cb(0, total, None, 0)`` emission). ``embedded`` is the
subset of ``done`` that actually produced rows — i.e. files we ran
through the model, *excluding* fast-skip / sha-skip / reject paths.
A UI can use ``embedded == 0 and done > 0`` to render a "scanning,
nothing new yet" indicator instead of the misleading "indexing N
files" copy that suggests the GPU is busy."""


def ingest_path(
    path: Path | str,
    table: lancedb.table.Table | None = None,
    *,
    progress_cb: ProgressCallback | None = None,
    pause_event: threading.Event | None = None,
) -> dict[str, int]:
    """Ingest a single file or recursively walk a directory.

    Returns ``{"files": <files_with_rows>, "chunks": <total_rows>,
    "skipped": <files_skipped>}``. Chunks counts **rows added** —
    text contributes 1-N chunks per file while image contributes
    exactly 1 row per file; this stays useful as a single
    cross-modality counter regardless.

    ``progress_cb`` (optional) fires once with ``(0, total, None)`` after
    the walk finishes, and once per file thereafter as ``(done, total,
    str(file))``. See ``ProgressCallback`` for the exact contract.

    ``pause_event`` (optional) is a ``threading.Event`` that — when *not
    set* — blocks the per-file loop just before each ingest. The
    worker mirrors its ``asyncio.Event`` pause gate into a threading
    one so Pause / Resume takes effect within ~one file even mid-folder
    (the previous behaviour only checked between jobs). Pass ``None``
    in CLI / test contexts to skip the gate entirely.
    """
    from ..store import (
        clear_dedup_cache,
        note_indexed,
        open_table,
        warm_dedup_cache,
    )

    table = table if table is not None else open_table()

    p = Path(path).expanduser().resolve()
    if p.is_file():
        targets: list[Path] = [p]
    elif p.is_dir():
        targets = _walk_supported_files(p)
    else:
        log.warning("path not found: %s", p)
        if progress_cb is not None:
            progress_cb(0, 0, None, 0)
        return {"files": 0, "chunks": 0, "skipped": 0}

    sorted_targets = sorted(targets)
    total = len(sorted_targets)
    if progress_cb is not None:
        progress_cb(0, total, None, 0)

    # Warm a process-wide doc_id set so per-file ``is_doc_indexed`` is
    # an O(1) memory lookup instead of a LanceDB ``count_rows`` query
    # each time.  On a 1131-image gallery this drops the "scanning"
    # phase from ~10 s of cumulative LanceDB roundtrips to under a
    # second of pure dict checks.  Only worth doing for directories;
    # single-file ingests don't benefit and pay the table-walk cost.
    cache_warmed = False
    if p.is_dir() and total > 0:
        n_loaded = warm_dedup_cache(table=table)
        cache_warmed = n_loaded > 0
        if cache_warmed:
            log.debug("warmed dedup cache with %d doc_ids", n_loaded)

    files = 0
    chunks = 0
    skipped = 0
    embedded = 0  # files that produced rows (vs fast-skipped / rejected)
    t0 = time.perf_counter()
    for idx, f in enumerate(sorted_targets, start=1):
        if pause_event is not None:
            pause_event.wait()
        rows = ingest_file(f)
        if rows:
            table.add(rows)
            files += 1
            chunks += len(rows)
            embedded += 1
            # Keep the warm cache accurate so two files with identical
            # content in the same run don't both pay the embed cost.
            for row in rows:
                note_indexed(row.get("doc_id", ""))
        else:
            skipped += 1
        if progress_cb is not None:
            progress_cb(idx, total, str(f), embedded)

        # Periodically drop the Metal allocator cache so a multi-thousand-
        # image folder doesn't pile up unreleased GPU buffers. Cheap
        # (<5 ms on warm M-series); mid-folder is fine because between
        # files there's no MLX work in flight.
        #
        # Same cadence flushes the path+mtime cache so a force-quit
        # mid-folder doesn't throw away the fast-skip table built up
        # so far.  Without this the user closes the app on a half-
        # ingested folder, comes back, and re-pays the full
        # `read_bytes + sha256` for every file we'd already seen —
        # i.e. exactly the slow path the cache was supposed to fix.
        if idx % _CLEAR_CACHE_EVERY_N_FILES == 0:
            try:
                from ..embed import clear_inference_cache

                clear_inference_cache()
            except Exception as e:  # noqa: BLE001 — cleanup is best-effort
                log.debug("clear_inference_cache failed: %s", e)
            try:
                from ..store import path_cache

                path_cache.flush()
            except Exception as e:  # noqa: BLE001
                log.debug("path_cache flush failed: %s", e)

    # Ensure FTS indexes exist for hybrid BM25 retrieval on ``raw_text``
    # and ``filename``. Materialize once per ingest_path call rather
    # than on first ``/search`` — cheap no-ops when indexes already
    # exist (``ensure_fts_indexes`` is idempotent).
    if chunks > 0:
        from ..store import ensure_fts_indexes

        ensure_fts_indexes(table)

    # Final cleanup pass at the end of every ingest_path call, regardless
    # of whether we hit a 50-file boundary; for the typical "drop one
    # folder" workflow this is the most important cache release point.
    try:
        from ..embed import clear_inference_cache

        clear_inference_cache()
    except Exception as e:  # noqa: BLE001
        log.debug("clear_inference_cache (final) failed: %s", e)

    # Flush the path+mtime cache to disk so the next app restart can
    # skip re-reading already-indexed files.
    try:
        from ..store import path_cache

        path_cache.flush()
    except Exception as e:  # noqa: BLE001
        log.debug("path_cache flush failed: %s", e)

    # Drop the warmed dedup cache so a follow-up call (e.g. a single-
    # file FSEvents kick) doesn't operate on stale doc_ids carried
    # over from the previous walk.
    if cache_warmed:
        clear_dedup_cache()

    elapsed = time.perf_counter() - t0
    log.info(
        "ingest_path done: %d files, %d chunks, %d skipped (%.2fs)",
        files,
        chunks,
        skipped,
        elapsed,
    )
    return {"files": files, "chunks": chunks, "skipped": skipped}


# === Walk ===================================================================


def _matches_user_pattern(name: str, patterns: list[str]) -> bool:
    """Return ``True`` if ``name`` matches **any** fnmatch ``patterns``.

    Pulled out so we can unit-test the per-name decision separately
    from the walk loop. ``patterns`` is the live
    ``settings.EXCLUSION_PATTERNS`` list — we read it fresh on every
    walk pass (no snapshotting) so a ``PATCH /settings`` update
    affects the very next ingest job without a restart.
    """
    return any(fnmatch.fnmatchcase(name, p) for p in patterns)


def _walk_supported_files(root: Path) -> list[Path]:
    """Recursively collect files whose extension is handled by *any*
    registered pipeline.

    Skips hidden directories (``.git`` / ``.venv`` / ``.cache`` / ...)
    and a small denylist of common build / dependency dirs. v1.x #3
    additionally honours user-supplied ``exclusion_patterns`` from
    ``settings`` — fnmatch globs applied to the **basename** of each
    directory or file the walker sees. ``secrets`` blocks any folder
    literally named ``secrets``; ``*.tmp`` blocks any file ending in
    ``.tmp``; ``backup-*`` blocks anything starting that way. No
    ``.gitignore`` semantics — anchored / negated / recursive ``**``
    patterns are deferred to a future iteration once we have user
    demand evidence.
    """
    found: list[Path] = []
    supported = SUPPORTED_EXTENSIONS
    user_patterns = _settings.EXCLUSION_PATTERNS
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [
            d
            for d in dirnames
            if not d.startswith(".")
            and d not in EXCLUDED_DIR_NAMES
            and not _matches_user_pattern(d, user_patterns)
        ]
        for name in filenames:
            if name.startswith("."):
                continue
            ext = Path(name).suffix.lower()
            if ext not in supported:
                continue
            if _matches_user_pattern(name, user_patterns):
                continue
            found.append(Path(dirpath) / name)
    return found


# === Compatibility re-exports ================================================

# Legacy re-exports for CLI/tests. Chunk-token knobs: ``settings.TARGET_TOKENS``
# / ``settings.HARD_MAX_TOKENS`` (not re-exported from this package).
from .text import (  # noqa: E402
    MAX_FILE_BYTES,
    SHORT_CHAR_THRESHOLD,
    chunk_text,
)

__all__ = [
    "EXCLUDED_DIR_NAMES",
    "MAX_FILE_BYTES",
    "SHORT_CHAR_THRESHOLD",
    "SUPPORTED_EXTENSIONS",
    "chunk_text",
    "ingest_file",
    "ingest_path",
]
