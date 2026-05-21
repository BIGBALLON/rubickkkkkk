"""Lightweight path → (mtime, doc_id) on-disk cache for fast ingest skip.

Problem this solves
-------------------
On every app restart, ``WatchService`` re-scans all watched folders.
For each file the ingest pipeline previously had to:

    1. p.read_bytes()           — reads the whole file (e.g. 5 MB JPEG)
    2. sha256(file_bytes)       — hashes it
    3. is_doc_indexed(doc_id)   — queries LanceDB

For a gallery of 1 000 images at 5 MB each that means reading ~5 GB
from disk just to discover "nothing changed" — that is the slow part
the user sees on restart.

How it works
------------
After any file is successfully indexed (or confirmed already indexed),
we record ``path → {mtime, doc_id}`` in a JSON file next to the LanceDB
table.  On subsequent scans the pipeline calls ``lookup(path, mtime)``
first:

- Hit (path known AND mtime unchanged) → return cached doc_id, check
  ``is_doc_indexed(doc_id)`` (cheap LanceDB query, no file read) →
  skip without reading a single byte of the image.
- Miss (path unknown or mtime changed) → fall through to the existing
  full read + sha256 path, then populate the cache for next time.

Correctness guarantees
----------------------
- mtime change (file edited) → cache miss → full re-read → correct.
- File deleted and re-created with same name but new content → mtime
  differs on any modern FS → cache miss → correct.
- Cache file corrupted or absent → all misses → falls through to the
  sha256 path, which is the pre-existing behaviour → correct.
- ``is_doc_indexed`` is still called on every cache hit, so if the
  LanceDB table is wiped externally the files get re-indexed.

Thread safety
-------------
``record()`` acquires ``_lock`` before mutating the dict.  ``flush()``
also acquires it.  ``lookup()`` reads without a lock — the GIL makes
dict reads safe, and a torn read is just a cache miss.
"""

from __future__ import annotations

import json
import logging
import threading
from pathlib import Path
from typing import TypedDict

log = logging.getLogger(__name__)


class _Entry(TypedDict):
    mtime: int
    doc_id: str


_cache: dict[str, _Entry] = {}
_cache_file: Path | None = None
_lock = threading.Lock()
_dirty = False
_initialised = False


# === Init ===================================================================


def _ensure_init() -> None:
    """Lazy-init: load cache from disk on first access.

    Acquires ``_lock`` for the double-checked init so two ingest threads
    racing on first call don't both run ``_init`` (which would issue two
    file reads and two JSON decodes).
    """
    global _initialised
    if _initialised:
        return
    with _lock:
        if _initialised:
            return
        from .. import settings

        _init_locked(settings.DATA_ROOT)


def _init_locked(data_dir: Path) -> None:
    """Caller must hold ``_lock``."""
    global _cache, _cache_file, _initialised
    _cache_file = data_dir / "path_mtime_cache.json"
    if _cache_file.exists():
        try:
            raw = _cache_file.read_text(encoding="utf-8")
            _cache = json.loads(raw)
            log.debug("path_cache: loaded %d entries from %s", len(_cache), _cache_file)
        except Exception as e:  # noqa: BLE001
            log.warning("path_cache: load failed (%s) — starting empty", e)
            _cache = {}
    _initialised = True


# === Public API =============================================================


def lookup(path: str, mtime: int) -> str | None:
    """Return the cached ``doc_id`` if ``path`` was last seen with ``mtime``.

    Returns ``None`` on any cache miss, letting the caller fall through to
    the full read + sha256 path.
    """
    _ensure_init()
    entry = _cache.get(path)
    if entry and entry.get("mtime") == mtime:
        return entry.get("doc_id")
    return None


def record(path: str, mtime: int, doc_id: str) -> None:
    """Store or refresh a path → (mtime, doc_id) mapping.

    Call this after a file is confirmed indexed (either freshly embedded
    or found already present via ``is_doc_indexed``).  Writes are
    batched in memory; call ``flush()`` to persist to disk.
    """
    global _dirty
    _ensure_init()
    with _lock:
        existing = _cache.get(path)
        if existing and existing.get("mtime") == mtime and existing.get("doc_id") == doc_id:
            return  # already current, skip dirty mark
        _cache[path] = {"mtime": mtime, "doc_id": doc_id}
        _dirty = True


def flush() -> None:
    """Write dirty cache entries to disk (atomic rename).

    Safe to call from any thread.  No-op when nothing has changed.

    Implementation note: we snapshot the dict + clear the dirty flag under
    the lock, then do the (potentially-slow) JSON encode + file write
    outside the lock so concurrent ``record()`` callers don't block on
    disk IO.  Worst case on a flush failure: ``_dirty`` gets set back to
    ``True`` and the next flush retries.
    """
    global _dirty
    if _cache_file is None:
        return
    with _lock:
        if not _dirty:
            return
        snapshot = dict(_cache)
        target = _cache_file
        _dirty = False

    try:
        tmp = target.with_suffix(".tmp")
        tmp.write_text(
            json.dumps(snapshot, ensure_ascii=False, separators=(",", ":")),
            encoding="utf-8",
        )
        tmp.replace(target)
        log.debug("path_cache: flushed %d entries to %s", len(snapshot), target)
    except Exception as e:  # noqa: BLE001
        log.warning("path_cache: flush failed: %s", e)
        # Re-arm the dirty flag so the next flush retries.  Reads of
        # ``_dirty`` go through ``_lock`` everywhere else so we acquire it.
        with _lock:
            _dirty = True


# === Test helpers ============================================================


def _reset_for_tests(data_dir: Path | None = None) -> None:
    """Wipe module state for unit tests.

    Pass a ``data_dir`` to point the cache at an isolated tmp directory
    *and* re-load any cache file that already lives there (so tests can
    simulate a process restart by writing to disk, calling reset, then
    reading back via ``lookup``).  Pass ``None`` to fully un-initialise.

    Not for production use.
    """
    global _cache, _cache_file, _initialised, _dirty
    with _lock:
        _cache = {}
        _cache_file = None
        _initialised = False
        _dirty = False
        if data_dir is not None:
            _init_locked(data_dir)
