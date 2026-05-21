"""LanceDB schema for the unified multimodal documents table.

All modalities share **one** table:

    Per-chunk primary key:  id = "{doc_id}-{modality}-{chunk_idx}"
    Per-document key:       doc_id = sha256(file_bytes)[:16]

Per-doc fields (``file_paths``, ``mtime``, ``sha256``, ``filename``,
``created_at`` and the metadata fields like ``width`` / ``duration_s``)
are denormalized into every chunk row to keep v1 a single LanceDB call;
the space cost is negligible and saves us a join table.

Modality values (string enum):

  - ``text``               regular text-note chunk
  - ``image``              still image (1 row / file)
  - ``video``              video (1 row / file, 32-frame sample)
  - ``rejected``           placeholder for files we refused to embed
                           (e.g. > 2min video); prevents re-scan

Deprecated values still readable in old indexes (no new rows produced
in current builds):

  - ``audio``              audio chunk (retired: audio modality removed)
  - ``audio_transcript``   was the Whisper transcript of an audio file
                           (retired in v0.0.2 — Whisper transcript track removed)
  - ``video_transcript``   was the Whisper transcript of a video's
                           audio track (retired in v0.0.2)

These fall outside ``MODALITIES`` so ``make_row`` rejects them on
write, but ``modality`` is a free-form string column — old rows
surface in search results as long as they exist on disk. Run
``DELETE WHERE modality IN ('audio', 'audio_transcript',
'video_transcript')`` on the LanceDB table to retire them once you
no longer need them.
"""

from __future__ import annotations

import hashlib
import logging
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pyarrow as pa

from .. import settings

log = logging.getLogger(__name__)

if TYPE_CHECKING:
    import lancedb

MODALITIES: frozenset[str] = frozenset(
    {
        "text",
        "image",
        "video",
        "rejected",
    }
)


def _vector_type() -> pa.DataType:
    """Fixed-size float32 vector — LanceDB needs the dim baked into the type
    so it can build IVF_PQ indexes.
    """
    return pa.list_(pa.float32(), settings.EMBED_DIM)


SCHEMA: pa.Schema = pa.schema(
    [
        # --- Required for every row ---
        pa.field("id", pa.string(), nullable=False),
        pa.field("doc_id", pa.string(), nullable=False),
        pa.field("file_paths", pa.list_(pa.string()), nullable=False),
        pa.field("modality", pa.string(), nullable=False),
        pa.field("chunk_idx", pa.int32(), nullable=False),
        pa.field("embedding", _vector_type(), nullable=False),
        pa.field("created_at", pa.int64(), nullable=False),
        pa.field("mtime", pa.int64(), nullable=False),
        pa.field("sha256", pa.string(), nullable=False),
        pa.field("filename", pa.string(), nullable=False),
        # --- Optional, per-modality ---
        pa.field("raw_text", pa.string(), nullable=True),
        pa.field("thumbnail_path", pa.string(), nullable=True),
        pa.field("chunk_n_tokens", pa.int32(), nullable=True),
        pa.field("chunk_offset_s", pa.int32(), nullable=True),
        pa.field("chunk_duration_s", pa.int32(), nullable=True),
        pa.field("n_frames_sampled", pa.int32(), nullable=True),
        pa.field("duration_s", pa.float32(), nullable=True),
        pa.field("width", pa.int32(), nullable=True),
        pa.field("height", pa.int32(), nullable=True),
        pa.field("exif_taken_at", pa.int64(), nullable=True),
        pa.field("status", pa.string(), nullable=True),
        pa.field("rejected_reason", pa.string(), nullable=True),
    ]
)


# === Helpers ================================================================


def make_doc_id(file_bytes: bytes) -> str:
    """``sha256(file_bytes)[:16]`` — short stable document identity."""
    return hashlib.sha256(file_bytes).hexdigest()[:16]


def make_row(
    *,
    doc_id: str,
    modality: str,
    chunk_idx: int,
    embedding: list[float],
    file_path: str | Path,
    sha256: str,
    mtime: int,
    filename: str,
    raw_text: str | None = None,
    thumbnail_path: str | None = None,
    chunk_n_tokens: int | None = None,
    chunk_offset_s: int | None = None,
    chunk_duration_s: int | None = None,
    n_frames_sampled: int | None = None,
    duration_s: float | None = None,
    width: int | None = None,
    height: int | None = None,
    exif_taken_at: int | None = None,
    status: str | None = None,
    rejected_reason: str | None = None,
    created_at: int | None = None,
) -> dict[str, Any]:
    """Build a row dict that matches ``SCHEMA``. Validates the modality and
    fills sensible defaults (``created_at`` → now, single-element
    ``file_paths`` array from the given path).

    Multi-path docs (hardlinks / moved files) must merge rows manually via
    a writer that fetches the existing ``file_paths`` array.
    """
    if modality not in MODALITIES:
        raise ValueError(f"unknown modality {modality!r}; expected one of {sorted(MODALITIES)}")

    if len(embedding) != settings.EMBED_DIM:
        raise ValueError(f"embedding has dim {len(embedding)}, expected {settings.EMBED_DIM}")

    return {
        "id": f"{doc_id}-{modality}-{chunk_idx}",
        "doc_id": doc_id,
        "file_paths": [str(file_path)],
        "modality": modality,
        "chunk_idx": chunk_idx,
        "embedding": list(embedding),
        "created_at": created_at if created_at is not None else int(time.time()),
        "mtime": mtime,
        "sha256": sha256,
        "filename": filename,
        "raw_text": raw_text,
        "thumbnail_path": thumbnail_path,
        "chunk_n_tokens": chunk_n_tokens,
        "chunk_offset_s": chunk_offset_s,
        "chunk_duration_s": chunk_duration_s,
        "n_frames_sampled": n_frames_sampled,
        "duration_s": duration_s,
        "width": width,
        "height": height,
        "exif_taken_at": exif_taken_at,
        "status": status,
        "rejected_reason": rejected_reason,
    }


# === Connection / table management =========================================


def connect() -> lancedb.DBConnection:
    """Open (or create on first call) the LanceDB connection at ``settings.LANCEDB_DIR``.

    Ensures the on-disk data dirs exist; callers don't need to call
    ``settings.ensure_data_dirs()`` themselves.
    """
    import lancedb

    settings.ensure_data_dirs()
    return lancedb.connect(str(settings.LANCEDB_DIR))


def open_table(db: lancedb.DBConnection | None = None):
    """Open the documents table, creating it (empty) if absent.

    Returns a ``lancedb.table.Table`` object.
    """
    db = db if db is not None else connect()
    name = settings.LANCEDB_TABLE
    # ``list_tables()`` returns a paginated response object whose ``.tables``
    # is the list of table names. (LanceDB ≥ 0.13 deprecated the simpler
    # ``table_names()`` helper.)
    existing = db.list_tables().tables
    if name in existing:
        return db.open_table(name)
    return db.create_table(name, schema=SCHEMA)


# === Dedup gate =============================================================
#
# Two-tier strategy:
#
# 1. **Warm a process-wide set of doc_ids** at the start of an
#    ``ingest_path`` run.  ``is_doc_indexed`` then reduces to an O(1)
#    set membership check — critical for the Re-scan / restart path
#    where 99% of files are already in the index and the per-call
#    ``count_rows`` filter (a few ms each) used to dominate the
#    "scanning" phase wall-clock (1131 files × 10 ms = 11 s, exactly
#    what users perceived as "scanning is slow").
# 2. **Fall back to a per-call LanceDB query** when the cache hasn't
#    been warmed (CLI single-file ingest, tests, FSEvents-driven
#    one-shot kicks).  Behaviour matches the old function exactly so
#    nothing else changes.
#
# Pair ``warm_dedup_cache`` and ``clear_dedup_cache`` around an
# ``ingest_path`` run; the worker queue serialises ingest jobs so
# the module-level state doesn't need finer locking than already
# present.

_dedup_cache: set[str] | None = None
_dedup_cache_lock = threading.Lock()


def warm_dedup_cache(table=None) -> int:
    """Load every ``doc_id`` in the table into a process-wide set so
    subsequent ``is_doc_indexed`` calls are O(1) memory lookups.

    Returns the number of entries loaded.  Idempotent — calling twice
    just re-loads from the table (handy after a delete pass).  No-op
    on failure; ``is_doc_indexed`` will fall through to the per-call
    query path so ingest still works.

    Implementation note: ``LanceTable.to_pandas`` doesn't accept a
    ``columns=`` kwarg (used to in older lancedb versions, removed in
    ≥ 0.13).  We use ``search().select(["doc_id"]).limit(huge)`` so
    we only deserialise the one column we care about — pulling the
    full 768-dim ``embedding`` column would dominate at 100k rows.
    The very-large ``limit`` is the idiomatic LanceDB "no limit"
    pattern: when the request asks for more rows than exist, you
    just get all of them.
    """
    global _dedup_cache
    tbl = table if table is not None else open_table()
    try:
        df = (
            tbl.search()
            .select(["doc_id"])
            .limit(2**31 - 1)  # effectively unbounded
            .to_pandas()
        )
        loaded = set(df["doc_id"].tolist())
    except Exception as e:  # noqa: BLE001 — diagnostic only
        import logging

        logging.getLogger(__name__).warning(
            "warm_dedup_cache failed (%s); falling back to per-call queries", e
        )
        return 0
    with _dedup_cache_lock:
        _dedup_cache = loaded
    return len(loaded)


def note_indexed(doc_id: str) -> None:
    """Record a freshly-added ``doc_id`` in the warm cache (if any).

    Call after each successful ``table.add`` so the cache stays
    accurate within a single ``ingest_path`` run — without this, two
    files with identical content (rare but possible) would both pay
    the embed cost because the cache wouldn't know about the row the
    first one just added.
    """
    if not doc_id:
        return
    with _dedup_cache_lock:
        if _dedup_cache is not None:
            _dedup_cache.add(doc_id)


def clear_dedup_cache() -> None:
    """Drop the warm cache.  Pair with ``warm_dedup_cache``."""
    global _dedup_cache
    with _dedup_cache_lock:
        _dedup_cache = None


def drop_all_index_data() -> dict[str, Any]:
    """Drop the entire index table and recreate it empty.

    Also clears:
    - The path-mtime cache file (forces full re-walk on next ingest)
    - The nebula map file (stale after clearing)
    - The in-memory dedup cache

    Returns counts of what was removed.
    """
    import shutil

    tbl = open_table()
    total_chunks = 0
    total_docs = 0
    try:
        total_chunks = tbl.count_rows()
        df = tbl.to_pandas()
        total_docs = int(df["doc_id"].nunique()) if "doc_id" in df.columns and not df.empty else 0
    except Exception:  # noqa: BLE001
        pass

    # Drop and recreate the table
    db = connect()
    name = settings.LANCEDB_TABLE
    try:
        db.drop_table(name)
    except Exception:  # noqa: BLE001
        pass
    db.create_table(name, schema=SCHEMA)

    # Clear path-mtime cache
    cache_file = settings.DATA_ROOT / "path_mtime_cache.json"
    if cache_file.is_file():
        cache_file.unlink(missing_ok=True)

    # Clear nebula map (now stale)
    if settings.NEBULA_MAP_FILE.is_file():
        settings.NEBULA_MAP_FILE.unlink(missing_ok=True)

    # Clear thumbnails directory
    deleted_thumbnails = 0
    if settings.THUMBNAILS_DIR.is_dir():
        for f in settings.THUMBNAILS_DIR.iterdir():
            if f.is_file():
                deleted_thumbnails += 1
        shutil.rmtree(settings.THUMBNAILS_DIR, ignore_errors=True)
        settings.THUMBNAILS_DIR.mkdir(parents=True, exist_ok=True)

    # Clear in-memory dedup cache
    clear_dedup_cache()

    log.info("Dropped all index data: %d chunks, %d docs", total_chunks, total_docs)
    return {
        "deleted_chunks": total_chunks,
        "deleted_docs": total_docs,
        "deleted_thumbnails": deleted_thumbnails,
    }


def is_doc_indexed(doc_id: str, table=None) -> bool:
    """Return True if any row with ``doc_id`` already lives in the table.

    Pipelines call this right after computing ``doc_id = sha256(file_bytes)[:16]``
    and bail out (returning an empty row list) when it returns True.
    That keeps FSEvents-triggered re-ingest cheap: the disk read is
    already paid (we needed it for the sha), but the model forward
    pass is skipped.

    Re-ingesting a file whose contents *changed* still works because
    a different sha → different doc_id. The old (stale) row stays in
    the table; orphan-doc cleanup is a future garbage-collect pass.

    Rejected rows count too: a previously-rejected file (e.g.
    > 2 min video) doesn't get re-tried on every walk pass. That's
    the whole point of the rejected placeholder.

    Fast path: when ``warm_dedup_cache`` was called by the caller
    (typical for ``ingest_path``), this is a pure O(1) set lookup.
    Slow path: ``count_rows`` with filter — a few ms per call.
    """
    if not doc_id:
        return False
    with _dedup_cache_lock:
        cache = _dedup_cache
    if cache is not None:
        return doc_id in cache
    tbl = table if table is not None else open_table()
    # ``count_rows`` with a filter is the fastest path on lancedb ≥
    # 0.13 — no full table scan, leverages the manifest's per-fragment
    # min/max statistics for the string column.
    try:
        return tbl.count_rows(filter=f"doc_id = '{doc_id}'") > 0
    except Exception:  # noqa: BLE001
        # Defensive: if some LanceDB version doesn't support filtered
        # count, fall back to a search-then-where which always works.
        import numpy as np

        try:
            from .. import settings as _s

            probe = np.zeros(_s.EMBED_DIM, dtype=np.float32)
            df = tbl.search(probe).where(f"doc_id = '{doc_id}'").limit(1).to_pandas()
            return not df.empty
        except Exception:  # noqa: BLE001
            # Last resort — never block ingest on a buggy dedup check.
            import logging

            logging.getLogger(__name__).exception(
                "is_doc_indexed fell through both paths for doc_id=%r", doc_id
            )
            return False


# === Path-prefix delete (re-index / folder-scoped purge) =====================


def delete_by_path_prefix(prefix: str, table=None) -> dict[str, Any]:
    """Delete every chunk whose canonical first path starts with
    ``prefix``, cleaning up associated thumbnails on disk.

    Used when removing indexed chunks under a path prefix (e.g. re-index).
    Generic on purpose — dangerous prefixes are rejected in the helper.

    Returns::

        {
          "deleted_chunks":      <int>,    # rows removed from LanceDB
          "deleted_docs":        <int>,    # distinct doc_ids removed
          "deleted_thumbnails":  <int>,    # thumbnail files unlinked
        }

    Safety:

    - ``prefix`` must be non-empty and ≥ 2 characters; ``prefix == '/'``
      would delete the entire index, which is never what a caller
      means here. (A future "Reset all data" surface lives behind a
      different, more obvious affordance.)
    - Single quotes inside the prefix are SQL-doubled to prevent
      WHERE-clause injection — same trick ``_build_where`` uses.
    - Thumbnail unlinks are best-effort: a missing or already-deleted
      file just gets logged and counted-down, never raised. The
      ``deleted_thumbnails`` count reports what we actually removed.
    """
    cleaned = (prefix or "").strip()
    if len(cleaned) < 2:
        raise ValueError(
            f"refusing to delete with prefix={cleaned!r}: must be at least 2 characters "
            "(use a more specific path; clearing the whole index is not allowed here)"
        )
    if cleaned == "/" or cleaned.rstrip("/") == "":
        raise ValueError(
            f"refusing to delete with prefix={cleaned!r}: would match every indexed row"
        )

    tbl = table if table is not None else open_table()
    escaped = cleaned.replace("'", "''")
    where = f"starts_with(file_paths[1], '{escaped}')"

    # Collect doc_ids + thumbnail paths *before* deletion so we can
    # report stats and unlink thumbnails after the row delete succeeds.
    # We accept the small cost of an extra query in exchange for
    # accurate counts and clean thumbnail bookkeeping.
    try:
        df = tbl.search().where(where).limit(100_000).to_pandas()
    except Exception:  # noqa: BLE001
        # Fall back to a raw filter scan for older LanceDB versions
        # where ``search().where(...)`` returns nothing without a
        # vector to score against.
        try:
            df = tbl.to_pandas()
            if not df.empty and "file_paths" in df.columns:
                df = df[
                    df["file_paths"].apply(
                        lambda paths: (
                            bool(paths)
                            and isinstance(paths[0], str)
                            and paths[0].startswith(cleaned)
                        )
                    )
                ]
        except Exception:  # noqa: BLE001
            import logging

            logging.getLogger(__name__).exception(
                "failed to enumerate rows for prefix=%r; reporting zeros", cleaned
            )
            return {"deleted_chunks": 0, "deleted_docs": 0, "deleted_thumbnails": 0}

    if df.empty:
        return {"deleted_chunks": 0, "deleted_docs": 0, "deleted_thumbnails": 0}

    deleted_chunks = int(len(df))
    deleted_docs = int(df["doc_id"].nunique()) if "doc_id" in df.columns else 0
    thumbnail_paths: list[str] = []
    if "thumbnail_path" in df.columns:
        for raw in df["thumbnail_path"].tolist():
            if raw is None:
                continue
            try:
                # ``raw`` may be a numpy NaN for legacy rows; coerce
                # via str() and skip the literal "nan".
                s = str(raw)
            except Exception:  # noqa: BLE001
                continue
            if s and s.lower() != "nan":
                thumbnail_paths.append(s)

    # Now actually delete. We pass a ``id IN (...)`` predicate built
    # from the enumeration above instead of re-using the
    # ``starts_with(file_paths[1], ...)`` form because LanceDB's
    # ``Table.delete(where=...)`` doesn't accept the array-indexing
    # function (verified on lancedb 0.30: the search-side WHERE
    # supports it, but the delete-side doesn't and the call appears
    # to silently match nothing). Going via a flat string-equals list
    # is portable across LanceDB versions and unambiguous.
    if "id" in df.columns:
        ids = [str(x) for x in df["id"].tolist() if x is not None]
        if ids:
            quoted = ", ".join(f"'{i}'" for i in ids)
            try:
                tbl.delete(f"id IN ({quoted})")
            except Exception:  # noqa: BLE001
                import logging

                logging.getLogger(__name__).exception(
                    "tbl.delete by id IN list failed; aborting prefix delete"
                )
                raise

    # Best-effort thumbnail unlink. Deduplicate first — old indexes
    # may carry both a ``video`` row and a legacy ``video_transcript``
    # row that share one thumbnail, and we don't want to log "missing"
    # for the second pass.
    deleted_thumbnails = 0
    for path_str in set(thumbnail_paths):
        try:
            Path(path_str).unlink(missing_ok=True)
            deleted_thumbnails += 1
        except OSError as e:
            import logging

            logging.getLogger(__name__).warning("could not unlink thumbnail %s: %s", path_str, e)

    return {
        "deleted_chunks": deleted_chunks,
        "deleted_docs": deleted_docs,
        "deleted_thumbnails": deleted_thumbnails,
    }


# === FTS (BM25) index management ===========================================

FTS_FIELDS: tuple[str, ...] = ("raw_text", "filename")
"""Fields LanceDB FTS indexes for BM25 retrieval. Each column gets its
own index (LanceDB FTS is single-field per index); ``retrieve.bm25``
queries both and merges by max score."""


# === Aggregate stats (Settings → Index tab) =================================


def _items_per_modality(df) -> dict[str, int]:
    """Count distinct ``doc_id`` per modality from a materialized table.

    Done in pandas because LanceDB has no native
    ``COUNT(DISTINCT doc_id) GROUP BY modality``. We share the
    ``to_pandas()`` materialization with the global ``total_docs``
    computation in ``index_stats``, so this call is essentially free
    once that DataFrame exists.
    """
    out = {m: 0 for m in sorted(MODALITIES)}
    try:
        groups = df.groupby("modality")["doc_id"].nunique()
    except Exception:  # noqa: BLE001
        return out
    for modality, count in groups.items():
        out[str(modality)] = int(count)
    return out


def index_stats(table=None, *, path_prefix: str | None = None) -> dict[str, Any]:
    """Return aggregate index counts for the Settings → Index tab.

    Shape::

        {
          "total_chunks": <int>,        # rows in the table
          "total_docs":   <int>,        # distinct doc_ids
          "by_modality":       {<modality>: <chunk_count>, ...}   # alphabetized
          "by_modality_docs":  {<modality>: <doc_count>, ...}     # alphabetized
        }

    All MODALITIES keys are always present in both ``by_modality``
    and ``by_modality_docs`` so the SwiftUI side can render a stable
    row order even when one modality has zero hits.

    ``by_modality`` counts **chunks** (LanceDB rows). For text where
    one file produces multiple chunks this is larger than the file
    count. ``by_modality_docs`` counts **distinct files** (``doc_id``)
    so a markdown note that chunked into 5 contributes 1 to
    ``by_modality_docs["text"]`` and 5 to ``by_modality["text"]``.

    ``path_prefix`` (v1.x): when set, restrict the counts to docs
    whose canonical first path starts with this string. Used by the
    Watched-folders sidebar to re-derive each folder's
    ``items · chunks`` line after an app restart wipes the in-memory
    per-folder stats. Implemented as a pandas filter on the full
    materialised DataFrame — cheap for v1's expected corpus
    (typically a few thousand rows); if 100k+ chunks per index
    become normal we'll maintain incremental counters in the writer.
    """
    tbl = table if table is not None else open_table()

    by_modality = {m: 0 for m in sorted(MODALITIES)}
    by_modality_docs = {m: 0 for m in sorted(MODALITIES)}

    if path_prefix is None:
        # Whole-table fast path: ``count_rows`` per modality is
        # cheaper than a single materialisation on big tables, and
        # there's exactly one ``to_pandas()`` for the distinct-doc
        # counts.
        total_chunks = tbl.count_rows()
        if total_chunks == 0:
            return {
                "total_chunks": 0,
                "total_docs": 0,
                "by_modality": by_modality,
                "by_modality_docs": by_modality_docs,
            }
        for m in sorted(MODALITIES):
            try:
                by_modality[m] = tbl.count_rows(filter=f"modality = '{m}'")
            except Exception:  # noqa: BLE001 — best-effort per modality
                import logging

                logging.getLogger(__name__).exception(
                    "count_rows for modality=%r failed; reporting 0", m
                )
                by_modality[m] = 0
        try:
            df = tbl.to_pandas()
            total_docs = int(df["doc_id"].nunique())
            by_modality_docs.update(_items_per_modality(df))
        except Exception:  # noqa: BLE001
            total_docs = 0
        return {
            "total_chunks": total_chunks,
            "total_docs": total_docs,
            "by_modality": by_modality,
            "by_modality_docs": by_modality_docs,
        }

    # path_prefix path: pull the full table and boolean-filter on
    # ``file_paths[0]`` in pandas. LanceDB's SQL surface ``count_rows
    # (filter=starts_with(...))`` does work, but combining a starts_with
    # AND modality filter requires a string compose dance the pandas
    # path expresses more cleanly. The trade-off (one full
    # materialisation vs N count_rows) is negligible at v1 sizes.
    try:
        df = tbl.to_pandas()
    except Exception:  # noqa: BLE001 — surface as empty stats
        return {
            "total_chunks": 0,
            "total_docs": 0,
            "by_modality": by_modality,
            "by_modality_docs": by_modality_docs,
        }
    if df.empty:
        return {
            "total_chunks": 0,
            "total_docs": 0,
            "by_modality": by_modality,
            "by_modality_docs": by_modality_docs,
        }

    def _matches_prefix(file_paths: Any) -> bool:
        # ``file_paths`` round-trips as a numpy ObjectArray; first
        # element is the canonical path. Empty / malformed → drop.
        try:
            if file_paths is None or len(file_paths) == 0:
                return False
            return str(file_paths[0]).startswith(path_prefix)
        except (TypeError, IndexError):
            return False

    df = df[df["file_paths"].apply(_matches_prefix)]
    total_chunks = int(len(df))
    if total_chunks == 0:
        return {
            "total_chunks": 0,
            "total_docs": 0,
            "by_modality": by_modality,
            "by_modality_docs": by_modality_docs,
        }
    total_docs = int(df["doc_id"].nunique())
    by_modality_docs.update(_items_per_modality(df))
    for m, n in df["modality"].value_counts().items():
        if m in by_modality:
            by_modality[str(m)] = int(n)
    return {
        "total_chunks": total_chunks,
        "total_docs": total_docs,
        "by_modality": by_modality,
        "by_modality_docs": by_modality_docs,
    }


def ensure_fts_indexes(table) -> None:
    """Create FTS indexes on ``FTS_FIELDS`` if they're not already present.

    Idempotent — safe to call after every ingest batch. We deliberately
    *don't* pass ``replace=True`` here because rebuilding the entire
    index would O(N) every batch; LanceDB's FTS picks up newly added
    rows automatically (verified on lancedb 0.30: new rows surface
    in subsequent ``query_type='fts'`` searches without an explicit
    rebuild). ``optimize()`` cleans up fragmentation in the background.

    No-op on empty tables: ``create_fts_index`` raises on zero rows in
    some lancedb versions, so we guard with a row count check.
    """
    try:
        n_rows = table.count_rows()
    except Exception:  # noqa: BLE001 — best-effort
        n_rows = 0
    if n_rows == 0:
        return

    existing = {tuple(idx.columns) for idx in table.list_indices()}
    for field in FTS_FIELDS:
        if (field,) in existing:
            continue
        try:
            table.create_fts_index(field, replace=False)
        except Exception:  # noqa: BLE001
            # If create_fts_index raises (already-exists race, schema
            # mismatch, etc.) we don't want to break ingest — BM25 just
            # won't find this field until next call. Log via the
            # standard logger; callers can investigate.
            import logging

            logging.getLogger(__name__).exception("failed to create FTS index on %s", field)
