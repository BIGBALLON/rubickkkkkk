"""LanceDB-backed multimodal store.

See ``schema.py`` for the unified single-table layout and ``connect``
helpers; every modality (text / image / video / rejected — plus legacy
``audio`` / ``*_transcript`` rows from old indexes; see schema docstring)
shares one table so LanceDB can serve arbitrary cross-modal queries with
a single ``.search()``.
"""

from . import path_cache
from .schema import (
    FTS_FIELDS,
    MODALITIES,
    SCHEMA,
    clear_dedup_cache,
    connect,
    ensure_fts_indexes,
    is_doc_indexed,
    make_doc_id,
    make_row,
    note_indexed,
    open_table,
    warm_dedup_cache,
)

__all__ = [
    "FTS_FIELDS",
    "MODALITIES",
    "SCHEMA",
    "clear_dedup_cache",
    "connect",
    "ensure_fts_indexes",
    "is_doc_indexed",
    "make_doc_id",
    "make_row",
    "note_indexed",
    "open_table",
    "path_cache",
    "warm_dedup_cache",
]
