"""BM25 leg of hybrid retrieval.

LanceDB ≥ 0.13 ships a native BM25-style FTS index per text column.
The current limitation (probed on lancedb 0.30): one index per field,
no multi-field index. We run FTS on both ``raw_text`` and
``filename``, so we run **two** parallel FTS queries and merge them
by max-score before the caller hands them off to RRF.

Why merge by max instead of feeding RRF two separate BM25 lists?
RRF would over-weight a doc that scores in both fields, but the
fields are highly correlated (filename often shows up in raw_text
of the first chunk), so we'd double-count. ``max(scores)`` keeps the
BM25 leg honest and lets the vector leg provide independent signal
in RRF.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

log = logging.getLogger(__name__)

# Fields indexed for BM25. Order matters only for logging.
FTS_FIELDS: tuple[str, ...] = ("raw_text", "filename")


@dataclass(frozen=True, slots=True)
class BM25Hit:
    """One BM25 match. ``score`` is LanceDB's raw FTS score (positive,
    unbounded; higher is better). The field that matched is tracked
    in ``best_field`` for debugging / future reranking.
    """

    id: str
    score: float
    best_field: str
    row: dict[str, Any]


def bm25_search(
    query_text: str,
    *,
    limit: int = 50,
    where: str | None = None,
    table=None,
) -> list[BM25Hit]:
    """Run BM25/FTS on every indexed text field and merge results.

    - Empty / whitespace-only queries short-circuit to ``[]`` —
      LanceDB's FTS happily returns nothing on empty queries, but we
      can skip the round-trips entirely.
    - If no FTS index exists yet (fresh DB before first ingest),
      LanceDB raises; we catch and return ``[]`` so callers fall
      back to vector-only.

    Returns hits in score-descending order, deduped per ``id``.
    """
    from ..store import open_table

    if not query_text or not query_text.strip():
        return []

    table = table if table is not None else open_table()

    merged: dict[str, BM25Hit] = {}
    for field in FTS_FIELDS:
        try:
            q = table.search(query_text, query_type="fts", fts_columns=field).limit(limit)
        except TypeError:
            # Older lancedb releases don't accept ``fts_columns``; fall
            # back to relying on the column-specific index existing.
            q = table.search(query_text, query_type="fts").limit(limit)
        if where:
            q = q.where(where)
        try:
            df = q.to_pandas()
        except Exception as e:  # noqa: BLE001 — missing index, parse, ...
            log.debug("bm25 on field=%s failed: %s", field, e)
            continue

        if df.empty:
            continue

        # LanceDB FTS returns ``_score`` (BM25-like, higher = better).
        for _, row in df.iterrows():
            doc_id = str(row["id"])
            score = float(row["_score"])
            existing = merged.get(doc_id)
            if existing is None or score > existing.score:
                merged[doc_id] = BM25Hit(
                    id=doc_id,
                    score=score,
                    best_field=field,
                    row=row.to_dict(),
                )

    return sorted(merged.values(), key=lambda h: -h.score)
