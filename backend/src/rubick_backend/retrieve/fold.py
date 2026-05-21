"""Doc-level folding — collapse chunk hits by ``doc_id`` for the UI.

A single document often has multiple chunks (a markdown note can
split into 3-4 chunks; legacy v0.0.1 indexes additionally have one
``video_transcript`` row per video — see store/schema.py for the
deprecation note). When several of those chunks all rank highly
for the same query, the user sees the same file repeating in the
results list — visually noisy and unhelpful.

Folding collapses by ``doc_id`` while preserving:

- ``best_chunk`` — highest-ranked chunk we'll show by default
- ``hit_count`` — total chunks that matched (UI shows "+N more")
- ``all_chunk_ids`` — secondary chunks the UI can expand into
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class DocResult:
    """One doc-level hit (folds N chunks of the same file).

    ``best_chunk`` is the original raw chunk row dict, normalized to
    JSON-safe types at the API boundary (LanceDB returns NaN for
    nullable strings; ``hybrid.search`` handles that). ``score_rrf``
    is the doc's best chunk RRF score — comparable across docs."""

    doc_id: str
    best_chunk: dict[str, Any]
    score_rrf: float
    score_vector: float | None
    score_bm25: float | None
    hit_count: int
    all_chunk_ids: list[str]


def fold_to_docs(
    ranked_chunks: Sequence[dict[str, Any]],
    *,
    doc_limit: int,
) -> list[DocResult]:
    """Fold an ordered list of chunk dicts into doc-level results.

    ``ranked_chunks`` must be sorted by descending RRF score and
    each dict must include at least:

      - ``doc_id``
      - ``id`` (chunk id)
      - ``score_rrf`` (fused score, float)
      - ``score_vector`` (float | None)
      - ``score_bm25`` (float | None)
      - any LanceDB row fields needed by the UI

    Returns up to ``doc_limit`` ``DocResult`` objects, in the same
    order their best chunk first appeared.
    """
    by_doc: dict[str, DocResult] = {}
    for chunk in ranked_chunks:
        doc_id = chunk["doc_id"]
        existing = by_doc.get(doc_id)
        if existing is not None:
            # Same doc seen again — accumulate, never overwrite best.
            # Input is RRF-sorted descending, so the first chunk seen
            # per doc is already its best.
            existing.hit_count += 1
            existing.all_chunk_ids.append(chunk["id"])
            continue

        if len(by_doc) >= doc_limit:
            # We already collected ``doc_limit`` unique docs; further
            # chunks of *new* docs are dropped. Keep iterating in case
            # later chunks belong to a doc we already have (so hit_count
            # stays accurate).
            continue

        by_doc[doc_id] = DocResult(
            doc_id=doc_id,
            best_chunk=chunk,
            score_rrf=float(chunk["score_rrf"]),
            score_vector=_optional_float(chunk.get("score_vector")),
            score_bm25=_optional_float(chunk.get("score_bm25")),
            hit_count=1,
            all_chunk_ids=[chunk["id"]],
        )

    return list(by_doc.values())


def _optional_float(x: Any) -> float | None:
    if x is None:
        return None
    try:
        return float(x)
    except (TypeError, ValueError):
        return None
