"""Reciprocal Rank Fusion for hybrid retrieval.

Fuse 2+ ranked lists into one by summing ``1 / (k + rank)`` per
ranker, where ``k = 60`` is the industry-standard tuning constant
(Elasticsearch / Vespa default). RRF's appeal:

- No need to normalize cosine similarity vs BM25 scores (totally
  different scales).
- Naturally robust to one ranker returning zero hits (e.g. image
  queries that have no BM25 path).
- Pure Python, O(total_hits), microseconds in practice.

The fuser takes ranked lists of ``(id, ...)`` tuples and only cares
about the ranks; the per-ranker scores are kept for debug / UI
display by ``hybrid.search`` but don't influence the fused order.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

RRF_K: int = 60
"""Industry-standard default (k=60). Not exposed as a user knob in v1."""


def reciprocal_rank_fusion(
    ranked_lists: Sequence[Iterable[tuple[str, Any]] | Iterable[str]],
    *,
    k: int = RRF_K,
) -> list[tuple[str, float]]:
    """Fuse ``ranked_lists`` into one combined ranking.

    Each input is a ranked iterable. Elements are either bare ``id``
    strings or ``(id, ...)`` tuples (only the first element matters).
    Position in the iterable is the rank: index 0 → rank 1, etc.

    Returns ``[(id, fused_score), ...]`` sorted by descending fused
    score. Stable on ties — first-encountered id wins, which (because
    we iterate vector before BM25) gives vector-favoring tie-breaks.
    A doc that appears once in vector at rank 0 beats a doc that
    appears once in BM25 at rank 0, which roughly matches v1 user
    intent: cross-modal queries lean on the vector side.
    """
    if k < 1:
        raise ValueError(f"k must be ≥ 1, got {k}")

    scores: dict[str, float] = {}
    order: list[str] = []  # insertion order for stable tiebreaks
    for lst in ranked_lists:
        for rank, item in enumerate(lst):
            doc_id = item if isinstance(item, str) else item[0]
            if doc_id not in scores:
                scores[doc_id] = 0.0
                order.append(doc_id)
            scores[doc_id] += 1.0 / (k + rank + 1)

    # Sort by descending fused score, tiebreak on insertion order.
    order_rank = {d: i for i, d in enumerate(order)}
    return sorted(
        scores.items(),
        key=lambda kv: (-kv[1], order_rank[kv[0]]),
    )
