"""Vector (ANN) leg of hybrid retrieval.

Pulled out of ``api/search.py`` so RRF / hybrid orchestration can call
it independently of HTTP. Also lets the CLI use the same code path.

A vector hit returns the raw LanceDB row as a dict (we don't materialize
a Pydantic model in v1 — Lance returns numpy types / nullable strings
as NaN, and ``hybrid.search`` handles the JSON normalization once at
the end).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ..store import open_table


@dataclass(frozen=True, slots=True)
class VectorHit:
    """One ANN match. ``similarity`` is cosine in [-1, 1].

    Keep the raw row dict alongside the score so ``hybrid.search`` can
    fold doc rows without re-querying.
    """

    id: str
    similarity: float
    row: dict[str, Any]


def vector_search(
    qvec: np.ndarray,
    *,
    limit: int = 50,
    where: str | None = None,
    table=None,
) -> list[VectorHit]:
    """Run an ANN cosine search.

    - ``qvec``: 768-dim L2-normalized query vector (caller ensures this;
      ``embed.embed_query`` already normalizes).
    - ``limit``: top-K cap before fusion (default 50).
    - ``where``: optional LanceDB SQL filter (string), e.g.
      ``"modality != 'rejected'"``.
    - ``table``: optional pre-opened ``lancedb.table.Table``; for tests
      and tight loops. Defaults to the singleton via ``open_table()``.

    Returns hits in similarity-descending order.
    """
    table = table if table is not None else open_table()

    q = table.search(qvec).metric("cosine").limit(limit)
    if where:
        q = q.where(where)
    df = q.to_pandas()

    hits: list[VectorHit] = []
    for _, row in df.iterrows():
        # LanceDB cosine distance = 1 - cosine_similarity for L2-norm vecs.
        sim = 1.0 - float(row["_distance"])
        hits.append(
            VectorHit(
                id=str(row["id"]),
                similarity=sim,
                row=row.to_dict(),
            )
        )
    return hits
