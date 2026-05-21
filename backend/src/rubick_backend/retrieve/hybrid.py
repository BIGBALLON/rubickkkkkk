"""Public orchestrator: vector ANN + BM25 FTS → RRF → doc fold.

This is the single function the FastAPI route and the CLI both call.
Embedding is the *caller's* responsibility — it always happens before
``hybrid_search``, so the API handler can dispatch it to a worker
thread independently of the LanceDB queries.

The two ranker legs (vector / BM25) are run sequentially inside the
synchronous ``hybrid_search`` because LanceDB internally serializes
table reads on a single connection. The API handler wraps the whole
call in ``asyncio.to_thread`` so it doesn't block the event loop.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd

from .bm25 import bm25_search
from .fold import DocResult, fold_to_docs
from .rrf import RRF_K, reciprocal_rank_fusion
from .vector import vector_search

log = logging.getLogger(__name__)

# Default recall / doc-limit constants for hybrid search.
DEFAULT_RECALL: int = 50  # per-ranker top-K before fusion
DEFAULT_DOC_LIMIT: int = 20  # docs returned to the UI


@dataclass(slots=True)
class SearchResult:
    """A single hit returned to the API caller.

    Field naming matches the ``GET /search`` JSON shape — we serialize
    via ``asdict()`` so adding a field here surfaces it on the wire.

    Backwards compat: ``similarity`` remains the best-chunk cosine so
    the Swift ``SearchHit`` decoder keeps working unchanged.
    """

    id: str
    doc_id: str
    modality: str
    chunk_idx: int
    file_paths: list[str]
    filename: str
    raw_text: str | None
    thumbnail_path: str | None
    similarity: float  # vector similarity (cosine); back-compat field
    score_rrf: float
    score_vector: float | None
    score_bm25: float | None
    hit_count: int


def hybrid_search(
    *,
    qvec: np.ndarray | None,
    qtext: str | None,
    recall: int = DEFAULT_RECALL,
    doc_limit: int = DEFAULT_DOC_LIMIT,
    modality: str | None = None,
    include_rejected: bool = False,
    path_prefix: str | None = None,
    mtime_after: int | None = None,
    mtime_before: int | None = None,
    table=None,
) -> list[SearchResult]:
    """Run vector + BM25 + RRF + fold; return doc-level results.

    Either ``qvec`` or ``qtext`` (or both) must be provided. If only
    one is set, the other leg is skipped — e.g. image-by-image
    queries can pass ``qtext=None`` to get pure vector retrieval.

    The ``where`` clause is composed from filter args:

    - ``modality`` exact match (string).
    - ``include_rejected``: hide rows with ``modality == 'rejected'`` by
      default — UI never wants to surface "we couldn't process this"
      placeholders.
    - ``path_prefix``: keep only docs whose canonical first path starts
      with this prefix (matches the spec's "scope to a folder" facet).
    - ``mtime_after`` / ``mtime_before``: file mtime (POSIX epoch
      seconds) range, both inclusive. ``None`` means open-ended on
      that side. Spec mentions ``created_at / mtime / exif_taken_at``
      as candidate time fields; v1 picks ``mtime`` because that's what
      users intuitively mean by "files I touched recently"
      (``created_at`` is when *Rubick* ingested, exposed as a future
      ``ingested_*`` filter).
    """
    if qvec is None and (qtext is None or not qtext.strip()):
        raise ValueError("hybrid_search requires at least one of qvec / qtext")

    where = _build_where(
        modality=modality,
        include_rejected=include_rejected,
        path_prefix=path_prefix,
        mtime_after=mtime_after,
        mtime_before=mtime_before,
    )

    vec_hits = (
        vector_search(qvec, limit=recall, where=where, table=table) if qvec is not None else []
    )
    bm25_hits = (
        bm25_search(qtext, limit=recall, where=where, table=table)
        if qtext and qtext.strip()
        else []
    )

    # Build the RRF inputs as ranked lists of (id, ...) tuples.
    fused = reciprocal_rank_fusion(
        [
            [(h.id, h.similarity) for h in vec_hits],
            [(h.id, h.score) for h in bm25_hits],
        ],
        k=RRF_K,
    )

    # Re-attach the LanceDB row data so we can build SearchResult
    # objects without re-querying. ``vec_hits`` and ``bm25_hits`` both
    # carry the same row schema; vector wins on ties (better metadata
    # confidence — BM25's row can sometimes lack ``_distance``).
    row_by_id: dict[str, dict[str, Any]] = {}
    sim_by_id: dict[str, float] = {}
    bm25_by_id: dict[str, float] = {}
    for h in vec_hits:
        row_by_id[h.id] = h.row
        sim_by_id[h.id] = h.similarity
    for h in bm25_hits:
        if h.id not in row_by_id:
            row_by_id[h.id] = h.row
        bm25_by_id[h.id] = h.score

    # Build the per-chunk dicts the folder expects.
    ranked_chunks: list[dict[str, Any]] = []
    for chunk_id, rrf_score in fused:
        row = row_by_id.get(chunk_id)
        if row is None:
            log.warning("rrf produced id=%s with no backing row", chunk_id)
            continue
        ranked_chunks.append(
            {
                **row,
                "score_rrf": rrf_score,
                "score_vector": sim_by_id.get(chunk_id),
                "score_bm25": bm25_by_id.get(chunk_id),
            }
        )

    docs: list[DocResult] = fold_to_docs(ranked_chunks, doc_limit=doc_limit)
    return [_doc_to_searchresult(d) for d in docs]


# === Internals ==============================================================


def _build_where(
    *,
    modality: str | None,
    include_rejected: bool,
    path_prefix: str | None = None,
    mtime_after: int | None = None,
    mtime_before: int | None = None,
) -> str | None:
    """Compose the LanceDB SQL WHERE clause from filter args.

    - ``modality`` exact-match if provided.
    - ``rejected`` rows hidden unless explicitly requested. The
      placeholder rows from oversize video exist *only* to prevent
      re-scan, never as search hits.
    - ``path_prefix`` matches against ``file_paths[1]`` (Datafusion
      arrays are 1-indexed); single-quotes inside the prefix are
      escaped by doubling per SQL standard. Multi-path (hardlink) docs
      get filtered on their canonical first path only — acceptable v1
      trade-off for the typical case where a doc has exactly one path.
    - ``mtime_after`` / ``mtime_before`` are POSIX-epoch ints, both
      inclusive. Negative values raise — the caller is expected to
      reject obvious garbage before reaching the SQL builder.
    """
    clauses: list[str] = []
    if modality:
        # Comma-separated modality becomes SQL ``IN`` so multi-select
        # sidebar facets compose in one round-trip.
        tokens = [t.strip() for t in modality.split(",") if t.strip()]
        if not tokens:
            raise ValueError(f"empty modality filter: {modality!r}")
        for tok in tokens:
            # Sanitize: only allow simple identifier-like values. The
            # store schema validates against MODALITIES on write, so
            # any value the user could legitimately pass is
            # ``[a-zA-Z0-9_]+``.
            if not tok.replace("_", "").isalnum():
                raise ValueError(f"illegal modality filter: {tok!r}")
        if len(tokens) == 1:
            clauses.append(f"modality = '{tokens[0]}'")
        else:
            quoted = ", ".join(f"'{t}'" for t in tokens)
            clauses.append(f"modality IN ({quoted})")
    if not include_rejected:
        clauses.append("modality != 'rejected'")
    if path_prefix:
        escaped = path_prefix.replace("'", "''")
        clauses.append(f"starts_with(file_paths[1], '{escaped}')")
    if mtime_after is not None:
        if mtime_after < 0:
            raise ValueError(f"mtime_after must be >= 0, got {mtime_after}")
        clauses.append(f"mtime >= {int(mtime_after)}")
    if mtime_before is not None:
        if mtime_before < 0:
            raise ValueError(f"mtime_before must be >= 0, got {mtime_before}")
        if mtime_after is not None and mtime_before < mtime_after:
            raise ValueError(
                f"mtime_before ({mtime_before}) must be >= mtime_after ({mtime_after})"
            )
        clauses.append(f"mtime <= {int(mtime_before)}")
    if not clauses:
        return None
    return " AND ".join(clauses)


def _doc_to_searchresult(d: DocResult) -> SearchResult:
    """Convert a ``DocResult`` (LanceDB-row-flavored) into the JSON-safe
    ``SearchResult`` shape. Mostly NaN → None normalization for
    nullable string columns LanceDB round-trips via pandas.
    """
    row = d.best_chunk
    return SearchResult(
        id=str(row["id"]),
        doc_id=str(row["doc_id"]),
        modality=str(row["modality"]),
        chunk_idx=int(row["chunk_idx"]),
        file_paths=list(row["file_paths"]),
        filename=str(row["filename"]),
        raw_text=_normalize_text(row.get("raw_text")),
        thumbnail_path=_normalize_text(row.get("thumbnail_path")),
        similarity=float(d.score_vector) if d.score_vector is not None else 0.0,
        score_rrf=float(d.score_rrf),
        score_vector=d.score_vector,
        score_bm25=d.score_bm25,
        hit_count=int(d.hit_count),
    )


def _normalize_text(x: Any) -> str | None:
    """LanceDB nullable strings round-trip as pandas NaN — turn those
    into ``None`` for JSON. We also accept Python ``None`` directly
    (BM25's ``row.to_dict()`` preserves it cleanly)."""
    if x is None:
        return None
    try:
        if pd.isna(x):
            return None
    except (TypeError, ValueError):
        # ``pd.isna`` raises on numpy arrays etc.; not our case here.
        pass
    return str(x)


__all__ = ["DEFAULT_DOC_LIMIT", "DEFAULT_RECALL", "SearchResult", "asdict", "hybrid_search"]
