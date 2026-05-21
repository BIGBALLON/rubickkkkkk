"""Retrieval layer — vector + BM25 + RRF fusion + doc-level folding.

Public pipeline (``hybrid_search``)::

    embed → (ANN top-50 ∥ BM25 top-50)
          → RRF (k=60)
          → metadata filter
          → doc-level fold (best chunk per doc, hit_count)
          → top-K docs (default 20, max 50)

Building blocks live as small, individually testable modules; the
FastAPI route and CLI both call ``hybrid_search``.
"""

from .bm25 import BM25Hit, bm25_search
from .fold import DocResult, fold_to_docs
from .hybrid import SearchResult, hybrid_search
from .rrf import RRF_K, reciprocal_rank_fusion
from .vector import VectorHit, vector_search

__all__ = [
    "RRF_K",
    "BM25Hit",
    "DocResult",
    "SearchResult",
    "VectorHit",
    "bm25_search",
    "fold_to_docs",
    "hybrid_search",
    "reciprocal_rank_fusion",
    "vector_search",
]
