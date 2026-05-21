"""Unit tests for the RRF fuser and doc-level fold.

Pure-Python math — no LanceDB, no model load, runs in milliseconds.
End-to-end coverage (vector + BM25 + RRF + fold against a real
LanceDB index) lives in ``test_smoke_e2e.py`` behind ``@slow``.
"""

from __future__ import annotations

import pytest

from rubick_backend.retrieve.fold import fold_to_docs
from rubick_backend.retrieve.rrf import RRF_K, reciprocal_rank_fusion

# === RRF math ===============================================================


def test_rrf_single_list_preserves_order() -> None:
    """A single ranked input → RRF is monotone in rank, so order is
    preserved verbatim and scores are ``1/(k + 1 + i)``."""
    out = reciprocal_rank_fusion([["a", "b", "c"]], k=60)
    ids = [doc_id for doc_id, _ in out]
    assert ids == ["a", "b", "c"]
    assert out[0][1] == pytest.approx(1 / 61, rel=1e-6)
    assert out[1][1] == pytest.approx(1 / 62, rel=1e-6)
    assert out[2][1] == pytest.approx(1 / 63, rel=1e-6)


def test_rrf_two_lists_sum_when_doc_appears_in_both() -> None:
    """A doc that hits both rankers should outrank a doc that hits
    only one — modulo positions. ``a`` is rank-0 in both lists, so
    its RRF score is ``1/61 + 1/61``. ``b`` is rank-1 in one only:
    ``1/62``. So a > b > c."""
    vec = ["a", "b"]
    bm25 = ["a", "c"]
    out = reciprocal_rank_fusion([vec, bm25], k=60)
    ids = [doc_id for doc_id, _ in out]
    assert ids[0] == "a"
    assert set(ids) == {"a", "b", "c"}
    a_score = next(s for d, s in out if d == "a")
    b_score = next(s for d, s in out if d == "b")
    c_score = next(s for d, s in out if d == "c")
    assert a_score == pytest.approx(2 / 61, rel=1e-6)
    assert b_score == pytest.approx(1 / 62, rel=1e-6)
    assert c_score == pytest.approx(1 / 62, rel=1e-6)


def test_rrf_handles_empty_lists() -> None:
    """Empty rankers contribute nothing — image-only queries pass
    ``bm25=[]`` and should fall back to vector-only ordering."""
    out = reciprocal_rank_fusion([["a", "b"], []], k=60)
    assert [d for d, _ in out] == ["a", "b"]

    out2 = reciprocal_rank_fusion([[], []], k=60)
    assert out2 == []


def test_rrf_accepts_tuple_inputs() -> None:
    """Real callers pass ``(id, score)`` tuples; the fuser must use
    the first element as the id and ignore the score."""
    vec = [("a", 0.9), ("b", 0.5)]
    bm25 = [("c", 12.3), ("a", 8.0)]
    out = reciprocal_rank_fusion([vec, bm25], k=60)
    ids_only = {d for d, _ in out}
    assert ids_only == {"a", "b", "c"}
    top_id = out[0][0]
    assert top_id == "a"  # hits both rankers


def test_rrf_rejects_invalid_k() -> None:
    with pytest.raises(ValueError):
        reciprocal_rank_fusion([["a"]], k=0)
    with pytest.raises(ValueError):
        reciprocal_rank_fusion([["a"]], k=-5)


def test_rrf_default_k_matches_spec() -> None:
    """RRF k=60 default — guard against accidental constant drift."""
    assert RRF_K == 60


def test_rrf_stable_tiebreak_favors_first_list() -> None:
    """When two docs end up with the same fused score, the one that
    first appeared (vector before BM25 in our pipeline) should win.
    """
    # 'a' appears at rank 0 of list 1; 'b' at rank 0 of list 2.
    # Both get score 1/61. Tiebreak by insertion order → 'a' first.
    out = reciprocal_rank_fusion([["a"], ["b"]], k=60)
    assert [d for d, _ in out] == ["a", "b"]


# === Doc fold ==============================================================


def _chunk(
    *,
    chunk_id: str,
    doc_id: str,
    rrf: float,
    vector: float | None = 0.5,
    bm25: float | None = None,
) -> dict:
    """Build a minimal RRF-output chunk dict for fold tests."""
    return {
        "id": chunk_id,
        "doc_id": doc_id,
        "score_rrf": rrf,
        "score_vector": vector,
        "score_bm25": bm25,
    }


def test_fold_collapses_same_doc_chunks() -> None:
    """Three chunks of the same doc → one DocResult with hit_count=3."""
    chunks = [
        _chunk(chunk_id="x-text-0", doc_id="x", rrf=0.08),
        _chunk(chunk_id="x-text-1", doc_id="x", rrf=0.05),
        _chunk(chunk_id="x-text-2", doc_id="x", rrf=0.03),
    ]
    out = fold_to_docs(chunks, doc_limit=20)
    assert len(out) == 1
    doc = out[0]
    assert doc.doc_id == "x"
    assert doc.hit_count == 3
    assert doc.best_chunk["id"] == "x-text-0"
    assert doc.all_chunk_ids == ["x-text-0", "x-text-1", "x-text-2"]
    assert doc.score_rrf == pytest.approx(0.08)


def test_fold_preserves_best_chunk_per_doc() -> None:
    """The first chunk seen for each doc is the best (input is
    RRF-sorted). Folding must not overwrite that with a later
    (worse) chunk."""
    chunks = [
        _chunk(chunk_id="x-text-2", doc_id="x", rrf=0.08, vector=0.42),
        _chunk(chunk_id="y-image-0", doc_id="y", rrf=0.07, vector=0.31),
        _chunk(chunk_id="x-text-0", doc_id="x", rrf=0.04, vector=0.99),
    ]
    out = fold_to_docs(chunks, doc_limit=20)
    assert {d.doc_id for d in out} == {"x", "y"}
    x_doc = next(d for d in out if d.doc_id == "x")
    assert x_doc.best_chunk["id"] == "x-text-2"
    assert x_doc.score_vector == pytest.approx(0.42)
    assert x_doc.hit_count == 2


def test_fold_caps_at_doc_limit_but_keeps_existing_doc_counts() -> None:
    """``doc_limit`` caps the number of distinct docs, but additional
    chunks of *already-seen* docs still bump hit_count."""
    chunks = [
        _chunk(chunk_id="a-text-0", doc_id="a", rrf=0.10),
        _chunk(chunk_id="b-text-0", doc_id="b", rrf=0.09),
        _chunk(chunk_id="c-text-0", doc_id="c", rrf=0.08),  # past cap
        _chunk(chunk_id="a-text-1", doc_id="a", rrf=0.05),  # still counts
    ]
    out = fold_to_docs(chunks, doc_limit=2)
    assert [d.doc_id for d in out] == ["a", "b"]
    assert next(d for d in out if d.doc_id == "a").hit_count == 2


def test_fold_empty_input_returns_empty() -> None:
    assert fold_to_docs([], doc_limit=20) == []
