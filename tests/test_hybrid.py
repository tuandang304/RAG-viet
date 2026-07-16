import numpy as np
import pytest

import rag_vie.retrieval.hybrid as hybrid_mod
from rag_vie.retrieval.hybrid import HybridRetriever, _min_max_normalize


class _StubRetriever:
    """Duck-typed stand-in for Dense/BM25/Sparse retrievers."""

    def __init__(self, hits: list[tuple[str, str, float]]) -> None:
        self._hits = hits

    def search(self, *args, **kwargs):
        return self._hits


@pytest.fixture(autouse=True)
def _no_embedding_api(monkeypatch):
    """HybridRetriever.retrieve calls embed_query (FPT API) — stub it out."""
    monkeypatch.setattr(
        hybrid_mod, "embed_query", lambda q: np.zeros((1, 4), dtype=np.float32)
    )


def test_min_max_normalize():
    norm = _min_max_normalize({"a": 2.0, "b": 6.0, "c": 4.0})
    assert norm == {"a": 0.0, "b": 1.0, "c": 0.5}
    # Constant scores collapse to 0; empty dict passes through
    assert _min_max_normalize({"a": 3.0, "b": 3.0}) == {"a": 0.0, "b": 0.0}
    assert _min_max_normalize({}) == {}


def test_two_way_fusion_weighted_sum():
    dense = _StubRetriever([("d1", "passage d1", 0.9), ("shared", "passage s", 0.5)])
    bm25 = _StubRetriever([("shared", "passage s", 10.0), ("b1", "passage b1", 2.0)])
    retriever = HybridRetriever(dense, bm25, sparse=None)

    hits = retriever.retrieve("q", weights=(0.5, 0.5), k_dense=10, k_bm25=10, k_final=10)
    fused = {pid: score for pid, _, score in hits}

    # dense norm: d1=1.0, shared=0.0 | bm25 norm: shared=1.0, b1=0.0
    assert fused == pytest.approx({"d1": 0.5, "shared": 0.5, "b1": 0.0})


def test_three_way_fusion_includes_sparse():
    dense = _StubRetriever([("d1", "p", 1.0), ("x", "p", 0.0)])
    bm25 = _StubRetriever([("b1", "p", 1.0), ("x", "p", 0.0)])
    sparse = _StubRetriever([("s1", "p", 1.0), ("x", "p", 0.0)])
    retriever = HybridRetriever(dense, bm25, sparse)

    hits = retriever.retrieve("q", weights=(0.2, 0.3, 0.5), k_dense=10, k_bm25=10, k_final=10)
    fused = {pid: score for pid, _, score in hits}
    assert fused["d1"] == pytest.approx(0.2)
    assert fused["b1"] == pytest.approx(0.3)
    assert fused["s1"] == pytest.approx(0.5)
    # Top-1 must be the sparse hit given the largest weight
    assert hits[0][0] == "s1"


def test_sparse_weight_zero_contributes_nothing():
    # Sparse IS always queried (its hits feed retrieval-signal features),
    # but with c == 0 it must not contribute to fused scores.
    # Two hits per channel so min-max normalization keeps a non-zero top score.
    dense = _StubRetriever([("d1", "p", 1.0), ("d0", "p", 0.0)])
    bm25 = _StubRetriever([("b1", "p", 1.0), ("b0", "p", 0.0)])
    sparse = _StubRetriever([("s1", "p", 99.0), ("s0", "p", 0.0)])
    retriever = HybridRetriever(dense, bm25, sparse)
    hits = retriever.retrieve("q", weights=(0.5, 0.5, 0.0), k_dense=10, k_bm25=10, k_final=10)
    fused = {pid: score for pid, _, score in hits}
    assert fused["s1"] == 0.0
    assert fused["d1"] == pytest.approx(0.5)


def test_search_all_returns_all_channels():
    dense = _StubRetriever([("d1", "p", 1.0)])
    bm25 = _StubRetriever([("b1", "p", 1.0)])
    sparse = _StubRetriever([("s1", "p", 1.0)])
    hits = HybridRetriever(dense, bm25, sparse).search_all("q", k_dense=10, k_bm25=10)
    assert set(hits) == {"dense", "bm25", "sparse", "toneless"}
    assert hits["sparse"][0][0] == "s1"
    # Channels without a retriever are present but empty
    assert hits["toneless"] == []
    hits2 = HybridRetriever(dense, bm25).search_all("q", k_dense=10, k_bm25=10)
    assert hits2["sparse"] == []


def test_four_way_fusion_with_toneless():
    dense = _StubRetriever([("d1", "p", 1.0), ("x", "p", 0.0)])
    bm25 = _StubRetriever([("b1", "p", 1.0), ("x", "p", 0.0)])
    sparse = _StubRetriever([("s1", "p", 1.0), ("x", "p", 0.0)])
    toneless = _StubRetriever([("t1", "p", 1.0), ("x", "p", 0.0)])
    retriever = HybridRetriever(dense, bm25, sparse, toneless)

    hits = retriever.search_all("q", k_dense=10, k_bm25=10)
    assert hits["toneless"][0][0] == "t1"

    fused = {
        pid: score
        for pid, _, score in retriever.fuse(hits, (0.1, 0.2, 0.3, 0.4), k_final=10)
    }
    assert fused["d1"] == pytest.approx(0.1)
    assert fused["b1"] == pytest.approx(0.2)
    assert fused["s1"] == pytest.approx(0.3)
    assert fused["t1"] == pytest.approx(0.4)


def test_three_weight_tuple_still_works_with_toneless_retriever():
    # Legacy 3-way checkpoints emit 3 weights; the toneless channel must
    # silently get weight 0 instead of crashing.
    dense = _StubRetriever([("d1", "p", 1.0), ("d0", "p", 0.0)])
    bm25 = _StubRetriever([])
    toneless = _StubRetriever([("t1", "p", 5.0), ("t0", "p", 0.0)])
    retriever = HybridRetriever(dense, bm25, sparse=None, toneless=toneless)
    hits = retriever.retrieve("q", weights=(1.0, 0.0, 0.0), k_dense=10, k_bm25=10, k_final=10)
    fused = {pid: score for pid, _, score in hits}
    assert fused["d1"] == pytest.approx(1.0)
    assert fused["t1"] == 0.0


def test_k_final_truncates():
    dense = _StubRetriever([(f"d{i}", "p", float(10 - i)) for i in range(10)])
    bm25 = _StubRetriever([])
    retriever = HybridRetriever(dense, bm25)
    hits = retriever.retrieve("q", weights=(1.0, 0.0), k_dense=10, k_bm25=10, k_final=3)
    assert len(hits) == 3
