import numpy as np
import pytest

import rag_vie.pipeline as pipeline_mod
import rag_vie.retrieval.hybrid as hybrid_mod
from rag_vie.features.vietnamese import FEATURE_NAMES
from rag_vie.pipeline import BASELINE_METHODS, RAGPipeline


class _StubRetriever:
    """Duck-typed stand-in for Dense/BM25/Sparse retrievers."""

    def __init__(self, hits: list[tuple[str, str, float]]) -> None:
        self._hits = hits

    def search(self, *args, **kwargs):
        return self._hits


class _StubBM25(_StubRetriever):
    vocab: set[str] = set()


class _StubMLP:
    input_dim = 8  # matches len(FEATURE_NAMES) — no retrieval-signal branch

    def predict_weights(self, features):
        return (0.5, 0.3, 0.2, 0.0)


@pytest.fixture(autouse=True)
def _stub_externals(monkeypatch):
    monkeypatch.setattr(
        pipeline_mod, "extract_features", lambda *a, **k: np.zeros(8, dtype=np.float32)
    )
    monkeypatch.setattr(
        pipeline_mod, "generate", lambda query, passages, **k: f"answer for {query}"
    )
    monkeypatch.setattr(hybrid_mod, "embed_query", lambda q: np.zeros((1, 4), dtype=np.float32))


def _make_pipeline(use_generator: bool = True) -> RAGPipeline:
    dense = _StubRetriever([("d1", "dense passage", 1.0), ("shared", "shared passage", 0.5)])
    bm25 = _StubBM25([("shared", "shared passage", 10.0), ("b1", "bm25 passage", 2.0)])
    mlp = _StubMLP()
    return RAGPipeline(dense, bm25, mlp, sparse=None, use_generator=use_generator)


def test_run_uses_mlp_weights_and_generates():
    pipeline = _make_pipeline()
    result = pipeline.run("query")
    assert result.weights == (0.5, 0.3, 0.2, 0.0)
    assert result.answer == "answer for query"
    assert len(result.retrieved) > 0


def test_run_surfaces_all_linguistic_features():
    pipeline = _make_pipeline()
    result = pipeline.run("query")
    # The router's 8 linguistic features are exposed for the UI to explain routing.
    assert set(result.features) == set(FEATURE_NAMES)
    assert all(isinstance(v, float) for v in result.features.values())


def test_compare_shares_query_level_features_across_methods():
    pipeline = _make_pipeline()
    results = pipeline.compare("query")
    feats = [tuple(sorted(r.features.items())) for r in results.values()]
    assert all(f == feats[0] for f in feats)          # identical across methods
    assert set(results["mlp"].features) == set(FEATURE_NAMES)


def test_compare_covers_all_baselines_with_fixed_weights():
    pipeline = _make_pipeline()
    results = pipeline.compare("query")
    assert set(results) == set(BASELINE_METHODS)
    assert results["mlp"].weights == (0.5, 0.3, 0.2, 0.0)
    assert results["fixed_equal"].weights == (1 / 3, 1 / 3, 1 / 3, 0.0)
    assert results["dense"].weights == (1.0, 0.0, 0.0, 0.0)
    assert results["bm25"].weights == (0.0, 1.0, 0.0, 0.0)


def test_compare_only_generates_for_mlp_by_default():
    pipeline = _make_pipeline()
    results = pipeline.compare("query")
    assert results["mlp"].answer == "answer for query"
    for name, result in results.items():
        if name != "mlp":
            assert result.answer == ""


def test_compare_respects_custom_generate_for():
    pipeline = _make_pipeline()
    results = pipeline.compare("query", generate_for={"mlp", "dense"})
    assert results["dense"].answer == "answer for query"
    assert results["bm25"].answer == ""


def test_compare_skips_generation_when_generator_disabled():
    pipeline = _make_pipeline(use_generator=False)
    results = pipeline.compare("query")
    assert all(r.answer == "" for r in results.values())
