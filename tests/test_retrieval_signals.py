import numpy as np
import pytest

from rag_vie.features.retrieval_signals import (
    SIGNAL_NAMES,
    SIGNAL_NAMES_4WAY,
    extract_retrieval_signals,
)

# A "confident" channel: one hit stands far above the rest
CONFIDENT = {"p1": 10.0, "p2": 2.0, "p3": 1.9, "p4": 1.8, "p5": 1.7}
# A "flat" channel: near-ties everywhere
FLAT = {"q1": 5.0, "q2": 4.99, "q3": 4.98, "q4": 4.97, "q5": 0.0}


def _idx(name: str) -> int:
    return SIGNAL_NAMES.index(name)


def test_shape_and_range():
    sig = extract_retrieval_signals(CONFIDENT, FLAT, {})
    assert sig.shape == (len(SIGNAL_NAMES),)
    assert sig.dtype == np.float32
    assert np.all(sig >= 0.0) and np.all(sig <= 1.0)


def test_confident_channel_has_larger_top1_gap():
    sig = extract_retrieval_signals(CONFIDENT, FLAT, {})
    assert sig[_idx("dense_top1_gap")] > sig[_idx("bm25_top1_gap")]


def test_scale_and_shift_invariance():
    base = extract_retrieval_signals(CONFIDENT, FLAT, {})
    scaled = extract_retrieval_signals(
        {k: v * 37.5 + 4.2 for k, v in CONFIDENT.items()},
        {k: v * 0.001 + 100.0 for k, v in FLAT.items()},
        {},
    )
    np.testing.assert_allclose(base, scaled, atol=1e-6)


def test_empty_channel_yields_zero_stats():
    sig = extract_retrieval_signals(CONFIDENT, {}, {})
    for stat in ("top1_gap", "top10_std", "top10_mean", "coverage"):
        assert sig[_idx(f"bm25_{stat}")] == 0.0
        assert sig[_idx(f"sparse_{stat}")] == 0.0


def test_all_tied_scores_yield_zero_shape_stats():
    # e.g. BM25 with zero score for every candidate (OOV query)
    tied = {"a": 0.0, "b": 0.0, "c": 0.0}
    sig = extract_retrieval_signals(tied, {}, {})
    assert sig[_idx("dense_top1_gap")] == 0.0
    assert sig[_idx("dense_top10_std")] == 0.0
    assert sig[_idx("dense_coverage")] == pytest.approx(0.3)


def test_overlap_and_top1_agreement():
    a = {"x": 3.0, "y": 2.0, "z": 1.0}
    b = {"x": 9.0, "y": 8.0, "w": 7.0}   # shares {x, y} with a, same top-1
    c = {"m": 1.0}                        # disjoint
    sig = extract_retrieval_signals(a, b, c)
    assert sig[_idx("overlap_dense_bm25")] == pytest.approx(2 / 4)
    assert sig[_idx("overlap_dense_sparse")] == 0.0
    assert sig[_idx("top1_agree_dense_bm25")] == 1.0
    assert sig[_idx("top1_agree_dense_sparse")] == 0.0


def test_signal_names_unique_and_sized():
    assert len(SIGNAL_NAMES) == 18
    assert len(set(SIGNAL_NAMES)) == 18
    assert len(SIGNAL_NAMES_4WAY) == 28
    assert len(set(SIGNAL_NAMES_4WAY)) == 28
    # 3-way names are a strict prefix-compatible subset pattern: per-channel
    # stats for dense/bm25/sparse appear in both layouts
    assert SIGNAL_NAMES[:12] == SIGNAL_NAMES_4WAY[:12]


def test_4way_layout_includes_toneless():
    a = {"x": 3.0, "y": 2.0}
    t = {"x": 9.0, "z": 1.0}
    sig = extract_retrieval_signals(a, {}, {}, toneless_scores=t)
    assert sig.shape == (len(SIGNAL_NAMES_4WAY),)
    i4 = SIGNAL_NAMES_4WAY.index
    assert sig[i4("toneless_coverage")] == pytest.approx(0.2)
    assert sig[i4("overlap_dense_toneless")] == pytest.approx(1 / 3)
    assert sig[i4("top1_agree_dense_toneless")] == 1.0
    # 3-way call must be unchanged by the 4-way extension
    sig3 = extract_retrieval_signals(a, {}, {})
    assert sig3.shape == (len(SIGNAL_NAMES),)
