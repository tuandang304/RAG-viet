"""Signal-aware features (P0-A).

Hand-crafted query features (``features/vietnamese.py``) describe the *query* but tell
the fusion MLP nothing about whether the three retrievers actually *agree* on this
particular query. These features are computed from the already-normalized score dicts
that the hybrid retriever produces, giving the MLP direct evidence of inter-retriever
(dis)agreement so it can move off the centre of the simplex with justification.

All features are bounded in [0, 1] and degrade gracefully to 0 when a source is empty
(e.g. ``--no-sparse``). Pure functions of the score dicts → cheap and unit-testable.
"""

import numpy as np

SIGNAL_FEATURE_NAMES = [
    "overlap_dense_bm25",    # Jaccard top-k giữa dense và bm25
    "overlap_dense_sparse",  # Jaccard top-k giữa dense và sparse
    "overlap_bm25_sparse",   # Jaccard top-k giữa bm25 và sparse
    "sharp_dense",           # độ nhọn phân phối điểm dense (max - mean)
    "sharp_bm25",            # độ nhọn phân phối điểm bm25
    "sharp_sparse",          # độ nhọn phân phối điểm sparse
    "top1_agreement",        # tỉ lệ cặp nguồn có cùng document top-1
]


def _top_ids(scores: dict[str, float], k: int) -> set[str]:
    if not scores:
        return set()
    return {pid for pid, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]}


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    union = a | b
    return len(a & b) / len(union) if union else 0.0


def _sharpness(scores: dict[str, float]) -> float:
    """max - mean của điểm đã chuẩn hoá [0,1]; cao = phân phối nhọn (một vài doc trội hẳn)."""
    if not scores:
        return 0.0
    vals = np.fromiter(scores.values(), dtype=np.float64)
    return float(vals.max() - vals.mean())


def _top1(scores: dict[str, float]) -> str | None:
    if not scores:
        return None
    return max(scores.items(), key=lambda x: x[1])[0]


def extract_signal_features(
    dense_norm: dict[str, float],
    bm25_norm: dict[str, float],
    sparse_norm: dict[str, float],
    top_k: int = 10,
) -> np.ndarray:
    """Return a 1-D float32 array of length ``len(SIGNAL_FEATURE_NAMES)``."""
    d_top = _top_ids(dense_norm, top_k)
    b_top = _top_ids(bm25_norm, top_k)
    s_top = _top_ids(sparse_norm, top_k)

    overlap_db = _jaccard(d_top, b_top)
    overlap_ds = _jaccard(d_top, s_top)
    overlap_bs = _jaccard(b_top, s_top)

    sharp_d = _sharpness(dense_norm)
    sharp_b = _sharpness(bm25_norm)
    sharp_s = _sharpness(sparse_norm)

    tops = [t for t in (_top1(dense_norm), _top1(bm25_norm), _top1(sparse_norm)) if t is not None]
    if len(tops) < 2:
        top1_agreement = 0.0
    else:
        pairs = [(i, j) for i in range(len(tops)) for j in range(i + 1, len(tops))]
        agree = sum(1 for i, j in pairs if tops[i] == tops[j])
        top1_agreement = agree / len(pairs)

    return np.array(
        [overlap_db, overlap_ds, overlap_bs, sharp_d, sharp_b, sharp_s, top1_agreement],
        dtype=np.float32,
    )
