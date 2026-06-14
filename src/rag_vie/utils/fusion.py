"""Three-way weighted score fusion shared across retrieval and evaluation."""


def fuse_scores(
    dense_norm: dict[str, float],
    bm25_norm: dict[str, float],
    sparse_norm: dict[str, float],
    weights: tuple[float, float, float],
) -> dict[str, float]:
    """Fuse three normalized score dicts: a·dense + b·bm25 + c·sparse.

    Returns the full fused-score dict over the union of candidate ids; callers
    sort/truncate to top-k as needed.
    """
    a, b, c = weights
    ids = set(dense_norm) | set(bm25_norm) | set(sparse_norm)
    return {
        pid: a * dense_norm.get(pid, 0.0)
        + b * bm25_norm.get(pid, 0.0)
        + c * sparse_norm.get(pid, 0.0)
        for pid in ids
    }
