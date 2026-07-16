"""Weighted score fusion shared across retrieval and evaluation."""


def fuse_scores(
    dense_norm: dict[str, float],
    bm25_norm: dict[str, float],
    sparse_norm: dict[str, float],
    weights: tuple[float, ...],
    toneless_norm: dict[str, float] | None = None,
) -> dict[str, float]:
    """Fuse normalized score dicts: a·dense + b·bm25 + c·sparse (+ d·toneless).

    `weights` may carry 2–4 components in (dense, bm25, sparse, toneless)
    order; missing trailing weights are treated as 0. Returns the full
    fused-score dict over the union of candidate ids; callers sort/truncate
    to top-k as needed.
    """
    toneless_norm = toneless_norm or {}
    a, b, c, d = tuple(weights) + (0.0,) * (4 - len(weights))
    ids = set(dense_norm) | set(bm25_norm) | set(sparse_norm) | set(toneless_norm)
    return {
        pid: a * dense_norm.get(pid, 0.0)
        + b * bm25_norm.get(pid, 0.0)
        + c * sparse_norm.get(pid, 0.0)
        + d * toneless_norm.get(pid, 0.0)
        for pid in ids
    }
