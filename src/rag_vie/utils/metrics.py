"""Shared retrieval metrics and score normalization.

Single source of truth for the ranking metrics used by both training
(`scripts/train_mlp.py`) and evaluation (`scripts/evaluate_all.py`). Keeping one
implementation guarantees that the soft-label targets and the reported numbers
are computed by exactly the same code.
"""

import numpy as np


def ndcg_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    dcg = sum(1.0 / np.log2(i + 2) for i, p in enumerate(ranked[:k]) if p in relevant)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(relevant), k)))
    return dcg / idcg if idcg > 0 else 0.0


def mrr_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    for i, p in enumerate(ranked[:k]):
        if p in relevant:
            return 1.0 / (i + 1)
    return 0.0


def map_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    hits, s = 0, 0.0
    for i, p in enumerate(ranked[:k]):
        if p in relevant:
            hits += 1
            s += hits / (i + 1)
    denom = min(len(relevant), k)
    return s / denom if denom > 0 else 0.0


def recall_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    return sum(1 for p in ranked[:k] if p in relevant) / len(relevant) if relevant else 0.0


def hit_at_1(ranked: list[str], relevant: set[str]) -> float:
    return 1.0 if ranked and ranked[0] in relevant else 0.0


def min_max_normalize(scores: dict[str, float]) -> dict[str, float]:
    """Min-max normalize score values to [0, 1].

    BM25 and BGE-M3 sparse scores are unbounded, so every source must be
    normalized to a common range before weighted fusion. A flat distribution
    (span == 0) maps to all-zeros.
    """
    if not scores:
        return {}
    lo, hi = min(scores.values()), max(scores.values())
    span = hi - lo
    if span == 0:
        return {k: 0.0 for k in scores}
    return {k: (v - lo) / span for k, v in scores.items()}
