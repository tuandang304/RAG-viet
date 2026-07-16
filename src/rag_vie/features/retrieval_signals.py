"""Post-retrieval signal features for the fusion router (QPP-style).

The 8 linguistic features in `vietnamese.py` describe only the query itself.
Which retrieval channel wins, however, also depends on how the corpus/indexes
respond to the query. These features summarise each channel's *response*:
score-distribution shape per channel (how confident the channel is) and
top-k agreement across channels (how much the channels disagree — the only
case where routing weights actually matter).

Design constraints:
  * Scale-invariant — every per-channel statistic is computed on scores
    normalized within the channel's own top-k window, so raw dense (cosine),
    BM25 (unbounded) and sparse (unbounded) scores are all comparable and the
    features transfer zero-shot across corpora with different score scales.
  * Label-free — computed only from returned (id, score) pairs; never touches
    relevant_ids.
  * Free at inference — all channels are already retrieved before fusion;
    this adds a few numpy ops on lists that are already in memory.

Channel layouts:
  * 3-way (dense, bm25, sparse)            → 18 features (SIGNAL_NAMES)
  * 4-way (dense, bm25, sparse, toneless)  → 28 features (SIGNAL_NAMES_4WAY)
  The 4-way layout is selected by passing `toneless_scores`.

Usage:
    signals = extract_retrieval_signals(dense_scores, bm25_scores, sparse_scores)
    signals = extract_retrieval_signals(d, b, s, toneless_scores=t)   # 4-way
    features = np.concatenate([linguistic_features, signals])
"""

from __future__ import annotations

from itertools import combinations

import numpy as np

# Window size for score-shape statistics and overlap sets.
SIGNAL_TOP_K = 10

_CHANNELS_3WAY = ("dense", "bm25", "sparse")
_CHANNELS_4WAY = ("dense", "bm25", "sparse", "toneless")

_PER_CHANNEL = ("top1_gap", "top10_std", "top10_mean", "coverage")


def _signal_names(channels: tuple[str, ...]) -> list[str]:
    names = [f"{ch}_{stat}" for ch in channels for stat in _PER_CHANNEL]
    names += [f"overlap_{a}_{b}" for a, b in combinations(channels, 2)]
    names += [f"top1_agree_{a}_{b}" for a, b in combinations(channels, 2)]
    return names


SIGNAL_NAMES: list[str] = _signal_names(_CHANNELS_3WAY)          # 18
SIGNAL_NAMES_4WAY: list[str] = _signal_names(_CHANNELS_4WAY)     # 28


def _sorted_scores(scores: dict[str, float]) -> tuple[list[str], np.ndarray]:
    """Ids and scores sorted by score descending."""
    if not scores:
        return [], np.empty(0, dtype=np.float64)
    items = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    ids = [k for k, _ in items]
    vals = np.array([v for _, v in items], dtype=np.float64)
    return ids, vals


def _channel_stats(vals: np.ndarray, k: int) -> tuple[float, float, float, float]:
    """(top1_gap, top10_std, top10_mean, coverage) for one channel.

    Statistics are computed on the top-k window normalized to [0, 1] via
    (s - s_k) / (s_1 - s_k), making them invariant to any positive affine
    transform of the raw scores:
      top1_gap   — (s1 - s2) / span: does one hit stand out from the rest?
      top10_std  — std of the normalized window: curvature of the score drop.
      top10_mean — mean of the normalized window: high = plateau near the top
                   (many near-ties), low = steep single-peak drop.
      coverage   — fraction of the k requested hits actually returned.
    A flat window (s1 == sk, e.g. all-zero BM25 scores on an OOV query)
    carries no ranking information → all shape stats are 0.
    """
    n = min(len(vals), k)
    coverage = n / k
    if n < 2:
        return 0.0, 0.0, 0.0, coverage
    window = vals[:n]
    span = window[0] - window[-1]
    if span <= 0:
        return 0.0, 0.0, 0.0, coverage
    norm = (window - window[-1]) / span
    top1_gap = float((window[0] - window[1]) / span)
    return top1_gap, float(norm.std()), float(norm.mean()), coverage


def _jaccard(a: list[str], b: list[str], k: int) -> float:
    sa, sb = set(a[:k]), set(b[:k])
    if not sa and not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def _top1_agree(a: list[str], b: list[str]) -> float:
    return 1.0 if a and b and a[0] == b[0] else 0.0


def extract_retrieval_signals(
    dense_scores: dict[str, float],
    bm25_scores: dict[str, float],
    sparse_scores: dict[str, float],
    toneless_scores: dict[str, float] | None = None,
    k: int = SIGNAL_TOP_K,
) -> np.ndarray:
    """Return a float32 signal vector.

    3-way (toneless_scores=None): length len(SIGNAL_NAMES) = 18.
    4-way (toneless_scores given): length len(SIGNAL_NAMES_4WAY) = 28.

    Args are {passage_id: score} per channel — raw or normalized scores give
    identical output (see scale-invariance note in the module docstring).
    An empty dict (channel unavailable) yields all-zero stats for that channel.
    """
    channel_scores: list[dict[str, float]] = [dense_scores, bm25_scores, sparse_scores]
    if toneless_scores is not None:
        channel_scores.append(toneless_scores)

    sorted_channels = [_sorted_scores(s) for s in channel_scores]

    feats: list[float] = []
    for _, vals in sorted_channels:
        feats.extend(_channel_stats(vals, k))

    id_lists = [ids for ids, _ in sorted_channels]
    for a, b in combinations(range(len(id_lists)), 2):
        feats.append(_jaccard(id_lists[a], id_lists[b], k))
    for a, b in combinations(range(len(id_lists)), 2):
        feats.append(_top1_agree(id_lists[a], id_lists[b]))

    return np.array(feats, dtype=np.float32)
