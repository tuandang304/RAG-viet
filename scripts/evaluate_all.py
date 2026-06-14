"""evaluate_all.py — Unified retrieval evaluation for Dynamic Hybrid RAG (3-way).

Metric groups:
  1. Retrieval   — NDCG@10, MRR@10, MAP@10, Recall@10, Recall@100, Hit@1
  2. Significance — paired t-test, Wilcoxon signed-rank, 95% bootstrap CI
  3. Efficiency  — MLP param count, index sizes, MLP inference latency,
                   retrieval+fusion latency (p50/p95) and throughput
  4. Weights     — entropy, distribution stats, Pearson correlations
  + Stratified analysis (11 query-feature strata)

Each query embeds exactly once; all methods share the same dense/BM25/sparse hits.

Usage:
    uv run python scripts/evaluate_all.py \\
        --qas-path data/processed/viaquad_dev.jsonl \\
        --index-dir indexes/viaquad \\
        --mlp-path checkpoints/fusion_mlp_aug.pt \\
        --output results/eval_all_dev.json

    uv run python scripts/evaluate_all.py \\
        --qas-path data/processed/dangdocao_test.jsonl \\
        --index-dir indexes/dangdocao \\
        --mlp-path checkpoints/fusion_mlp_aug.pt \\
        --output results/eval_all_cross.json
"""

# IMPORTANT: pyarrow must be imported BEFORE torch / faiss on Windows + CUDA.
# Otherwise pyarrow's native DLL load triggers an access violation (0xC0000005)
# the first time something downstream (pandas → pyarrow, or HuggingFace
# datasets → pyarrow) tries to load it after PyTorch/CUDA has initialised.
# Pre-loading pyarrow grabs its DLL slots before the conflict can occur.
import pyarrow  # noqa: F401

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
from scipy import stats as sp_stats
from tqdm import tqdm

from rag_vie.config import settings
from rag_vie.features.combined import combine
from rag_vie.features.signal import extract_signal_features
from rag_vie.features.vietnamese import FEATURE_NAMES, extract_features
from rag_vie.fusion.mlp import FusionMLP
from rag_vie.retrieval.bm25 import BM25Retriever
from rag_vie.retrieval.embedder import embed_texts
from rag_vie.utils.fusion import fuse_scores
from rag_vie.utils.metrics import (
    hit_at_1,
    map_at_k,
    min_max_normalize as _minmax,
    mrr_at_k,
    ndcg_at_k,
    recall_at_k,
)

# NOTE: DenseRetriever (which transitively imports faiss) is intentionally NOT
# imported at runtime here. faiss's MKL/OpenMP runtime collides with the
# PyTorch/CUDA runtime used by BGE-M3 unless PyTorch is initialised first.
# main() imports DenseRetriever lazily AFTER BGE-M3 has been loaded.
# The TYPE_CHECKING import is purely for static type hints on `run_eval`.
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from rag_vie.retrieval.dense import DenseRetriever  # noqa: F401


# ── Strata ────────────────────────────────────────────────────────────────────

STRATA: list[dict] = [
    {"name": "diac_low",    "feature": "diacritic_ratio",   "lo": 0.0,  "hi": 0.3},
    {"name": "diac_mid",    "feature": "diacritic_ratio",   "lo": 0.3,  "hi": 0.7},
    {"name": "diac_high",   "feature": "diacritic_ratio",   "lo": 0.7,  "hi": 1.01},
    {"name": "comp_low",    "feature": "compound_ratio",    "lo": 0.0,  "hi": 0.2},
    {"name": "comp_high",   "feature": "compound_ratio",    "lo": 0.2,  "hi": 1.01},
    {"name": "eng_none",    "feature": "english_ratio",     "lo": 0.0,  "hi": 0.01},
    {"name": "eng_mixed",   "feature": "english_ratio",     "lo": 0.01, "hi": 1.01},
    {"name": "short_query", "feature": "query_length_norm", "lo": 0.0,  "hi": 0.4},
    {"name": "long_query",  "feature": "query_length_norm", "lo": 0.4,  "hi": 1.01},
    {"name": "simple",      "feature": "clause_count_norm", "lo": 0.0,  "hi": 0.01},
    {"name": "complex",     "feature": "clause_count_norm", "lo": 0.01, "hi": 1.01},
]

# Three-way methods: (w_dense, w_bm25, w_sparse)
METHODS: dict[str, tuple[float, float, float] | None] = {
    "mlp":          None,
    "fixed_equal":  (1/3, 1/3, 1/3),
    "dense_bm25":   (0.5, 0.5, 0.0),    # former 2-way hybrid (for backward comparison)
    "dense":        (1.0, 0.0, 0.0),
    "bm25":         (0.0, 1.0, 0.0),
    "sparse":       (0.0, 0.0, 1.0),
}

# Baselines to compare MLP against in significance tests
_SIG_BASELINES = ("fixed_equal", "dense_bm25", "dense", "bm25", "sparse")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _r4(x) -> float: return round(float(x), 4)
def _r2(x) -> float: return round(float(x), 2)


# ── Score fusion ──────────────────────────────────────────────────────────────

def fuse(
    dense_norm:  dict[str, float],
    bm25_norm:   dict[str, float],
    sparse_norm: dict[str, float],
    weights: tuple[float, float, float],
    k_final: int,
) -> list[str]:
    scored = fuse_scores(dense_norm, bm25_norm, sparse_norm, weights)
    return [p for p, _ in sorted(scored.items(), key=lambda x: x[1], reverse=True)[:k_final]]


def _ranks(scores: dict[str, float]) -> dict[str, int]:
    """id → 0-based rank within a source (descending score)."""
    return {pid: r for r, (pid, _) in
            enumerate(sorted(scores.items(), key=lambda x: x[1], reverse=True))}


def fuse_rrf(
    dense_norm:  dict[str, float],
    bm25_norm:   dict[str, float],
    sparse_norm: dict[str, float],
    k_final: int,
    k_rrf: int = 60,
) -> list[str]:
    """Reciprocal Rank Fusion (parameter-free hybrid baseline): score = Σ 1/(k_rrf + rank)."""
    ranks = [_ranks(s) for s in (dense_norm, bm25_norm, sparse_norm) if s]
    ids = set().union(*[set(r) for r in ranks]) if ranks else set()
    scored = {
        pid: sum(1.0 / (k_rrf + r[pid]) for r in ranks if pid in r)
        for pid in ids
    }
    return [p for p, _ in sorted(scored.items(), key=lambda x: x[1], reverse=True)[:k_final]]


# Coarse simplex grid (step 0.1, 66 points) used only for the per-query ORACLE upper bound.
_ORACLE_GRID: list[tuple[float, float, float]] = [
    (i / 10, j / 10, (10 - i - j) / 10)
    for i in range(11) for j in range(11 - i)
]


# ── Statistics ────────────────────────────────────────────────────────────────

def bootstrap_ci(
    scores: np.ndarray,
    n_boot: int = 2000,
    level:  float = 0.95,
    seed:   int = 42,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    means = [rng.choice(scores, len(scores), replace=True).mean() for _ in range(n_boot)]
    lo, hi = np.percentile(means, [(1 - level) / 2 * 100, (1 + level) / 2 * 100])
    return float(lo), float(hi)


def significance_tests(a: np.ndarray, b: np.ndarray) -> dict:
    diff = a - b
    _, t_p = sp_stats.ttest_rel(a, b)
    try:
        _, w_p = sp_stats.wilcoxon(diff, alternative="two-sided")
    except ValueError:
        w_p = 1.0
    ci_lo, ci_hi = bootstrap_ci(diff)
    return {
        "mean_delta":  _r4(diff.mean()),
        "ttest_p":     float(t_p),
        "wilcoxon_p":  float(w_p),
        "delta_CI95":  [round(ci_lo, 4), round(ci_hi, 4)],
    }


# ── Core evaluation loop ──────────────────────────────────────────────────────

def run_eval(
    qas:       list[dict],
    dense:     "DenseRetriever",
    bm25_r:    BM25Retriever,
    mlp:       FusionMLP,
    k_dense:   int,
    k_bm25:    int,
    k_final:   int,
    sparse_r=None,   # SparseRetriever | None
    k_sparse:  int = 100,
) -> dict:
    # rrf = parameter-free hybrid baseline; oracle = per-query best simplex point (headroom).
    method_names = list(METHODS) + ["rrf", "oracle"]
    per_query: dict[str, dict[str, list]] = {
        m: {mt: [] for mt in ("ndcg10", "mrr10", "map10", "rec10", "rec100", "hit1")}
        for m in method_names
    }
    feat_idx = {name: i for i, name in enumerate(FEATURE_NAMES)}
    feat_rows: list[np.ndarray] = []          # query-only features (for strata/correlation)
    bm25_idf = bm25_r._bm25.idf

    # Weight accumulators (MLP 3-way)
    w_dense_list:  list[float] = []
    w_bm25_list:   list[float] = []
    w_sparse_list: list[float] = []
    entropies:     list[float] = []
    mlp_us:        list[float] = []
    e2e_ms:        list[float] = []           # retrieval+fusion latency (excl. embedding API)

    def _record(method: str, ranked: list[str], relevant: set[str]) -> None:
        acc = per_query[method]
        acc["ndcg10"].append(ndcg_at_k(ranked, relevant, 10))
        acc["mrr10"].append(mrr_at_k(ranked, relevant, 10))
        acc["map10"].append(map_at_k(ranked, relevant, 10))
        acc["rec10"].append(recall_at_k(ranked, relevant, 10))
        acc["rec100"].append(recall_at_k(ranked, relevant, 100))
        acc["hit1"].append(hit_at_1(ranked, relevant))

    # Batch pre-embed all queries once (avoids per-query FPT API round-trips).
    queries = [qa["question"] for qa in qas]
    print(f"Embedding {len(queries):,} queries via FPT API (batched)…", flush=True)
    qembs = embed_texts(queries, batch_size=32)

    for i, qa in enumerate(tqdm(qas, desc="Evaluating")):
        query    = qa["question"]
        relevant = set(qa["relevant_ids"])

        qfeats = extract_features(query, bm25_idf=bm25_idf)
        feat_rows.append(qfeats)

        # ── Timed MLP path: retrieval + signal features + weight prediction + fusion ──
        t0     = time.perf_counter()
        d_hits = dense.search(qembs[i:i + 1], k_dense)
        b_hits = bm25_r.search(query, k_bm25)
        s_hits = sparse_r.search(query, k_sparse) if sparse_r is not None else []

        d_norm = _minmax({pid: s for pid, _, s in d_hits})
        b_norm = _minmax({pid: s for pid, _, s in b_hits})
        s_norm = _minmax({pid: s for pid, _, s in s_hits})

        sig_feats = extract_signal_features(d_norm, b_norm, s_norm)
        feats     = combine(qfeats, sig_feats)

        tw    = time.perf_counter()
        w_mlp = mlp.predict_weights(feats)   # (w_dense, w_bm25, w_sparse)
        mlp_us.append((time.perf_counter() - tw) * 1e6)

        w = w_mlp if len(w_mlp) == 3 else (*w_mlp, 0.0)
        ranked_mlp = fuse(d_norm, b_norm, s_norm, w, k_final)
        e2e_ms.append((time.perf_counter() - t0) * 1e3)
        _record("mlp", ranked_mlp, relevant)

        w_dense_list.append(float(w_mlp[0]))
        w_bm25_list.append(float(w_mlp[1]))
        w_sparse_list.append(float(w_mlp[2]) if len(w_mlp) > 2 else 0.0)
        eps = 1e-9
        entropies.append(float(-sum(wt * np.log(wt + eps) for wt in w_mlp)))

        # ── Untimed: fixed-weight baselines, RRF, and per-query oracle ──
        for method, fixed_w in METHODS.items():
            if fixed_w is None:
                continue   # mlp already recorded
            _record(method, fuse(d_norm, b_norm, s_norm, fixed_w, k_final), relevant)

        _record("rrf", fuse_rrf(d_norm, b_norm, s_norm, k_final), relevant)

        best_w = max(
            _ORACLE_GRID,
            key=lambda gw: ndcg_at_k(fuse(d_norm, b_norm, s_norm, gw, k_final), relevant, 10),
        )
        _record("oracle", fuse(d_norm, b_norm, s_norm, best_w, k_final), relevant)

    features_arr = np.array(feat_rows, dtype=np.float32)
    w_d_arr = np.array(w_dense_list)
    w_b_arr = np.array(w_bm25_list)
    w_s_arr = np.array(w_sparse_list)

    # ── 1. Aggregate retrieval metrics ────────────────────────────────────────
    methods_out: dict = {}
    for method, acc in per_query.items():
        ndcg_arr = np.array(acc["ndcg10"])
        ci       = bootstrap_ci(ndcg_arr)
        methods_out[method] = {
            "NDCG@10":     _r4(ndcg_arr.mean()),
            "MRR@10":      _r4(np.mean(acc["mrr10"])),
            "MAP@10":      _r4(np.mean(acc["map10"])),
            "Recall@10":   _r4(np.mean(acc["rec10"])),
            "Recall@100":  _r4(np.mean(acc["rec100"])),
            "Hit@1":       _r4(np.mean(acc["hit1"])),
            "NDCG@10_std": _r4(ndcg_arr.std()),
            "NDCG@10_CI95": [round(ci[0], 4), round(ci[1], 4)],
        }

    # ── 2. Statistical significance (MLP vs each baseline) ───────────────────
    mlp_ndcg = np.array(per_query["mlp"]["ndcg10"])
    significance = {
        f"mlp_vs_{b}": significance_tests(mlp_ndcg, np.array(per_query[b]["ndcg10"]))
        for b in _SIG_BASELINES
    }

    # ── 3. Efficiency ─────────────────────────────────────────────────────────
    e2e_arr = np.array(e2e_ms)
    p50, p95 = (float(np.percentile(e2e_arr, 50)), float(np.percentile(e2e_arr, 95))) if len(e2e_arr) else (0.0, 0.0)
    efficiency = {
        "mlp_params": mlp.net.count_params(),
        "mlp_inference_us": {
            "mean": _r2(np.mean(mlp_us)),
            "std":  _r2(np.std(mlp_us)),
        },
        # Retrieval + fusion latency per query (excludes the batched embedding API call).
        "retrieval_latency_ms": {
            "p50": _r2(p50),
            "p95": _r2(p95),
            "mean": _r2(np.mean(e2e_arr)) if len(e2e_arr) else 0.0,
        },
        "throughput_qps": _r2(1000.0 / np.mean(e2e_arr)) if len(e2e_arr) and np.mean(e2e_arr) > 0 else 0.0,
    }

    # ── 4. Weight analysis (3-way) ────────────────────────────────────────────
    di   = feat_idx["diacritic_ratio"]
    ci_i = feat_idx["compound_ratio"]
    en_i = feat_idx["english_ratio"]

    r_diac, p_diac = sp_stats.pearsonr(features_arr[:, di],   w_d_arr)
    r_comp, p_comp = sp_stats.pearsonr(features_arr[:, ci_i], w_b_arr)
    r_eng,  p_eng  = sp_stats.pearsonr(features_arr[:, en_i], w_s_arr)

    weight_analysis = {
        "w_dense":  {"mean": _r4(w_d_arr.mean()), "std": _r4(w_d_arr.std()),
                     "min": _r4(w_d_arr.min()),    "max": _r4(w_d_arr.max())},
        "w_bm25":   {"mean": _r4(w_b_arr.mean()), "std": _r4(w_b_arr.std()),
                     "min": _r4(w_b_arr.min()),    "max": _r4(w_b_arr.max())},
        "w_sparse": {"mean": _r4(w_s_arr.mean()), "std": _r4(w_s_arr.std()),
                     "min": _r4(w_s_arr.min()),    "max": _r4(w_s_arr.max())},
        "entropy": {
            "mean": _r4(np.mean(entropies)),
            "std":  _r4(np.std(entropies)),
            "max_possible": _r4(np.log(3)),    # ln(3) ≈ 1.099 for uniform 3-way
        },
        "pearson_diacritic_vs_wdense":  {"r": _r4(r_diac), "p": float(p_diac)},
        "pearson_compound_vs_wbm25":    {"r": _r4(r_comp), "p": float(p_comp)},
        "pearson_english_vs_wsparse":   {"r": _r4(r_eng),  "p": float(p_eng)},
    }

    # ── Stratified analysis ───────────────────────────────────────────────────
    mlp_ndcg_l = per_query["mlp"]["ndcg10"]
    eq_ndcg_l  = per_query["fixed_equal"]["ndcg10"]
    stratified: dict = {}
    for s in STRATA:
        fi   = feat_idx[s["feature"]]
        idxs = [i for i, v in enumerate(features_arr[:, fi]) if s["lo"] <= v < s["hi"]]
        if not idxs:
            stratified[s["name"]] = {"n": 0}
            continue
        m_s = [mlp_ndcg_l[i] for i in idxs]
        e_s = [eq_ndcg_l[i] for i in idxs]
        wd  = [w_dense_list[i]  for i in idxs]
        wb  = [w_bm25_list[i]   for i in idxs]
        ws  = [w_sparse_list[i] for i in idxs]
        stratified[s["name"]] = {
            "n":             len(idxs),
            "mlp_ndcg":     _r4(np.mean(m_s)),
            "fixed_ndcg":   _r4(np.mean(e_s)),
            "delta":         _r4(np.mean(m_s) - np.mean(e_s)),
            "mean_w_dense":  round(float(np.mean(wd)), 3),
            "mean_w_bm25":   round(float(np.mean(wb)), 3),
            "mean_w_sparse": round(float(np.mean(ws)), 3),
        }

    return {
        "n_queries":       len(qas),
        "methods":         methods_out,
        "significance":    significance,
        "efficiency":      efficiency,
        "weight_analysis": weight_analysis,
        "stratified":      stratified,
    }


# ── Pretty print ──────────────────────────────────────────────────────────────

def print_results(results: dict) -> None:
    W = 90

    def sep(c="-"): print(c * W)

    print("\n" + "=" * W)
    print(f"  EVALUATION RESULTS   (n = {results['n_queries']:,} queries)")
    print("=" * W)

    # 1. Retrieval metrics
    cols = ["NDCG@10", "MRR@10", "MAP@10", "Recall@10", "Recall@100", "Hit@1"]
    header = f"  {'Method':<16}" + "".join(f"  {c:>10}" for c in cols)
    print(f"\n{header}")
    sep()
    for method, m in results["methods"].items():
        row = f"  {method:<16}" + "".join(f"  {m[c]:>10.4f}" for c in cols)
        print(row)
    print()
    for method, m in results["methods"].items():
        lo, hi = m["NDCG@10_CI95"]
        print(f"  {method:<16}  NDCG@10 CI95 [{lo:.4f}, {hi:.4f}]  ±{m['NDCG@10_std']:.4f}")

    # 2. Significance
    print(f"\n{'─' * W}")
    print("  Statistical Significance  (MLP vs baselines, NDCG@10 per query)")
    print(f"{'─' * W}")
    print(f"  {'Comparison':<22}  {'Δ NDCG':>8}  {'95% CI':>18}  {'t-test p':>10}  {'Wilcoxon p':>12}")
    sep("·")
    for key, v in results["significance"].items():
        label  = key.replace("mlp_vs_", "MLP > ")
        lo, hi = v["delta_CI95"]
        print(
            f"  {label:<22}  {v['mean_delta']:>+8.4f}  [{lo:+.4f},{hi:+.4f}]"
            f"  {v['ttest_p']:>10.2e}  {v['wilcoxon_p']:>12.2e}"
        )

    # 3. Efficiency
    e = results["efficiency"]
    print(f"\n{'─' * W}")
    print("  Efficiency")
    print(f"{'─' * W}")
    print(f"  MLP parameters:        {e['mlp_params']:,}")
    lat = e["mlp_inference_us"]
    print(f"  MLP inference latency: {lat['mean']:.1f} ± {lat['std']:.1f} μs")
    if "retrieval_latency_ms" in e:
        rl = e["retrieval_latency_ms"]
        print(f"  Retrieval+fusion lat.: p50={rl['p50']:.2f} ms  p95={rl['p95']:.2f} ms  "
              f"(mean={rl['mean']:.2f} ms, excl. embedding API)")
        print(f"  Throughput:            {e['throughput_qps']:.1f} q/s")
    if "faiss_index_mb" in e:
        print(f"  FAISS index size:      {e['faiss_index_mb']:.1f} MB")
        print(f"  BM25 index size:       {e['bm25_pkl_mb']:.1f} MB")
        if e.get("sparse_pkl_mb", -1) >= 0:
            print(f"  Sparse index size:     {e['sparse_pkl_mb']:.1f} MB")

    # 4. Weight analysis
    w = results["weight_analysis"]
    print(f"\n{'─' * W}")
    print("  Weight Analysis  (3-way MLP predictions)")
    print(f"{'─' * W}")
    for key in ("w_dense", "w_bm25", "w_sparse"):
        d = w[key]
        print(f"  {key:<10}  mean={d['mean']:.4f}  std={d['std']:.4f}  "
              f"[{d['min']:.4f}, {d['max']:.4f}]")
    ent = w["entropy"]
    print(f"  Entropy  mean={ent['mean']:.4f} ± {ent['std']:.4f}  "
          f"(max_possible={ent['max_possible']:.4f})")
    for k_label, pkey in [
        ("Pearson(diacritic_ratio, w_dense)", "pearson_diacritic_vs_wdense"),
        ("Pearson(compound_ratio,  w_bm25)",  "pearson_compound_vs_wbm25"),
        ("Pearson(english_ratio,   w_sparse)","pearson_english_vs_wsparse"),
    ]:
        v = w[pkey]
        print(f"  {k_label:42}  r={v['r']:+.4f}  p={v['p']:.3e}")

    # 5. Stratified
    print(f"\n{'─' * W}")
    print("  Stratified Analysis  (MLP vs Fixed-Equal 1/3/1/3/1/3)")
    print(f"{'─' * W}")
    hdr = (f"  {'Stratum':<18}  {'N':>5}  {'Fixed':>8}  {'MLP':>8}"
           f"  {'Δ':>8}  {'w_d':>6}  {'w_b':>6}  {'w_s':>6}")
    print(hdr)
    sep("·")
    for name, s in results["stratified"].items():
        if s.get("n", 0) == 0:
            continue
        print(
            f"  {name:<18}  {s['n']:>5}  {s['fixed_ndcg']:>8.4f}  {s['mlp_ndcg']:>8.4f}"
            f"  {s['delta']:>+8.4f}  {s['mean_w_dense']:>6.3f}"
            f"  {s['mean_w_bm25']:>6.3f}  {s['mean_w_sparse']:>6.3f}"
        )
    print()


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Unified evaluation — Dynamic Hybrid RAG (3-way)")
    parser.add_argument("--qas-path",    required=True,  help="QA JSONL file")
    parser.add_argument("--index-dir",   required=True,  help="Index directory")
    parser.add_argument("--mlp-path",    required=True,  help="FusionMLP checkpoint (.pt)")
    parser.add_argument("--sparse-path", default=None,   help="sparse.pkl path (default: <index-dir>/sparse.pkl)")
    parser.add_argument("--no-sparse",   action="store_true", help="Skip sparse signal")
    parser.add_argument("--k-dense",     type=int, default=None)
    parser.add_argument("--k-bm25",      type=int, default=None)
    parser.add_argument("--k-sparse",    type=int, default=100)
    parser.add_argument("--k-final",     type=int, default=100)
    parser.add_argument("--max-samples", type=int, default=None,
                        help="If set, randomly subsample N queries (seeded). "
                             "Use this instead of head -n N to avoid HF-grouped-by-article bias.")
    parser.add_argument("--subset-seed", type=int, default=42,
                        help="Seed for --max-samples subsetting (default: 42)")
    parser.add_argument("--output",      default=None, help="Save JSON to this path")
    args = parser.parse_args()

    k_dense = args.k_dense or settings.top_k_dense
    k_bm25  = args.k_bm25  or settings.top_k_bm25

    # NB: BGE-M3 (PyTorch / CUDA) MUST be initialised BEFORE FAISS, otherwise
    # FAISS' MKL/OpenMP runtime collides with PyTorch's and the process
    # segfaults on the first sparse.encode call. See CLAUDE.md "OMP deadlock"
    # note — same fix applies on Windows (manifests as exit 0xC0000005).
    sparse_r = None
    if not args.no_sparse:
        sparse_path = args.sparse_path or str(Path(args.index_dir) / "sparse.pkl")
        if Path(sparse_path).exists():
            from rag_vie.retrieval.sparse import SparseRetriever, _encode_sparse
            sparse_r = SparseRetriever.load(sparse_path)
            print(f"Sparse index loaded from {sparse_path}", flush=True)
            # Pre-warm BGE-M3 (triggers model load + CUDA init before FAISS).
            _ = _encode_sparse(["warmup"])
            print("BGE-M3 warmed up", flush=True)
        else:
            print(f"NOTE: sparse.pkl not found at {sparse_path} — running 2-way fallback.")

    # Lazy import: faiss only loads here, AFTER PyTorch/CUDA is initialised.
    from rag_vie.retrieval.dense import DenseRetriever  # noqa: E402

    dense  = DenseRetriever.load(args.index_dir)
    bm25_r = BM25Retriever.load(str(Path(args.index_dir) / "bm25.pkl"))
    mlp    = FusionMLP.load(args.mlp_path)

    with open(args.qas_path, encoding="utf-8") as f:
        qas = [json.loads(line) for line in f if line.strip()]
    print(f"Loaded {len(qas):,} queries from {args.qas_path}")

    if args.max_samples is not None and args.max_samples < len(qas):
        import random as _random
        _rng = _random.Random(args.subset_seed)
        qas = _rng.sample(qas, args.max_samples)
        print(f"Subsampled to {len(qas):,} queries (seed={args.subset_seed})")

    results = run_eval(
        qas, dense, bm25_r, mlp,
        k_dense, k_bm25, args.k_final,
        sparse_r=sparse_r, k_sparse=args.k_sparse,
    )

    # Index file sizes
    def _mb(path: str) -> float:
        try:
            return round(os.path.getsize(path) / 1e6, 2)
        except OSError:
            return -1.0

    results["efficiency"]["faiss_index_mb"] = _mb(str(Path(args.index_dir) / "index.faiss"))
    results["efficiency"]["bm25_pkl_mb"]    = _mb(str(Path(args.index_dir) / "bm25.pkl"))
    results["efficiency"]["sparse_pkl_mb"]  = _mb(str(Path(args.index_dir) / "sparse.pkl"))
    results["meta"] = {
        "qas_path":  args.qas_path,
        "index_dir": args.index_dir,
        "mlp_path":  args.mlp_path,
        "sparse":    sparse_r is not None,
    }

    print_results(results)

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"Saved -> {args.output}")


if __name__ == "__main__":
    main()
