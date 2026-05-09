"""evaluate_all.py — Unified retrieval evaluation for Dynamic Hybrid RAG.

Metric groups:
  1. Retrieval   — NDCG@10, MRR@10, MAP@10, Recall@10, Recall@100, Hit@1
  2. Significance — paired t-test, Wilcoxon signed-rank, 95% bootstrap CI
  3. Efficiency  — MLP param count, index sizes, MLP inference latency
  4. Weights     — entropy, distribution stats, Pearson correlations
  + Stratified analysis (11 query-feature strata)

Each query embeds exactly once; all methods share the same dense/BM25 hits.

Usage:
    uv run python scripts/evaluate_all.py \\
        --qas-path data/processed/viaquad_dev.jsonl \\
        --index-dir indexes/viaquad \\
        --mlp-path checkpoints/fusion_mlp_aug.pt \\
        --output results/eval_all_dev.json

    uv run python scripts/evaluate_all.py \\
        --qas-path data/processed/viaquad_dev_noisy.jsonl \\
        --index-dir indexes/viaquad \\
        --mlp-path checkpoints/fusion_mlp_aug.pt \\
        --output results/eval_all_dev_noisy.json

    uv run python scripts/evaluate_all.py \\
        --qas-path data/processed/dangdocao_test.jsonl \\
        --index-dir indexes/dangdocao \\
        --mlp-path checkpoints/fusion_mlp_aug.pt \\
        --output results/eval_all_cross.json
"""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
from scipy import stats as sp_stats
from tqdm import tqdm

from rag_vie.config import settings
from rag_vie.features.vietnamese import FEATURE_NAMES, extract_features
from rag_vie.fusion.mlp import FusionMLP
from rag_vie.retrieval.bm25 import BM25Retriever
from rag_vie.retrieval.dense import DenseRetriever
from rag_vie.retrieval.embedder import embed_query


# ── Strata definition ─────────────────────────────────────────────────────────

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

METHODS: dict[str, tuple[float, float] | None] = {
    "mlp":       None,          # dynamic weights from MLP
    "fixed_0.5": (0.5, 0.5),
    "dense":     (1.0, 0.0),
    "bm25":      (0.0, 1.0),
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _r4(x) -> float: return round(float(x), 4)
def _r2(x) -> float: return round(float(x), 2)


# ── Retrieval metrics ─────────────────────────────────────────────────────────

def ndcg_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    dcg  = sum(1.0 / np.log2(i + 2) for i, p in enumerate(ranked[:k]) if p in relevant)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(relevant), k)))
    return dcg / idcg if idcg > 0 else 0.0


def mrr_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    for i, p in enumerate(ranked[:k]):
        if p in relevant:
            return 1.0 / (i + 1)
    return 0.0


def map_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    """Mean Average Precision @ k: AP = (1/min(R,k)) * Σ P@i * rel(i)."""
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


# ── Score fusion ──────────────────────────────────────────────────────────────

def _minmax(scores: dict[str, float]) -> dict[str, float]:
    if not scores:
        return scores
    lo, hi = min(scores.values()), max(scores.values())
    span = hi - lo
    return {k: (v - lo) / span if span > 0 else 0.0 for k, v in scores.items()}


def fuse(
    dense_norm: dict[str, float],
    bm25_norm:  dict[str, float],
    weights:    tuple[float, float],
    k_final:    int,
) -> list[str]:
    a, b = weights
    ids = set(dense_norm) | set(bm25_norm)
    scored = {p: a * dense_norm.get(p, 0.0) + b * bm25_norm.get(p, 0.0) for p in ids}
    return [p for p, _ in sorted(scored.items(), key=lambda x: x[1], reverse=True)[:k_final]]


# ── Statistical significance ──────────────────────────────────────────────────

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
        w_p = 1.0   # all differences are zero → no evidence of difference
    ci_lo, ci_hi = bootstrap_ci(diff)
    return {
        "mean_delta":  _r4(diff.mean()),
        "ttest_p":     float(t_p),
        "wilcoxon_p":  float(w_p),
        "delta_CI95":  [round(ci_lo, 4), round(ci_hi, 4)],
    }


# ── Core evaluation loop ──────────────────────────────────────────────────────

def run_eval(
    qas:     list[dict],
    dense:   DenseRetriever,
    bm25_r:  BM25Retriever,
    mlp:     FusionMLP,
    k_dense: int,
    k_bm25:  int,
    k_final: int,
) -> dict:
    per_query: dict[str, dict[str, list]] = {
        m: {mt: [] for mt in ("ndcg10", "mrr10", "map10", "rec10", "rec100", "hit1")}
        for m in METHODS
    }
    feat_idx   = {name: i for i, name in enumerate(FEATURE_NAMES)}
    feat_rows:  list[np.ndarray] = []
    w_dense_list: list[float]    = []
    entropies:    list[float]    = []
    mlp_us:       list[float]    = []   # MLP inference time in microseconds

    for qa in tqdm(qas, desc="Evaluating"):
        query    = qa["question"]
        relevant = set(qa["relevant_ids"])

        features = extract_features(query)
        feat_rows.append(features)

        # MLP inference + timing
        t0    = time.perf_counter()
        w_mlp = mlp.predict_weights(features)
        mlp_us.append((time.perf_counter() - t0) * 1e6)

        w_dense_list.append(float(w_mlp[0]))
        eps = 1e-9
        entropies.append(float(-sum(w * np.log(w + eps) for w in w_mlp)))

        # Embed once, reuse for all methods
        qemb   = embed_query(query)
        d_hits = dense.search(qemb, k_dense)
        b_hits = bm25_r.search(query, k_bm25)
        d_norm = _minmax({pid: s for pid, _, s in d_hits})
        b_norm = _minmax({pid: s for pid, _, s in b_hits})

        for method, fixed_w in METHODS.items():
            w      = w_mlp if fixed_w is None else fixed_w
            ranked = fuse(d_norm, b_norm, w, k_final)

            acc = per_query[method]
            acc["ndcg10"].append(ndcg_at_k(ranked, relevant, 10))
            acc["mrr10"].append(mrr_at_k(ranked, relevant, 10))
            acc["map10"].append(map_at_k(ranked, relevant, 10))
            acc["rec10"].append(recall_at_k(ranked, relevant, 10))
            acc["rec100"].append(recall_at_k(ranked, relevant, 100))
            acc["hit1"].append(hit_at_1(ranked, relevant))

    features_arr = np.array(feat_rows, dtype=np.float32)
    w_arr        = np.array(w_dense_list)

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
    mlp_ndcg   = np.array(per_query["mlp"]["ndcg10"])
    significance = {
        f"mlp_vs_{b}": significance_tests(mlp_ndcg, np.array(per_query[b]["ndcg10"]))
        for b in ("fixed_0.5", "dense", "bm25")
    }

    # ── 3. Efficiency ─────────────────────────────────────────────────────────
    efficiency = {
        "mlp_params": sum(p.numel() for p in mlp.parameters()),
        "mlp_inference_us": {
            "mean": _r2(np.mean(mlp_us)),
            "std":  _r2(np.std(mlp_us)),
        },
    }

    # ── 4. Weight analysis ────────────────────────────────────────────────────
    di   = feat_idx["diacritic_ratio"]
    ci_i = feat_idx["compound_ratio"]
    r1, p1 = sp_stats.pearsonr(features_arr[:, di],   w_arr)
    r2, p2 = sp_stats.pearsonr(features_arr[:, ci_i], 1 - w_arr)   # compound vs w_bm25

    weight_analysis = {
        "w_dense": {
            "mean": _r4(w_arr.mean()),
            "std":  _r4(w_arr.std()),
            "min":  _r4(w_arr.min()),
            "max":  _r4(w_arr.max()),
        },
        "entropy": {
            "mean": _r4(np.mean(entropies)),
            "std":  _r4(np.std(entropies)),
        },
        "pearson_diacritic_vs_wdense": {"r": _r4(r1), "p": float(p1)},
        "pearson_compound_vs_wbm25":   {"r": _r4(r2), "p": float(p2)},
    }

    # ── Stratified analysis ───────────────────────────────────────────────────
    mlp_ndcg_l = per_query["mlp"]["ndcg10"]
    fix_ndcg_l = per_query["fixed_0.5"]["ndcg10"]
    stratified: dict = {}
    for s in STRATA:
        fi   = feat_idx[s["feature"]]
        idxs = [i for i, v in enumerate(features_arr[:, fi]) if s["lo"] <= v < s["hi"]]
        if not idxs:
            stratified[s["name"]] = {"n": 0}
            continue
        m_s = [mlp_ndcg_l[i] for i in idxs]
        f_s = [fix_ndcg_l[i] for i in idxs]
        wd  = [w_dense_list[i] for i in idxs]
        stratified[s["name"]] = {
            "n":            len(idxs),
            "mlp_ndcg":    _r4(np.mean(m_s)),
            "fixed_ndcg":  _r4(np.mean(f_s)),
            "delta":        _r4(np.mean(m_s) - np.mean(f_s)),
            "mean_w_dense": round(float(np.mean(wd)), 3),
        }

    return {
        "n_queries":       len(qas),
        "methods":         methods_out,
        "significance":    significance,
        "efficiency":      efficiency,
        "weight_analysis": weight_analysis,
        "stratified":      stratified,
    }


# ── Pretty printing ───────────────────────────────────────────────────────────

def print_results(results: dict) -> None:
    n = results["n_queries"]
    W = 78

    def sep(char="-"): print(char * W)

    print("\n" + "=" * W)
    print(f"  EVALUATION RESULTS   (n = {n:,} queries)")
    print("=" * W)

    # 1. Retrieval metrics table
    cols = ["NDCG@10", "MRR@10", "MAP@10", "Recall@10", "Recall@100", "Hit@1"]
    header = f"  {'Method':<16}" + "".join(f"  {c:>10}" for c in cols)
    print(f"\n{header}")
    sep()
    for method, m in results["methods"].items():
        row = f"  {method:<16}" + "".join(f"  {m[c]:>10.4f}" for c in cols)
        print(row)
    # NDCG@10 CI95 sub-rows
    print()
    print("  95% CI for NDCG@10:")
    for method, m in results["methods"].items():
        lo, hi = m["NDCG@10_CI95"]
        print(f"    {method:<14}  [{lo:.4f}, {hi:.4f}]  (±{m['NDCG@10_std']:.4f})")

    # 2. Statistical significance
    print(f"\n{'─' * W}")
    print("  Statistical Significance  (MLP vs baselines, NDCG@10 per query)")
    print(f"{'─' * W}")
    print(f"  {'Comparison':<22}  {'Δ NDCG':>8}  {'95% CI':>16}  {'t-test p':>10}  {'Wilcoxon p':>12}")
    sep("·")
    for key, v in results["significance"].items():
        label  = key.replace("mlp_vs_", "MLP > ")
        lo, hi = v["delta_CI95"]
        ci_str = f"[{lo:+.4f},{hi:+.4f}]"
        print(
            f"  {label:<22}  {v['mean_delta']:>+8.4f}  {ci_str:>16}"
            f"  {v['ttest_p']:>10.2e}  {v['wilcoxon_p']:>12.2e}"
        )

    # 3. Efficiency
    e = results["efficiency"]
    print(f"\n{'─' * W}")
    print("  Efficiency")
    print(f"{'─' * W}")
    print(f"  MLP parameters:        {e['mlp_params']:,}")
    lat = e["mlp_inference_us"]
    print(f"  MLP inference latency: {lat['mean']:.1f} ± {lat['std']:.1f} μs  (negligible vs ~100ms embed)")
    if "faiss_index_mb" in e:
        print(f"  FAISS index size:      {e['faiss_index_mb']:.1f} MB")
        print(f"  BM25 index size:       {e['bm25_pkl_mb']:.1f} MB")

    # 4. Weight analysis
    w = results["weight_analysis"]
    print(f"\n{'─' * W}")
    print("  Weight Analysis  (MLP predictions)")
    print(f"{'─' * W}")
    wd = w["w_dense"]
    print(f"  w_dense  mean={wd['mean']:.4f}  std={wd['std']:.4f}  "
          f"min={wd['min']:.4f}  max={wd['max']:.4f}")
    ent = w["entropy"]
    print(f"  Entropy  mean={ent['mean']:.4f}  std={ent['std']:.4f}  "
          f"(max entropy ln2 ≈ {np.log(2):.4f})")
    pd_ = w["pearson_diacritic_vs_wdense"]
    pc_ = w["pearson_compound_vs_wbm25"]
    print(f"  Pearson(diacritic_ratio, w_dense) = {pd_['r']:+.4f}  p={pd_['p']:.3e}")
    print(f"  Pearson(compound_ratio,  w_bm25)  = {pc_['r']:+.4f}  p={pc_['p']:.3e}")

    # 5. Stratified
    print(f"\n{'─' * W}")
    print("  Stratified Analysis  (MLP vs Fixed 0.5/0.5)")
    print(f"{'─' * W}")
    print(f"  {'Stratum':<18}  {'N':>5}  {'Fixed':>8}  {'MLP':>8}  {'Δ':>8}  {'w_dense':>8}")
    sep("·")
    for name, s in results["stratified"].items():
        if s.get("n", 0) == 0:
            continue
        delta_s = f"{s['delta']:+.4f}"
        print(
            f"  {name:<18}  {s['n']:>5}  {s['fixed_ndcg']:>8.4f}"
            f"  {s['mlp_ndcg']:>8.4f}  {delta_s:>8}  {s['mean_w_dense']:>8.3f}"
        )
    print()


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Unified evaluation — Dynamic Hybrid RAG")
    parser.add_argument("--qas-path",  required=True,  help="QA JSONL file")
    parser.add_argument("--index-dir", required=True,  help="Index directory (FAISS + BM25)")
    parser.add_argument("--mlp-path",  required=True,  help="FusionMLP checkpoint (.pt)")
    parser.add_argument("--k-dense",   type=int, default=None)
    parser.add_argument("--k-bm25",    type=int, default=None)
    parser.add_argument("--k-final",   type=int, default=100)
    parser.add_argument("--output",    default=None, help="Save JSON to this path")
    args = parser.parse_args()

    k_dense = args.k_dense or settings.top_k_dense
    k_bm25  = args.k_bm25  or settings.top_k_bm25

    dense  = DenseRetriever.load(args.index_dir)
    bm25_r = BM25Retriever.load(str(Path(args.index_dir) / "bm25.pkl"))
    mlp    = FusionMLP.load(args.mlp_path)

    with open(args.qas_path, encoding="utf-8") as f:
        qas = [json.loads(line) for line in f if line.strip()]
    print(f"Loaded {len(qas):,} queries from {args.qas_path}")

    results = run_eval(qas, dense, bm25_r, mlp, k_dense, k_bm25, args.k_final)

    # Add index file sizes (needs paths, computed after run_eval)
    def _mb(path: str) -> float:
        try:
            return round(os.path.getsize(path) / 1e6, 2)
        except FileNotFoundError:
            return -1.0

    results["efficiency"]["faiss_index_mb"] = _mb(str(Path(args.index_dir) / "index.faiss"))
    results["efficiency"]["bm25_pkl_mb"]    = _mb(str(Path(args.index_dir) / "bm25.pkl"))
    results["meta"] = {
        "qas_path":  args.qas_path,
        "index_dir": args.index_dir,
        "mlp_path":  args.mlp_path,
    }

    print_results(results)

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"Saved → {args.output}")


if __name__ == "__main__":
    main()
