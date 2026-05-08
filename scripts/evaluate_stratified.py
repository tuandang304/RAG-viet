"""Stratified evaluation: NDCG@10 per query-feature group (MLP vs fixed 0.5/0.5).

Proves the central hypothesis: dynamic fusion helps most for queries where
one signal clearly dominates (low diacritic, high compound, etc.).

Output: a markdown/JSON table for paper Section 5.3.

Usage:
    uv run python scripts/evaluate_stratified.py \\
        --qas-path data/processed/viaquad_dev.jsonl \\
        --index-dir indexes/viaquad \\
        --mlp-path checkpoints/fusion_mlp_aug.pt \\
        --output results/stratified.json

    # Also test diacritic robustness:
    uv run python scripts/evaluate_stratified.py \\
        --qas-path data/processed/viaquad_dev_noisy.jsonl \\
        --index-dir indexes/viaquad \\
        --mlp-path checkpoints/fusion_mlp_aug.pt \\
        --output results/stratified_noisy.json
"""

import argparse
import json
from pathlib import Path

import numpy as np
from tqdm import tqdm

from rag_vie.config import settings
from rag_vie.features.vietnamese import FEATURE_NAMES, extract_features
from rag_vie.fusion.mlp import FusionMLP
from rag_vie.retrieval.bm25 import BM25Retriever
from rag_vie.retrieval.dense import DenseRetriever
from rag_vie.retrieval.hybrid import HybridRetriever


# ── Stratification groups ──────────────────────────────────────────────────

STRATA: list[dict] = [
    # diacritic_ratio buckets (key feature for diacritic robustness claim)
    {"name": "diac_low",      "feature": "diacritic_ratio",  "lo": 0.0,  "hi": 0.3},
    {"name": "diac_mid",      "feature": "diacritic_ratio",  "lo": 0.3,  "hi": 0.7},
    {"name": "diac_high",     "feature": "diacritic_ratio",  "lo": 0.7,  "hi": 1.01},
    # compound_ratio buckets
    {"name": "comp_low",      "feature": "compound_ratio",   "lo": 0.0,  "hi": 0.2},
    {"name": "comp_high",     "feature": "compound_ratio",   "lo": 0.2,  "hi": 1.01},
    # english_ratio buckets
    {"name": "eng_none",      "feature": "english_ratio",    "lo": 0.0,  "hi": 0.01},
    {"name": "eng_mixed",     "feature": "english_ratio",    "lo": 0.01, "hi": 1.01},
    # query length
    {"name": "short_query",   "feature": "query_length_norm","lo": 0.0,  "hi": 0.4},
    {"name": "long_query",    "feature": "query_length_norm","lo": 0.4,  "hi": 1.01},
    # clause complexity
    {"name": "simple",        "feature": "clause_count_norm","lo": 0.0,  "hi": 0.01},
    {"name": "complex",       "feature": "clause_count_norm","lo": 0.01, "hi": 1.01},
]


# ── Metrics ────────────────────────────────────────────────────────────────

def ndcg_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    dcg = sum(1.0 / np.log2(i + 2) for i, p in enumerate(retrieved[:k]) if p in relevant)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(relevant), k)))
    return dcg / idcg if idcg > 0 else 0.0


# ── Core eval ─────────────────────────────────────────────────────────────

def retrieve_all(
    hybrid: HybridRetriever,
    qas: list[dict],
    features: np.ndarray,
    mlp: FusionMLP,
    fixed_w: tuple[float, float],
    k_final: int = 100,
) -> tuple[list[list[str]], list[list[str]], list[np.ndarray]]:
    """Return (mlp_ranked, fixed_ranked, predicted_weights) for all queries."""
    mlp_ranked, fixed_ranked, pred_weights = [], [], []
    for i, qa in enumerate(tqdm(qas, desc="Retrieving")):
        w_mlp = mlp.predict_weights(features[i])
        hits_mlp = hybrid.retrieve(qa["question"], w_mlp, settings.top_k_dense, settings.top_k_bm25, k_final)
        mlp_ranked.append([p for p, _, _ in hits_mlp])

        hits_fix = hybrid.retrieve(qa["question"], fixed_w, settings.top_k_dense, settings.top_k_bm25, k_final)
        fixed_ranked.append([p for p, _, _ in hits_fix])

        pred_weights.append(np.array(w_mlp))
    return mlp_ranked, fixed_ranked, pred_weights


def stratum_metrics(
    indices: list[int],
    mlp_ranked: list[list[str]],
    fixed_ranked: list[list[str]],
    pred_weights: list[np.ndarray],
    qas: list[dict],
) -> dict:
    if not indices:
        return {"n": 0, "mlp_ndcg": None, "fixed_ndcg": None, "delta": None,
                "mean_w_dense": None, "mean_w_bm25": None}
    mlp_s, fix_s, w_d, w_b = [], [], [], []
    for i in indices:
        rel = set(qas[i]["relevant_ids"])
        mlp_s.append(ndcg_at_k(mlp_ranked[i], rel, 10))
        fix_s.append(ndcg_at_k(fixed_ranked[i], rel, 10))
        w_d.append(float(pred_weights[i][0]))
        w_b.append(float(pred_weights[i][1]))
    return {
        "n": len(indices),
        "mlp_ndcg":   round(float(np.mean(mlp_s)), 4),
        "fixed_ndcg": round(float(np.mean(fix_s)), 4),
        "delta":      round(float(np.mean(mlp_s)) - float(np.mean(fix_s)), 4),
        "mean_w_dense": round(float(np.mean(w_d)), 3),
        "mean_w_bm25":  round(float(np.mean(w_b)), 3),
    }


def print_table(results: dict) -> None:
    header = f"{'Stratum':<18} {'N':>5}  {'Fixed':>7}  {'MLP':>7}  {'Δ':>7}  {'w_dense':>8}  {'w_bm25':>7}"
    print("\n" + header)
    print("-" * len(header))
    for name, m in results["strata"].items():
        if m["n"] == 0:
            continue
        delta_str = f"+{m['delta']:.4f}" if m["delta"] >= 0 else f"{m['delta']:.4f}"
        print(
            f"{name:<18} {m['n']:>5}  {m['fixed_ndcg']:>7.4f}  {m['mlp_ndcg']:>7.4f}"
            f"  {delta_str:>7}  {m['mean_w_dense']:>8.3f}  {m['mean_w_bm25']:>7.3f}"
        )
    print()
    ov = results["overall"]
    print(f"{'OVERALL':<18} {ov['n']:>5}  {ov['fixed_ndcg']:>7.4f}  {ov['mlp_ndcg']:>7.4f}"
          f"  {('+' if ov['delta']>=0 else '')}{ov['delta']:.4f}")


# ── Main ──────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qas-path",   required=True)
    parser.add_argument("--index-dir",  default="indexes/viaquad")
    parser.add_argument("--mlp-path",   required=True)
    parser.add_argument("--fixed-weights", default="0.5,0.5")
    parser.add_argument("--output",     default=None)
    args = parser.parse_args()

    fixed_w = tuple(map(float, args.fixed_weights.split(",")))
    assert len(fixed_w) == 2

    # Load
    dense  = DenseRetriever.load(args.index_dir)
    bm25   = BM25Retriever.load(str(Path(args.index_dir) / "bm25.pkl"))
    hybrid = HybridRetriever(dense, bm25)
    mlp    = FusionMLP.load(args.mlp_path)

    with open(args.qas_path, encoding="utf-8") as f:
        qas = [json.loads(l) for l in f if l.strip()]
    print(f"Queries: {len(qas)}")

    # Extract features
    features = np.array([extract_features(q["question"]) for q in qas], dtype=np.float32)
    feat_idx = {name: i for i, name in enumerate(FEATURE_NAMES)}

    # Retrieve once for all
    mlp_ranked, fixed_ranked, pred_weights = retrieve_all(hybrid, qas, features, mlp, fixed_w)

    # Stratified metrics
    strata_results = {}
    for s in STRATA:
        fi = feat_idx[s["feature"]]
        idxs = [i for i, fv in enumerate(features[:, fi]) if s["lo"] <= fv < s["hi"]]
        strata_results[s["name"]] = stratum_metrics(idxs, mlp_ranked, fixed_ranked, pred_weights, qas)

    # Overall
    all_idxs = list(range(len(qas)))
    overall = stratum_metrics(all_idxs, mlp_ranked, fixed_ranked, pred_weights, qas)

    results = {"overall": overall, "strata": strata_results,
               "fixed_weights": list(fixed_w), "mlp_path": args.mlp_path}

    print_table(results)

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"Saved → {args.output}")


if __name__ == "__main__":
    main()
