"""Evaluate retrieval metrics (NDCG@10, MRR@10, Recall@100) on a QA dataset.

Usage:
    python scripts/evaluate.py --qas-path data/processed/viaquad_dev.jsonl \
        --index-dir indexes --bm25-path indexes/bm25.pkl \
        --mlp-path checkpoints/fusion_mlp.pt
"""

import argparse
import json
from pathlib import Path

import numpy as np
from tqdm import tqdm

from rag_vie.config import settings
from rag_vie.features.vietnamese import extract_features
from rag_vie.fusion.mlp import FusionMLP
from rag_vie.retrieval.bm25 import BM25Retriever
from rag_vie.retrieval.dense import DenseRetriever
from rag_vie.retrieval.hybrid import HybridRetriever


def ndcg_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    dcg = sum(
        1.0 / np.log2(i + 2)
        for i, pid in enumerate(retrieved_ids[:k])
        if pid in relevant_ids
    )
    idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(relevant_ids), k)))
    return dcg / idcg if idcg > 0 else 0.0


def mrr_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    for i, pid in enumerate(retrieved_ids[:k]):
        if pid in relevant_ids:
            return 1.0 / (i + 1)
    return 0.0


def recall_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    hits = sum(1 for pid in retrieved_ids[:k] if pid in relevant_ids)
    return hits / len(relevant_ids) if relevant_ids else 0.0


def evaluate(
    hybrid: HybridRetriever,
    mlp: FusionMLP,
    qas: list[dict],
    fixed_weights: tuple[float, float] | None = None,
) -> dict[str, float]:
    ndcg_scores, mrr_scores, recall_scores = [], [], []

    for qa in tqdm(qas, desc="Evaluating"):
        query = qa["question"]
        relevant_ids: set[str] = set(qa["relevant_ids"])

        if fixed_weights is not None:
            weights = fixed_weights
        else:
            features = extract_features(query)
            weights = mlp.predict_weights(features)

        hits = hybrid.retrieve(
            query=query,
            weights=weights,
            k_dense=settings.top_k_dense,
            k_bm25=settings.top_k_bm25,
            k_final=100,
        )
        retrieved_ids = [pid for pid, _, _ in hits]

        ndcg_scores.append(ndcg_at_k(retrieved_ids, relevant_ids, 10))
        mrr_scores.append(mrr_at_k(retrieved_ids, relevant_ids, 10))
        recall_scores.append(recall_at_k(retrieved_ids, relevant_ids, 100))

    return {
        "NDCG@10": float(np.mean(ndcg_scores)),
        "MRR@10": float(np.mean(mrr_scores)),
        "Recall@100": float(np.mean(recall_scores)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qas-path", required=True)
    parser.add_argument("--index-dir", default=settings.index_dir)
    parser.add_argument("--bm25-path", default=None)
    parser.add_argument("--mlp-path", default=None)
    parser.add_argument("--fixed-weights", default=None, help="e.g. 0.5,0.5")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    bm25_path = args.bm25_path or str(Path(args.index_dir) / "bm25.pkl")

    dense = DenseRetriever.load(args.index_dir)
    bm25 = BM25Retriever.load(bm25_path)
    hybrid = HybridRetriever(dense, bm25)

    mlp = FusionMLP.load(args.mlp_path) if args.mlp_path else FusionMLP()

    fixed_weights = None
    if args.fixed_weights:
        parts = list(map(float, args.fixed_weights.split(",")))
        assert len(parts) == 2, "--fixed-weights cần đúng 2 giá trị, ví dụ: 0.5,0.5"
        fixed_weights = (parts[0], parts[1])

    with open(args.qas_path, encoding="utf-8") as f:
        qas = [json.loads(line) for line in f]

    metrics = evaluate(hybrid, mlp, qas, fixed_weights)

    print("\n--- Results ---")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
