"""Train FusionMLP trên ViQuAD training set.

Pipeline:
  1. Load training QAs, trích đặc trưng Vietnamese-aware
  2. Batch-embed tất cả queries qua FPT API (batch=32)
  3. Với mỗi query: dense FAISS search + BM25 search
  4. Grid search (a ∈ {0.0, 0.1, …, 1.0}) để tìm (a, b) tối ưu NDCG@10
  5. Train MLP (features → optimal weights) với MSE loss
  6. Lưu checkpoint tại checkpoints/fusion_mlp.pt

Usage:
    uv run python scripts/train_mlp.py \
        --qas-path data/processed/viaquad_train.jsonl \
        --index-dir indexes/viaquad \
        --output checkpoints/fusion_mlp.pt
"""

import argparse
import json
import random
from pathlib import Path

import faiss
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from rag_vie.features.vietnamese import extract_features
from rag_vie.fusion.mlp import FusionMLP
from rag_vie.retrieval.bm25 import BM25Retriever
from rag_vie.retrieval.dense import DenseRetriever
from rag_vie.retrieval.embedder import embed_texts

# Grid: a ∈ {0.0, 0.1, ..., 1.0}, b = 1.0 - a
WEIGHT_GRID = [(round(a / 10, 1), round(1.0 - a / 10, 1)) for a in range(11)]
TOP_K = 100


def _min_max_normalize(scores: dict[str, float]) -> dict[str, float]:
    if not scores:
        return {}
    vals = list(scores.values())
    lo, hi = min(vals), max(vals)
    if hi == lo:
        return {k: 0.0 for k in scores}
    return {k: (v - lo) / (hi - lo) for k, v in scores.items()}


def ndcg_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    dcg = sum(
        1.0 / np.log2(i + 2)
        for i, pid in enumerate(retrieved_ids[:k])
        if pid in relevant_ids
    )
    idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(relevant_ids), k)))
    return dcg / idcg if idcg > 0 else 0.0


def find_optimal_weights(
    dense_scores: dict[str, float],
    bm25_scores: dict[str, float],
    passage_map: dict[str, str],
    relevant_ids: set[str],
) -> tuple[float, float]:
    """Grid search trả về (a, b) tối ưu NDCG@10."""
    dense_norm = _min_max_normalize(dense_scores)
    bm25_norm = _min_max_normalize(bm25_scores)
    all_ids = list(set(dense_norm) | set(bm25_norm))

    best_ndcg, best_a, best_b = -1.0, 0.5, 0.5
    for a, b in WEIGHT_GRID:
        fused = {
            pid: a * dense_norm.get(pid, 0.0) + b * bm25_norm.get(pid, 0.0)
            for pid in all_ids
        }
        ranked = sorted(fused, key=fused.get, reverse=True)
        score = ndcg_at_k(ranked, relevant_ids, 10)
        if score > best_ndcg:
            best_ndcg, best_a, best_b = score, a, b
    return best_a, best_b


def collect_training_pairs(
    qas: list[dict],
    dense: DenseRetriever,
    bm25: BM25Retriever,
    query_embeddings: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Trả về (X: features, Y: optimal weights) cho tất cả queries.

    Tách thành 3 phase để tránh segfault khi underthesea + FAISS chạy xen kẽ:
      Phase 1 — BM25 search (underthesea)
      Phase 2 — FAISS batch search
      Phase 3 — Grid search + feature extraction
    """
    # Phase 1: BM25 + features (underthesea) — gọi tất cả trước khi dùng torch/faiss
    print("  Phase 1/3: BM25 search + feature extraction…")
    bm25_results: list[list[tuple[str, str, float]]] = []
    features_list: list[np.ndarray] = []
    for qa in tqdm(qas, desc="  BM25+feat"):
        bm25_results.append(bm25.search(qa["question"], TOP_K))
        features_list.append(extract_features(qa["question"]))

    # Phase 2: FAISS batch search
    print("  Phase 2/3: FAISS batch search…")
    embs = query_embeddings.copy().astype(np.float32)
    faiss.normalize_L2(embs)
    scores_arr, indices_arr = dense._index.search(embs, TOP_K)  # (N, TOP_K)

    # Phase 3: Grid search only (no external library calls)
    print("  Phase 3/3: Grid search…")
    X_list, Y_list = [], []
    for i, qa in enumerate(tqdm(qas, desc="  Grid")):
        relevant_ids = set(qa["relevant_ids"])
        if not relevant_ids:
            continue

        dense_scores = {
            dense._ids[idx]: float(scores_arr[i, j])
            for j, idx in enumerate(indices_arr[i])
            if idx != -1
        }
        bm25_scores = {pid: s for pid, _, s in bm25_results[i]}

        a_opt, b_opt = find_optimal_weights(dense_scores, bm25_scores, {}, relevant_ids)

        X_list.append(features_list[i])
        Y_list.append([a_opt, b_opt])

    return np.array(X_list, dtype=np.float32), np.array(Y_list, dtype=np.float32)


def train_mlp(
    X: np.ndarray,
    Y: np.ndarray,
    epochs: int,
    batch_size: int,
    lr: float,
) -> FusionMLP:
    X_t = torch.from_numpy(X.copy())
    Y_t = torch.from_numpy(Y.copy())
    dataset = TensorDataset(X_t, Y_t)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    mlp = FusionMLP(input_dim=X.shape[1], output_dim=2)
    optimizer = torch.optim.Adam(mlp.parameters(), lr=lr)
    criterion = nn.MSELoss()

    mlp.train()
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        for xb, yb in loader:
            optimizer.zero_grad()
            pred = mlp(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(xb)
        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{epochs} — MSE loss: {total_loss / len(X):.6f}")

    return mlp


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qas-path", default="data/processed/viaquad_train.jsonl")
    parser.add_argument("--index-dir", default="indexes/viaquad")
    parser.add_argument("--output", default="checkpoints/fusion_mlp.pt")
    parser.add_argument("--max-samples", type=int, default=5000,
                        help="Số query tối đa dùng để train (0 = dùng tất cả)")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Load QAs
    with open(args.qas_path, encoding="utf-8") as f:
        qas = [json.loads(line) for line in f if line.strip()]
    qas = [q for q in qas if q.get("relevant_ids")]

    if args.max_samples and args.max_samples < len(qas):
        qas = random.sample(qas, args.max_samples)
    print(f"Training queries: {len(qas)}")

    # Load indexes
    print("Loading indexes…")
    dense = DenseRetriever.load(args.index_dir)
    bm25 = BM25Retriever.load(f"{args.index_dir}/bm25.pkl")

    # Batch embed
    queries = [qa["question"] for qa in qas]
    print(f"Embedding {len(queries)} queries via FPT API…")
    query_embeddings = embed_texts(queries, batch_size=32)  # (N, dim) float32

    # Grid search → training pairs
    print("Collecting training pairs via grid search…")
    X, Y = collect_training_pairs(qas, dense, bm25, query_embeddings)
    print(f"Training pairs: {len(X)}  |  feature dim: {X.shape[1]}")
    print(f"Mean optimal weights — dense: {Y[:, 0].mean():.3f}  bm25: {Y[:, 1].mean():.3f}")

    # Train MLP
    print(f"\nTraining MLP ({args.epochs} epochs)…")
    mlp = train_mlp(X, Y, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)

    # Save
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    mlp.save(out_path)
    print(f"\nCheckpoint saved → {out_path}")


if __name__ == "__main__":
    main()
