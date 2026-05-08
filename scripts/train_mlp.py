"""Train FusionMLP trên ViQuAD training set.

Pipeline:
  1. Load training QAs, trích đặc trưng Vietnamese-aware
  2. Batch-embed queries qua FPT API (có cache để tránh gọi lại)
  3. Với mỗi query: dense FAISS search + BM25 search
  4. Soft-label grid search: softmax(NDCG_scores / temperature) → expected (a, b)
  5. Train MLP với MSE loss trên soft labels (in a fresh subprocess — avoids OMP deadlock)
  6. Lưu checkpoint

Usage:
    # Train từ đầu với soft labels
    uv run python scripts/train_mlp.py \
        --qas-path data/processed/viaquad_train.jsonl \
        --index-dir indexes/viaquad \
        --output checkpoints/fusion_mlp.pt \
        --emb-cache checkpoints/train_embeddings.npy

    # Fine-tune từ checkpoint cũ
    uv run python scripts/train_mlp.py \
        --init-from checkpoints/fusion_mlp.pt \
        --emb-cache checkpoints/train_embeddings.npy \
        --lr 1e-4 --epochs 50
"""

import argparse
import json
import os
import random
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from tqdm import tqdm

# ── Mode: pure PyTorch training (no FAISS) ────────────────────────────────
# This block runs when the script is re-spawned as a subprocess for training only.
# It must appear before any FAISS/underthesea imports so they are never loaded.

def _train_only_mode() -> None:
    """Load X/Y .npz, train MLP, save checkpoint. No FAISS involved."""
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    from rag_vie.fusion.mlp import FusionMLP

    parser = argparse.ArgumentParser()
    parser.add_argument("--xy-path",    required=True)
    parser.add_argument("--output",     required=True)
    parser.add_argument("--init-from",  default=None)
    parser.add_argument("--epochs",     type=int,   default=100)
    parser.add_argument("--batch-size", type=int,   default=256)
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--seed",       type=int,   default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    data = np.load(args.xy_path)
    X, Y = data["X"].astype(np.float32), data["Y"].astype(np.float32)
    print(f"Training pairs: {len(X)}  |  feature dim: {X.shape[1]}", flush=True)
    print(f"Mean soft weights — dense: {Y[:, 0].mean():.3f}  bm25: {Y[:, 1].mean():.3f}", flush=True)

    X_t = torch.from_numpy(X)
    Y_t = torch.from_numpy(Y)
    dataset = TensorDataset(X_t, Y_t)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)

    if args.init_from:
        print(f"  Fine-tuning từ checkpoint: {args.init_from}", flush=True)
        mlp = FusionMLP.load(args.init_from)
    else:
        mlp = FusionMLP(input_dim=X.shape[1], output_dim=2)

    optimizer = torch.optim.Adam(mlp.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    print(f"\nTraining MLP ({args.epochs} epochs, lr={args.lr})…", flush=True)
    mlp.train()
    for epoch in range(1, args.epochs + 1):
        total_loss = 0.0
        for xb, yb in loader:
            optimizer.zero_grad()
            pred = mlp(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(xb)
        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{args.epochs} — MSE loss: {total_loss / len(X):.6f}", flush=True)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    mlp.save(out_path)
    print(f"\nCheckpoint saved → {out_path}", flush=True)


if "--train-only" in sys.argv:
    sys.argv.remove("--train-only")
    _train_only_mode()
    sys.exit(0)

# ── Data-collection phase (FAISS + underthesea) ───────────────────────────
# Only imported when NOT in --train-only mode so the subprocess stays clean.

import faiss  # noqa: E402

from rag_vie.features.vietnamese import extract_features  # noqa: E402
from rag_vie.retrieval.bm25 import BM25Retriever  # noqa: E402
from rag_vie.retrieval.dense import DenseRetriever  # noqa: E402
from rag_vie.retrieval.embedder import embed_texts  # noqa: E402

# Grid: a ∈ {0.0, 0.05, ..., 1.0} (21 điểm), b = 1.0 - a
WEIGHT_GRID = [(round(a / 20, 2), round(1.0 - a / 20, 2)) for a in range(21)]
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


def find_soft_weights(
    dense_scores: dict[str, float],
    bm25_scores: dict[str, float],
    relevant_ids: set[str],
    temperature: float = 0.3,
) -> tuple[float, float]:
    """Soft-label grid search.

    Tính NDCG@10 cho mỗi grid point, áp softmax(NDCG / T),
    rồi lấy kỳ vọng E[a] và E[b] làm soft label.
    Tránh hard boundary và nhiễu từ ties.
    """
    dense_norm = _min_max_normalize(dense_scores)
    bm25_norm = _min_max_normalize(bm25_scores)
    all_ids = list(set(dense_norm) | set(bm25_norm))

    ndcg_scores = []
    for a, b in WEIGHT_GRID:
        fused = {
            pid: a * dense_norm.get(pid, 0.0) + b * bm25_norm.get(pid, 0.0)
            for pid in all_ids
        }
        ranked = sorted(fused, key=fused.get, reverse=True)
        ndcg_scores.append(ndcg_at_k(ranked, relevant_ids, 10))

    scores_arr = np.array(ndcg_scores, dtype=np.float64)
    # Softmax với temperature
    scores_arr = scores_arr - scores_arr.max()
    exp = np.exp(scores_arr / temperature)
    probs = exp / exp.sum()

    expected_a = sum(p * a for p, (a, _) in zip(probs, WEIGHT_GRID))
    expected_b = sum(p * b for p, (_, b) in zip(probs, WEIGHT_GRID))

    # Re-normalize để đảm bảo a + b = 1
    total = expected_a + expected_b
    return float(expected_a / total), float(expected_b / total)


def collect_training_pairs(
    qas: list[dict],
    dense: DenseRetriever,
    bm25: BM25Retriever,
    query_embeddings: np.ndarray,
    temperature: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Trả về (X: features, Y: soft-label weights).

    3 phases tách biệt để tránh segfault (underthesea + FAISS + torch):
      Phase 1 — BM25 search + feature extraction (underthesea)
      Phase 2 — FAISS batch search
      Phase 3 — Soft-label grid search (pure numpy)
    """
    # Phase 1: BM25 + features
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
    scores_arr, indices_arr = dense._index.search(embs, TOP_K)

    # Phase 3: Soft-label grid search
    print(f"  Phase 3/3: Soft-label grid search (T={temperature})…")
    X_list, Y_list = [], []
    for i, qa in enumerate(tqdm(qas, desc="  Soft-grid")):
        relevant_ids = set(qa["relevant_ids"])
        if not relevant_ids:
            continue

        dense_scores = {
            dense._ids[idx]: float(scores_arr[i, j])
            for j, idx in enumerate(indices_arr[i])
            if idx != -1
        }
        bm25_scores = {pid: s for pid, _, s in bm25_results[i]}

        a_soft, b_soft = find_soft_weights(dense_scores, bm25_scores, relevant_ids, temperature)

        X_list.append(features_list[i])
        Y_list.append([a_soft, b_soft])

    return np.array(X_list, dtype=np.float32), np.array(Y_list, dtype=np.float32)


def spawn_train_mlp(
    X: np.ndarray,
    Y: np.ndarray,
    output: str,
    epochs: int,
    batch_size: int,
    lr: float,
    seed: int,
    init_from: str | None = None,
) -> None:
    """Save X/Y to a temp file, then re-spawn this script with --train-only.

    This avoids the OMP/FAISS+PyTorch deadlock on macOS ARM64 by ensuring
    PyTorch is initialized in a fresh process that never loaded FAISS.
    """
    with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
        xy_path = f.name
    try:
        np.savez(xy_path, X=X, Y=Y)
        cmd = [
            sys.executable, __file__, "--train-only",
            "--xy-path", xy_path,
            "--output", output,
            "--epochs", str(epochs),
            "--batch-size", str(batch_size),
            "--lr", str(lr),
            "--seed", str(seed),
        ]
        if init_from:
            cmd += ["--init-from", init_from]
        env = os.environ.copy()
        env["OMP_NUM_THREADS"] = "1"
        env["MKL_NUM_THREADS"] = "1"
        env["KMP_DUPLICATE_LIB_OK"] = "TRUE"
        env["PYTHONUNBUFFERED"] = "1"
        result = subprocess.run(cmd, env=env)
        if result.returncode != 0:
            raise RuntimeError(f"Training subprocess failed (exit {result.returncode})")
    finally:
        Path(xy_path).unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qas-path", default="data/processed/viaquad_train.jsonl")
    parser.add_argument("--index-dir", default="indexes/viaquad")
    parser.add_argument("--output", default="checkpoints/fusion_mlp.pt")
    parser.add_argument("--emb-cache", default=None,
                        help="Path để save/load query embeddings (tránh gọi lại API)")
    parser.add_argument("--init-from", default=None,
                        help="Fine-tune từ checkpoint này thay vì train từ đầu")
    parser.add_argument("--max-samples", type=int, default=5000)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--temperature", type=float, default=0.3,
                        help="Softmax temperature cho soft labels (nhỏ hơn = sharper)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

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

    # Embeddings (với cache)
    queries = [qa["question"] for qa in qas]
    if args.emb_cache and Path(args.emb_cache).exists():
        print(f"Loading embeddings from cache: {args.emb_cache}")
        query_embeddings = np.load(args.emb_cache)
        if len(query_embeddings) != len(queries):
            print("  Cache size mismatch — re-embedding…")
            query_embeddings = embed_texts(queries, batch_size=32)
            np.save(args.emb_cache, query_embeddings)
    else:
        print(f"Embedding {len(queries)} queries via FPT API…")
        query_embeddings = embed_texts(queries, batch_size=32)
        if args.emb_cache:
            Path(args.emb_cache).parent.mkdir(parents=True, exist_ok=True)
            np.save(args.emb_cache, query_embeddings)
            print(f"  Embeddings cached → {args.emb_cache}")

    # Soft-label grid search → training pairs
    print("Collecting soft-label training pairs…")
    X, Y = collect_training_pairs(qas, dense, bm25, query_embeddings, args.temperature)
    print(f"Training pairs collected: {len(X)}  |  feature dim: {X.shape[1]}", flush=True)
    print(f"Mean soft weights — dense: {Y[:, 0].mean():.3f}  bm25: {Y[:, 1].mean():.3f}", flush=True)

    # Spawn a fresh subprocess for MLP training (avoids OMP/FAISS+PyTorch deadlock on macOS)
    print("\nSpawning training subprocess (isolated from FAISS OMP)…", flush=True)
    spawn_train_mlp(
        X, Y,
        output=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
        init_from=args.init_from,
    )


if __name__ == "__main__":
    main()
