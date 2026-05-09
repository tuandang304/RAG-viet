"""Train FusionMLP (3-way) on ViQuAD training set.

Pipeline:
  1. BM25 search + feature extraction (underthesea)
  2. Sparse search (BGE-M3 local, PyTorch) — initialised BEFORE FAISS to avoid OMP deadlock
  3. FAISS dense batch search
  4. 3D soft-label grid search on simplex: softmax(NDCG / T) → expected (a, b, c)
  5. Spawn training subprocess (isolates PyTorch init from FAISS)
  6. Save checkpoint

Usage:
    uv run python scripts/train_mlp.py \\
        --qas-path data/processed/viaquad_train_aug.jsonl \\
        --index-dir indexes/viaquad \\
        --output checkpoints/fusion_mlp_aug.pt \\
        --emb-cache checkpoints/train_aug_embeddings.npy

    # Fine-tune from existing checkpoint:
    uv run python scripts/train_mlp.py \\
        --init-from checkpoints/fusion_mlp_aug.pt \\
        --emb-cache checkpoints/train_aug_embeddings.npy \\
        --lr 1e-4 --epochs 50
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
import json
import random
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from tqdm import tqdm


# ── Simplex grid: step = 0.1 → 66 points (a, b, c) with a+b+c=1 ─────────────
_N = 10
WEIGHT_GRID: list[tuple[float, float, float]] = [
    (i / _N, j / _N, (_N - i - j) / _N)
    for i in range(_N + 1)
    for j in range(_N + 1 - i)
]
TOP_K = 100


# ── Train-only subprocess ─────────────────────────────────────────────────────

def _train_only_mode() -> None:
    """Load X/Y .npz, train MLP, save checkpoint. Isolated from FAISS."""
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
    output_dim = Y.shape[1]
    print(f"Training pairs: {len(X)}  |  feature dim: {X.shape[1]}  |  output dim: {output_dim}", flush=True)
    print(f"Mean weights — dense: {Y[:, 0].mean():.3f}  bm25: {Y[:, 1].mean():.3f}  sparse: {Y[:, 2].mean():.3f}", flush=True)

    X_t, Y_t = torch.from_numpy(X), torch.from_numpy(Y)
    loader = DataLoader(TensorDataset(X_t, Y_t), batch_size=args.batch_size, shuffle=True, num_workers=0)

    if args.init_from:
        print(f"  Fine-tuning from: {args.init_from}", flush=True)
        mlp = FusionMLP.load(args.init_from)
    else:
        mlp = FusionMLP(input_dim=X.shape[1], output_dim=output_dim)

    optimizer = torch.optim.Adam(mlp.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    print(f"\nTraining MLP ({args.epochs} epochs, lr={args.lr})…", flush=True)
    mlp.train()
    for epoch in range(1, args.epochs + 1):
        total_loss = 0.0
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = criterion(mlp(xb), yb)
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


# ── Data-collection phase ─────────────────────────────────────────────────────
# Imports only reached when NOT in --train-only mode.

import faiss  # noqa: E402

from rag_vie.features.vietnamese import extract_features  # noqa: E402
from rag_vie.retrieval.bm25 import BM25Retriever          # noqa: E402
from rag_vie.retrieval.dense import DenseRetriever         # noqa: E402
from rag_vie.retrieval.embedder import embed_texts         # noqa: E402


def _minmax(scores: dict[str, float]) -> dict[str, float]:
    if not scores:
        return {}
    lo, hi = min(scores.values()), max(scores.values())
    if hi == lo:
        return {k: 0.0 for k in scores}
    return {k: (v - lo) / (hi - lo) for k, v in scores.items()}


def ndcg_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    dcg  = sum(1.0 / np.log2(i + 2) for i, p in enumerate(ranked[:k]) if p in relevant)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(relevant), k)))
    return dcg / idcg if idcg > 0 else 0.0


def find_soft_weights(
    dense_scores: dict[str, float],
    bm25_scores:  dict[str, float],
    sparse_scores: dict[str, float],
    relevant_ids: set[str],
    temperature: float = 0.3,
) -> tuple[float, float, float]:
    """3D soft-label grid search on the simplex.

    Computes NDCG@10 for each of the 66 simplex grid points,
    applies softmax(NDCG / T), returns expected (a, b, c).
    """
    dense_n  = _minmax(dense_scores)
    bm25_n   = _minmax(bm25_scores)
    sparse_n = _minmax(sparse_scores)
    all_ids  = list(set(dense_n) | set(bm25_n) | set(sparse_n))

    ndcg_vals = []
    for a, b, c in WEIGHT_GRID:
        fused = {
            pid: a * dense_n.get(pid, 0.0) + b * bm25_n.get(pid, 0.0) + c * sparse_n.get(pid, 0.0)
            for pid in all_ids
        }
        ranked = sorted(fused, key=fused.get, reverse=True)
        ndcg_vals.append(ndcg_at_k(ranked, relevant_ids, 10))

    arr = np.array(ndcg_vals, dtype=np.float64)
    arr -= arr.max()
    probs = np.exp(arr / temperature)
    probs /= probs.sum()

    ea = float(sum(p * a for p, (a, b, c) in zip(probs, WEIGHT_GRID)))
    eb = float(sum(p * b for p, (a, b, c) in zip(probs, WEIGHT_GRID)))
    ec = float(sum(p * c for p, (a, b, c) in zip(probs, WEIGHT_GRID)))
    total = ea + eb + ec
    return ea / total, eb / total, ec / total


def collect_training_pairs(
    qas: list[dict],
    dense: DenseRetriever,
    bm25: BM25Retriever,
    sparse,                         # SparseRetriever | None
    query_embeddings: np.ndarray,
    temperature: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (X: features, Y: 3-way soft-label weights).

    Phase ordering to avoid OMP deadlock on macOS:
      1 — BM25 + features (underthesea, pure Python)
      2 — Sparse search (BGE-M3 / PyTorch — initialised BEFORE FAISS ops)
      3 — FAISS dense batch search
      4 — 3D soft-label grid search (pure numpy)
    """
    # Phase 1: BM25 + features
    print("  Phase 1/4: BM25 search + feature extraction…")
    bm25_results:     list[list[tuple[str, str, float]]] = []
    features_list:    list[np.ndarray] = []
    for qa in tqdm(qas, desc="  BM25+feat"):
        bm25_results.append(bm25.search(qa["question"], TOP_K))
        features_list.append(extract_features(qa["question"]))

    # Phase 2: Sparse search (PyTorch — must come BEFORE faiss.normalize_L2)
    print("  Phase 2/4: Sparse search (BGE-M3)…")
    sparse_results: list[dict[str, float]] = []
    if sparse is not None:
        for qa in tqdm(qas, desc="  Sparse"):
            hits = sparse.search(qa["question"], TOP_K)
            sparse_results.append({pid: s for pid, _, s in hits})
    else:
        sparse_results = [{} for _ in qas]

    # Phase 3: FAISS dense batch search
    print("  Phase 3/4: FAISS batch search…")
    embs = query_embeddings.copy().astype(np.float32)
    faiss.normalize_L2(embs)
    scores_arr, indices_arr = dense._index.search(embs, TOP_K)

    # Phase 4: 3D soft-label grid search
    print(f"  Phase 4/4: 3D soft-label grid search (T={temperature}, {len(WEIGHT_GRID)} grid pts)…")
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
        bm25_scores  = {pid: s for pid, _, s in bm25_results[i]}
        sparse_scores = sparse_results[i]

        a, b, c = find_soft_weights(dense_scores, bm25_scores, sparse_scores, relevant_ids, temperature)

        X_list.append(features_list[i])
        Y_list.append([a, b, c])

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
    """Save X/Y then re-spawn this script with --train-only (fresh process, no FAISS)."""
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
        env["KMP_DUPLICATE_LIB_OK"] = "TRUE"
        env["OMP_NUM_THREADS"] = "1"
        env["MKL_NUM_THREADS"] = "1"
        env["PYTHONUNBUFFERED"] = "1"
        result = subprocess.run(cmd, env=env)
        if result.returncode != 0:
            raise RuntimeError(f"Training subprocess failed (exit {result.returncode})")
    finally:
        Path(xy_path).unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qas-path",   default="data/processed/viaquad_train_aug.jsonl")
    parser.add_argument("--index-dir",  default="indexes/viaquad")
    parser.add_argument("--output",     default="checkpoints/fusion_mlp_aug.pt")
    parser.add_argument("--emb-cache",  default=None,
                        help="Cache path for query embeddings (avoids re-calling FPT API)")
    parser.add_argument("--sparse-path", default=None,
                        help="Path to sparse.pkl (default: <index-dir>/sparse.pkl)")
    parser.add_argument("--no-sparse",  action="store_true",
                        help="Disable sparse signal (train 3-way with c=0 — uses 2-way effectively)")
    parser.add_argument("--init-from",  default=None)
    parser.add_argument("--max-samples", type=int, default=5000)
    parser.add_argument("--epochs",     type=int,   default=100)
    parser.add_argument("--batch-size", type=int,   default=256)
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--seed",       type=int,   default=42)
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
    bm25  = BM25Retriever.load(f"{args.index_dir}/bm25.pkl")

    sparse = None
    if not args.no_sparse:
        sparse_path = args.sparse_path or f"{args.index_dir}/sparse.pkl"
        if Path(sparse_path).exists():
            from rag_vie.retrieval.sparse import SparseRetriever
            sparse = SparseRetriever.load(sparse_path)
            print(f"Sparse index loaded from {sparse_path}")
        else:
            print(f"WARNING: sparse.pkl not found at {sparse_path}. "
                  "Run build_index.py first, or use --no-sparse.")

    # Query embeddings (cached to avoid re-calling FPT API)
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
            print(f"  Cached → {args.emb_cache}")

    # Collect soft-label training pairs
    print("Collecting 3-way soft-label training pairs…")
    X, Y = collect_training_pairs(qas, dense, bm25, sparse, query_embeddings, args.temperature)
    print(f"Pairs: {len(X)}  |  dim: {X.shape[1]}  |  output: {Y.shape[1]}", flush=True)
    print(f"Mean weights — dense: {Y[:, 0].mean():.3f}  bm25: {Y[:, 1].mean():.3f}  sparse: {Y[:, 2].mean():.3f}", flush=True)

    # Spawn training subprocess (fresh process — no FAISS OMP conflict with PyTorch)
    print("\nSpawning training subprocess…", flush=True)
    spawn_train_mlp(X, Y, args.output, args.epochs, args.batch_size, args.lr, args.seed, args.init_from)


if __name__ == "__main__":
    main()
