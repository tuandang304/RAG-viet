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

import pyarrow  # noqa: F401
import argparse
import json
import random
import subprocess
import sys
import tempfile
from pathlib import Path

# Windows consoles default to cp1252, which can't encode characters like '→'
# used in progress messages — reconfigure instead of crashing mid-training.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import tensorflow as tf
from tqdm import tqdm


# ── Simplex grids ────────────────────────────────────────────────────────────
# Three-way: full 2-simplex, step = 0.1 → 66 points (a, b, c) with a+b+c=1
# Two-way:   edge of simplex where c=0, step = 0.1 → 11 points (a, b, 0) with a+b=1
_N = 10
WEIGHT_GRID_3WAY: list[tuple[float, float, float]] = [
    (i / _N, j / _N, (_N - i - j) / _N)
    for i in range(_N + 1)
    for j in range(_N + 1 - i)
]
WEIGHT_GRID_2WAY: list[tuple[float, float, float]] = [
    (i / _N, (_N - i) / _N, 0.0) for i in range(_N + 1)
]
# Four-way: (w_dense, w_bm25, w_sparse, w_toneless), step = 0.1 → 286 points
WEIGHT_GRID_4WAY: list[tuple[float, float, float, float]] = [
    (i / _N, j / _N, k / _N, (_N - i - j - k) / _N)
    for i in range(_N + 1)
    for j in range(_N + 1 - i)
    for k in range(_N + 1 - i - j)
]
TOP_K = 200


# ── Train-only subprocess ─────────────────────────────────────────────────────

def expected_ndcg_loss(y_true, y_pred):
    """Negative Expected NDCG Loss.
    
    y_true: Ground-truth NDCG scores of grid points (batch_size, output_dim)
    y_pred: Predicted probabilities of grid points (batch_size, output_dim)
    """
    # Maximize Expected NDCG = Minimize Negative Expected NDCG
    return -tf.reduce_mean(tf.reduce_sum(y_true * y_pred, axis=-1))


def _train_only_mode() -> None:
    """Load X/Y .npz, train MLP with Keras, save checkpoint. Isolated from FAISS."""
    import keras
    from rag_vie.fusion.mlp import FusionMLP

    parser = argparse.ArgumentParser()
    parser.add_argument("--xy-path",    required=True)
    parser.add_argument("--output",     required=True)
    parser.add_argument("--init-from",  default=None)
    parser.add_argument("--epochs",     type=int,   default=100)
    parser.add_argument("--batch-size", type=int,   default=256)
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--seed",       type=int,   default=42)
    parser.add_argument("--keep-zero-rows", action="store_true",
                        help="Keep queries where no channel found a relevant doc. "
                             "With raw NDCG labels these flat rows teach the model to "
                             "predict flat surfaces (→ ~equal weights) on dead queries.")
    args = parser.parse_args()

    keras.utils.set_random_seed(args.seed)

    data = np.load(args.xy_path)
    X, Y = data["X"].astype(np.float32), data["Y"].astype(np.float32)
    print(f"Loaded training pairs: {len(X)}  |  feature dim: {X.shape[1]}  |  output dim: {Y.shape[1]}", flush=True)

    if not args.keep_zero_rows:
        # Filter out queries where no retriever found any relevant document
        # (all-zero NDCG rows contribute zero gradient and dilute the routing signal).
        max_ndcg = Y.max(axis=1)
        valid = max_ndcg > 0
        n_before = len(X)
        X, Y = X[valid], Y[valid]
        print(f"  Filtered: {n_before} → {len(X)} queries (removed {n_before - len(X)} zero-NDCG rows)", flush=True)
    output_dim = Y.shape[1]

    if args.init_from:
        print(f"  Fine-tuning from: {args.init_from}", flush=True)
        mlp = FusionMLP.load(args.init_from)
    else:
        mlp = FusionMLP(input_dim=X.shape[1], output_dim=output_dim)

    # Adapt the input Normalization layer to training feature statistics
    mlp.adapt_features(X)

    mlp.compile(
        optimizer=keras.optimizers.Adam(learning_rate=args.lr),
        loss="mse",
    )

    print(f"\nTraining MLP ({args.epochs} epochs, lr={args.lr}) with MSE Grid Predictor Loss...", flush=True)
    mlp.fit(X, Y, epochs=args.epochs, batch_size=args.batch_size, verbose=1)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    mlp.save(out_path)
    print(f"\nCheckpoint saved -> {out_path}", flush=True)


if "--train-only" in sys.argv:
    sys.argv.remove("--train-only")
    _train_only_mode()
    sys.exit(0)


# ── Data-collection phase ─────────────────────────────────────────────────────
# Imports only reached when NOT in --train-only mode.

import faiss  # noqa: E402

from rag_vie.features.retrieval_signals import extract_retrieval_signals  # noqa: E402
from rag_vie.features.vietnamese import extract_features  # noqa: E402
from rag_vie.retrieval.bm25 import BM25Retriever          # noqa: E402
from rag_vie.retrieval.dense import DenseRetriever         # noqa: E402
from rag_vie.retrieval.embedder import embed_texts         # noqa: E402
from rag_vie.utils.metrics import min_max_normalize as _minmax  # noqa: E402


def find_ndcg_grid(
    dense_scores: dict[str, float],
    bm25_scores:  dict[str, float],
    sparse_scores: dict[str, float],
    relevant_ids: set[str],
    include_sparse: bool = True,
    toneless_scores: dict[str, float] | None = None,
) -> list[float]:
    """Calculate NDCG@10 for all simplex grid points (vectorized).

    toneless given + include_sparse → 3-simplex, 286 grid points (4-way).
    include_sparse=True            → full 2-simplex, 66 grid points.
    include_sparse=False           → 1-simplex edge with c=0, 11 grid points.

    Fusion for all grid points at once: S (n_ids × n_ch) @ Gᵀ (n_ch × G)
    → per-column top-10 via argsort → DCG with shared discount vector.
    """
    if toneless_scores is not None and include_sparse:
        grid = np.asarray(WEIGHT_GRID_4WAY, dtype=np.float32)
        channels = [dense_scores, bm25_scores, sparse_scores, toneless_scores]
    else:
        grid = np.asarray(
            WEIGHT_GRID_3WAY if include_sparse else WEIGHT_GRID_2WAY, dtype=np.float32
        )
        channels = [dense_scores, bm25_scores, sparse_scores if include_sparse else {}]

    norms = [_minmax(ch) for ch in channels]
    all_ids = list(set().union(*(n.keys() for n in norms)))
    if not all_ids:
        return [0.0] * len(grid)

    S = np.zeros((len(all_ids), len(channels)), dtype=np.float32)
    for j, n in enumerate(norms):
        for i, pid in enumerate(all_ids):
            S[i, j] = n.get(pid, 0.0)

    fused = S @ grid.T                       # (n_ids, G)
    top = min(10, len(all_ids))
    order = np.argsort(-fused, axis=0, kind="stable")[:top]  # (top, G)

    rel = np.fromiter((pid in relevant_ids for pid in all_ids), dtype=bool, count=len(all_ids))
    discounts = 1.0 / np.log2(np.arange(top) + 2)
    dcg = (rel[order] * discounts[:, None]).sum(axis=0)   # (G,)

    idcg_n = min(len(relevant_ids), 10)
    idcg = float((1.0 / np.log2(np.arange(idcg_n) + 2)).sum()) if idcg_n else 0.0
    if idcg == 0.0:
        return [0.0] * len(grid)
    return (dcg / idcg).tolist()


def collect_training_pairs(
    qas: list[dict],
    dense: DenseRetriever,
    bm25: BM25Retriever,
    sparse,                         # SparseRetriever | None
    query_embeddings: np.ndarray,
    include_signals: bool = True,
    normalize_labels: bool = True,
    toneless: BM25Retriever | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (X: features, Y: simplex grid NDCG scores).

    include_signals=True appends 18 post-retrieval signal features (QPP-style
    channel confidence + cross-channel agreement) to the 8 linguistic features,
    reusing the per-channel hits already computed for grid labeling — X becomes
    (n, 26). Pass include_signals=False for the legacy 8-dim feature set.

    normalize_labels=False keeps RAW NDCG@10 targets. Per-query min-max
    normalization stretches even a 0.001 raw NDCG spread to the full [0, 1]
    range, training the model to route confidently on queries where weights
    barely matter; raw targets keep those surfaces flat, so expected-weight
    inference gracefully falls back to ~equal weights there.

    Phase ordering to avoid OMP deadlock on macOS:
      1 — BM25 + features (underthesea, pure Python)
      2 — Sparse search (BGE-M3 / PyTorch — initialised BEFORE FAISS ops)
      3 — FAISS dense batch search
      4 — 3D simplex grid NDCG calculation (pure numpy)
    """
    # Phase 1: BM25 (+ toneless BM25) + features
    print("  Phase 1/4: BM25 search + feature extraction…")
    bm25_vocab = bm25.vocab
    bm25_results:     list[list[tuple[str, str, float]]] = []
    toneless_results: list[list[tuple[str, str, float]]] = []
    features_list:    list[np.ndarray] = []
    for qa in tqdm(qas, desc="  BM25+feat"):
        bm25_results.append(bm25.search(qa["question"], TOP_K))
        if toneless is not None:
            toneless_results.append(toneless.search(qa["question"], TOP_K))
        features_list.append(extract_features(qa["question"], bm25_vocab=bm25_vocab))

    # Phase 2: Sparse search (PyTorch — must come BEFORE faiss.normalize_L2)
    sparse_results: list[dict[str, float]] = []
    if sparse is not None:
        print("  Phase 2/4: Sparse search (BGE-M3, batched on GPU when available)…")
        all_queries = [qa["question"] for qa in qas]
        batched = sparse.search_batch(all_queries, TOP_K)
        for hits in batched:
            sparse_results.append({pid: s for pid, _, s in hits})
    else:
        print("  Phase 2/4: Sparse search SKIPPED (no sparse retriever — 2-way training)")
        sparse_results = [{} for _ in qas]

    # Phase 3: FAISS dense batch search
    print("  Phase 3/4: FAISS batch search…")
    embs = query_embeddings.copy().astype(np.float32)
    faiss.normalize_L2(embs)
    scores_arr, indices_arr = dense._index.search(embs, TOP_K)

    # Phase 4: grid NDCG calculation
    include_sparse = sparse is not None
    include_toneless = toneless is not None and include_sparse
    if include_toneless:
        grid_dim, n_grid = "3-simplex (4-way + toneless)", len(WEIGHT_GRID_4WAY)
    elif include_sparse:
        grid_dim, n_grid = "2-simplex (3-way)", len(WEIGHT_GRID_3WAY)
    else:
        grid_dim, n_grid = "1-simplex edge (c=0, sparse disabled)", len(WEIGHT_GRID_2WAY)
    print(f"  Phase 4/4: Simplex NDCG grid search — {grid_dim} ({n_grid} grid pts)…")
    X_list, Y_list = [], []
    for i, qa in enumerate(tqdm(qas, desc="  NDCG-grid")):
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
        toneless_scores = (
            {pid: s for pid, _, s in toneless_results[i]} if include_toneless else None
        )

        ndcgs = find_ndcg_grid(
            dense_scores, bm25_scores, sparse_scores, relevant_ids,
            include_sparse=include_sparse,
            toneless_scores=toneless_scores,
        )

        ndcgs = np.array(ndcgs, dtype=np.float32)
        if normalize_labels:
            min_v, max_v = np.min(ndcgs), np.max(ndcgs)
            if max_v > min_v:
                ndcgs = (ndcgs - min_v) / (max_v - min_v)
            else:
                ndcgs = np.zeros_like(ndcgs)

        feats = features_list[i]
        if include_signals:
            signals = extract_retrieval_signals(
                dense_scores, bm25_scores, sparse_scores,
                toneless_scores=toneless_scores,
            )
            feats = np.concatenate([feats, signals])

        X_list.append(feats)
        Y_list.append(ndcgs)

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
    keep_zero_rows: bool = False,
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
        if keep_zero_rows:
            cmd += ["--keep-zero-rows"]
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
    parser.add_argument("--xy-cache",   default=None,
                        help="Cache path for training pairs X/Y (avoids re-running search and grid NDCG calculations)")
    parser.add_argument("--sparse-path", default=None,
                        help="Path to sparse.pkl (default: <index-dir>/sparse.pkl)")
    parser.add_argument("--no-sparse",  action="store_true",
                        help="Disable sparse signal (train 3-way with c=0 — uses 2-way effectively)")
    parser.add_argument("--no-toneless", action="store_true",
                        help="Disable the toneless BM25 channel even when "
                             "<index-dir>/bm25_toneless.pkl exists (3-way training).")
    parser.add_argument("--init-from",  default=None)
    parser.add_argument("--max-samples", type=int, default=5000)
    parser.add_argument("--epochs",     type=int,   default=100)
    parser.add_argument("--batch-size", type=int,   default=256)
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--hard-label",  action="store_true",
                        help="Use argmax over the simplex grid instead of "
                             "temperature-scaled soft expectation (§5.6 ablation).")
    parser.add_argument("--no-signals",  action="store_true",
                        help="Train on the 8 linguistic features only, without the "
                             "18 post-retrieval signal features (ablation / legacy).")
    parser.add_argument("--raw-labels",  action="store_true",
                        help="Keep raw NDCG@10 grid targets instead of per-query min-max "
                             "normalization, and keep all-zero rows during training — flat "
                             "surfaces then route to ~equal weights under expected-mode "
                             "inference instead of being amplified into confident routing.")
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

    X, Y = None, None
    if args.xy_cache and Path(args.xy_cache).exists():
        print(f"Loading training pairs from cache: {args.xy_cache}")
        xy_data = np.load(args.xy_cache)
        X, Y = xy_data["X"], xy_data["Y"]
    else:
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

        toneless = None
        if not args.no_toneless:
            toneless_path = Path(args.index_dir) / "bm25_toneless.pkl"
            if toneless_path.exists():
                toneless = BM25Retriever.load(toneless_path)
                print(f"Toneless BM25 index loaded from {toneless_path} — 4-way training")
            else:
                print(f"NOTE: {toneless_path} not found — training 3-way. "
                      "Run build_index.py to add the toneless channel.")

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

        # Collect training pairs
        print("Collecting 3-way training pairs with Simplex NDCG grid...")
        X, Y = collect_training_pairs(
            qas, dense, bm25, sparse, query_embeddings,
            include_signals=not args.no_signals,
            normalize_labels=not args.raw_labels,
            toneless=toneless,
        )
        
        if args.xy_cache:
            Path(args.xy_cache).parent.mkdir(parents=True, exist_ok=True)
            np.savez(args.xy_cache, X=X, Y=Y)
            print(f"  Training pairs saved to cache -> {args.xy_cache}")

    print(f"Pairs: {len(X)}  |  dim: {X.shape[1]}  |  output: {Y.shape[1]}", flush=True)

    # Spawn training subprocess (fresh process — no FAISS OMP conflict with PyTorch)
    print("\nSpawning training subprocess…", flush=True)
    spawn_train_mlp(
        X, Y, args.output, args.epochs, args.batch_size, args.lr, args.seed,
        args.init_from, keep_zero_rows=args.raw_labels,
    )


if __name__ == "__main__":
    main()
