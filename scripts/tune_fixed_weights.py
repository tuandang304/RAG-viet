"""Tune the best FIXED fusion weight vector on a dev set (grid search).

Produces the "best fixed-weight (dev-tuned)" baseline: the single (a, b, c, d)
simplex point with the highest mean NDCG@10 across the dev queries. Tune on a
mix of clean + noisy dev files so the static baseline gets the same regime
exposure as the dynamic router — the strongest fair static competitor.

Usage:
    uv run python scripts/tune_fixed_weights.py \\
        --index-dir indexes/viaquad \\
        --qas-paths data/processed/viaquad_dev.jsonl data/processed/viaquad_dev_noisy.jsonl \\
        --n-per-file 500 \\
        --emb-cache checkpoints/tune_viaquad_dev_emb.npy \\
        --output results/tuned_fixed_viaquad.json

Then evaluate with:
    uv run python scripts/evaluate_all.py ... --fixed-extra "tuned_fixed=a,b,c,d"
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import pyarrow  # noqa: F401
import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

from rag_vie.retrieval.bm25 import BM25Retriever
from rag_vie.utils.metrics import min_max_normalize

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

TOP_K = 200

_N = 10
GRID_4WAY = np.array(
    [
        (i / _N, j / _N, k / _N, (_N - i - j - k) / _N)
        for i in range(_N + 1)
        for j in range(_N + 1 - i)
        for k in range(_N + 1 - i - j)
    ],
    dtype=np.float32,
)


def ndcg_grid_for_query(
    channel_scores: list[dict[str, float]],
    relevant_ids: set[str],
    grid: np.ndarray,
) -> np.ndarray:
    """Vectorized NDCG@10 for every grid point (same math as train_mlp labeling)."""
    norms = [min_max_normalize(ch) for ch in channel_scores]
    all_ids = list(set().union(*(n.keys() for n in norms)))
    if not all_ids:
        return np.zeros(len(grid), dtype=np.float32)

    S = np.zeros((len(all_ids), len(norms)), dtype=np.float32)
    for j, n in enumerate(norms):
        for i, pid in enumerate(all_ids):
            S[i, j] = n.get(pid, 0.0)

    fused = S @ grid.T
    top = min(10, len(all_ids))
    order = np.argsort(-fused, axis=0, kind="stable")[:top]
    rel = np.fromiter((pid in relevant_ids for pid in all_ids), dtype=bool, count=len(all_ids))
    discounts = 1.0 / np.log2(np.arange(top) + 2)
    dcg = (rel[order] * discounts[:, None]).sum(axis=0)
    idcg_n = min(len(relevant_ids), 10)
    idcg = float((1.0 / np.log2(np.arange(idcg_n) + 2)).sum()) if idcg_n else 0.0
    return (dcg / idcg).astype(np.float32) if idcg > 0 else np.zeros(len(grid), dtype=np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index-dir", required=True)
    parser.add_argument("--qas-paths", nargs="+", required=True,
                        help="Dev JSONL files (e.g. clean + noisy) — sampled and concatenated")
    parser.add_argument("--n-per-file", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--emb-cache", default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    qas: list[dict] = []
    for path in args.qas_paths:
        with open(path, encoding="utf-8") as f:
            rows = [json.loads(line) for line in f if line.strip()]
        rows = [q for q in rows if q.get("relevant_ids")]
        if args.n_per_file and args.n_per_file < len(rows):
            rows = random.Random(args.seed).sample(rows, args.n_per_file)
        print(f"{path}: {len(rows)} queries")
        qas.extend(rows)
    queries = [q["question"] for q in qas]

    # BGE-M3 (PyTorch) must initialise BEFORE faiss — see CLAUDE.md OMP note.
    sparse_path = Path(args.index_dir) / "sparse.pkl"
    if not sparse_path.exists():
        raise SystemExit(f"sparse.pkl required for 4-way tuning, not found in {args.index_dir}")
    from rag_vie.retrieval.sparse import SparseRetriever, _encode_sparse
    sparse = SparseRetriever.load(sparse_path)
    _ = _encode_sparse(["warmup"])

    from rag_vie.retrieval.dense import DenseRetriever
    import faiss

    dense = DenseRetriever.load(args.index_dir)
    bm25 = BM25Retriever.load(Path(args.index_dir) / "bm25.pkl")
    toneless_path = Path(args.index_dir) / "bm25_toneless.pkl"
    if not toneless_path.exists():
        raise SystemExit(f"bm25_toneless.pkl required for 4-way tuning, not found in {args.index_dir}")
    toneless = BM25Retriever.load(toneless_path)

    # Query embeddings (cached)
    embs = None
    if args.emb_cache and Path(args.emb_cache).exists():
        embs = np.load(args.emb_cache)
        if len(embs) != len(queries):
            embs = None
    if embs is None:
        from rag_vie.retrieval.embedder import embed_texts
        print(f"Embedding {len(queries)} dev queries via FPT API…")
        embs = embed_texts(queries, batch_size=32)
        if args.emb_cache:
            Path(args.emb_cache).parent.mkdir(parents=True, exist_ok=True)
            np.save(args.emb_cache, embs)

    print("Sparse batch search…")
    sparse_hits = sparse.search_batch(queries, TOP_K)

    print("FAISS batch search…")
    embs32 = embs.copy().astype(np.float32)
    faiss.normalize_L2(embs32)
    d_scores_arr, d_idx_arr = dense._index.search(embs32, TOP_K)

    print(f"Grid search over {len(GRID_4WAY)} points…")
    grid_sum = np.zeros(len(GRID_4WAY), dtype=np.float64)
    n_scored = 0
    for i, qa in enumerate(tqdm(qas, desc="NDCG grid")):
        relevant = set(qa["relevant_ids"])
        d_sc = {
            dense._ids[idx]: float(d_scores_arr[i, j])
            for j, idx in enumerate(d_idx_arr[i]) if idx != -1
        }
        b_sc = {pid: s for pid, _, s in bm25.search(qa["question"], TOP_K)}
        t_sc = {pid: s for pid, _, s in toneless.search(qa["question"], TOP_K)}
        s_sc = {pid: s for pid, _, s in sparse_hits[i]}
        grid_sum += ndcg_grid_for_query([d_sc, b_sc, s_sc, t_sc], relevant, GRID_4WAY)
        n_scored += 1

    mean_ndcg = grid_sum / max(n_scored, 1)
    order = np.argsort(-mean_ndcg)
    best = GRID_4WAY[order[0]]

    print(f"\nDev queries scored: {n_scored}")
    print("Top-5 fixed weight vectors (dense, bm25, sparse, toneless):")
    for rank, gi in enumerate(order[:5], 1):
        w = GRID_4WAY[gi]
        print(f"  {rank}. ({w[0]:.1f}, {w[1]:.1f}, {w[2]:.1f}, {w[3]:.1f})  NDCG@10={mean_ndcg[gi]:.4f}")

    result = {
        "best_weights": [float(v) for v in best],
        "best_dev_ndcg10": float(mean_ndcg[order[0]]),
        "top5": [
            {"weights": [float(v) for v in GRID_4WAY[gi]], "ndcg10": float(mean_ndcg[gi])}
            for gi in order[:5]
        ],
        "fixed_extra_arg": "tuned_fixed=" + ",".join(f"{v:.1f}" for v in best),
        "config": {
            "index_dir": args.index_dir,
            "qas_paths": args.qas_paths,
            "n_per_file": args.n_per_file,
            "seed": args.seed,
            "top_k": TOP_K,
        },
    }
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"\nSaved -> {args.output}")
    print(f"\nUse with: --fixed-extra \"{result['fixed_extra_arg']}\"")


if __name__ == "__main__":
    main()
