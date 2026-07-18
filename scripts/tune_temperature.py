"""Tune the router's expected-weight softmax temperature — fully offline sweep.

Bottleneck-aware: retrieval (BGE-M3 + FAISS + BM25×2) runs ONCE per query; the
per-query predicted-NDCG surface and normalized channel scores are cached, then
every temperature is scored in pure numpy. This makes a fine T sweep cheap.

For each query it stores:
  * the MLP's 286-dim predicted-NDCG surface (needs features + signals),
  * an (n_docs × 4) matrix of min-max normalized channel scores,
  * the relevance mask over those docs.
Then, per temperature T: weights = softmax(surface / T) @ grid → fuse → NDCG@10.

Tune on a clean+noisy dev mix so the chosen T serves both regimes.

Usage:
    uv run python scripts/tune_temperature.py \\
        --index-dir indexes/viaquad \\
        --qas-path data/processed/viaquad_devmix1000.jsonl \\
        --mlp-path checkpoints/fusion_mlp_4way_aug.keras \\
        --emb-cache checkpoints/tune_viaquad_dev_emb.npy \\
        --output results/temp_sweep_viaquad.json
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import pyarrow  # noqa: F401
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

from rag_vie.features.retrieval_signals import (
    SIGNAL_NAMES,
    SIGNAL_NAMES_4WAY,
    extract_retrieval_signals,
)
from rag_vie.features.vietnamese import FEATURE_NAMES, extract_features
from rag_vie.retrieval.bm25 import BM25Retriever
from rag_vie.utils.metrics import min_max_normalize

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

TOP_K = 200
CHANNELS = ("dense", "bm25", "sparse", "toneless")

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

TEMPERATURES = [0.01, 0.02, 0.03, 0.05, 0.08, 0.12, 0.2, 0.35, 0.6, 1.0]


def _expected_weights(surface: np.ndarray, grid: np.ndarray, T: float) -> np.ndarray:
    logits = surface / max(T, 1e-8)
    logits = logits - logits.max()
    p = np.exp(logits)
    p /= p.sum()
    return p @ grid


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index-dir", required=True)
    parser.add_argument("--qas-path", required=True)
    parser.add_argument("--mlp-path", required=True)
    parser.add_argument("--emb-cache", default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    with open(args.qas_path, encoding="utf-8") as f:
        qas = [json.loads(line) for line in f if line.strip()]
    qas = [q for q in qas if q.get("relevant_ids")]
    if args.max_samples and args.max_samples < len(qas):
        import random
        qas = random.Random(args.seed).sample(qas, args.max_samples)
    queries = [q["question"] for q in qas]
    print(f"Tuning on {len(qas)} queries from {args.qas_path}")

    # BGE-M3 (PyTorch) before FAISS — OMP ordering (see CLAUDE.md).
    sparse_path = Path(args.index_dir) / "sparse.pkl"
    from rag_vie.retrieval.sparse import SparseRetriever, _encode_sparse
    sparse = SparseRetriever.load(sparse_path)
    _ = _encode_sparse(["warmup"])

    import faiss

    from rag_vie.fusion.mlp import FusionMLP
    from rag_vie.retrieval.dense import DenseRetriever

    dense = DenseRetriever.load(args.index_dir)
    bm25 = BM25Retriever.load(Path(args.index_dir) / "bm25.pkl")
    toneless = BM25Retriever.load(Path(args.index_dir) / "bm25_toneless.pkl")
    bm25_vocab = bm25.vocab
    mlp = FusionMLP.load(args.mlp_path)

    n_base = len(FEATURE_NAMES)
    use_4way = mlp.input_dim == n_base + len(SIGNAL_NAMES_4WAY)
    use_signals = use_4way or mlp.input_dim == n_base + len(SIGNAL_NAMES)

    # Query embeddings (cached → no API).
    embs = None
    if args.emb_cache and Path(args.emb_cache).exists():
        embs = np.load(args.emb_cache)
        if len(embs) != len(queries):
            embs = None
    if embs is None:
        from rag_vie.retrieval.embedder import embed_texts
        print(f"Embedding {len(queries)} queries via FPT API…")
        embs = embed_texts(queries, batch_size=32)
        if args.emb_cache:
            np.save(args.emb_cache, embs)

    print("Sparse batch search…")
    sparse_hits = sparse.search_batch(queries, TOP_K)
    print("FAISS batch search…")
    embs32 = embs.copy().astype(np.float32)
    faiss.normalize_L2(embs32)
    d_scores_arr, d_idx_arr = dense._index.search(embs32, TOP_K)

    # Per-query cached tensors for the numpy sweep.
    surfaces: list[np.ndarray] = []
    score_mats: list[np.ndarray] = []
    rel_masks: list[np.ndarray] = []

    print("Retrieval + surface caching…")
    for i, qa in enumerate(tqdm(qas, desc="  cache")):
        d_raw = {dense._ids[idx]: float(d_scores_arr[i, j])
                 for j, idx in enumerate(d_idx_arr[i]) if idx != -1}
        b_raw = {pid: s for pid, _, s in bm25.search(qa["question"], TOP_K)}
        s_raw = {pid: s for pid, _, s in sparse_hits[i]}
        t_raw = {pid: s for pid, _, s in toneless.search(qa["question"], TOP_K)}

        norms = [min_max_normalize(x) for x in (d_raw, b_raw, s_raw, t_raw)]
        all_ids = list(set().union(*(n.keys() for n in norms)))
        if not all_ids:
            continue
        S = np.zeros((len(all_ids), 4), dtype=np.float32)
        for cj, nrm in enumerate(norms):
            for ri, pid in enumerate(all_ids):
                S[ri, cj] = nrm.get(pid, 0.0)

        feats = extract_features(qa["question"], bm25_vocab=bm25_vocab)
        if use_signals:
            sig = extract_retrieval_signals(
                d_raw, b_raw, s_raw, toneless_scores=t_raw if use_4way else None
            )
            feats = np.concatenate([feats, sig])
        surface = mlp(np.expand_dims(feats, 0).astype(np.float32), training=False).numpy().squeeze(0)

        relevant = set(qa["relevant_ids"])
        rel = np.fromiter((pid in relevant for pid in all_ids), dtype=bool, count=len(all_ids))

        surfaces.append(surface)
        score_mats.append(S)
        rel_masks.append(rel)

    print(f"Cached {len(surfaces)} queries. Sweeping {len(TEMPERATURES)} temperatures…")
    results = {}
    for T in TEMPERATURES:
        ndcgs = []
        for surface, S, rel in zip(surfaces, score_mats, rel_masks, strict=True):
            w = _expected_weights(surface, GRID_4WAY, T)
            fused = S @ w
            order = np.argsort(-fused, kind="stable")
            ranked_rel = rel[order]
            # NDCG@10 directly from the boolean relevance of the top-10 ranking
            top = ranked_rel[:10]
            dcg = float((top / np.log2(np.arange(2, 2 + len(top)))).sum())
            idcg_n = min(int(rel.sum()), 10)
            idcg = float((1.0 / np.log2(np.arange(2, 2 + idcg_n))).sum()) if idcg_n else 0.0
            ndcgs.append(dcg / idcg if idcg > 0 else 0.0)
        results[T] = round(float(np.mean(ndcgs)), 4)
        print(f"  T={T:<5}: NDCG@10 = {results[T]:.4f}")

    best_T = max(results, key=results.get)
    print(f"\nBest T = {best_T}  (NDCG@10 = {results[best_T]:.4f}); "
          f"current default T=0.05 → {results.get(0.05)}")

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump({"by_temperature": results, "best_T": best_T,
                       "qas_path": args.qas_path, "n": len(surfaces)}, f, indent=2)
        print(f"Saved → {args.output}")


if __name__ == "__main__":
    main()
