"""Build FAISS dense index, BM25 index, and BGE-M3 sparse index from a processed dataset.

Usage:
    uv run python scripts/build_index.py \\
        --data-path data/processed/viaquad_passages.jsonl \\
        --index-dir indexes/viaquad

    uv run python scripts/build_index.py \\
        --data-path data/processed/dangdocao_passages.jsonl \\
        --index-dir indexes/dangdocao

    # Skip sparse index (no BGE-M3 download required):
    uv run python scripts/build_index.py \\
        --data-path data/processed/viaquad_passages.jsonl \\
        --index-dir indexes/viaquad \\
        --no-sparse
"""

# Set before any lib imports to avoid OMP conflicts between FAISS and PyTorch on macOS.
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
import json
from pathlib import Path

import numpy as np
from tqdm import tqdm

from rag_vie.config import settings
from rag_vie.retrieval.bm25 import BM25Retriever
from rag_vie.retrieval.dense import DenseRetriever
from rag_vie.retrieval.embedder import embed_texts


def load_passages(data_path: Path) -> tuple[list[str], list[str]]:
    passages, ids = [], []
    with open(data_path, encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            passages.append(record["passage"])
            ids.append(record["id"])
    return passages, ids


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path",  required=True, help="Path to passages JSONL file")
    parser.add_argument("--index-dir",  required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--no-sparse",  action="store_true",
                        help="Skip BGE-M3 sparse index (no model download required)")
    args = parser.parse_args()

    data_path = Path(args.data_path)
    index_dir = Path(args.index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading passages from {data_path} ...")
    passages, ids = load_passages(data_path)
    print(f"  {len(passages):,} passages loaded.")

    # ── 1. Sparse index (BGE-M3 / PyTorch) — built BEFORE FAISS to avoid OMP conflict
    if not args.no_sparse:
        sparse_path = index_dir / "sparse.pkl"
        if sparse_path.exists():
            print(f"Sparse index already exists at {sparse_path}, skipping.")
        else:
            print("Building sparse index (BGE-M3 — first run downloads ~570 MB) ...")
            from rag_vie.retrieval.sparse import SparseRetriever
            sparse = SparseRetriever()
            sparse.build(passages, ids, batch_size=args.batch_size)
            sparse.save(sparse_path)
            print(f"  Sparse index → {sparse_path}")

    # ── 2. Dense embeddings via FPT API
    print("Embedding passages via FPT API ...")
    all_embeddings = []
    for i in tqdm(range(0, len(passages), args.batch_size), desc="  Embedding"):
        batch = passages[i : i + args.batch_size]
        all_embeddings.append(embed_texts(batch))
    embeddings = np.vstack(all_embeddings)

    # ── 3. FAISS index
    print("Building dense index ...")
    dense = DenseRetriever()
    dense.add(embeddings, passages, ids)
    dense.save(index_dir)
    print(f"  Dense index → {index_dir}")

    # ── 4. BM25 index
    print("Building BM25 index ...")
    bm25 = BM25Retriever()
    bm25.build(passages, ids)
    bm25.save(index_dir / "bm25.pkl")
    print(f"  BM25 index → {index_dir / 'bm25.pkl'}")

    print("Done.")


if __name__ == "__main__":
    main()
