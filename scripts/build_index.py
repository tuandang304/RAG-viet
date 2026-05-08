"""Build FAISS dense index and BM25 index from a processed dataset.

Usage:
    python scripts/build_index.py --dataset viaquad --split train
"""

import argparse
import json
from pathlib import Path

from tqdm import tqdm

from rag_vie.config import settings
from rag_vie.retrieval.bm25 import BM25Retriever
from rag_vie.retrieval.dense import DenseRetriever
from rag_vie.retrieval.sparse import SparseRetriever
from rag_vie.retrieval.embedder import embed_texts


def load_passages(data_path: Path) -> tuple[list[str], list[str]]:
    """Load passages from a JSONL file with {id, passage} records."""
    passages, ids = [], []
    with open(data_path, encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            passages.append(record["passage"])
            ids.append(record["id"])
    return passages, ids


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", required=True, help="Path to passages JSONL file")
    parser.add_argument("--index-dir", default=settings.index_dir)
    parser.add_argument("--bm25-path", default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    data_path = Path(args.data_path)
    index_dir = Path(args.index_dir)
    bm25_path = Path(args.bm25_path) if args.bm25_path else index_dir / "bm25.pkl"

    print(f"Loading passages from {data_path} ...")
    passages, ids = load_passages(data_path)
    print(f"  {len(passages)} passages loaded.")

    # Dense index
    print("Building dense index ...")
    dense = DenseRetriever()
    all_embeddings = []
    for i in tqdm(range(0, len(passages), args.batch_size), desc="Embedding"):
        batch = passages[i : i + args.batch_size]
        all_embeddings.append(embed_texts(batch))

    import numpy as np
    embeddings = np.vstack(all_embeddings)
    dense.add(embeddings, passages, ids)
    dense.save(index_dir)
    print(f"  Dense index saved to {index_dir}")

    # BM25 index
    print("Building BM25 index ...")
    bm25 = BM25Retriever()
    bm25.build(passages, ids)
    bm25.save(bm25_path)
    print(f"  BM25 index saved to {bm25_path}")

    # Sparse index (BGE-M3 lexical weights)
    sparse_path = index_dir / "sparse.pkl"
    print("Building sparse index (BGE-M3 lexical weights) ...")
    print("  Note: BAAI/bge-m3 (~570MB) sẽ được download lần đầu nếu chưa có cache.")
    sparse = SparseRetriever()
    sparse.build(passages, ids, batch_size=args.batch_size)
    sparse.save(sparse_path)
    print(f"  Sparse index saved to {sparse_path}")

    print("Done.")


if __name__ == "__main__":
    main()
