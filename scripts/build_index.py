"""Build FAISS dense index and BM25 index from a processed dataset.

Usage:
    uv run python scripts/build_index.py --data-path data/processed/viaquad_passages.jsonl --index-dir indexes/viaquad
    uv run python scripts/build_index.py --data-path data/processed/dangdocao_passages.jsonl --index-dir indexes/dangdocao
"""

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
    parser.add_argument("--data-path", required=True, help="Path to passages JSONL file")
    parser.add_argument("--index-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    data_path = Path(args.data_path)
    index_dir = Path(args.index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading passages from {data_path} ...")
    passages, ids = load_passages(data_path)
    print(f"  {len(passages):,} passages loaded.")

    # Dense index
    print("Building dense index ...")
    dense = DenseRetriever()
    all_embeddings = []
    for i in tqdm(range(0, len(passages), args.batch_size), desc="  Embedding"):
        batch = passages[i : i + args.batch_size]
        all_embeddings.append(embed_texts(batch))
    embeddings = np.vstack(all_embeddings)
    dense.add(embeddings, passages, ids)
    dense.save(index_dir)
    print(f"  Dense index → {index_dir}")

    # BM25 index
    print("Building BM25 index ...")
    bm25 = BM25Retriever()
    bm25.build(passages, ids)
    bm25.save(index_dir / "bm25.pkl")
    print(f"  BM25 index → {index_dir / 'bm25.pkl'}")

    print("Done.")


if __name__ == "__main__":
    main()
