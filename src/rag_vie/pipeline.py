from dataclasses import dataclass
from pathlib import Path

from .config import settings
from .features.vietnamese import extract_features
from .fusion.mlp import FusionMLP
from .generator.llm import generate
from .retrieval.bm25 import BM25Retriever
from .retrieval.dense import DenseRetriever
from .retrieval.hybrid import HybridRetriever
from .retrieval.sparse import SparseRetriever


@dataclass
class RAGResult:
    query: str
    weights: tuple[float, ...]            # (w_dense, w_bm25, w_sparse)
    retrieved: list[tuple[str, str, float]]   # (id, passage, fused_score)
    answer: str


class RAGPipeline:
    """End-to-end Dynamic Hybrid RAG — three-way fusion (dense + BM25 + sparse)."""

    def __init__(
        self,
        dense: DenseRetriever,
        bm25: BM25Retriever,
        fusion_mlp: FusionMLP,
        sparse: SparseRetriever | None = None,
        use_generator: bool = True,
    ) -> None:
        self._hybrid = HybridRetriever(dense, bm25, sparse)
        self._mlp = fusion_mlp
        self._use_generator = use_generator

    def run(self, query: str) -> RAGResult:
        features = extract_features(query)
        weights = self._mlp.predict_weights(features)   # (a, b, c)

        retrieved = self._hybrid.retrieve(
            query=query,
            weights=weights,
            k_dense=settings.top_k_dense,
            k_bm25=settings.top_k_bm25,
            k_final=settings.top_k_final,
        )

        answer = ""
        if self._use_generator:
            passages = [p for _, p, _ in retrieved]
            answer = generate(query, passages)

        return RAGResult(query=query, weights=weights, retrieved=retrieved, answer=answer)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Dynamic Hybrid RAG for Vietnamese")
    parser.add_argument("--index-dir", default=settings.index_dir)
    parser.add_argument("--bm25-path", default=None)
    parser.add_argument("--sparse-path", default=None)
    parser.add_argument("--mlp-path", default=f"{settings.checkpoint_dir}/fusion_mlp.pt")
    parser.add_argument("--query", required=True)
    parser.add_argument("--no-generate", action="store_true")
    args = parser.parse_args()

    bm25_path   = args.bm25_path   or f"{args.index_dir}/bm25.pkl"
    sparse_path = args.sparse_path or f"{args.index_dir}/sparse.pkl"

    dense  = DenseRetriever.load(args.index_dir)
    bm25   = BM25Retriever.load(bm25_path)
    mlp    = FusionMLP.load(args.mlp_path)

    sparse = None
    if Path(sparse_path).exists():
        sparse = SparseRetriever.load(sparse_path)

    pipeline = RAGPipeline(dense, bm25, mlp, sparse, use_generator=not args.no_generate)
    result = pipeline.run(args.query)

    n_weights = len(result.weights)
    print(f"\nQuery: {result.query}")
    if n_weights == 3:
        print(
            f"Weights — dense: {result.weights[0]:.3f} | "
            f"bm25: {result.weights[1]:.3f} | "
            f"sparse: {result.weights[2]:.3f}"
        )
    else:
        print(f"Weights — dense: {result.weights[0]:.3f} | bm25: {result.weights[1]:.3f}")

    print("\nTop passages:")
    for i, (pid, passage, score) in enumerate(result.retrieved[:3], 1):
        print(f"  [{i}] (id={pid}, score={score:.4f}) {passage[:120]}...")
    if result.answer:
        print(f"\nAnswer: {result.answer}")
