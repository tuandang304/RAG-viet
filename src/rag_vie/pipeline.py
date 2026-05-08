from dataclasses import dataclass

from .config import settings
from .features.vietnamese import extract_features
from .fusion.mlp import FusionMLP
from .generator.llm import generate
from .retrieval.bm25 import BM25Retriever
from .retrieval.dense import DenseRetriever
from .retrieval.hybrid import HybridRetriever


@dataclass
class RAGResult:
    query: str
    weights: tuple[float, float]        # (w_dense, w_bm25)
    retrieved: list[tuple[str, str, float]]  # (id, passage, fused_score)
    answer: str


class RAGPipeline:
    """End-to-end Dynamic Hybrid RAG — two-way fusion (dense + BM25)."""

    def __init__(
        self,
        dense: DenseRetriever,
        bm25: BM25Retriever,
        fusion_mlp: FusionMLP,
        use_generator: bool = True,
    ) -> None:
        self._hybrid = HybridRetriever(dense, bm25)
        self._mlp = fusion_mlp
        self._use_generator = use_generator

    def run(self, query: str) -> RAGResult:
        features = extract_features(query)
        weights = self._mlp.predict_weights(features)  # (a, b)

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
    parser.add_argument("--mlp-path", default=f"{settings.checkpoint_dir}/fusion_mlp.pt")
    parser.add_argument("--query", required=True)
    parser.add_argument("--no-generate", action="store_true")
    args = parser.parse_args()

    bm25_path = args.bm25_path or f"{args.index_dir}/bm25.pkl"

    dense = DenseRetriever.load(args.index_dir)
    bm25 = BM25Retriever.load(bm25_path)
    mlp = FusionMLP.load(args.mlp_path)

    pipeline = RAGPipeline(dense, bm25, mlp, use_generator=not args.no_generate)
    result = pipeline.run(args.query)

    print(f"\nQuery: {result.query}")
    print(f"Weights — dense: {result.weights[0]:.3f} | bm25: {result.weights[1]:.3f}")
    print("\nTop passages:")
    for i, (pid, passage, score) in enumerate(result.retrieved[:3], 1):
        print(f"  [{i}] (id={pid}, score={score:.4f}) {passage[:120]}...")
    if result.answer:
        print(f"\nAnswer: {result.answer}")
