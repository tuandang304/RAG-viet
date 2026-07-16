from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import settings
from .features.neural import NeuralFeatureExtractor
from .features.retrieval_signals import (
    SIGNAL_NAMES,
    SIGNAL_NAMES_4WAY,
    extract_retrieval_signals,
)
from .features.vietnamese import FEATURE_NAMES, extract_features
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
    """End-to-end Dynamic Hybrid RAG — dynamic fusion over dense + BM25 +
    sparse (+ optional toneless BM25 channel for tone-robust retrieval)."""

    def __init__(
        self,
        dense: DenseRetriever,
        bm25: BM25Retriever,
        fusion_mlp: FusionMLP,
        sparse: SparseRetriever | None = None,
        use_generator: bool = True,
        neural_extractor: NeuralFeatureExtractor | None = None,
        toneless: BM25Retriever | None = None,
    ) -> None:
        self._hybrid = HybridRetriever(dense, bm25, sparse, toneless)
        self._mlp = fusion_mlp
        self._use_generator = use_generator
        self._neural_extractor = neural_extractor
        self._bm25_vocab: set[str] | None = bm25.vocab

    def run(self, query: str) -> RAGResult:
        # Retrieve all channels first — fusion needs the hits anyway, and
        # checkpoints trained with post-retrieval signals need them as input.
        hits = self._hybrid.search_all(
            query,
            k_dense=settings.top_k_dense,
            k_bm25=settings.top_k_bm25,
        )

        features = extract_features(
            query,
            bm25_vocab=self._bm25_vocab,
            neural_extractor=self._neural_extractor,
        )
        n_base = len(FEATURE_NAMES)
        if self._mlp.input_dim in (n_base + len(SIGNAL_NAMES), n_base + len(SIGNAL_NAMES_4WAY)):
            four_way = self._mlp.input_dim == n_base + len(SIGNAL_NAMES_4WAY)
            signals = extract_retrieval_signals(
                {pid: s for pid, _, s in hits["dense"]},
                {pid: s for pid, _, s in hits["bm25"]},
                {pid: s for pid, _, s in hits["sparse"]},
                toneless_scores=(
                    {pid: s for pid, _, s in hits["toneless"]} if four_way else None
                ),
            )
            features = np.concatenate([features, signals])

        weights = self._mlp.predict_weights(features)   # (a, b, c[, d])
        retrieved = self._hybrid.fuse(hits, weights, k_final=settings.top_k_final)

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
    parser.add_argument("--mlp-path", default=f"{settings.checkpoint_dir}/fusion_mlp_multidomain.keras")
    parser.add_argument("--query", required=True)
    parser.add_argument("--no-generate", action="store_true")
    parser.add_argument("--neural-extractor-path", default=None,
                        help="Path to trained NeuralFeatureExtractor projection weights (.pt). "
                             "When provided, replaces heuristic feature extraction.")
    args = parser.parse_args()

    bm25_path     = args.bm25_path   or f"{args.index_dir}/bm25.pkl"
    sparse_path   = args.sparse_path or f"{args.index_dir}/sparse.pkl"
    toneless_path = f"{args.index_dir}/bm25_toneless.pkl"

    dense  = DenseRetriever.load(args.index_dir)
    bm25   = BM25Retriever.load(bm25_path)
    mlp    = FusionMLP.load(args.mlp_path)

    sparse = None
    if Path(sparse_path).exists():
        sparse = SparseRetriever.load(sparse_path)

    toneless = None
    if Path(toneless_path).exists():
        toneless = BM25Retriever.load(toneless_path)

    neural_extractor = None
    if args.neural_extractor_path and Path(args.neural_extractor_path).exists():
        neural_extractor = NeuralFeatureExtractor.load(args.neural_extractor_path)

    pipeline = RAGPipeline(
        dense, bm25, mlp, sparse,
        use_generator=not args.no_generate,
        neural_extractor=neural_extractor,
        toneless=toneless,
    )
    result = pipeline.run(args.query)

    print(f"\nQuery: {result.query}")
    labels = ("dense", "bm25", "sparse", "toneless")
    print("Weights — " + " | ".join(
        f"{name}: {w:.3f}" for name, w in zip(labels, result.weights, strict=False)
    ))

    print("\nTop passages:")
    for i, (pid, passage, score) in enumerate(result.retrieved[:3], 1):
        print(f"  [{i}] (id={pid}, score={score:.4f}) {passage[:120]}...")
    if result.answer:
        print(f"\nAnswer: {result.answer}")
