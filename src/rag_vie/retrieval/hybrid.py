from dataclasses import dataclass, field

from ..utils.fusion import fuse_scores
from ..utils.metrics import min_max_normalize
from .bm25 import BM25Retriever
from .dense import DenseRetriever
from .embedder import embed_query
from .sparse import SparseRetriever


@dataclass
class Candidates:
    """Per-query candidate sets after min-max normalization, before weighted fusion."""

    dense_norm: dict[str, float]
    bm25_norm: dict[str, float]
    sparse_norm: dict[str, float] = field(default_factory=dict)
    passage_map: dict[str, str] = field(default_factory=dict)


class HybridRetriever:
    def __init__(
        self,
        dense: DenseRetriever,
        bm25: BM25Retriever,
        sparse: SparseRetriever | None = None,
    ) -> None:
        self.dense = dense
        self.bm25 = bm25
        self.sparse = sparse

    def retrieve_candidates(
        self,
        query: str,
        k_dense: int,
        k_bm25: int,
        k_sparse: int = 100,
    ) -> Candidates:
        """Retrieve and min-max normalize each source separately.

        The sparse source is always retrieved when available so that signal-aware
        features (and fusion) see a consistent candidate set regardless of weights.
        """
        query_emb = embed_query(query)
        dense_hits = self.dense.search(query_emb, k_dense)
        bm25_hits = self.bm25.search(query, k_bm25)

        dense_norm = min_max_normalize({pid: s for pid, _, s in dense_hits})
        bm25_norm = min_max_normalize({pid: s for pid, _, s in bm25_hits})
        passage_map = {pid: psg for pid, psg, _ in dense_hits + bm25_hits}

        sparse_norm: dict[str, float] = {}
        if self.sparse is not None:
            sparse_hits = self.sparse.search(query, k_sparse)
            sparse_norm = min_max_normalize({pid: s for pid, _, s in sparse_hits})
            passage_map.update({pid: psg for pid, psg, _ in sparse_hits})

        return Candidates(dense_norm, bm25_norm, sparse_norm, passage_map)

    def fuse(
        self,
        candidates: Candidates,
        weights: tuple[float, ...],   # (a, b) or (a, b, c)
        k_final: int,
    ) -> list[tuple[str, str, float]]:
        """Weighted fusion of pre-retrieved candidates → top-k (id, passage, score)."""
        if len(weights) == 2:
            a, b, c = weights[0], weights[1], 0.0
        else:
            a, b, c = weights[0], weights[1], weights[2]

        fused = fuse_scores(
            candidates.dense_norm, candidates.bm25_norm, candidates.sparse_norm, (a, b, c)
        )
        top = sorted(fused.items(), key=lambda x: x[1], reverse=True)[:k_final]
        pm = candidates.passage_map
        return [(pid, pm[pid], score) for pid, score in top if pid in pm]

    def retrieve(
        self,
        query: str,
        weights: tuple[float, ...],
        k_dense: int,
        k_bm25: int,
        k_final: int,
        k_sparse: int = 100,
    ) -> list[tuple[str, str, float]]:
        """Convenience: retrieve candidates then fuse with the given weights."""
        candidates = self.retrieve_candidates(query, k_dense, k_bm25, k_sparse)
        return self.fuse(candidates, weights, k_final)
