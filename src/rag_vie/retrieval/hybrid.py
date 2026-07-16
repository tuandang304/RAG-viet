from ..utils.metrics import min_max_normalize as _min_max_normalize
from .bm25 import BM25Retriever
from .dense import DenseRetriever
from .embedder import embed_query
from .sparse import SparseRetriever

# (id, passage, score) triple returned by every retriever's search()
Hit = tuple[str, str, float]

# Fusion channel order: weights are (w_dense, w_bm25, w_sparse, w_toneless)
CHANNELS = ("dense", "bm25", "sparse", "toneless")


class HybridRetriever:
    def __init__(
        self,
        dense: DenseRetriever,
        bm25: BM25Retriever,
        sparse: SparseRetriever | None = None,
        toneless: BM25Retriever | None = None,
    ) -> None:
        self.dense = dense
        self.bm25 = bm25
        self.sparse = sparse
        self.toneless = toneless

    def search_all(
        self,
        query: str,
        k_dense: int,
        k_bm25: int,
        k_sparse: int = 100,
        k_toneless: int = 100,
    ) -> dict[str, list[Hit]]:
        """Run all available channels once and return their raw hit lists.

        Always queries every channel that has a retriever — the hits are
        needed both for fusion and for post-retrieval signal features,
        regardless of what weight the router later assigns to each channel.
        """
        query_emb = embed_query(query)
        return {
            "dense": self.dense.search(query_emb, k_dense),
            "bm25": self.bm25.search(query, k_bm25),
            "sparse": self.sparse.search(query, k_sparse) if self.sparse is not None else [],
            "toneless": self.toneless.search(query, k_toneless) if self.toneless is not None else [],
        }

    @staticmethod
    def fuse(
        hits: dict[str, list[Hit]],
        weights: tuple[float, ...],   # 2, 3 or 4 weights, CHANNELS order
        k_final: int,
    ) -> list[Hit]:
        """Weighted fusion of already-retrieved hits.

        score = Σ w_ch · s_ch, each channel min-max normalized first
        (BM25/sparse scores have no upper bound). Missing trailing weights
        are treated as 0. Returns (id, passage, fused_score), descending.
        """
        w = tuple(weights) + (0.0,) * (len(CHANNELS) - len(weights))

        norms = {
            ch: _min_max_normalize({pid: s for pid, _, s in hits.get(ch, [])})
            for ch in CHANNELS
        }
        passage_map = {
            pid: psg
            for ch in CHANNELS
            for pid, psg, _ in hits.get(ch, [])
        }

        all_ids = set().union(*(norms[ch].keys() for ch in CHANNELS))
        fused = {
            pid: sum(w[i] * norms[ch].get(pid, 0.0) for i, ch in enumerate(CHANNELS))
            for pid in all_ids
        }

        top = sorted(fused.items(), key=lambda x: x[1], reverse=True)[:k_final]
        return [(pid, passage_map[pid], score) for pid, score in top if pid in passage_map]

    def retrieve(
        self,
        query: str,
        weights: tuple[float, ...],
        k_dense: int,
        k_bm25: int,
        k_final: int,
        k_sparse: int = 100,
        k_toneless: int = 100,
    ) -> list[Hit]:
        """search_all + fuse in one call (kept for backward compatibility)."""
        hits = self.search_all(query, k_dense, k_bm25, k_sparse, k_toneless)
        return self.fuse(hits, weights, k_final)
