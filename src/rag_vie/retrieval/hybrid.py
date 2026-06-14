from ..utils.fusion import fuse_scores
from ..utils.metrics import min_max_normalize
from .bm25 import BM25Retriever
from .dense import DenseRetriever
from .embedder import embed_query
from .sparse import SparseRetriever


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

    def retrieve(
        self,
        query: str,
        weights: tuple[float, ...],   # (a, b) or (a, b, c)
        k_dense: int,
        k_bm25: int,
        k_final: int,
        k_sparse: int = 100,
    ) -> list[tuple[str, str, float]]:
        """Three-way fusion: score = a·s_dense + b·s_bm25 + c·s_sparse.

        Falls back to two-way (c=0) when sparse is None or len(weights)==2.
        Returns list of (id, passage, fused_score), sorted descending.
        """
        if len(weights) == 2:
            a, b, c = weights[0], weights[1], 0.0
        else:
            a, b, c = weights[0], weights[1], weights[2]

        query_emb = embed_query(query)
        dense_hits = self.dense.search(query_emb, k_dense)
        bm25_hits = self.bm25.search(query, k_bm25)

        dense_norm = min_max_normalize({pid: s for pid, _, s in dense_hits})
        bm25_norm = min_max_normalize({pid: s for pid, _, s in bm25_hits})

        passage_map = {pid: psg for pid, psg, _ in dense_hits + bm25_hits}

        sparse_norm: dict[str, float] = {}
        if c > 0 and self.sparse is not None:
            sparse_hits = self.sparse.search(query, k_sparse)
            sparse_norm = min_max_normalize({pid: s for pid, _, s in sparse_hits})
            passage_map.update({pid: psg for pid, psg, _ in sparse_hits})

        fused = fuse_scores(dense_norm, bm25_norm, sparse_norm, (a, b, c))

        top = sorted(fused.items(), key=lambda x: x[1], reverse=True)[:k_final]
        return [(pid, passage_map[pid], score) for pid, score in top if pid in passage_map]
