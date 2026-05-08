import numpy as np

from .bm25 import BM25Retriever
from .dense import DenseRetriever
from .embedder import embed_query


def _min_max_normalize(scores: dict[str, float]) -> dict[str, float]:
    if not scores:
        return scores
    min_s, max_s = min(scores.values()), max(scores.values())
    span = max_s - min_s
    if span == 0:
        return {k: 0.0 for k in scores}
    return {k: (v - min_s) / span for k, v in scores.items()}


class HybridRetriever:
    def __init__(self, dense: DenseRetriever, bm25: BM25Retriever) -> None:
        self.dense = dense
        self.bm25 = bm25

    def retrieve(
        self,
        query: str,
        weights: tuple[float, float],
        k_dense: int,
        k_bm25: int,
        k_final: int,
    ) -> list[tuple[str, str, float]]:
        """
        Two-way fusion: score = a·s_dense + b·s_bm25.
        Returns list of (id, passage, fused_score), sorted descending.
        """
        a, b = weights

        query_emb = embed_query(query)
        dense_hits = self.dense.search(query_emb, k_dense)
        bm25_hits = self.bm25.search(query, k_bm25)

        dense_norm = _min_max_normalize({pid: s for pid, _, s in dense_hits})
        bm25_norm = _min_max_normalize({pid: s for pid, _, s in bm25_hits})

        all_ids = set(dense_norm) | set(bm25_norm)
        passage_map = {pid: psg for pid, psg, _ in dense_hits + bm25_hits}

        fused = {
            pid: a * dense_norm.get(pid, 0.0) + b * bm25_norm.get(pid, 0.0)
            for pid in all_ids
        }

        top = sorted(fused.items(), key=lambda x: x[1], reverse=True)[:k_final]
        return [(pid, passage_map[pid], score) for pid, score in top]
