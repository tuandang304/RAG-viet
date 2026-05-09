"""BGE-M3 sparse (lexical) retriever — local model, inverted index.

Uses FlagEmbedding BGEM3FlagModel to compute SPLADE-style lexical weights.
Model is downloaded once (~570 MB) and cached by HuggingFace hub.
"""

import pickle
from pathlib import Path

_MODEL_NAME = "BAAI/bge-m3"
_model = None   # lazy singleton — loaded on first call


def _get_model():
    global _model
    if _model is None:
        from FlagEmbedding import BGEM3FlagModel
        _model = BGEM3FlagModel(_MODEL_NAME, use_fp16=True)
    return _model


def _encode_sparse(texts: list[str], batch_size: int = 32) -> list[dict[str, float]]:
    """Return list of lexical weight dicts: [{token_string: weight, ...}]."""
    model = _get_model()
    all_weights: list[dict[str, float]] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        out = model.encode(
            batch,
            return_dense=False,
            return_sparse=True,
            return_colbert_vecs=False,
        )
        all_weights.extend(out["lexical_weights"])
    return all_weights


class SparseRetriever:
    """BGE-M3 lexical sparse retriever — inverted index over token-level weights."""

    def __init__(self) -> None:
        self._inverted_index: dict[str, list[tuple[str, float]]] = {}
        self._passage_map: dict[str, str] = {}

    def build(self, passages: list[str], ids: list[str], batch_size: int = 32) -> None:
        from tqdm import tqdm

        self._inverted_index = {}
        self._passage_map = dict(zip(ids, passages))

        for i in tqdm(range(0, len(passages), batch_size), desc="  Sparse index"):
            batch = passages[i : i + batch_size]
            batch_ids = ids[i : i + batch_size]
            vecs = _encode_sparse(batch, batch_size=batch_size)
            for doc_id, vec in zip(batch_ids, vecs):
                for token, weight in vec.items():
                    if weight > 0:
                        self._inverted_index.setdefault(token, []).append(
                            (doc_id, float(weight))
                        )

    def search(self, query: str, k: int) -> list[tuple[str, str, float]]:
        """Returns list of (id, passage, score)."""
        query_vec = _encode_sparse([query])[0]
        scores: dict[str, float] = {}
        for token, q_w in query_vec.items():
            for doc_id, d_w in self._inverted_index.get(token, []):
                scores[doc_id] = scores.get(doc_id, 0.0) + q_w * d_w

        top_k = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]
        return [(pid, self._passage_map[pid], s) for pid, s in top_k]

    def save(self, path: str | Path) -> None:
        with open(path, "wb") as f:
            pickle.dump(
                {"index": self._inverted_index, "passages": self._passage_map}, f
            )

    @classmethod
    def load(cls, path: str | Path) -> "SparseRetriever":
        obj = cls()
        with open(path, "rb") as f:
            data = pickle.load(f)
        obj._inverted_index = data["index"]
        obj._passage_map = data["passages"]
        return obj
