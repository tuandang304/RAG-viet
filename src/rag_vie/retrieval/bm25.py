import pickle
from pathlib import Path

from rank_bm25 import BM25Okapi
from underthesea import word_tokenize


def _tokenize(text: str) -> list[str]:
    return word_tokenize(text, format="text").split()


class BM25Retriever:
    def __init__(self) -> None:
        self._bm25: BM25Okapi | None = None
        self._passages: list[str] = []
        self._ids: list[str] = []

    def build(self, passages: list[str], ids: list[str] | None = None) -> None:
        self._passages = passages
        self._ids = ids if ids is not None else [str(i) for i in range(len(passages))]
        tokenized = [_tokenize(p) for p in passages]
        self._bm25 = BM25Okapi(tokenized)

    def search(self, query: str, k: int) -> list[tuple[str, str, float]]:
        """Returns list of (id, passage, score)."""
        assert self._bm25 is not None, "Call build() first."
        tokens = _tokenize(query)
        scores = self._bm25.get_scores(tokens)
        top_k = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [(self._ids[i], self._passages[i], float(scores[i])) for i in top_k]

    def save(self, path: str | Path) -> None:
        with open(path, "wb") as f:
            pickle.dump({"bm25": self._bm25, "passages": self._passages, "ids": self._ids}, f)

    @classmethod
    def load(cls, path: str | Path) -> "BM25Retriever":
        obj = cls()
        with open(path, "rb") as f:
            data = pickle.load(f)
        obj._bm25 = data["bm25"]
        obj._passages = data["passages"]
        obj._ids = data["ids"]
        return obj
