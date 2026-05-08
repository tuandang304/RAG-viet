import json
from pathlib import Path

import faiss
import numpy as np

from ..config import settings


class DenseRetriever:
    def __init__(self, dim: int | None = None) -> None:
        self._dim = dim or settings.embedding_dim
        self._index: faiss.Index = faiss.IndexFlatIP(self._dim)
        self._passages: list[str] = []
        self._ids: list[str] = []

    def add(self, embeddings: np.ndarray, passages: list[str], ids: list[str] | None = None) -> None:
        """Add passages to the index. Embeddings are L2-normalized before adding."""
        emb = embeddings.astype(np.float32).copy()
        faiss.normalize_L2(emb)
        self._index.add(emb)
        self._passages.extend(passages)
        self._ids.extend(ids if ids is not None else [str(i) for i in range(len(passages))])

    def search(self, query_emb: np.ndarray, k: int) -> list[tuple[str, str, float]]:
        """Returns list of (id, passage, score)."""
        emb = query_emb.astype(np.float32).copy()
        faiss.normalize_L2(emb)
        scores, indices = self._index.search(emb, k)
        return [
            (self._ids[idx], self._passages[idx], float(s))
            for idx, s in zip(indices[0], scores[0])
            if idx != -1
        ]

    def save(self, dir_path: str | Path) -> None:
        p = Path(dir_path)
        p.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(p / "index.faiss"))
        with open(p / "meta.json", "w", encoding="utf-8") as f:
            json.dump({"passages": self._passages, "ids": self._ids, "dim": self._dim}, f, ensure_ascii=False)

    @classmethod
    def load(cls, dir_path: str | Path) -> "DenseRetriever":
        p = Path(dir_path)
        obj = cls()
        obj._index = faiss.read_index(str(p / "index.faiss"))
        with open(p / "meta.json", encoding="utf-8") as f:
            meta = json.load(f)
        obj._passages = meta["passages"]
        obj._ids = meta["ids"]
        obj._dim = meta["dim"]
        return obj
