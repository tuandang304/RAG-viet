import pickle
from pathlib import Path

from rank_bm25 import BM25Okapi
from underthesea import word_tokenize

from ..utils.text import remove_diacritics


def _tokenize(text: str) -> list[str]:
    return word_tokenize(text, format="text").split()


def _tokenize_toneless(text: str) -> list[str]:
    """Lowercase, diacritic-stripped, syllable-level tokens.

    Deliberately NOT underthesea: word segmentation quality collapses on
    toneless text, and segmenting the (toned) passages differently from the
    (toneless) queries would create a new mismatch. Whitespace syllables are
    applied identically on both sides — fully symmetric and deterministic.
    """
    return remove_diacritics(text.lower()).split()


_TOKENIZERS = {
    "underthesea": _tokenize,
    "toneless_syllable": _tokenize_toneless,
}


class BM25Retriever:
    """BM25Okapi retriever with a pluggable tokenizer.

    tokenizer="underthesea"        — word segmentation, diacritics kept (default).
    tokenizer="toneless_syllable"  — lowercase syllables with diacritics removed;
                                     use for the tone-robust retrieval channel.
    The tokenizer name is persisted in the pickle so load() restores it.
    """

    def __init__(self, tokenizer: str = "underthesea") -> None:
        if tokenizer not in _TOKENIZERS:
            raise ValueError(f"Unknown tokenizer: {tokenizer!r} (choose from {sorted(_TOKENIZERS)})")
        self.tokenizer = tokenizer
        self._tokenize = _TOKENIZERS[tokenizer]
        self._bm25: BM25Okapi | None = None
        self._passages: list[str] = []
        self._ids: list[str] = []

    def build(self, passages: list[str], ids: list[str] | None = None) -> None:
        self._passages = passages
        self._ids = ids if ids is not None else [str(i) for i in range(len(passages))]
        tokenized = [self._tokenize(p) for p in passages]
        self._bm25 = BM25Okapi(tokenized)

    @property
    def vocab(self) -> set[str] | None:
        """Corpus vocabulary (tokens with an IDF entry), or None before build()/load()."""
        return set(self._bm25.idf.keys()) if self._bm25 is not None else None

    def search(self, query: str, k: int) -> list[tuple[str, str, float]]:
        """Returns list of (id, passage, score)."""
        assert self._bm25 is not None, "Call build() first."
        tokens = self._tokenize(query)
        scores = self._bm25.get_scores(tokens)
        top_k = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [(self._ids[i], self._passages[i], float(scores[i])) for i in top_k]

    def save(self, path: str | Path) -> None:
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "bm25": self._bm25,
                    "passages": self._passages,
                    "ids": self._ids,
                    "tokenizer": self.tokenizer,
                },
                f,
            )

    @classmethod
    def load(cls, path: str | Path) -> "BM25Retriever":
        with open(path, "rb") as f:
            data = pickle.load(f)
        # Older pickles predate the tokenizer field — they were all underthesea.
        obj = cls(tokenizer=data.get("tokenizer", "underthesea"))
        obj._bm25 = data["bm25"]
        obj._passages = data["passages"]
        obj._ids = data["ids"]
        return obj
