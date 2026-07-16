from rag_vie.retrieval.bm25 import BM25Retriever

PASSAGES = [
    "Hà Nội là thủ đô của Việt Nam.",
    "Thành phố Hồ Chí Minh là thành phố lớn nhất Việt Nam.",
    "Phở là món ăn truyền thống của người Việt.",
]
IDS = ["p1", "p2", "p3"]


def _build() -> BM25Retriever:
    r = BM25Retriever()
    r.build(PASSAGES, IDS)
    return r


def test_search_returns_relevant_passage_first():
    r = _build()
    hits = r.search("thủ đô của Việt Nam", k=3)
    assert hits[0][0] == "p1"
    assert len(hits) == 3
    # (id, passage, score) triples, scores descending
    scores = [s for _, _, s in hits]
    assert scores == sorted(scores, reverse=True)


def test_vocab_property():
    assert BM25Retriever().vocab is None
    vocab = _build().vocab
    assert vocab is not None and len(vocab) > 0
    # underthesea joins compounds with "_"
    assert any("_" in tok for tok in vocab)


def test_save_load_roundtrip(tmp_path):
    r = _build()
    path = tmp_path / "bm25.pkl"
    r.save(path)
    loaded = BM25Retriever.load(path)
    assert loaded.tokenizer == "underthesea"
    assert loaded.search("thủ đô", k=1)[0][0] == r.search("thủ đô", k=1)[0][0]


def test_toneless_tokenizer_matches_toneless_query():
    r = BM25Retriever(tokenizer="toneless_syllable")
    r.build(PASSAGES, IDS)
    # Toneless query must match the (originally toned) passage
    hits = r.search("thu do cua viet nam la gi", k=3)
    assert hits[0][0] == "p1"
    assert hits[0][2] > 0
    # Vocabulary must contain no diacritics and no compound joins
    assert all("_" not in tok for tok in r.vocab)
    assert "thu" in r.vocab and "do" in r.vocab


def test_toneless_tokenizer_survives_save_load(tmp_path):
    r = BM25Retriever(tokenizer="toneless_syllable")
    r.build(PASSAGES, IDS)
    path = tmp_path / "bm25_toneless.pkl"
    r.save(path)
    loaded = BM25Retriever.load(path)
    assert loaded.tokenizer == "toneless_syllable"
    assert loaded.search("thu do", k=1)[0][0] == "p1"


def test_unknown_tokenizer_rejected():
    import pytest

    with pytest.raises(ValueError):
        BM25Retriever(tokenizer="nope")
