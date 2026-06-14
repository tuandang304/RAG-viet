import numpy as np
import pytest

# extract_features depends on underthesea → transformers → torch. Skip cleanly if that
# native stack fails to load (e.g. missing MSVC redistributable on Windows) rather than
# erroring out the whole test session.
try:
    from rag_vie.features.vietnamese import FEATURE_NAMES, extract_features
except (ImportError, OSError) as exc:  # pragma: no cover - environment dependent
    pytest.skip(f"underthesea/torch unavailable: {exc}", allow_module_level=True)


def _f(name, vec):
    return float(vec[FEATURE_NAMES.index(name)])


def test_shape_and_bounds():
    vec = extract_features("Thủ đô của Việt Nam là gì?")
    assert len(vec) == len(FEATURE_NAMES) == 13
    assert vec.dtype == np.float32
    assert np.all(vec >= 0.0) and np.all(vec <= 1.0)


def test_question_word_and_length():
    assert _f("has_question_word", extract_features("Hà Nội ở đâu")) == 1.0
    assert _f("has_question_word", extract_features("Hà Nội là thủ đô")) == 0.0
    assert _f("query_length_norm", extract_features("a " * 30)) == 1.0  # capped at 20


def test_digit_and_acronym():
    assert _f("digit_ratio", extract_features("Năm 2020 có sự kiện gì")) > 0.0
    assert _f("acronym_ratio", extract_features("WTO là tổ chức gì")) > 0.0


def test_idf_features_off_without_corpus():
    vec = extract_features("Hà Nội ở đâu")
    assert _f("oov_ratio", vec) == 0.0
    assert _f("avg_idf_norm", vec) == 0.0
    assert _f("max_idf_norm", vec) == 0.0


def test_idf_features_with_corpus():
    # underthesea segments "Hà Nội" → "Hà_Nội"; supply a matching idf entry.
    idf = {"Hà_Nội": 6.0, "ở": 0.1, "đâu": 3.0}
    vec = extract_features("Hà Nội ở đâu", bm25_idf=idf)
    assert 0.0 <= _f("avg_idf_norm", vec) <= 1.0
    assert _f("max_idf_norm", vec) >= _f("avg_idf_norm", vec)
