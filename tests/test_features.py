import numpy as np

from rag_vie.features.vietnamese import FEATURE_NAMES, extract_features


def test_feature_vector_shape_and_range():
    feats = extract_features("Thủ đô của Việt Nam là gì?")
    assert feats.shape == (len(FEATURE_NAMES),)
    assert feats.dtype == np.float32
    assert np.all(feats >= 0.0) and np.all(feats <= 1.0)


def test_diacritic_ratio():
    idx = FEATURE_NAMES.index("diacritic_ratio")
    assert extract_features("thu do cua Viet Nam la gi")[idx] == 0.0
    assert extract_features("Thủ đô của Việt Nam là gì?")[idx] > 0.5


def test_english_ratio_code_switching():
    idx = FEATURE_NAMES.index("english_ratio")
    pure_vi = extract_features("Thủ đô của Việt Nam là gì?")
    mixed = extract_features("Cách config server database như thế nào?")
    assert mixed[idx] > pure_vi[idx]


def test_question_word_detection():
    idx = FEATURE_NAMES.index("has_question_word")
    assert extract_features("Thủ đô của Việt Nam là gì?")[idx] == 1.0
    assert extract_features("Hà Nội là thủ đô của Việt Nam.")[idx] == 0.0


def test_oov_ratio_uses_vocab():
    idx = FEATURE_NAMES.index("oov_ratio")
    # Without a vocab, oov_ratio must default to 0
    assert extract_features("Thủ đô của Việt Nam là gì?")[idx] == 0.0
    # Empty vocab → everything is OOV
    feats = extract_features("Thủ đô của Việt Nam là gì?", bm25_vocab=set())
    assert feats[idx] == 1.0


def test_empty_query_does_not_crash():
    feats = extract_features("")
    assert feats.shape == (len(FEATURE_NAMES),)
    assert np.all(np.isfinite(feats))
