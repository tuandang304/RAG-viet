from rag_vie.features.signal import SIGNAL_FEATURE_NAMES, extract_signal_features


def _feat(name, vec):
    return float(vec[SIGNAL_FEATURE_NAMES.index(name)])


def test_length_and_dtype():
    vec = extract_signal_features({"a": 1.0}, {"a": 1.0}, {})
    assert len(vec) == len(SIGNAL_FEATURE_NAMES) == 7
    assert vec.dtype.name == "float32"


def test_full_agreement():
    d = {"a": 1.0, "b": 0.5}
    vec = extract_signal_features(d, dict(d), dict(d))
    assert _feat("overlap_dense_bm25", vec) == 1.0
    assert _feat("overlap_bm25_sparse", vec) == 1.0
    assert _feat("top1_agreement", vec) == 1.0


def test_disjoint_sources():
    vec = extract_signal_features({"a": 1.0}, {"b": 1.0}, {"c": 1.0})
    assert _feat("overlap_dense_bm25", vec) == 0.0
    assert _feat("top1_agreement", vec) == 0.0


def test_empty_sparse_graceful():
    vec = extract_signal_features({"a": 1.0}, {"a": 1.0}, {})
    assert _feat("overlap_dense_sparse", vec) == 0.0
    assert _feat("sharp_sparse", vec) == 0.0
    # only two sources present but they agree on top-1
    assert _feat("top1_agreement", vec) == 1.0


def test_sharpness():
    # values {1.0, 0.0} → max - mean = 1.0 - 0.5 = 0.5
    vec = extract_signal_features({"a": 1.0, "b": 0.0}, {}, {})
    assert abs(_feat("sharp_dense", vec) - 0.5) < 1e-6


def test_all_empty_is_zero():
    vec = extract_signal_features({}, {}, {})
    assert vec.sum() == 0.0
