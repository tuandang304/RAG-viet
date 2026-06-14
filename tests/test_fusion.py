from rag_vie.utils.fusion import fuse_scores


def test_fuse_weighted_sum():
    d = {"a": 1.0, "b": 0.0}
    b = {"a": 0.0, "b": 1.0}
    s: dict[str, float] = {}
    fused = fuse_scores(d, b, s, (0.7, 0.3, 0.0))
    assert fused["a"] == 0.7
    assert fused["b"] == 0.3


def test_fuse_union_and_missing_ids():
    d = {"a": 1.0}
    b = {"c": 1.0}
    s = {"a": 0.5}
    fused = fuse_scores(d, b, s, (1.0, 1.0, 1.0))
    assert set(fused) == {"a", "c"}
    assert fused["a"] == 1.0 + 0.0 + 0.5
    assert fused["c"] == 0.0 + 1.0 + 0.0
