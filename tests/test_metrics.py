"""Known-answer tests for the ranking metrics that back the paper's numbers."""

import math

from rag_vie.utils.metrics import (
    hit_at_1,
    map_at_k,
    min_max_normalize,
    mrr_at_k,
    ndcg_at_k,
    recall_at_k,
)


def test_ndcg_perfect_and_empty():
    assert ndcg_at_k(["a", "b"], {"a", "b"}, 10) == 1.0
    assert ndcg_at_k([], {"a"}, 10) == 0.0
    assert ndcg_at_k(["x", "y"], set(), 10) == 0.0


def test_ndcg_discounts_lower_ranks():
    # One relevant doc at rank 2: DCG = 1/log2(3); IDCG (1 relevant) = 1/log2(2) = 1.
    got = ndcg_at_k(["x", "a"], {"a"}, 10)
    assert math.isclose(got, (1 / math.log2(3)) / 1.0, rel_tol=1e-9)


def test_mrr():
    assert mrr_at_k(["x", "a", "b"], {"a"}, 10) == 0.5
    assert mrr_at_k(["a"], {"a"}, 10) == 1.0
    assert mrr_at_k(["x"], {"a"}, 10) == 0.0


def test_map():
    # relevant at ranks 1 and 3 → (1/1 + 2/3) / 2
    got = map_at_k(["a", "x", "b"], {"a", "b"}, 10)
    assert math.isclose(got, (1.0 + 2 / 3) / 2, rel_tol=1e-9)


def test_recall_and_hit():
    assert recall_at_k(["a", "x"], {"a", "b"}, 10) == 0.5
    assert recall_at_k(["x"], set(), 10) == 0.0
    assert hit_at_1(["a", "b"], {"a"}) == 1.0
    assert hit_at_1(["x", "a"], {"a"}) == 0.0
    assert hit_at_1([], {"a"}) == 0.0


def test_min_max_normalize():
    out = min_max_normalize({"a": 0.0, "b": 10.0, "c": 5.0})
    assert out == {"a": 0.0, "b": 1.0, "c": 0.5}
    assert min_max_normalize({}) == {}
    assert min_max_normalize({"a": 3.0, "b": 3.0}) == {"a": 0.0, "b": 0.0}  # flat → zeros
