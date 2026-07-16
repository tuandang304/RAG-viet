import numpy as np
import pytest

from rag_vie.fusion.mlp import (
    SIMPLEX_GRID_2WAY,
    SIMPLEX_GRID_3WAY,
    SIMPLEX_GRID_4WAY,
    WEIGHT_GRID_2WAY,
    WEIGHT_GRID_3WAY,
    WEIGHT_GRID_4WAY,
)


def test_3way_grid_covers_simplex():
    assert len(WEIGHT_GRID_3WAY) == 66
    for a, b, c in WEIGHT_GRID_3WAY:
        assert a >= 0 and b >= 0 and c >= 0
        assert a + b + c == pytest.approx(1.0)
    # Grid points are unique
    assert len(set(WEIGHT_GRID_3WAY)) == 66


def test_2way_grid_is_simplex_edge():
    assert len(WEIGHT_GRID_2WAY) == 11
    for a, b, c in WEIGHT_GRID_2WAY:
        assert c == 0.0
        assert a + b == pytest.approx(1.0)


def test_4way_grid_covers_simplex():
    # Compositions of 10 into 4 parts: C(13, 3) = 286
    assert len(WEIGHT_GRID_4WAY) == 286
    for a, b, c, d in WEIGHT_GRID_4WAY:
        assert min(a, b, c, d) >= 0
        assert a + b + c + d == pytest.approx(1.0)
    assert len(set(WEIGHT_GRID_4WAY)) == 286
    assert SIMPLEX_GRID_4WAY.shape == (286, 4)
    np.testing.assert_allclose(SIMPLEX_GRID_4WAY.sum(axis=1), 1.0, atol=1e-6)


def test_numpy_grids_match_python_grids():
    assert SIMPLEX_GRID_3WAY.shape == (66, 3)
    assert SIMPLEX_GRID_2WAY.shape == (11, 3)
    np.testing.assert_allclose(SIMPLEX_GRID_3WAY.sum(axis=1), 1.0, atol=1e-6)


@pytest.mark.slow
def test_untrained_mlp_predicts_valid_grid_point():
    """Loads TensorFlow/Keras — slow. Deselect with -m 'not slow'."""
    from rag_vie.features.vietnamese import FEATURE_NAMES
    from rag_vie.fusion.mlp import FusionMLP

    model = FusionMLP(output_dim=66)
    features = np.random.default_rng(0).random(len(FEATURE_NAMES)).astype(np.float32)
    weights = model.predict_weights(features, mode="argmax")

    assert len(weights) == 3
    assert sum(weights) == pytest.approx(1.0, abs=1e-6)
    # argmax mode must return an exact grid point (float32 tolerance)
    dists = np.abs(SIMPLEX_GRID_3WAY - np.array(weights, dtype=np.float32)).sum(axis=1)
    assert dists.min() == pytest.approx(0.0, abs=1e-6)


@pytest.mark.slow
def test_expected_mode_flat_surface_gives_near_equal_weights():
    """A flat predicted-NDCG surface must fall back to ~fixed-equal weights."""
    from rag_vie.features.vietnamese import FEATURE_NAMES
    from rag_vie.fusion.mlp import FusionMLP

    model = FusionMLP(output_dim=66)
    features = np.random.default_rng(2).random(len(FEATURE_NAMES)).astype(np.float32)
    # Huge temperature flattens softmax regardless of predictions →
    # expected weights = centroid of the 66-point simplex grid = (1/3, 1/3, 1/3)
    weights = model.predict_weights(features, mode="expected", temperature=1e6)
    assert weights == pytest.approx((1 / 3, 1 / 3, 1 / 3), abs=1e-3)


@pytest.mark.slow
def test_expected_mode_sharp_temperature_converges_to_argmax():
    from rag_vie.features.vietnamese import FEATURE_NAMES
    from rag_vie.fusion.mlp import FusionMLP

    model = FusionMLP(output_dim=66)
    features = np.random.default_rng(3).random(len(FEATURE_NAMES)).astype(np.float32)
    argmax_w = model.predict_weights(features, mode="argmax")
    sharp_w = model.predict_weights(features, mode="expected", temperature=1e-6)
    assert sharp_w == pytest.approx(argmax_w, abs=1e-3)


@pytest.mark.slow
def test_expected_weights_are_valid_simplex_point():
    from rag_vie.features.vietnamese import FEATURE_NAMES
    from rag_vie.fusion.mlp import FusionMLP

    model = FusionMLP(output_dim=66)
    features = np.random.default_rng(4).random(len(FEATURE_NAMES)).astype(np.float32)
    weights = model.predict_weights(features)   # default mode="expected"
    assert len(weights) == 3
    assert all(w >= 0 for w in weights)
    assert sum(weights) == pytest.approx(1.0, abs=1e-5)


@pytest.mark.slow
def test_mlp_save_load_roundtrip(tmp_path):
    from rag_vie.features.vietnamese import FEATURE_NAMES
    from rag_vie.fusion.mlp import FusionMLP

    model = FusionMLP(output_dim=66)
    features = np.random.default_rng(1).random(len(FEATURE_NAMES)).astype(np.float32)
    before = model.predict_weights(features)

    path = tmp_path / "mlp.keras"
    model.save(path)
    loaded = FusionMLP.load(path)
    assert loaded.output_dim == 66
    assert loaded.predict_weights(features) == before
