import numpy as np
import keras
from keras import layers
from pathlib import Path

from ..features.vietnamese import FEATURE_NAMES

_INPUT_DIM = len(FEATURE_NAMES)

_N = 10
WEIGHT_GRID_3WAY = [
    (i / _N, j / _N, (_N - i - j) / _N)
    for i in range(_N + 1)
    for j in range(_N + 1 - i)
]
SIMPLEX_GRID_3WAY = np.array(WEIGHT_GRID_3WAY, dtype=np.float32)

WEIGHT_GRID_2WAY = [
    (i / _N, (_N - i) / _N, 0.0) for i in range(_N + 1)
]
SIMPLEX_GRID_2WAY = np.array(WEIGHT_GRID_2WAY, dtype=np.float32)

# Four-way: (w_dense, w_bm25, w_sparse, w_toneless), step = 0.1 → 286 points
WEIGHT_GRID_4WAY = [
    (i / _N, j / _N, k / _N, (_N - i - j - k) / _N)
    for i in range(_N + 1)
    for j in range(_N + 1 - i)
    for k in range(_N + 1 - i - j)
]
SIMPLEX_GRID_4WAY = np.array(WEIGHT_GRID_4WAY, dtype=np.float32)


class FusionMLP(keras.Model):
    """Lightweight MLP — Grid NDCG Predictor.

    Instead of predicting a probability distribution over the simplex,
    the model directly regresses NDCG@10 scores for all 66 grid points.
    At inference, weights are derived from the predicted NDCG surface —
    either the softmax-expected grid point (default, robust to flat
    surfaces) or the argmax point (see predict_weights).

    Input:  feature vector — 8 linguistic features, optionally + retrieval
            signals (dim is stored in the checkpoint; see `input_dim`)
    Output: 66-dim sigmoid (predicted NDCG for each simplex grid point)

    Architecture: Normalization (Z-score, adapted)
                  → Dense(64) → LayerNorm → GELU → Dropout(0.1)
                  → Dense(32) → LayerNorm → GELU → Dropout(0.1)
                  → Dense(output_dim)  → sigmoid

    The Normalization layer auto-standardises input features (mean/variance)
    and persists its statistics inside the Keras checkpoint, so inference
    code needs zero external scaling.
    """

    def __init__(self, input_dim: int = _INPUT_DIM, output_dim: int = 66) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self._input_norm = layers.Normalization(axis=-1)
        self.net = keras.Sequential([
            layers.Input(shape=(input_dim,)),
            self._input_norm,
            layers.Dense(64),
            layers.LayerNormalization(),
            layers.Activation("gelu"),
            layers.Dropout(0.1),
            layers.Dense(32),
            layers.LayerNormalization(),
            layers.Activation("gelu"),
            layers.Dropout(0.1),
            layers.Dense(output_dim, activation="sigmoid"),
        ])

    def adapt_features(self, X: np.ndarray) -> None:
        """Adapt the input Normalization layer to feature statistics.

        Must be called once before training with the training feature matrix.
        Not needed at inference time — the adapted mean/variance are stored
        inside the Keras checkpoint.
        """
        self._input_norm.adapt(X)

    def call(self, x, training=False):
        return self.net(x, training=training)

    def predict_weights(
        self,
        features: np.ndarray,
        mode: str = "expected",
        temperature: float = 0.05,
    ) -> tuple[float, ...]:
        """Convenience method: numpy features → routing weights (inference mode).

        The model predicts (per-query normalized) NDCG scores for each simplex
        grid point. Two ways to turn that surface into weights:

        mode="expected" (default):
            w = Σᵢ softmax(ndcg / T)ᵢ · gridᵢ — the softmax-weighted average of
            grid points. On a flat predicted surface this converges to the grid
            centroid ≈ (1/3, 1/3, 1/3), i.e. it degrades gracefully to the
            fixed-equal baseline instead of jumping to an arbitrary extreme
            point the way argmax does. Lower `temperature` → sharper routing.

        mode="argmax":
            The single grid point with the highest predicted NDCG (legacy
            behaviour; brittle when the surface is flat).
        """
        x = np.expand_dims(features, axis=0).astype(np.float32)
        predicted_ndcg = self(x, training=False).numpy().squeeze(0)

        # Legacy direct 3-way weights
        if self.output_dim == 3:
            return tuple(float(v) for v in predicted_ndcg)

        if self.output_dim == 286:
            grid = SIMPLEX_GRID_4WAY     # (w_dense, w_bm25, w_sparse, w_toneless)
        elif self.output_dim == 66:
            grid = SIMPLEX_GRID_3WAY
        elif self.output_dim == 11:
            grid = SIMPLEX_GRID_2WAY
        else:
            raise ValueError(f"Unsupported output_dim: {self.output_dim}")

        if mode == "argmax":
            best_idx = int(np.argmax(predicted_ndcg))
            return tuple(float(v) for v in grid[best_idx])
        elif mode == "expected":
            logits = predicted_ndcg / max(temperature, 1e-8)
            logits -= logits.max()                     # numerical stability
            p = np.exp(logits)
            p /= p.sum()
            w = p @ grid                               # (3,)
            return tuple(float(v) for v in w)
        else:
            raise ValueError(f"Unknown mode: {mode!r} (use 'expected' or 'argmax')")

    def save(self, path: str | Path) -> None:
        self.net.save(str(path))

    @classmethod
    def load(cls, path: str | Path) -> "FusionMLP":
        loaded_net = keras.models.load_model(str(path), compile=False)
        output_dim = loaded_net.output_shape[-1]
        input_dim = loaded_net.input_shape[-1]

        model = cls(input_dim=input_dim, output_dim=output_dim)
        model.net = loaded_net
        return model


