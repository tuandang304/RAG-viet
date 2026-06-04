import numpy as np
import tensorflow as tf
import keras
from keras import layers
from pathlib import Path

from ..features.vietnamese import FEATURE_NAMES

_INPUT_DIM = len(FEATURE_NAMES)


class FusionMLP(keras.Model):
    """Lightweight MLP that predicts three-way fusion weights.

    Input:  Vietnamese-aware feature vector (dim = len(FEATURE_NAMES) = 8)
    Output: 3-dim softmax → (w_dense, w_bm25, w_sparse)

    Architecture: Dense(64) → LayerNorm → GELU → Dropout(0.1)
                  → Dense(32) → LayerNorm → GELU → Dropout(0.1)
                  → Dense(3)  → softmax
    """

    def __init__(self, input_dim: int = _INPUT_DIM, output_dim: int = 3) -> None:
        super().__init__()
        self.output_dim = output_dim
        self.net = keras.Sequential([
            layers.Input(shape=(input_dim,)),
            layers.Dense(64),
            layers.LayerNormalization(),
            layers.Activation("gelu"),
            layers.Dropout(0.1),
            layers.Dense(32),
            layers.LayerNormalization(),
            layers.Activation("gelu"),
            layers.Dropout(0.1),
            layers.Dense(output_dim, activation="softmax"),
        ])

    def call(self, x, training=False):
        return self.net(x, training=training)

    def predict_weights(self, features: np.ndarray) -> tuple[float, ...]:
        """Convenience method: numpy features → tuple of weights (inference mode)."""
        x = np.expand_dims(features, axis=0).astype(np.float32)
        w = self(x, training=False).numpy().squeeze(0)
        return tuple(float(v) for v in w)

    def save(self, path: str | Path) -> None:
        self.net.save(str(path))

    @classmethod
    def load(cls, path: str | Path) -> "FusionMLP":
        model = cls()
        model.net = keras.models.load_model(str(path))
        return model

