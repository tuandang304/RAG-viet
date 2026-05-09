from pathlib import Path

import torch
import torch.nn as nn
import numpy as np

from ..features.vietnamese import FEATURE_NAMES

_INPUT_DIM = len(FEATURE_NAMES)


class FusionMLP(nn.Module):
    """Lightweight MLP that predicts three-way fusion weights.
    Input: Vietnamese-aware feature vector (dim = len(FEATURE_NAMES))
    Output: 3-dim softmax → (w_dense, w_bm25, w_sparse)
    """

    def __init__(self, input_dim: int = _INPUT_DIM, output_dim: int = 3) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, output_dim),
        )
        self.output_dim = output_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, input_dim) → weights: (B, output_dim) after softmax."""
        return torch.softmax(self.net(x), dim=-1)

    def predict_weights(self, features: np.ndarray) -> tuple[float, ...]:
        """Convenience method: numpy features → tuple of weights."""
        self.eval()
        with torch.no_grad():
            x = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
            w = self.forward(x).squeeze(0).numpy()
        return tuple(float(v) for v in w)

    def save(self, path: str | Path) -> None:
        torch.save({"state_dict": self.state_dict(), "output_dim": self.output_dim}, path)

    @classmethod
    def load(cls, path: str | Path) -> "FusionMLP":
        data = torch.load(path, weights_only=True)
        model = cls(output_dim=data["output_dim"])
        model.load_state_dict(data["state_dict"])
        return model
