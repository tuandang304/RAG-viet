"""Neural feature extractor: frozen PhoBERT-base + trainable projection head.

Replaces heuristic feature extraction in extract_features() with learned
representations distilled from heuristic soft-labels.

Usage (inference):
    extractor = NeuralFeatureExtractor.load("checkpoints/feature_extractor.pt")
    features  = extractor.extract("Thủ đô của Việt Nam là gì?")   # float32[8]

Usage (training): see scripts/train_feature_extractor.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

from .vietnamese import FEATURE_NAMES

_DEFAULT_MODEL = "vinai/phobert-base"
_N_FEATURES = len(FEATURE_NAMES)  # 8


class _ProjectionHead(nn.Module):
    """Linear(hidden_dim → 64) → ReLU → Linear(64 → N_FEATURES) → Sigmoid."""

    def __init__(self, hidden_dim: int, n_features: int = _N_FEATURES) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, n_features),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class NeuralFeatureExtractor:
    """Frozen PhoBERT encoder + trainable projection head → 8-dim feature vector.

    The encoder (~135 M params) is always frozen; only the projection head
    (~3 K params) is updated during distillation training.

    Args:
        model_name: HuggingFace model id for the encoder (default: vinai/phobert-base).
        device:     "cuda" / "cpu" / None (auto-detect).
    """

    def __init__(
        self,
        model_name: str = _DEFAULT_MODEL,
        device: str | None = None,
    ) -> None:
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        encoder = AutoModel.from_pretrained(model_name)
        encoder.eval()
        for p in encoder.parameters():
            p.requires_grad = False
        self.encoder = encoder.to(self.device)

        hidden_dim: int = encoder.config.hidden_size
        self.projection = _ProjectionHead(hidden_dim).to(self.device)

    @torch.no_grad()
    def _encode(self, query: str) -> torch.Tensor:
        inputs = self.tokenizer(
            query,
            return_tensors="pt",
            truncation=True,
            max_length=128,
            padding=True,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        outputs = self.encoder(**inputs)
        mask = inputs["attention_mask"].unsqueeze(-1).float()
        pooled = (outputs.last_hidden_state * mask).sum(1) / mask.sum(1)
        return pooled  # (1, hidden_dim)

    def extract(self, query: str) -> np.ndarray:
        """Return float32 array of shape (len(FEATURE_NAMES),)."""
        pooled = self._encode(query)
        with torch.no_grad():
            features = self.projection(pooled).squeeze(0).cpu().numpy()
        return features.astype(np.float32)

    def save(self, path: str | Path) -> None:
        """Save projection head weights only (encoder is reloaded from HF)."""
        torch.save(self.projection.state_dict(), str(path))

    @classmethod
    def load(
        cls,
        path: str | Path,
        model_name: str = _DEFAULT_MODEL,
        device: str | None = None,
    ) -> NeuralFeatureExtractor:
        extractor = cls(model_name=model_name, device=device)
        state = torch.load(str(path), map_location=extractor.device, weights_only=True)
        extractor.projection.load_state_dict(state)
        extractor.projection.eval()
        return extractor
