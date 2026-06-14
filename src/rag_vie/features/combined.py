"""Combine Vietnamese query features with signal-aware features into the MLP input.

The concatenation order is fixed (query features first, then signal features) so that
training, evaluation, and inference all build the MLP input identically, and so that the
query-feature indices used by stratified / correlation analysis stay stable.
"""

import numpy as np

from .signal import SIGNAL_FEATURE_NAMES
from .vietnamese import FEATURE_NAMES

ALL_FEATURE_NAMES = FEATURE_NAMES + SIGNAL_FEATURE_NAMES


def combine(query_feats: np.ndarray, signal_feats: np.ndarray) -> np.ndarray:
    """Concatenate query features and signal features into one float32 vector."""
    return np.concatenate([query_feats, signal_feats]).astype(np.float32)
