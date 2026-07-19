"""Derive ablation training caches OFFLINE from the full 4-way xy cache.

The full cache (`multidomain_xy_4way_aug.npz`) stores X (n, 36) = 8 linguistic
features + 28 four-channel retrieval signals, and Y (n, 286) = raw NDCG@10 over
the 286-point four-simplex grid. Three ablation variants are exact column /
row-wise transforms of it — no retrieval, no API:

  1. no-signals      : X → X[:, :8]                 (linguistic only), Y unchanged
  2. norm-labels     : X unchanged, Y → per-query min-max (the harmful variant)
  3. three-way       : X → 26 cols (drop toneless per-channel stats and every
                       pair involving toneless), Y → the 66 grid points with
                       w_toneless = 0 (an exact subset of the 286-point grid,
                       in the same (i, j) order as WEIGHT_GRID_3WAY)

Column layout of the 28 signals (see rag_vie.features.retrieval_signals):
  [0:16]  per-channel stats, 4 per channel, channel order (dense, bm25, sparse, toneless)
  [16:22] jaccard overlaps, pair order (d,b) (d,s) (d,t) (b,s) (b,t) (s,t)
  [22:28] top1 agreement, same pair order

Usage:
    uv run python scripts/derive_ablation_caches.py \\
        --src checkpoints/multidomain_xy_4way_aug.npz --out-dir checkpoints
"""

import argparse
from pathlib import Path

import numpy as np

_N = 10


def four_way_grid() -> np.ndarray:
    return np.array(
        [
            (i / _N, j / _N, k / _N, (_N - i - j - k) / _N)
            for i in range(_N + 1)
            for j in range(_N + 1 - i)
            for k in range(_N + 1 - i - j)
        ],
        dtype=np.float32,
    )


def three_way_cols(n_base: int = 8) -> list[int]:
    """X columns for the 3-way (26-dim) feature layout inside the 36-dim one."""
    cols = list(range(n_base))                       # 8 linguistic
    cols += list(range(n_base, n_base + 12))         # dense/bm25/sparse per-channel stats
    jac = n_base + 16
    cols += [jac + 0, jac + 1, jac + 3]              # (d,b) (d,s) (b,s)
    top1 = n_base + 22
    cols += [top1 + 0, top1 + 1, top1 + 3]
    return cols


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", default="checkpoints/multidomain_xy_4way_aug.npz")
    parser.add_argument("--out-dir", default="checkpoints")
    args = parser.parse_args()

    data = np.load(args.src)
    X, Y = data["X"].astype(np.float32), data["Y"].astype(np.float32)
    assert X.shape[1] == 36 and Y.shape[1] == 286, (X.shape, Y.shape)
    out = Path(args.out_dir)

    # 1. no-signals
    np.savez(out / "abl_xy_nosignals.npz", X=X[:, :8], Y=Y)
    print(f"no-signals : X {X[:, :8].shape}  Y {Y.shape}")

    # 2. normalized labels (per-query min-max; flat rows -> zeros, as in train_mlp)
    lo = Y.min(axis=1, keepdims=True)
    hi = Y.max(axis=1, keepdims=True)
    span = hi - lo
    Yn = np.where(span > 0, (Y - lo) / np.where(span > 0, span, 1.0), 0.0).astype(np.float32)
    np.savez(out / "abl_xy_normlabels.npz", X=X, Y=Yn)
    print(f"norm-labels: X {X.shape}  Y {Yn.shape}  (flat rows: {int((span == 0).sum())})")

    # 3. three-way: grid subset where w_toneless == 0, plus the 26-col X
    grid = four_way_grid()
    idx3 = np.where(grid[:, 3] == 0.0)[0]
    assert len(idx3) == 66, len(idx3)
    # Sanity: extracted (a, b, c) sequence must equal WEIGHT_GRID_3WAY order
    expect = np.array(
        [(i / _N, j / _N, (_N - i - j) / _N) for i in range(_N + 1) for j in range(_N + 1 - i)],
        dtype=np.float32,
    )
    np.testing.assert_allclose(grid[idx3][:, :3], expect, atol=1e-6)

    cols = three_way_cols()
    X3, Y3 = X[:, cols], Y[:, idx3]
    np.savez(out / "abl_xy_3way.npz", X=X3, Y=Y3)
    print(f"three-way  : X {X3.shape}  Y {Y3.shape}  (grid subset verified)")


if __name__ == "__main__":
    main()
