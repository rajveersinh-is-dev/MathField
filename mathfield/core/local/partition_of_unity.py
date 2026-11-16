"""
mathfield/core/local/partition_of_unity.py

Stage 3: Smooth C2 Wendland Partition of Unity Blending.
Constructs smooth overlapping localized weighting windows with Shepard normalization:
sum_{k=1}^K w_k(x, y) == 1.0 everywhere on Omega.
Completely eliminates patch boundary seams and step discontinuities.
"""

from __future__ import annotations

from typing import List

import numpy as np

from mathfield.core.local.quadtree import QuadtreeNode


class WendlandPartitionOfUnity:
    """
    Constructs normalized C2 Wendland weighting windows across spatial patches.
    """

    def __init__(self, overlap_ratio: float = 0.33) -> None:
        if overlap_ratio < 0.0:
            raise ValueError(f"overlap_ratio must be non-negative, got {overlap_ratio}")
        self.overlap_ratio = overlap_ratio

    def compute_weights(
        self,
        leaves: List[QuadtreeNode],
        X: np.ndarray,
        Y: np.ndarray,
    ) -> np.ndarray:
        """
        Compute normalized partition of unity weights for each leaf node.
        Returns array of shape (K, H, W) where K == len(leaves).
        Guarantees sum_{k=0}^{K-1} weights[k, :, :] == 1.0 everywhere.
        """
        K = len(leaves)
        if K == 0:
            raise ValueError("Leaves list cannot be empty")
        if K == 1:
            return np.ones((1, *X.shape), dtype=np.float64)

        X_arr = np.asarray(X, dtype=np.float64)
        Y_arr = np.asarray(Y, dtype=np.float64)
        orig_shape = X_arr.shape

        x_flat = X_arr.ravel()
        y_flat = Y_arr.ravel()
        N = x_flat.size

        # Unnormalized weights: W[k, :]
        W = np.zeros((K, N), dtype=np.float64)

        for k, leaf in enumerate(leaves):
            cx, cy = leaf.center
            hx, hy = leaf.half_widths

            # Expand radius by overlap ratio
            Rx = (1.0 + self.overlap_ratio) * max(hx, 1e-6)
            Ry = (1.0 + self.overlap_ratio) * max(hy, 1e-6)

            dx = (x_flat - cx) / Rx
            dy = (y_flat - cy) / Ry
            r = np.sqrt(dx * dx + dy * dy)

            mask = r < 1.0
            if np.any(mask):
                sr = r[mask]
                W[k, mask] = ((1.0 - sr) ** 4.0) * (4.0 * sr + 1.0)

        # Shepard Normalization
        sum_W = np.sum(W, axis=0)

        # Protection for points outside patch overlap supports:
        # assign to closest leaf center if sum_W is zero
        zero_mask = sum_W < 1e-12
        if np.any(zero_mask):
            zero_x = x_flat[zero_mask]
            zero_y = y_flat[zero_mask]
            # Find nearest leaf center for each zero point
            centers = np.array([leaf.center for leaf in leaves])  # (K, 2)
            for pt_idx in range(len(zero_x)):
                px, py = zero_x[pt_idx], zero_y[pt_idx]
                dists = (centers[:, 0] - px) ** 2 + (centers[:, 1] - py) ** 2
                best_k = int(np.argmin(dists))
                # Global index in flat array
                glob_idx = np.where(zero_mask)[0][pt_idx]
                W[best_k, glob_idx] = 1.0
            sum_W = np.sum(W, axis=0)

        weights = W / sum_W

        # Reshape to (K, H, W)
        return weights.reshape(K, *orig_shape)
