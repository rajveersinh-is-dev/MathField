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

    def __init__(self, overlap_ratio: float = 0.33, separable: bool = True) -> None:
        if overlap_ratio < 0.0:
            raise ValueError(f"overlap_ratio must be non-negative, got {overlap_ratio}")
        self.overlap_ratio = overlap_ratio
        self.separable = separable

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

        centers = np.array([leaf.center for leaf in leaves], dtype=np.float64)
        half_widths = np.array([leaf.half_widths for leaf in leaves], dtype=np.float64)

        def build(factor: float) -> np.ndarray:
            W_local = np.zeros((K, N), dtype=np.float64)
            Rx = factor * np.maximum(half_widths[:, 0], 1e-6)
            Ry = factor * np.maximum(half_widths[:, 1], 1e-6)
            dx = (x_flat[None, :] - centers[:, 0:1]) / Rx[:, None]
            dy = (y_flat[None, :] - centers[:, 1:2]) / Ry[:, None]
            if self.separable:
                adx = np.minimum(np.abs(dx), 1.0)
                ady = np.minimum(np.abs(dy), 1.0)
                inside = (np.abs(dx) <= 1.0 + 1e-12) & (np.abs(dy) <= 1.0 + 1e-12)
                wx = ((1.0 - adx) ** 4.0) * (4.0 * adx + 1.0)
                wy = ((1.0 - ady) ** 4.0) * (4.0 * ady + 1.0)
                W_local = np.where(inside, wx * wy, 0.0)
            else:
                r = np.sqrt(dx * dx + dy * dy)
                sr = np.minimum(r, 1.0)
                W_local = np.where(r <= 1.0 + 1e-12, ((1.0 - sr) ** 4.0) * (4.0 * sr + 1.0), 0.0)
            return W_local

        # Grow supports until every sample lies inside at least one window.
        # A hard nearest-centre assignment would reintroduce the very seams
        # this partition of unity exists to remove.
        factor = 1.0 + max(self.overlap_ratio, 1e-6)
        W = build(factor)
        for _ in range(12):
            if np.all(np.sum(W, axis=0) > 1e-12):
                break
            factor *= 1.25
            W = build(factor)

        zero_mask = np.sum(W, axis=0) <= 1e-12
        if np.any(zero_mask):
            # Smooth (Gaussian) fallback for samples outside the convex hull
            # of all supports - continuous, unlike nearest-centre assignment.
            zx, zy = x_flat[zero_mask], y_flat[zero_mask]
            d2 = (centers[:, 0:1] - zx[None, :]) ** 2 + (centers[:, 1:2] - zy[None, :]) ** 2
            scale = max(float(np.max(d2)), 1e-12)
            W[:, zero_mask] = np.exp(-d2 / scale)

        sum_W = np.sum(W, axis=0)
        weights = W / np.maximum(sum_W, 1e-300)

        # Reshape to (K, H, W)
        return weights.reshape(K, *orig_shape)
