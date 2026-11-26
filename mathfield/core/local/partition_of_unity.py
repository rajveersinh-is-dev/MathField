"""
mathfield/core/local/partition_of_unity.py

Stage 3: Smooth C2 Wendland Partition of Unity Blending.
Constructs smooth overlapping localized weighting windows with Shepard normalization:
sum_{k=1}^K w_k(x, y) == 1.0 everywhere on Omega.
Completely eliminates patch boundary seams and step discontinuities.
"""

from __future__ import annotations

from typing import Iterator, List, Tuple

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

    def iter_weights(
        self,
        leaves: List[QuadtreeNode],
        X: np.ndarray,
        Y: np.ndarray,
    ) -> Iterator[Tuple[int, np.ndarray, np.ndarray]]:
        """Yield each leaf's nonzero flat indices and normalized weights."""
        K = len(leaves)
        if K == 0:
            raise ValueError("Leaves list cannot be empty")

        X_arr = np.asarray(X, dtype=np.float64)
        Y_arr = np.asarray(Y, dtype=np.float64)
        if X_arr.shape != Y_arr.shape:
            raise ValueError("X and Y must have the same shape")
        x_flat = X_arr.ravel()
        y_flat = Y_arr.ravel()
        N = x_flat.size
        if K == 1:
            yield 0, np.arange(N), np.ones(N, dtype=np.float64)
            return

        centers = np.array([leaf.center for leaf in leaves], dtype=np.float64)
        half_widths = np.array([leaf.half_widths for leaf in leaves], dtype=np.float64)

        def build(factor: float) -> Tuple[np.ndarray, List[Tuple[np.ndarray, np.ndarray]]]:
            sum_weights = np.zeros(N, dtype=np.float64)
            sparse: List[Tuple[np.ndarray, np.ndarray]] = []
            for k in range(K):
                rx = factor * max(half_widths[k, 0], 1e-6)
                ry = factor * max(half_widths[k, 1], 1e-6)
                dx = (x_flat - centers[k, 0]) / rx
                dy = (y_flat - centers[k, 1]) / ry
                if self.separable:
                    adx = np.minimum(np.abs(dx), 1.0)
                    ady = np.minimum(np.abs(dy), 1.0)
                    inside = (np.abs(dx) <= 1.0 + 1e-12) & (np.abs(dy) <= 1.0 + 1e-12)
                    wx = ((1.0 - adx) ** 4.0) * (4.0 * adx + 1.0)
                    wy = ((1.0 - ady) ** 4.0) * (4.0 * ady + 1.0)
                    values = np.where(inside, wx * wy, 0.0)
                else:
                    radius = np.sqrt(dx * dx + dy * dy)
                    sr = np.minimum(radius, 1.0)
                    values = np.where(radius <= 1.0 + 1e-12, ((1.0 - sr) ** 4.0) * (4.0 * sr + 1.0), 0.0)
                indices = np.flatnonzero(values > 0.0)
                selected = values[indices]
                sparse.append((indices, selected))
                sum_weights[indices] += selected
            return sum_weights, sparse

        # Grow supports until every sample lies inside at least one window.
        # A hard nearest-centre assignment would reintroduce the very seams
        # this partition of unity exists to remove.
        factor = 1.0 + max(self.overlap_ratio, 1e-6)
        sum_weights, sparse = build(factor)
        for _ in range(12):
            if np.all(sum_weights > 1e-12):
                break
            factor *= 1.25
            sum_weights, sparse = build(factor)

        zero_mask = sum_weights <= 1e-12
        if np.any(zero_mask):
            # Smooth (Gaussian) fallback for samples outside the convex hull
            # of all supports - continuous, unlike nearest-centre assignment.
            zero_indices = np.flatnonzero(zero_mask)
            zx, zy = x_flat[zero_indices], y_flat[zero_indices]
            d2 = (centers[:, 0:1] - zx[None, :]) ** 2 + (centers[:, 1:2] - zy[None, :]) ** 2
            scale = max(float(np.max(d2)), 1e-12)
            fallback = np.exp(-d2 / scale)
            for k in range(K):
                selected = fallback[k] > 0.0
                indices, values = sparse[k]
                sparse[k] = (np.concatenate([indices, zero_indices[selected]]), np.concatenate([values, fallback[k, selected]]))
                sum_weights[zero_indices[selected]] += fallback[k, selected]

        for k, (indices, values) in enumerate(sparse):
            yield k, indices, values / np.maximum(sum_weights[indices], 1e-300)

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
        X_arr = np.asarray(X, dtype=np.float64)
        Y_arr = np.asarray(Y, dtype=np.float64)
        weights = np.zeros((len(leaves), *X_arr.shape), dtype=np.float64)
        flat_weights = weights.reshape(len(leaves), -1)
        for k, indices, values in self.iter_weights(leaves, X_arr, Y_arr):
            flat_weights[k, indices] = values
        return weights
