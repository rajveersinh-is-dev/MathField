"""
mathfield/core/local/quadtree.py

Stage 3: Spatial Adaptive Quadtree Decomposition.
Subdivides domain guided by localized Sobolev residual error metric:
E(Q_d) = 1/|Q_d| * iint (||R||^2 + mu * ||grad R||^2) dx dy.
Focuses higher resolution on sharp image edges and high-contrast boundaries.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional, Tuple

import numpy as np


@dataclass
class QuadtreeNode:
    """A bounding region node within an adaptive spatial quadtree."""
    bounds: Tuple[float, float, float, float]  # (xmin, ymin, xmax, ymax)
    depth: int = 0
    is_leaf: bool = True
    error: float = 0.0
    children: Optional[List[QuadtreeNode]] = None
    patch: Optional[Any] = None

    @property
    def x_min(self) -> float:
        return self.bounds[0]

    @property
    def y_min(self) -> float:
        return self.bounds[1]

    @property
    def x_max(self) -> float:
        return self.bounds[2]

    @property
    def y_max(self) -> float:
        return self.bounds[3]

    @property
    def center(self) -> Tuple[float, float]:
        return (0.5 * (self.x_min + self.x_max), 0.5 * (self.y_min + self.y_max))

    @property
    def half_widths(self) -> Tuple[float, float]:
        return (0.5 * (self.x_max - self.x_min), 0.5 * (self.y_max - self.y_min))


class QuadtreeDecomposition:
    """Adaptive spatial quadtree decomposition guided by Sobolev error."""

    def __init__(
        self,
        max_depth: int = 3,
        split_threshold: float = 0.1,
        min_patch_size: int = 4,
        sobolev_mu: float = 0.1,
    ) -> None:
        self.max_depth = max_depth
        self.split_threshold = split_threshold
        self.min_patch_size = min_patch_size
        self.sobolev_mu = sobolev_mu

    def compute_sobolev_error(self, residual_patch: np.ndarray) -> float:
        """
        Compute localized Sobolev residual metric: E = MSE + mu * ||grad R||^2.
        """
        patch = np.asarray(residual_patch, dtype=np.float64)
        if patch.size == 0:
            return 0.0

        mse = float(np.mean(patch ** 2))
        if patch.shape[0] < 2 or patch.shape[1] < 2 or self.sobolev_mu <= 0.0:
            return mse

        # Discrete spatial gradient
        dy, dx = np.gradient(patch)
        grad_norm = float(np.mean(dx * dx + dy * dy))
        return mse + self.sobolev_mu * grad_norm

    def decompose(
        self,
        residual: np.ndarray,
        domain: Optional[Any] = None,
    ) -> QuadtreeNode:
        """
        Recursively decompose residual image into adaptive quadtree leaves.
        """
        res = np.asarray(residual, dtype=np.float64)
        if res.ndim == 3:
            # Grayscale magnitude for multi-channel residuals
            res = np.sqrt(np.mean(res ** 2, axis=-1))

        h, w = res.shape
        if domain is not None:
            bounds = (float(domain.x_min), float(domain.y_min), float(domain.x_max), float(domain.y_max))
        else:
            bounds = (-1.0, -1.0, 1.0, 1.0)

        root = QuadtreeNode(bounds=bounds, depth=0)
        self._subdivide_node(root, res, 0, h, 0, w)
        return root

    def _subdivide_node(
        self,
        node: QuadtreeNode,
        full_res: np.ndarray,
        r0: int,
        r1: int,
        c0: int,
        c1: int,
    ) -> None:
        patch = full_res[r0:r1, c0:c1]
        err = self.compute_sobolev_error(patch)
        node.error = err

        h = r1 - r0
        w = c1 - c0

        # Check stopping criteria
        if (
            node.depth >= self.max_depth
            or err <= self.split_threshold
            or h < 2 * self.min_patch_size
            or w < 2 * self.min_patch_size
        ):
            node.is_leaf = True
            node.children = None
            return

        # Subdivide into 4 quadrants
        node.is_leaf = False
        r_mid = r0 + h // 2
        c_mid = c0 + w // 2

        x_min, y_min, x_max, y_max = node.bounds
        x_mid = 0.5 * (x_min + x_max)
        y_mid = 0.5 * (y_min + y_max)

        # In domain coordinates (matching domain.py where row 0 is y_min and row h is y_max):
        # Top-Left (NW): r0..r_mid, c0..c_mid -> continuous x: x_min..x_mid, y: y_min..y_mid
        # Top-Right (NE): r0..r_mid, c_mid..c1 -> continuous x: x_mid..x_max, y: y_min..y_mid
        # Bottom-Left (SW): r_mid..r1, c0..c_mid -> continuous x: x_min..x_mid, y: y_mid..y_max
        # Bottom-Right (SE): r_mid..r1, c_mid..c1 -> continuous x: x_mid..x_max, y: y_mid..y_max

        nw = QuadtreeNode(bounds=(x_min, y_min, x_mid, y_mid), depth=node.depth + 1)
        ne = QuadtreeNode(bounds=(x_mid, y_min, x_max, y_mid), depth=node.depth + 1)
        sw = QuadtreeNode(bounds=(x_min, y_mid, x_mid, y_max), depth=node.depth + 1)
        se = QuadtreeNode(bounds=(x_mid, y_mid, x_max, y_max), depth=node.depth + 1)

        node.children = [nw, ne, sw, se]

        self._subdivide_node(nw, full_res, r0, r_mid, c0, c_mid)
        self._subdivide_node(ne, full_res, r0, r_mid, c_mid, c1)
        self._subdivide_node(sw, full_res, r_mid, r1, c0, c_mid)
        self._subdivide_node(se, full_res, r_mid, r1, c_mid, c1)

    def get_leaf_nodes(self, node: QuadtreeNode) -> List[QuadtreeNode]:
        """Extract all leaf nodes from quadtree root."""
        leaves: List[QuadtreeNode] = []
        stack = [node]
        while stack:
            curr = stack.pop()
            if curr.is_leaf or not curr.children:
                leaves.append(curr)
            else:
                for child in reversed(curr.children):
                    stack.append(child)
        return leaves
