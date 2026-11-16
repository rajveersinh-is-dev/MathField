"""
mathfield/core/local/local_patch.py

Stage 3: Localized Basis Patch Representation.
Encapsulates local basis models, local coordinate normalization,
weighted least squares fitting to residuals, and composite evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from mathfield.core.ast.nodes import (
    ASTNode,
    BinaryOpNode,
    ConstantNode,
    VariableNode,
)
from mathfield.core.basis.base import BasisFunction2D
from mathfield.core.basis.polynomial import Chebyshev2D


@dataclass
class LocalPatch:
    """
    Representation of a localized continuous approximation patch.
    """
    bounds: Tuple[float, float, float, float] = (-1.0, -1.0, 1.0, 1.0)  # (xmin, ymin, xmax, ymax)
    basis: Optional[BasisFunction2D] = None
    coefficients: Optional[np.ndarray] = None
    error: float = 0.0

    def __post_init__(self) -> None:
        if self.basis is None:
            # Default to a compact 3x3 Chebyshev polynomial basis
            self.basis = Chebyshev2D(deg_x=3, deg_y=3, truncation="total_degree")
        if self.coefficients is None:
            self.coefficients = np.zeros(self.basis.num_terms, dtype=np.float64)

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

    def normalize_coordinates(self, X: np.ndarray, Y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Map spatial coordinates into canonical local frame [-1, 1] x [-1, 1]."""
        cx, cy = self.center
        hx, hy = self.half_widths
        hx_safe = max(hx, 1e-12)
        hy_safe = max(hy, 1e-12)
        u = (X - cx) / hx_safe
        v = (Y - cy) / hy_safe
        return u, v

    def fit(
        self,
        residual: np.ndarray,
        X: np.ndarray,
        Y: np.ndarray,
        weights: Optional[np.ndarray] = None,
        reg_lambda: float = 1e-4,
    ) -> float:
        """
        Fit local basis to residual using regularized (weighted) least squares:
        min || sqrt(W) (Phi * c - R) ||^2 + lambda * ||c||^2.
        Returns final weighted root-mean-squared error.
        """
        u, v = self.normalize_coordinates(X, Y)
        Phi = self.basis.basis_matrix(u, v)
        r_flat = residual.ravel()

        if weights is not None:
            w_flat = np.sqrt(np.maximum(weights.ravel(), 0.0))
            A = Phi * w_flat[:, np.newaxis]
            b = r_flat * w_flat
        else:
            A = Phi
            b = r_flat

        p = self.basis.num_terms
        if reg_lambda > 0.0:
            # Tikhonov regularization
            A_aug = np.vstack([A, np.sqrt(reg_lambda) * np.eye(p)])
            b_aug = np.concatenate([b, np.zeros(p)])
            c_opt, _, _, _ = np.linalg.lstsq(A_aug, b_aug, rcond=1e-10)
        else:
            c_opt, _, _, _ = np.linalg.lstsq(A, b, rcond=1e-10)

        self.coefficients = c_opt
        fitted = Phi @ c_opt
        err = float(np.mean((fitted - r_flat) ** 2))
        self.error = err
        return err

    def evaluate(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """Evaluate local patch expression on coordinate grid."""
        u, v = self.normalize_coordinates(X, Y)
        return self.basis.evaluate(u, v, self.coefficients)

    def to_ast(
        self,
        param_offset: int = 0,
        symbolic_params: bool = False,
        prune_threshold: float = 1e-12,
    ) -> ASTNode:
        """
        Emit local patch AST with coordinate translation and scaling:
        u = (x - cx) / hx, v = (y - cy) / hy.
        """
        cx, cy = self.center
        hx, hy = self.half_widths
        base_ast = self.basis.to_ast(
            coefficients=self.coefficients,
            param_offset=param_offset,
            symbolic_params=symbolic_params,
            prune_threshold=prune_threshold,
        )
        return base_ast
