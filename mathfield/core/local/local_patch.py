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
        Supports single-channel (H, W) or multi-channel (H, W, C) residuals.
        Returns final weighted root-mean-squared error.
        """
        u, v = self.normalize_coordinates(X, Y)

        # Check active region if weights provided
        if weights is not None:
            w_flat = weights.ravel()
            active = w_flat > 1e-12
            if np.sum(active) >= self.basis.num_terms:
                u_fit = u.ravel()[active]
                v_fit = v.ravel()[active]
                w_fit = np.sqrt(np.maximum(w_flat[active], 0.0))
                Phi = self.basis.basis_matrix(u_fit, v_fit)
                A = Phi * w_fit[:, np.newaxis]
                if residual.ndim == 3:
                    r_fit = residual.reshape(-1, residual.shape[2])[active, :]
                else:
                    r_fit = residual.ravel()[active]
            else:
                Phi = self.basis.basis_matrix(u, v)
                w_sq = np.sqrt(np.maximum(w_flat, 0.0))
                A = Phi * w_sq[:, np.newaxis]
                if residual.ndim == 3:
                    r_fit = residual.reshape(-1, residual.shape[2])
                else:
                    r_fit = residual.ravel()
                w_fit = w_sq
        else:
            Phi = self.basis.basis_matrix(u, v)
            A = Phi
            w_fit = None
            if residual.ndim == 3:
                r_fit = residual.reshape(-1, residual.shape[2])
            else:
                r_fit = residual.ravel()

        p = self.basis.num_terms
        if residual.ndim == 3:
            c_channels = []
            c = residual.shape[2]
            for ch in range(c):
                b_ch = r_fit[:, ch] * (w_fit if w_fit is not None else 1.0)
                if reg_lambda > 0.0:
                    A_aug = np.vstack([A, np.sqrt(reg_lambda) * np.eye(p)])
                    b_aug = np.concatenate([b_ch, np.zeros(p)])
                    c_opt, _, _, _ = np.linalg.lstsq(A_aug, b_aug, rcond=1e-10)
                else:
                    c_opt, _, _, _ = np.linalg.lstsq(A, b_ch, rcond=1e-10)
                c_channels.append(c_opt)
            self.coefficients = np.column_stack(c_channels)
            fitted = Phi @ self.coefficients
            err = float(np.mean((fitted - r_fit) ** 2))
        else:
            b = r_fit * (w_fit if w_fit is not None else 1.0)
            if reg_lambda > 0.0:
                A_aug = np.vstack([A, np.sqrt(reg_lambda) * np.eye(p)])
                b_aug = np.concatenate([b, np.zeros(p)])
                c_opt, _, _, _ = np.linalg.lstsq(A_aug, b_aug, rcond=1e-10)
            else:
                c_opt, _, _, _ = np.linalg.lstsq(A, b, rcond=1e-10)
            self.coefficients = c_opt
            fitted = Phi @ c_opt
            err = float(np.mean((fitted - r_fit) ** 2))

        self.error = err
        return err

    def evaluate(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """Evaluate local patch expression on coordinate grid."""
        u, v = self.normalize_coordinates(X, Y)
        if self.coefficients is not None and self.coefficients.ndim == 2:
            ch_outs = []
            for ch in range(self.coefficients.shape[1]):
                ch_outs.append(self.basis.evaluate(u, v, self.coefficients[:, ch]))
            return np.stack(ch_outs, axis=-1)
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
        coeffs = (
            self.coefficients[:, 0]
            if (self.coefficients is not None and self.coefficients.ndim == 2)
            else self.coefficients
        )
        base_ast = self.basis.to_ast(
            coefficients=coeffs,
            param_offset=param_offset,
            symbolic_params=symbolic_params,
            prune_threshold=prune_threshold,
        )
        return base_ast
