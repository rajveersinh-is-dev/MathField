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
        sample_weights: Optional[np.ndarray] = None,
    ) -> float:
        """
        Fit local basis to residual using regularized (weighted) least squares:
        min || sqrt(W) (Phi * c - R) ||^2 + lambda * ||c||^2.
        Supports single-channel (H, W) or multi-channel (H, W, C) residuals.
        Returns final weighted root-mean-squared error.
        """
        u, v = self.normalize_coordinates(X, Y)

        if weights is None:
            active = np.ones(u.size, dtype=bool)
            base_sqrt_weights = np.ones(u.size, dtype=np.float64)
        else:
            weight_flat = np.asarray(weights, dtype=np.float64).ravel()
            if weight_flat.size != u.size:
                raise ValueError("Weight count must match the coordinate grid")
            if not np.all(np.isfinite(weight_flat)) or np.any(weight_flat < 0.0):
                raise ValueError("Weights must be finite and non-negative")
            active = weight_flat > 1e-12
            base_sqrt_weights = np.sqrt(np.maximum(weight_flat, 0.0))

        use_active = int(np.count_nonzero(active)) >= self.basis.num_terms
        if use_active:
            Phi = self.basis.basis_matrix(u.ravel()[active], v.ravel()[active])
            base_fit = base_sqrt_weights[active]
            if residual.ndim == 3:
                r_fit = residual.reshape(-1, residual.shape[2])[active, :]
            else:
                r_fit = residual.ravel()[active]
        else:
            Phi = self.basis.basis_matrix(u, v)
            base_fit = base_sqrt_weights
            if residual.ndim == 3:
                r_fit = residual.reshape(-1, residual.shape[2])
            else:
                r_fit = residual.ravel()

        sample_sqrt_weights = None
        if sample_weights is not None:
            sample = np.asarray(sample_weights, dtype=np.float64)
            if sample.ndim == 2:
                sample = sample[:, :, np.newaxis]
            if sample.ndim != 3 or sample.shape[:2] != residual.shape[:2]:
                raise ValueError("Sample weights must match the residual spatial shape")
            expected_channels = residual.shape[2] if residual.ndim == 3 else 1
            if sample.shape[2] not in (1, expected_channels):
                raise ValueError("Sample weight channel count must be one or match the residual channels")
            if not np.all(np.isfinite(sample)) or np.any(sample < 0.0):
                raise ValueError("Sample weights must be finite and non-negative")
            sample_flat = sample.reshape(-1, sample.shape[2])
            sample_fit = sample_flat[active] if use_active else sample_flat
            sample_sqrt_weights = base_fit[:, np.newaxis] * np.sqrt(sample_fit)

        p = self.basis.num_terms
        if residual.ndim == 3:
            c = residual.shape[2]
            if sample_sqrt_weights is None:
                row_weights = base_fit
                design = Phi * row_weights[:, np.newaxis]
                targets = r_fit * row_weights[:, np.newaxis]
                if reg_lambda > 0.0:
                    design = np.vstack([design, np.sqrt(reg_lambda) * np.eye(p)])
                    targets = np.vstack([targets, np.zeros((p, c))])
                c_opt, _, _, _ = np.linalg.lstsq(design, targets, rcond=1e-10)
            elif sample_sqrt_weights.shape[1] == 1 or np.array_equal(sample_sqrt_weights, sample_sqrt_weights[:, :1]):
                row_weights = sample_sqrt_weights[:, 0]
                design = Phi * row_weights[:, np.newaxis]
                targets = r_fit * row_weights[:, np.newaxis]
                if reg_lambda > 0.0:
                    design = np.vstack([design, np.sqrt(reg_lambda) * np.eye(p)])
                    targets = np.vstack([targets, np.zeros((p, c))])
                c_opt, _, _, _ = np.linalg.lstsq(design, targets, rcond=1e-10)
            else:
                channels = []
                for ch in range(c):
                    row_weights = sample_sqrt_weights[:, ch]
                    design = Phi * row_weights[:, np.newaxis]
                    targets = r_fit[:, ch] * row_weights
                    if reg_lambda > 0.0:
                        design = np.vstack([design, np.sqrt(reg_lambda) * np.eye(p)])
                        targets = np.concatenate([targets, np.zeros(p)])
                    coefficient, _, _, _ = np.linalg.lstsq(design, targets, rcond=1e-10)
                    channels.append(coefficient)
                c_opt = np.column_stack(channels)
            self.coefficients = c_opt
            fitted = Phi @ c_opt
            err = float(np.mean((fitted - r_fit) ** 2))
        else:
            row_weights = sample_sqrt_weights[:, 0] if sample_sqrt_weights is not None else base_fit
            design = Phi * row_weights[:, np.newaxis]
            targets = r_fit * row_weights
            if reg_lambda > 0.0:
                design = np.vstack([design, np.sqrt(reg_lambda) * np.eye(p)])
                targets = np.concatenate([targets, np.zeros(p)])
            c_opt, _, _, _ = np.linalg.lstsq(design, targets, rcond=1e-10)
            self.coefficients = c_opt
            fitted = Phi @ c_opt
            err = float(np.mean((fitted - r_fit) ** 2))

        self.error = err
        return err

    def evaluate(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """Evaluate local patch expression on coordinate grid."""
        u, v = self.normalize_coordinates(X, Y)
        coefficients = self.coefficients
        values = self.basis.basis_matrix(u, v) @ coefficients
        if coefficients.ndim == 2:
            return values.reshape(*u.shape, coefficients.shape[1])
        return values.reshape(u.shape)

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
