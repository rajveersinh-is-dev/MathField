"""
mathfield/core/optimize/conditioning.py

Real-Time SVD Condition Number Monitoring and Adaptive Regularization.
Protects least-squares and VarPro solvers from numerical blow-up on collinear bases.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict

import numpy as np


@dataclass(frozen=True, slots=True)
class ConditionReport:
    """Quantitative matrix conditioning diagnostic report."""

    condition_number: float
    max_singular_value: float
    min_singular_value: float
    effective_rank: int
    is_well_conditioned: bool
    is_severely_ill_conditioned: bool
    is_singular: bool
    recommended_tikhonov: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "condition_number": self.condition_number,
            "max_singular_value": self.max_singular_value,
            "min_singular_value": self.min_singular_value,
            "effective_rank": self.effective_rank,
            "is_well_conditioned": self.is_well_conditioned,
            "is_severely_ill_conditioned": self.is_severely_ill_conditioned,
            "is_singular": self.is_singular,
            "recommended_tikhonov": self.recommended_tikhonov,
        }


class ConditionMonitor:
    """
    Monitors SVD condition number kappa(A) = sigma_max / sigma_min and stabilizes matrices.
    """

    def __init__(
        self,
        max_condition_number: float = 1e10,
        tikhonov_alpha: float = 1e-4,
    ) -> None:
        self.max_condition_number = max_condition_number
        self.tikhonov_alpha = tikhonov_alpha

    def analyze_matrix(self, A: np.ndarray) -> Dict[str, Any]:
        """
        Analyze singular values and condition number of matrix A.
        Returns diagnostic dictionary.
        """
        A_mat = np.asarray(A, dtype=np.float64)
        if A_mat.size == 0:
            return {
                "condition_number": 1.0,
                "is_ill_conditioned": False,
                "max_singular_value": 0.0,
                "min_singular_value": 0.0,
                "effective_rank": 0,
            }

        s = np.linalg.svd(A_mat, compute_uv=False)
        s_max = float(s[0]) if s.size > 0 else 0.0
        s_min = float(s[-1]) if s.size > 0 else 0.0

        if s_min > 1e-16:
            cond = float(s_max / s_min)
        else:
            cond = float("inf")

        is_ill = (cond >= self.max_condition_number) or math.isinf(cond) or math.isnan(cond)
        rank = int(np.sum(s > 1e-14 * max(s_max, 1e-12)))

        # Recommended Tikhonov damping to cap kappa <= 1e8
        rec_lambda = max(1e-12 * (s_max**2), (s_max / 1e8) ** 2)

        return {
            "condition_number": cond,
            "is_ill_conditioned": is_ill,
            "max_singular_value": s_max,
            "min_singular_value": s_min,
            "effective_rank": rank,
            "recommended_tikhonov": rec_lambda,
            "singular_values": s,
        }

    def regularize(self, A: np.ndarray) -> np.ndarray:
        """
        Stabilize matrix A by appending Tikhonov augmentation:
        A_reg = [A; sqrt(alpha) * I_n]
        Guarantees kappa(A_reg) is safely bounded.
        """
        A_mat = np.asarray(A, dtype=np.float64)
        m, n = A_mat.shape
        alpha = self.tikhonov_alpha
        damping = math.sqrt(alpha) * np.eye(n, dtype=np.float64)
        A_reg = np.vstack([A_mat, damping])
        return A_reg

    def solve_tsvd(
        self,
        A: np.ndarray,
        b: np.ndarray,
        rcond: float = 1e-10,
    ) -> np.ndarray:
        """Truncated SVD pseudoinverse solution discarding small singular values."""
        A_mat = np.asarray(A, dtype=np.float64)
        b_vec = np.asarray(b, dtype=np.float64).ravel()

        U, s, Vt = np.linalg.svd(A_mat, full_matrices=False)
        s_max = s[0] if s.size > 0 else 0.0
        cutoff = rcond * s_max
        active = s >= cutoff

        s_inv = np.zeros_like(s)
        s_inv[active] = 1.0 / s[active]

        c = Vt.T @ (s_inv * (U.T @ b_vec))
        return c
