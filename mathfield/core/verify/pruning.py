"""
mathfield/core/verify/pruning.py

Stage 5: Complexity Pruning via Bayesian Information Criterion (BIC).
Performs backward elimination to prune insignificant basis terms
without degrading mathematical reconstruction quality.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from mathfield.core.optimize.solvers import solve_linear_lstsq


@dataclass(frozen=True, slots=True)
class PruningResult:
    """Outcome of BIC backward elimination pruning."""

    retained_indices: List[int]
    eliminated_indices: List[int]
    initial_bic: float
    final_bic: float
    initial_mse: float
    final_mse: float
    coefficients: np.ndarray


class BICPruner:
    """
    Bayesian Information Criterion (BIC) parameter pruner.
    """

    @staticmethod
    def compute_bic(rss: float, n_samples: int, k_parameters: int) -> float:
        """
        Compute standard Bayesian Information Criterion:
        BIC = n * ln(RSS / n) + k * ln(n)
        """
        n = max(int(n_samples), 1)
        k = max(int(k_parameters), 1)
        variance = max(float(rss) / float(n), 1e-15)
        return float(n * math.log(variance) + k * math.log(n))

    def prune_matrix(
        self,
        Phi: np.ndarray,
        target: np.ndarray,
        reg_lambda: float = 1e-6,
        weights: Optional[np.ndarray] = None,
    ) -> PruningResult:
        """
        Execute backward elimination on design matrix Phi of shape (N, K).
        """
        Phi_mat = np.asarray(Phi, dtype=np.float64)
        y = np.asarray(target, dtype=np.float64).ravel()
        N, K = Phi_mat.shape
        if weights is None:
            weight_vec = np.ones(N, dtype=np.float64)
        else:
            weight_vec = np.asarray(weights, dtype=np.float64).ravel()
            if weight_vec.shape[0] != N:
                raise ValueError("Weight count must match the design matrix rows")
            if not np.all(np.isfinite(weight_vec)) or np.any(weight_vec < 0.0):
                raise ValueError("Weights must be finite and non-negative")
        weight_sum = float(np.sum(weight_vec))
        n_samples = int(np.count_nonzero(weight_vec > 0.0))
        if weight_sum <= 0.0:
            return PruningResult(list(range(K)), [], 0.0, 0.0, 0.0, 0.0, np.zeros(K, dtype=np.float64))

        sqrt_weights = np.sqrt(weight_vec)
        Phi_fit = Phi_mat * sqrt_weights[:, np.newaxis]
        y_fit = y * sqrt_weights
        active_indices: List[int] = list(range(K))

        # Solve initial model
        c_init, _ = solve_linear_lstsq(Phi_fit, y_fit, reg_lambda=reg_lambda, method="svd")
        res_init = y_fit - Phi_fit @ c_init
        rss_init = float(np.sum(res_init**2))
        mse_init = rss_init / weight_sum
        bic_init = self.compute_bic(rss_init, n_samples, len(active_indices))

        curr_bic = bic_init
        curr_c = c_init

        while len(active_indices) > 1:
            best_cand_bic = float("inf")
            best_cand_idx = -1
            best_cand_c: Optional[np.ndarray] = None

            for term_idx in active_indices:
                cand_sub = [idx for idx in active_indices if idx != term_idx]
                Phi_sub = Phi_fit[:, cand_sub]

                c_sub, _ = solve_linear_lstsq(Phi_sub, y_fit, reg_lambda=reg_lambda, method="svd")
                res_sub = y_fit - Phi_sub @ c_sub
                rss_sub = float(np.sum(res_sub**2))
                cand_bic = self.compute_bic(rss_sub, n_samples, len(cand_sub))

                if cand_bic < best_cand_bic:
                    best_cand_bic = cand_bic
                    best_cand_idx = term_idx
                    best_cand_c = c_sub

            # Check if dropping best_cand_idx improves BIC
            if best_cand_bic < curr_bic and best_cand_idx >= 0:
                active_indices.remove(best_cand_idx)
                curr_bic = best_cand_bic
                curr_c = best_cand_c
            else:
                # No further improvement
                break

        eliminated = [idx for idx in range(K) if idx not in active_indices]

        # Reconstruct full-length coefficients array with zeros on pruned terms
        full_c = np.zeros(K, dtype=np.float64)
        for i, idx in enumerate(active_indices):
            full_c[idx] = curr_c[i]

        res_final = y_fit - Phi_fit @ full_c
        mse_final = float(np.sum(res_final**2) / weight_sum)

        return PruningResult(
            retained_indices=active_indices,
            eliminated_indices=eliminated,
            initial_bic=bic_init,
            final_bic=curr_bic,
            initial_mse=mse_init,
            final_mse=mse_final,
            coefficients=full_c,
        )
