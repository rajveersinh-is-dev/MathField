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

        # Phi_fit = Q R. Every candidate is a column subset of the same R, so
        # only the component of y_fit orthogonal to Q is a fixed RSS offset.
        q_mat, r_mat = np.linalg.qr(Phi_fit, mode="reduced")
        projected_y = q_mat.T @ y_fit
        orthogonal_res = y_fit - q_mat @ projected_y
        fixed_rss = float(orthogonal_res @ orthogonal_res)
        rcond = 1e-12

        def rank_needs_dense_fallback(diagnostics: dict) -> bool:
            singular_values = diagnostics["singular_values"]
            sigma_max = float(diagnostics["sigma_max"])
            if singular_values.size == 0:
                return False
            if sigma_max == 0.0:
                return True
            # Include numerical rank deficiency as well as the band where QR
            # roundoff could move a singular value across the SVD cutoff.
            return bool(np.any(singular_values < 2.0 * rcond * sigma_max))

        def evaluate_model(columns: List[int]) -> tuple[np.ndarray, float, bool]:
            design = r_mat[:, columns]
            response = projected_y
            rss_offset = fixed_rss
            coefficients, diagnostics = solve_linear_lstsq(
                design,
                response,
                reg_lambda=reg_lambda,
                method="svd",
                rcond=rcond,
            )
            residual = response - design @ coefficients
            rss = rss_offset + float(residual @ residual)
            bic = self.compute_bic(rss, n_samples, len(columns))
            return coefficients, bic, rank_needs_dense_fallback(diagnostics)

        def dense_fallback() -> PruningResult:
            """Replay the original dense elimination from the initial model."""
            dense_active = list(range(K))
            dense_c, _ = solve_linear_lstsq(Phi_fit, y_fit, reg_lambda=reg_lambda, method="svd", rcond=rcond)
            res_init = y_fit - Phi_fit @ dense_c
            rss_init = float(np.sum(res_init**2))
            mse_init = rss_init / weight_sum
            bic_init = self.compute_bic(rss_init, n_samples, len(dense_active))
            curr_bic = bic_init

            while len(dense_active) > 1:
                best_cand_bic = float("inf")
                best_cand_idx = -1
                best_cand_c: Optional[np.ndarray] = None
                for term_idx in dense_active:
                    cand_sub = [idx for idx in dense_active if idx != term_idx]
                    phi_sub = Phi_fit[:, cand_sub]
                    c_sub, _ = solve_linear_lstsq(
                        phi_sub,
                        y_fit,
                        reg_lambda=reg_lambda,
                        method="svd",
                        rcond=rcond,
                    )
                    res_sub = y_fit - phi_sub @ c_sub
                    rss_sub = float(np.sum(res_sub**2))
                    cand_bic = self.compute_bic(rss_sub, n_samples, len(cand_sub))
                    if cand_bic < best_cand_bic:
                        best_cand_bic = cand_bic
                        best_cand_idx = term_idx
                        best_cand_c = c_sub

                if best_cand_bic < curr_bic and best_cand_idx >= 0:
                    dense_active.remove(best_cand_idx)
                    curr_bic = best_cand_bic
                    dense_c = best_cand_c
                else:
                    break

            assert dense_c is not None
            dense_eliminated = [idx for idx in range(K) if idx not in dense_active]
            full_c = np.zeros(K, dtype=np.float64)
            full_c[dense_active] = dense_c
            res_final = y_fit - Phi_fit @ full_c
            mse_final = float(np.sum(res_final**2) / weight_sum)
            return PruningResult(
                retained_indices=dense_active,
                eliminated_indices=dense_eliminated,
                initial_bic=bic_init,
                final_bic=curr_bic,
                initial_mse=mse_init,
                final_mse=mse_final,
                coefficients=full_c,
            )

        # The reduced solve preserves the dense SVD cutoff, ridge filter, and
        # minimum-norm solution. Rank-deficient or cutoff-adjacent systems use
        # the original dense path because their elimination ties are sensitive
        # to roundoff.
        curr_c, curr_bic, rank_ambiguous = evaluate_model(active_indices)
        if rank_ambiguous:
            return dense_fallback()
        res_init = projected_y - r_mat @ curr_c
        rss_init = fixed_rss + float(np.sum(res_init**2))
        mse_init = rss_init / weight_sum
        bic_init = curr_bic

        def select_candidate() -> tuple[int, np.ndarray, float, float, bool]:
            best_cand_bic = float("inf")
            runner_up_bic = float("inf")
            best_cand_idx = -1
            best_cand_c: Optional[np.ndarray] = None

            for term_idx in active_indices:
                cand_sub = [idx for idx in active_indices if idx != term_idx]
                c_sub, cand_bic, cand_ambiguous = evaluate_model(cand_sub)
                if cand_ambiguous:
                    return -1, np.empty(0, dtype=np.float64), float("inf"), float("inf"), True

                if cand_bic < best_cand_bic:
                    runner_up_bic = best_cand_bic
                    best_cand_bic = cand_bic
                    best_cand_idx = term_idx
                    best_cand_c = c_sub
                elif cand_bic < runner_up_bic:
                    runner_up_bic = cand_bic

            assert best_cand_c is not None
            return best_cand_idx, best_cand_c, best_cand_bic, runner_up_bic, False

        def numerically_near(first: float, second: float) -> bool:
            if not math.isfinite(first) or not math.isfinite(second):
                return False
            tolerance = 1e-10 * max(1.0, abs(first), abs(second))
            return abs(first - second) <= tolerance

        while len(active_indices) > 1:
            best_cand_idx, best_cand_c, best_cand_bic, runner_up_bic, ambiguous = select_candidate()

            # Resolve close candidate decisions, and close accept/reject
            # decisions, by replaying the original dense-SVD elimination from
            # its initial model so earlier tie semantics cannot diverge.
            if ambiguous or numerically_near(best_cand_bic, runner_up_bic) or numerically_near(best_cand_bic, curr_bic):
                return dense_fallback()

            # Check if dropping best_cand_idx improves BIC
            if best_cand_bic < curr_bic and best_cand_idx >= 0:
                active_indices.remove(best_cand_idx)
                curr_bic = best_cand_bic
                curr_c = best_cand_c
            else:
                # No further improvement
                break

        eliminated = [idx for idx in range(K) if idx not in active_indices]

        # Refit the retained model once on the original rows. This keeps final
        # minimum-norm coefficients identical to the dense-SVD implementation
        # even for extremely ill-scaled columns, without slowing candidate
        # selection. Reconstruct zeros on pruned terms as before.
        active_phi = Phi_fit[:, active_indices]
        curr_c, _ = solve_linear_lstsq(
            active_phi,
            y_fit,
            reg_lambda=reg_lambda,
            method="svd",
            rcond=rcond,
        )
        full_c = np.zeros(K, dtype=np.float64)
        full_c[active_indices] = curr_c

        res_final = y_fit - Phi_fit @ full_c
        rss_final = float(np.sum(res_final**2))
        mse_final = rss_final / weight_sum
        # Score the retained system itself, as candidate scoring did, rather
        # than a full-width product with zeros that can round differently when
        # the residual is near machine precision.
        bic_design = Phi_fit if len(active_indices) == K else active_phi
        res_bic = y_fit - bic_design @ curr_c
        bic_final = self.compute_bic(float(np.sum(res_bic**2)), n_samples, len(active_indices))
        if len(active_indices) == K:
            # With no accepted deletion, initial and final are the same model.
            bic_init = bic_final
            mse_init = mse_final

        return PruningResult(
            retained_indices=active_indices,
            eliminated_indices=eliminated,
            initial_bic=bic_init,
            final_bic=bic_final,
            initial_mse=mse_init,
            final_mse=mse_final,
            coefficients=full_c,
        )
