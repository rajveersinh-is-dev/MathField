"""
mathfield/core/optimize/varpro.py

Stage 4: Separable Nonlinear Least Squares via Golub-Pereyra Variable Projection.
Eliminates linear coefficients analytically via regularized pseudoinverse,
optimizing only nonlinear parameters over the reduced residual manifold:
r(theta) = P_Phi^perp(theta) * y.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Literal, Optional, Tuple

import numpy as np

from mathfield.core.optimize.solvers import solve_linear_lstsq


@dataclass(frozen=True, slots=True)
class VarProResult:
    """Outcome of Variable Projection optimization."""

    theta_opt: np.ndarray
    c_opt: np.ndarray
    residual: np.ndarray
    cost: float
    n_iter: int
    condition_number: float
    converged: bool
    message: str


class VarProOptimizer:
    """
    Separable Nonlinear Least Squares Solver using Golub-Pereyra Variable Projection.
    """

    def __init__(
        self,
        max_iter: int = 100,
        reg_lambda: float = 0.0,
        jacobian_method: Literal["golub_pereyra", "kaufman"] = "golub_pereyra",
        tol_g: float = 1e-7,
        tol_x: float = 1e-7,
        tol_f: float = 1e-7,
    ) -> None:
        self.max_iter = max_iter
        self.reg_lambda = reg_lambda
        self.jacobian_method = jacobian_method
        self.tol_g = tol_g
        self.tol_x = tol_x
        self.tol_f = tol_f

    def fit(
        self,
        basis_fn: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray],
        X: np.ndarray,
        Y: np.ndarray,
        y: np.ndarray,
        theta_init: np.ndarray,
        deriv_fn: Optional[Callable[[np.ndarray, np.ndarray, np.ndarray], List[np.ndarray]]] = None,
        bounds: Optional[List[Tuple[Optional[float], Optional[float]]]] = None,
    ) -> VarProResult:
        """
        Fit separable model y ~ Phi(theta) * c.
        basis_fn(theta, X, Y) -> Phi of shape (N, p).
        deriv_fn(theta, X, Y) -> [dPhi/dtheta_0, ..., dPhi/dtheta_{q-1}], each shape (N, p).
        """
        theta = np.asarray(theta_init, dtype=np.float64).copy()
        q = len(theta)
        y_vec = np.asarray(y, dtype=np.float64).ravel()
        N = y_vec.size

        # Apply bounds to initial theta
        if bounds is not None:
            for i, (lb, ub) in enumerate(bounds):
                if lb is not None:
                    theta[i] = max(theta[i], lb)
                if ub is not None:
                    theta[i] = min(theta[i], ub)

        mu = 1e-3
        nu = 2.0

        def eval_subproblem(th: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
            Phi = basis_fn(th, X, Y)
            c, diag = solve_linear_lstsq(Phi, y_vec, reg_lambda=self.reg_lambda, method="svd")
            res = y_vec - Phi @ c
            cost_val = 0.5 * float(np.sum(res * res))
            cond_val = float(diag["condition_number"])
            return Phi, c, res, cost_val, cond_val

        Phi, c, r, cost, cond = eval_subproblem(theta)
        if q == 0:
            return VarProResult(
                theta_opt=theta,
                c_opt=c,
                residual=r,
                cost=cost,
                n_iter=0,
                condition_number=cond,
                converged=True,
                message="No nonlinear parameters to optimize",
            )
        converged = False
        message = "max_iter reached"

        last_it = -1
        for it in range(self.max_iter):
            last_it = it
            # Compute derivatives of Phi with respect to theta
            dPhi_list: List[np.ndarray] = []
            if deriv_fn is not None:
                dPhi_list = deriv_fn(theta, X, Y)
            else:
                # Central finite differences for dPhi / dtheta_k
                eps = 1e-6
                for k in range(q):
                    th_plus = theta.copy()
                    th_plus[k] += eps
                    th_minus = theta.copy()
                    th_minus[k] -= eps
                    Phi_p = basis_fn(th_plus, X, Y)
                    Phi_m = basis_fn(th_minus, X, Y)
                    dPhi_list.append((Phi_p - Phi_m) / (2.0 * eps))

            # Form reduced Jacobian J_red of shape (N, q)
            # Golub-Pereyra primary term: - P_Phi^perp (dPhi/dtheta_k * c)
            # P_Phi^perp v = v - Phi * (Phi^+ v)
            J_red = np.zeros((N, q), dtype=np.float64)

            # Precompute SVD of Phi for fast orthogonal projections
            U, s, Vt = np.linalg.svd(Phi, full_matrices=False)
            s_inv = np.zeros_like(s)
            active = s > 1e-12
            s_inv[active] = 1.0 / s[active]

            for k in range(q):
                dPhi_k = dPhi_list[k]
                dPhi_c = dPhi_k @ c  # shape (N,)
                # Project onto range of Phi: P_Phi dPhi_c = U (U^T dPhi_c)
                proj = U @ (U.T @ dPhi_c)
                perp = dPhi_c - proj
                col = -perp

                if self.jacobian_method == "golub_pereyra":
                    # Secondary term: - (Phi^+)^T (dPhi_k^T r)
                    # Phi^+ = Vt.T @ diag(s_inv) @ U.T
                    # (Phi^+)^T = U @ diag(s_inv) @ Vt
                    dPhi_T_r = dPhi_k.T @ r
                    sec_term = U @ (s_inv * (Vt @ dPhi_T_r))
                    col -= sec_term

                J_red[:, k] = col

            g = J_red.T @ r
            if np.max(np.abs(g)) < self.tol_g:
                converged = True
                message = "Gradient tolerance met"
                break

            # LM step: (J_red^T J_red + mu * I) delta = - g
            H = J_red.T @ J_red
            diag_H = np.diag(H)
            damped_H = H + mu * np.diag(np.maximum(diag_H, 1e-6))

            try:
                delta = np.linalg.solve(damped_H, -g)
            except np.linalg.LinAlgError:
                delta = np.linalg.lstsq(damped_H, -g, rcond=1e-10)[0]

            if np.linalg.norm(delta) < self.tol_x * (np.linalg.norm(theta) + self.tol_x):
                converged = True
                message = "Step tolerance met"
                break

            theta_cand = theta + delta
            if bounds is not None:
                for i, (lb, ub) in enumerate(bounds):
                    if lb is not None:
                        theta_cand[i] = max(theta_cand[i], lb)
                    if ub is not None:
                        theta_cand[i] = min(theta_cand[i], ub)

            Phi_cand, c_cand, r_cand, cost_cand, cond_cand = eval_subproblem(theta_cand)

            actual_reduction = cost - cost_cand
            denom = float(delta.T @ (mu * delta - g))
            rho = (actual_reduction / denom) if abs(denom) > 1e-15 else 0.0

            if rho > 0:
                theta = theta_cand
                Phi = Phi_cand
                c = c_cand
                r = r_cand
                cost = cost_cand
                cond = cond_cand
                mu *= max(1.0 / 3.0, 1.0 - (2.0 * rho - 1.0) ** 3)
                nu = 2.0
            else:
                mu *= nu
                nu *= 2.0

        return VarProResult(
            theta_opt=theta,
            c_opt=c,
            residual=r,
            cost=cost,
            n_iter=last_it + 1,
            condition_number=cond,
            converged=converged,
            message=message,
        )
