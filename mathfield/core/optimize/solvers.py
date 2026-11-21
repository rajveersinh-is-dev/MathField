"""
mathfield/core/optimize/solvers.py

Linear and Nonlinear Optimization Solvers:
- Regularized linear least squares (SVD, QR, Cholesky) with condition reporting
- In-process Levenberg-Marquardt optimizer with adaptive damping and gain ratio rho
- Bound-constrained L-BFGS-B wrapper
"""

from __future__ import annotations

import math
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import minimize


def solve_linear_lstsq(
    A: np.ndarray,
    b: np.ndarray,
    reg_lambda: float = 0.0,
    method: str = "svd",
    rcond: float = 1e-12,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Solve regularized linear least squares (A^T A + lambda I) x = A^T b.
    Returns (solution x, diagnostics dict).
    """
    A_mat = np.asarray(A, dtype=np.float64)
    b_vec = np.asarray(b, dtype=np.float64).ravel()
    m, n = A_mat.shape

    # SVD decomposition: A = U * S * Vt
    U, s, Vt = np.linalg.svd(A_mat, full_matrices=False)
    sigma_max = float(s[0]) if s.size > 0 else 0.0
    sigma_min = float(s[-1]) if s.size > 0 else 0.0
    cond = (sigma_max / max(sigma_min, 1e-16)) if sigma_min > 0 else float("inf")

    diag: Dict[str, Any] = {
        "condition_number": cond,
        "singular_values": s,
        "sigma_max": sigma_max,
        "sigma_min": sigma_min,
        "method": method,
    }

    if method == "svd":
        # Tikhonov filtered SVD: x = sum_i (s_i / (s_i^2 + lambda)) * (u_i^T b) * v_i
        threshold = rcond * sigma_max
        filter_factors = s / (s * s + reg_lambda)
        filter_factors[s < threshold] = 0.0
        c = Vt.T @ (filter_factors * (U.T @ b_vec))
        return c, diag

    elif method == "qr":
        if reg_lambda > 0.0:
            A_aug = np.vstack([A_mat, math.sqrt(reg_lambda) * np.eye(n)])
            b_aug = np.concatenate([b_vec, np.zeros(n)])
        else:
            A_aug, b_aug = A_mat, b_vec
        Q, R = np.linalg.qr(A_aug)
        c = np.linalg.solve(R, Q.T @ b_aug)
        return c, diag

    elif method == "cholesky":
        H = A_mat.T @ A_mat
        if reg_lambda > 0.0:
            H += reg_lambda * np.eye(n)
        g = A_mat.T @ b_vec
        try:
            L = np.linalg.cholesky(H)
            y = np.linalg.solve(L, g)
            c = np.linalg.solve(L.T, y)
            return c, diag
        except np.linalg.LinAlgError:
            # Fallback to SVD if Cholesky fails due to non-positive definiteness
            diag["fallback_to_svd"] = True
            return solve_linear_lstsq(A_mat, b_vec, reg_lambda, method="svd", rcond=rcond)
    else:
        raise ValueError(f"Unknown least squares method: {method}")


class LevenbergMarquardtOptimizer:
    """
    In-process Levenberg-Marquardt optimizer with adaptive Marquardt damping
    and gain ratio rho step acceptance.
    """

    def __init__(
        self,
        initial_lambda: float = 1e-3,
        max_iter: int = 100,
        tol_g: float = 1e-7,
        tol_x: float = 1e-7,
        tol_f: float = 1e-7,
    ) -> None:
        self.initial_lambda = initial_lambda
        self.max_iter = max_iter
        self.tol_g = tol_g
        self.tol_x = tol_x
        self.tol_f = tol_f

    def minimize(
        self,
        residual_fn: Callable[[np.ndarray], np.ndarray],
        jacobian_fn: Callable[[np.ndarray], np.ndarray],
        x0: np.ndarray,
        bounds: Optional[List[Tuple[Optional[float], Optional[float]]]] = None,
        callback: Optional[Callable[[int, np.ndarray, float], None]] = None,
    ) -> Dict[str, Any]:
        """
        Solve min 0.5 * ||r(x)||^2.
        """
        x = np.asarray(x0, dtype=np.float64).copy()

        # Apply bounds to initial point
        if bounds is not None:
            for i, (lb, ub) in enumerate(bounds):
                if lb is not None:
                    x[i] = max(x[i], lb)
                if ub is not None:
                    x[i] = min(x[i], ub)

        mu = float(self.initial_lambda)
        nu = 2.0

        r = residual_fn(x)
        cost = 0.5 * float(np.sum(r * r))

        if len(x) == 0:
            return {
                "x": x,
                "cost": cost,
                "residual": r,
                "n_iter": 0,
                "converged": True,
                "message": "No parameters to optimize",
            }

        converged = False
        message = "max_iter reached"

        last_it = -1
        for it in range(self.max_iter):
            last_it = it
            J = jacobian_fn(x)  # shape (m, n)
            g = J.T @ r  # shape (n,)

            if np.max(np.abs(g)) < self.tol_g:
                converged = True
                message = "Gradient tolerance met"
                break

            # Normal equations: (J^T J + mu * I) delta = - g
            H = J.T @ J
            diag_H = np.diag(H)
            damped_H = H + mu * np.diag(np.maximum(diag_H, 1e-6))

            try:
                delta = np.linalg.solve(damped_H, -g)
            except np.linalg.LinAlgError:
                delta = np.linalg.lstsq(damped_H, -g, rcond=1e-10)[0]

            if np.linalg.norm(delta) < self.tol_x * (np.linalg.norm(x) + self.tol_x):
                converged = True
                message = "Step tolerance met"
                break

            x_cand = x + delta
            if bounds is not None:
                for i, (lb, ub) in enumerate(bounds):
                    if lb is not None:
                        x_cand[i] = max(x_cand[i], lb)
                    if ub is not None:
                        x_cand[i] = min(x_cand[i], ub)

            r_cand = residual_fn(x_cand)
            cost_cand = 0.5 * float(np.sum(r_cand * r_cand))

            # Gain ratio rho
            actual_reduction = cost - cost_cand
            # Predicted reduction of the *damped* subproblem that was solved:
            # 0.5*d'H*d + d'D*d, where D = mu*max(diag(H), 1e-6).
            damping = mu * np.maximum(diag_H, 1e-6)
            denom = 0.5 * float(delta.T @ (H @ delta)) + float(delta.T @ (damping * delta))
            rho = (actual_reduction / denom) if abs(denom) > 1e-15 else 0.0

            if rho > 0:
                # Successful step: accept update and decrease damping
                x = x_cand
                r = r_cand
                cost = cost_cand
                mu *= max(1.0 / 3.0, 1.0 - (2.0 * rho - 1.0) ** 3)
                nu = 2.0
                if callback is not None:
                    callback(it, x, cost)
            else:
                # Failed step: reject update and increase damping
                mu *= nu
                nu *= 2.0

        return {
            "x": x,
            "cost": cost,
            "residual": r,
            "n_iter": last_it + 1,
            "converged": converged,
            "message": message,
        }


class LBFGSBOptimizer:
    """
    Bound-constrained L-BFGS-B optimizer wrapping scipy.optimize.minimize.
    """

    def __init__(
        self,
        bounds: Optional[List[Tuple[Optional[float], Optional[float]]]] = None,
        max_iter: int = 100,
        tol: float = 1e-7,
    ) -> None:
        self.bounds = bounds
        self.max_iter = max_iter
        self.tol = tol

    def minimize(
        self,
        fun: Callable[[np.ndarray], float],
        x0: np.ndarray,
        jac: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        callback: Optional[Callable[[np.ndarray], None]] = None,
    ) -> Any:
        res = minimize(
            fun=fun,
            x0=x0,
            jac=jac,
            method="L-BFGS-B",
            bounds=self.bounds,
            callback=callback,
            options={"maxiter": self.max_iter, "ftol": self.tol, "gtol": self.tol},
        )
        return res
