"""
mathfield/core/basis/rational.py

2D Rational Function Expansion R(x, y) = P(x, y) / Q(x, y):
- Regularized positive denominator parameterization Q(x, y) = 1 + (Q_lin)^2 >= 1.0
- Regularized division protecting against poles and division-by-zero
- Sanathanan-Koerner (SK) iterative linear least-squares fitting
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from mathfield.core.ast.nodes import (
    ASTNode,
    BinaryOpNode,
    ConstantNode,
    ParameterNode,
    VariableNode,
)
from mathfield.core.basis.base import BasisFunction2D
from mathfield.core.basis.polynomial import _build_balanced_addition


class Rational2D(BasisFunction2D):
    """
    Bivariate Rational Function:
    R(x, y) = P(x, y) / Q(x, y)
    Where P(x, y) is polynomial with numerator coefficients,
    and Q(x, y) = 1.0 + (Q_lin(x, y))^2 >= 1.0 or regularized 1.0 + den(x, y).
    Guarantees finite evaluations without asymptotes or division-by-zero.
    """

    def __init__(
        self,
        num_coeffs: Optional[np.ndarray] = None,
        den_coeffs: Optional[np.ndarray] = None,
        deg_x: int = 1,
        deg_y: int = 1,
        positive_denominator: bool = True,
    ) -> None:
        self.deg_x = deg_x
        self.deg_y = deg_y
        self.positive_denominator = positive_denominator

        # Build monomial index list (i, j)
        self.indices: List[Tuple[int, int]] = []
        for i in range(deg_x + 1):
            for j in range(deg_y + 1):
                self.indices.append((i, j))

        k_terms = len(self.indices)

        if num_coeffs is not None:
            self.num_coeffs = np.asarray(num_coeffs, dtype=np.float64).ravel()
        else:
            self.num_coeffs = np.zeros(k_terms, dtype=np.float64)
            self.num_coeffs[0] = 1.0

        if den_coeffs is not None:
            self.den_coeffs = np.asarray(den_coeffs, dtype=np.float64).ravel()
        else:
            self.den_coeffs = np.zeros(k_terms, dtype=np.float64)

    @property
    def num_terms(self) -> int:
        return len(self.num_coeffs) + len(self.den_coeffs)

    def _eval_monomials(self, X: np.ndarray, Y: np.ndarray, n_terms: int) -> np.ndarray:
        x_flat = X.ravel()
        y_flat = Y.ravel()
        N = x_flat.size

        Phi = np.empty((N, n_terms), dtype=np.float64)
        for k in range(n_terms):
            if k < len(self.indices):
                i, j = self.indices[k]
            else:
                i, j = 0, 0
            term = np.ones(N, dtype=np.float64)
            if i > 0:
                term *= x_flat**i
            if j > 0:
                term *= y_flat**j
            Phi[:, k] = term
        return Phi

    def basis_matrix(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """Evaluates numerator basis matrix."""
        return self._eval_monomials(X, Y, len(self.num_coeffs))

    def evaluate(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        coefficients: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        if coefficients is not None:
            c = np.asarray(coefficients, dtype=np.float64).ravel()
            n_num = len(self.num_coeffs)
            num_c = c[:n_num]
            den_c = c[n_num:] if len(c) > n_num else self.den_coeffs
        else:
            num_c = self.num_coeffs
            den_c = self.den_coeffs

        Phi_num = self._eval_monomials(X, Y, len(num_c))
        P = Phi_num @ num_c

        Phi_den = self._eval_monomials(X, Y, len(den_c))
        Q_lin = Phi_den @ den_c

        if self.positive_denominator:
            Q = 1.0 + (Q_lin**2)
        else:
            Q = 1.0 + Q_lin
            # Protected division
            eps = 1e-12
            Q = np.where(np.abs(Q) < eps, np.copysign(eps, Q) + (Q == 0) * eps, Q)

        vals = P / Q
        return vals.reshape(X.shape)

    def fit_sanathanan_koerner(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        target: np.ndarray,
        max_iter: int = 5,
        tol: float = 1e-6,
    ) -> None:
        """
        Fit the rational model that evaluate()/to_ast() actually implement.

        For the pole-free parameterization Q = 1 + q^2 the model is nonlinear in
        the denominator coefficients. We use the Sanathanan-Koerner weighting
        together with a first-order expansion of Q around the previous iterate:

            Q(q) ~= 1 - q_prev^2 + 2 * q_prev * q

        so each step is still a linear least-squares problem in [a; b], while
        the fitted coefficients remain consistent with evaluation and export.
        """
        y_flat = np.asarray(target, dtype=np.float64).ravel()
        n_num = len(self.num_coeffs)
        n_den = len(self.den_coeffs)

        Phi_num = self._eval_monomials(X, Y, n_num)
        Phi_den = self._eval_monomials(X, Y, n_den)

        q_lin = Phi_den @ self.den_coeffs
        # A zero initial linear denominator makes all denominator columns of
        # the linearized system vanish. Seed it with a small deterministic
        # perturbation so the denominator is identifiable from the first step.
        if np.max(np.abs(q_lin)) < 1e-12:
            q_lin = np.full(y_flat.size, 0.05, dtype=np.float64)

        for _ in range(max_iter):
            q_prev = q_lin
            q_prev_sq = q_prev * q_prev
            q_den = np.maximum(1.0 + q_prev_sq, 1e-6)

            if self.positive_denominator:
                # Q = 1 + q^2 is nonlinear in b, so solve the weighted
                # nonlinear least-squares problem directly. The weights are
                # refreshed from the previous SK iterate (iteratively
                # reweighted least squares), matching the SK objective.
                from scipy.optimize import least_squares

                weights = np.sqrt(q_den)

                def residual(theta: np.ndarray, w: np.ndarray = weights) -> np.ndarray:
                    q = Phi_den @ theta[n_num:]
                    model_q = 1.0 + q * q
                    return (Phi_num @ theta[:n_num] - y_flat * model_q) / w

                theta0 = np.concatenate([self.num_coeffs, self.den_coeffs])
                sol = least_squares(residual, theta0, max_nfev=200, xtol=tol, ftol=tol, gtol=tol)
                self.num_coeffs = sol.x[:n_num]
                self.den_coeffs = sol.x[n_num:]
            else:
                # Linear denominator: standard SK step
                # min ||(Phi_num a - y Phi_den b - y) / Q_prev||^2.
                q_abs = np.maximum(np.abs(1.0 + q_lin), 1e-6)
                A_num = Phi_num / q_abs[:, None]
                A_den = -(Phi_den * y_flat[:, None]) / q_abs[:, None]
                b_vec = y_flat / q_abs

                sol, _, _, _ = np.linalg.lstsq(np.hstack([A_num, A_den]), b_vec, rcond=1e-10)
                self.num_coeffs = sol[:n_num]
                self.den_coeffs = sol[n_num:]

            q_new = Phi_den @ self.den_coeffs
            if np.max(np.abs(q_new - q_lin)) < tol:
                q_lin = q_new
                break
            q_lin = q_new

    def to_ast(
        self,
        coefficients: Optional[np.ndarray] = None,
        param_offset: int = 0,
        symbolic_params: bool = False,
        prune_threshold: float = 1e-12,
    ) -> ASTNode:
        if coefficients is not None:
            c = np.asarray(coefficients, dtype=np.float64).ravel()
            n_num = len(self.num_coeffs)
            num_c = c[:n_num]
            den_c = c[n_num:] if len(c) > n_num else self.den_coeffs
        else:
            num_c = self.num_coeffs
            den_c = self.den_coeffs

        var_x = VariableNode("x")
        var_y = VariableNode("y")

        def _build_poly_ast(coeffs: np.ndarray, offset: int) -> ASTNode:
            nodes: List[ASTNode] = []
            for k in range(len(coeffs)):
                val = float(coeffs[k])
                if abs(val) < prune_threshold and not symbolic_params:
                    continue
                p_node: ASTNode = ParameterNode(f"c{offset + k}", offset + k) if symbolic_params else ConstantNode(val)
                i, j = self.indices[k] if k < len(self.indices) else (0, 0)
                term = p_node
                if i > 0:
                    term = BinaryOpNode("*", term, BinaryOpNode("^", var_x, ConstantNode(float(i))))
                if j > 0:
                    term = BinaryOpNode("*", term, BinaryOpNode("^", var_y, ConstantNode(float(j))))
                nodes.append(term)
            return _build_balanced_addition(nodes)

        p_ast = _build_poly_ast(num_c, param_offset)
        q_lin_ast = _build_poly_ast(den_c, param_offset + len(num_c))

        if self.positive_denominator:
            q_ast = BinaryOpNode("+", ConstantNode(1.0), BinaryOpNode("^", q_lin_ast, ConstantNode(2.0)))
        else:
            q_ast = BinaryOpNode("+", ConstantNode(1.0), q_lin_ast)

        return BinaryOpNode("/", p_ast, q_ast)

    def get_complexity(self, coefficients: Optional[np.ndarray] = None) -> int:
        num_active = sum(1 for v in self.num_coeffs if abs(v) > 1e-12)
        den_active = sum(1 for v in self.den_coeffs if abs(v) > 1e-12)
        return (num_active + den_active) * 4 + 2
