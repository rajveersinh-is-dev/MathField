"""
mathfield/core/basis/splines.py

2D Bicubic Tensor-Product B-Spline Surface Representation:
- Evaluated via Cox-de Boor recurrence with partition-of-unity preservation
- Numerical stability: pure convex combinations without cancellation errors
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from mathfield.core.ast.nodes import (
    ASTNode,
    BinaryOpNode,
    ConstantNode,
    FunctionCallNode,
    ParameterNode,
    VariableNode,
)
from mathfield.core.basis.base import BasisFunction2D
from mathfield.core.basis.polynomial import _build_balanced_addition


def _cox_de_boor_1d(u: np.ndarray, knots: np.ndarray, degree: int) -> np.ndarray:
    """
    Evaluate all 1D B-spline basis functions B_{i, degree}(u) for i = 0 .. n_basis-1
    using standard numerically stable Cox-de Boor recursion.
    Returns array of shape (u.size, n_basis).
    """
    u_flat = np.asarray(u, dtype=np.float64).ravel()
    m = len(knots) - 1
    p = degree
    n_basis = m - p
    if n_basis <= 0:
        raise ValueError(f"Number of knots {len(knots)} is too small for degree {degree}")

    n_pts = u_flat.size
    # Level 0 basis: B_{i, 0}(u)
    # B[i] is of shape (n_pts,)
    # Find the last interval with non-zero length
    non_deg = [i for i in range(m) if knots[i + 1] > knots[i] + 1e-12]
    last_valid_i = non_deg[-1] if non_deg else m - 1

    B = [np.zeros(n_pts, dtype=np.float64) for _ in range(m)]
    for i in range(m):
        if i == last_valid_i:
            mask = (u_flat >= knots[i]) & (u_flat <= knots[i + 1] + 1e-12)
        else:
            mask = (u_flat >= knots[i]) & (u_flat < knots[i + 1])
        B[i][mask] = 1.0

    # Recurse up to degree p
    for d in range(1, p + 1):
        B_next = []
        for i in range(m - d):
            term1 = np.zeros(n_pts, dtype=np.float64)
            term2 = np.zeros(n_pts, dtype=np.float64)

            d1 = knots[i + d] - knots[i]
            if d1 > 1e-12:
                term1 = ((u_flat - knots[i]) / d1) * B[i]

            d2 = knots[i + d + 1] - knots[i + 1]
            if d2 > 1e-12:
                term2 = ((knots[i + d + 1] - u_flat) / d2) * B[i + 1]

            B_next.append(term1 + term2)
        B = B_next

    # B has length m - p == n_basis
    out = np.column_stack(B[:n_basis])
    return out


class BSplineSurface2D(BasisFunction2D):
    """
    Tensor-product B-spline surface over 2D knot vectors:
    S(x, y) = sum_{i=0}^{Nx-1} sum_{j=0}^{Ny-1} c_{i, j} * B_{i, p}(x) * B_{j, p}(y)
    """

    def __init__(
        self,
        knots_x: np.ndarray,
        knots_y: np.ndarray,
        control_points: Optional[np.ndarray] = None,
        degree: int = 3,
    ) -> None:
        self.knots_x = np.asarray(knots_x, dtype=np.float64).ravel()
        self.knots_y = np.asarray(knots_y, dtype=np.float64).ravel()
        self.degree = degree

        if control_points is not None:
            ctrl_arr = np.asarray(control_points, dtype=np.float64)
            if ctrl_arr.ndim == 2:
                self.nx = ctrl_arr.shape[0]
                self.ny = ctrl_arr.shape[1]
                self._coefficients = ctrl_arr.ravel()
            else:
                self._coefficients = ctrl_arr.ravel()
                # Estimate nx, ny from knots
                self.nx = max(len(self.knots_x) - degree - 1, 1)
                self.ny = max(len(self.knots_y) - degree - 1, 1)
        else:
            self.nx = max(len(self.knots_x) - degree - 1, 1)
            self.ny = max(len(self.knots_y) - degree - 1, 1)
            self._coefficients = np.ones(self.nx * self.ny, dtype=np.float64)

        # Ensure knots have required length: nx + degree + 1
        req_x = self.nx + self.degree + 1
        if len(self.knots_x) < req_x:
            self.knots_x = np.pad(self.knots_x, (0, req_x - len(self.knots_x)), mode="edge")
        req_y = self.ny + self.degree + 1
        if len(self.knots_y) < req_y:
            self.knots_y = np.pad(self.knots_y, (0, req_y - len(self.knots_y)), mode="edge")

        self._num_terms = self.nx * self.ny
        if len(self._coefficients) != self._num_terms:
            raise ValueError(
                f"Control points count {len(self._coefficients)} does not match nx * ny = {self._num_terms}"
            )

    @property
    def num_terms(self) -> int:
        return self._num_terms

    def basis_matrix(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        Bx = _cox_de_boor_1d(X, self.knots_x, self.degree)  # (N, nx)
        By = _cox_de_boor_1d(Y, self.knots_y, self.degree)  # (N, ny)

        N = X.size
        Phi = np.empty((N, self._num_terms), dtype=np.float64)
        col = 0
        for i in range(self.nx):
            bx_i = Bx[:, i]
            for j in range(self.ny):
                Phi[:, col] = bx_i * By[:, j]
                col += 1
        return Phi

    def evaluate(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        coefficients: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        c = self._coefficients if coefficients is None else np.asarray(coefficients, dtype=np.float64).ravel()
        Phi = self.basis_matrix(X, Y)
        vals = Phi @ c
        return vals.reshape(X.shape)

    def to_ast(
        self,
        coefficients: Optional[np.ndarray] = None,
        param_offset: int = 0,
        symbolic_params: bool = False,
        prune_threshold: float = 1e-12,
    ) -> ASTNode:
        """
        Emit the *exact* B-spline surface as a symbolic piecewise polynomial.

        On every non-degenerate knot interval a degree-p B-spline surface is an
        exact bivariate polynomial. The local monomial coefficients are solved
        from sampled Cox-de Boor values, and narrow clamp-based indicators
        stitch the intervals together. This reproduces evaluate() (including
        repeated boundary knots), unlike the Gaussian surrogate that a previous
        version emitted.
        """
        c = self._coefficients if coefficients is None else np.asarray(coefficients, dtype=np.float64).ravel()
        if len(c) < self.num_terms:
            raise ValueError(f"Coefficients length {len(c)} is less than required basis terms {self.num_terms}")
        term_nodes: List[ASTNode] = []

        var_x = VariableNode("x")
        var_y = VariableNode("y")

        def intervals(knots: np.ndarray) -> List[Tuple[float, float]]:
            out: List[Tuple[float, float]] = []
            for i in range(len(knots) - 1):
                if knots[i + 1] > knots[i] + 1e-12:
                    out.append((float(knots[i]), float(knots[i + 1])))
            return out

        def local_poly(knots: np.ndarray, a: float, b: float) -> np.ndarray:
            """Monomial coefficients (degree+1, n_basis) of every basis on [a, b]."""
            p = self.degree
            nodes = 0.5 * (b - a) * (1.0 + np.cos(np.pi * (2 * np.arange(p + 1) + 1) / (2 * (p + 1)))) + a
            local_nodes = (nodes - a) / (b - a)
            vander = np.vander(local_nodes, p + 1, increasing=True)
            samples = _cox_de_boor_1d(nodes, knots, p)  # (p+1, n_basis)
            return np.linalg.solve(vander, samples)

        def indicator(variable: VariableNode, a: float, b: float) -> ASTNode:
            eps = max((b - a) * 1e-9, 1e-12)
            lo = FunctionCallNode("clamp", (BinaryOpNode("+", BinaryOpNode("/", BinaryOpNode("-", variable, ConstantNode(a)), ConstantNode(eps)), ConstantNode(0.5)), ConstantNode(0.0), ConstantNode(1.0)))
            hi = FunctionCallNode("clamp", (BinaryOpNode("+", BinaryOpNode("/", BinaryOpNode("-", ConstantNode(b), variable), ConstantNode(eps)), ConstantNode(0.5)), ConstantNode(0.0), ConstantNode(1.0)))
            return BinaryOpNode("*", lo, hi)

        cx_all = {iv: local_poly(self.knots_x, *iv) for iv in intervals(self.knots_x)}
        cy_all = {iv: local_poly(self.knots_y, *iv) for iv in intervals(self.knots_y)}

        for ix, cy in cx_all.items():
            for iy, cy_coeffs in cy_all.items():
                a, b = ix
                d, e = iy
                coef = np.zeros((self.degree + 1, self.degree + 1), dtype=np.float64)
                symbolic_coefs: List[Tuple[Tuple[int, int], ASTNode]] = []
                col = 0
                for i in range(self.nx):
                    for j in range(self.ny):
                        w = float(c[col])
                        if symbolic_params:
                            symbolic_coefs.append(
                                ((i, j), ParameterNode(f"c{param_offset + col}", param_offset + col))
                            )
                        elif abs(w) >= prune_threshold:
                            coef += w * np.outer(cy[:, i], cy_coeffs[:, j])
                        col += 1
                pieces: List[ASTNode] = []
                u = BinaryOpNode("/", BinaryOpNode("-", var_x, ConstantNode(a)), ConstantNode(b - a))
                v = BinaryOpNode("/", BinaryOpNode("-", var_y, ConstantNode(d)), ConstantNode(e - d))

                def poly_ast(coeffs: np.ndarray, variable: ASTNode) -> ASTNode:
                    terms = [
                        BinaryOpNode("*", ConstantNode(float(val)), BinaryOpNode("^", variable, ConstantNode(float(k))))
                        for k, val in enumerate(coeffs)
                        if abs(val) > prune_threshold or symbolic_params
                    ]
                    return _build_balanced_addition(terms)

                if symbolic_params:
                    for (i, j), p_node in symbolic_coefs:
                        pieces.append(
                            BinaryOpNode(
                                "*",
                                p_node,
                                BinaryOpNode("*", poly_ast(cx_all[ix][:, i], u), poly_ast(cy_all[iy][:, j], v)),
                            )
                        )
                else:
                    for m in range(self.degree + 1):
                        u_m: ASTNode = ConstantNode(1.0) if m == 0 else BinaryOpNode("^", u, ConstantNode(float(m)))
                        for n in range(self.degree + 1):
                            val = float(coef[m, n])
                            if abs(val) < prune_threshold:
                                continue
                            v_n: ASTNode = ConstantNode(1.0) if n == 0 else BinaryOpNode("^", v, ConstantNode(float(n)))
                            pieces.append(BinaryOpNode("*", ConstantNode(val), BinaryOpNode("*", u_m, v_n)))
                if pieces:
                    term_nodes.append(
                        BinaryOpNode("*", _build_balanced_addition(pieces), BinaryOpNode("*", indicator(var_x, a, b), indicator(var_y, d, e)))
                    )

        if not term_nodes:
            return ConstantNode(0.0)
        return _build_balanced_addition(term_nodes)

    def get_complexity(self, coefficients: Optional[np.ndarray] = None) -> int:
        c = self._coefficients if coefficients is None else np.asarray(coefficients, dtype=np.float64).ravel()
        active = sum(1 for val in c if abs(val) > 1e-12)
        return active * 6
