"""
mathfield/core/basis/polynomial.py

2D Orthogonal Polynomial Basis Expansions:
- Chebyshev2D (Clenshaw recurrence, total degree, hyperbolic cross, tensor truncations)
- Legendre2D (Bonnet 3-term recurrence, L2 orthogonality)
All recurrences and polynomial expansions are computed dynamically from first principles.
"""

from __future__ import annotations

from typing import List, Literal, Optional, Tuple

import numpy as np

from mathfield.core.ast.nodes import (
    ASTNode,
    BinaryOpNode,
    ConstantNode,
    ParameterNode,
    VariableNode,
)
from mathfield.core.basis.base import BasisFunction2D


def _build_balanced_addition(nodes: List[ASTNode]) -> ASTNode:
    """Combine a list of AST nodes into a balanced binary addition tree."""
    if not nodes:
        return ConstantNode(0.0)
    current = nodes
    while len(current) > 1:
        next_level = []
        for i in range(0, len(current), 2):
            if i + 1 < len(current):
                next_level.append(BinaryOpNode("+", current[i], current[i + 1]))
            else:
                next_level.append(current[i])
        current = next_level
    return current[0]


def _build_horner_poly(coeffs: np.ndarray, var_name: str) -> ASTNode:
    """
    Build Horner polynomial AST from ascending coefficient array:
    p(x) = c[0] + x * (c[1] + x * (c[2] + ...))
    """
    n = len(coeffs)
    # Find highest non-zero coefficient
    last_idx = n - 1
    while last_idx > 0 and abs(coeffs[last_idx]) < 1e-12:
        last_idx -= 1

    if last_idx == 0:
        return ConstantNode(float(coeffs[0]))

    var = VariableNode(var_name)
    expr: ASTNode = ConstantNode(float(coeffs[last_idx]))
    for k in range(last_idx - 1, -1, -1):
        c_k = float(coeffs[k])
        expr = BinaryOpNode("*", var, expr)
        if abs(c_k) > 1e-12:
            expr = BinaryOpNode("+", ConstantNode(c_k), expr)
    return expr


class Chebyshev2D(BasisFunction2D):
    """
    Bivariate Chebyshev Orthogonal Polynomial Expansion:
    phi_{i, j}(x, y) = T_i(x) * T_j(y)
    Evaluated with Clenshaw recurrence and dynamically derived Horner AST.
    """

    def __init__(
        self,
        deg_x: int = 4,
        deg_y: int = 4,
        truncation: Literal["total_degree", "hyperbolic_cross", "full_tensor", "tensor"] = "total_degree",
        coefficients: Optional[np.ndarray] = None,
    ) -> None:
        if deg_x < 0 or deg_y < 0:
            raise ValueError(f"Polynomial degrees must be non-negative, got ({deg_x}, {deg_y})")

        self.deg_x = deg_x
        self.deg_y = deg_y
        self.truncation = truncation

        # Build list of active degree index pairs (i, j)
        self.indices: List[Tuple[int, int]] = []
        max_deg = max(deg_x, deg_y)

        for i in range(deg_x + 1):
            for j in range(deg_y + 1):
                if truncation == "total_degree":
                    if i + j <= max_deg:
                        self.indices.append((i, j))
                elif truncation == "hyperbolic_cross":
                    if (i + 1) * (j + 1) <= max_deg + 1:
                        self.indices.append((i, j))
                elif truncation in ("full_tensor", "tensor"):
                    self.indices.append((i, j))
                else:
                    raise ValueError(f"Unknown truncation mode: {truncation}")

        if coefficients is not None:
            c = np.asarray(coefficients, dtype=np.float64).ravel()
            if len(c) != len(self.indices):
                raise ValueError(
                    f"Coefficients length {len(c)} does not match number of basis terms {len(self.indices)}"
                )
            self._coefficients = c
        else:
            self._coefficients = np.ones(len(self.indices), dtype=np.float64)

    @property
    def num_terms(self) -> int:
        return len(self.indices)

    def _eval_1d_chebyshev(self, coords: np.ndarray, max_deg: int) -> np.ndarray:
        """
        Dynamically compute 1D Chebyshev polynomials T_0(x) .. T_max_deg(x).
        Returns array of shape (coords.size, max_deg + 1).
        """
        flat = coords.ravel()
        n_pts = flat.size
        T = np.zeros((n_pts, max_deg + 1), dtype=np.float64)
        T[:, 0] = 1.0
        if max_deg >= 1:
            T[:, 1] = flat
            two_x = 2.0 * flat
            for d in range(2, max_deg + 1):
                T[:, d] = two_x * T[:, d - 1] - T[:, d - 2]
        return T

    def basis_matrix(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        Tx = self._eval_1d_chebyshev(X, self.deg_x)
        Ty = self._eval_1d_chebyshev(Y, self.deg_y)

        N = X.size
        Phi = np.empty((N, self.num_terms), dtype=np.float64)
        for k, (i, j) in enumerate(self.indices):
            Phi[:, k] = Tx[:, i] * Ty[:, j]
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

    @staticmethod
    def _compute_chebyshev_poly_coeffs(order: int) -> np.ndarray:
        """Dynamically compute polynomial coefficient vector for T_n(x)."""
        if order == 0:
            return np.array([1.0], dtype=np.float64)
        if order == 1:
            return np.array([0.0, 1.0], dtype=np.float64)

        c_prev2 = np.zeros(order + 1, dtype=np.float64)
        c_prev2[0] = 1.0
        c_prev1 = np.zeros(order + 1, dtype=np.float64)
        c_prev1[1] = 1.0

        for _ in range(2, order + 1):
            c_curr = np.zeros(order + 1, dtype=np.float64)
            # 2 * x * T_{n-1}(x): shift c_prev1 right by 1 and multiply by 2
            c_curr[1:] = 2.0 * c_prev1[:-1]
            # - T_{n-2}(x)
            c_curr -= c_prev2
            c_prev2 = c_prev1
            c_prev1 = c_curr

        return c_curr

    def to_ast(
        self,
        coefficients: Optional[np.ndarray] = None,
        param_offset: int = 0,
        symbolic_params: bool = False,
        prune_threshold: float = 1e-12,
    ) -> ASTNode:
        c = self._coefficients if coefficients is None else np.asarray(coefficients, dtype=np.float64).ravel()
        if len(c) < self.num_terms:
            raise ValueError(f"Coefficients length {len(c)} is less than required basis terms {self.num_terms}")

        term_nodes: List[ASTNode] = []
        for k, (i, j) in enumerate(self.indices):
            coeff_val = float(c[k])
            if abs(coeff_val) < prune_threshold and not symbolic_params:
                continue

            param_node: ASTNode
            if symbolic_params:
                param_node = ParameterNode(f"c{param_offset + k}", param_offset + k)
            else:
                param_node = ConstantNode(coeff_val)

            # Build T_i(x)
            poly_i = self._compute_chebyshev_poly_coeffs(i)
            tx_ast = _build_horner_poly(poly_i, "x")

            # Build T_j(y)
            poly_j = self._compute_chebyshev_poly_coeffs(j)
            ty_ast = _build_horner_poly(poly_j, "y")

            term_ast = BinaryOpNode("*", param_node, BinaryOpNode("*", tx_ast, ty_ast))
            term_nodes.append(term_ast)

        if not term_nodes:
            return ConstantNode(0.0)
        return _build_balanced_addition(term_nodes)

    def get_complexity(self, coefficients: Optional[np.ndarray] = None) -> int:
        c = self._coefficients if coefficients is None else np.asarray(coefficients, dtype=np.float64).ravel()
        active = sum(1 for val in c if abs(val) > 1e-12)
        return active * 4


class Legendre2D(BasisFunction2D):
    """
    Bivariate Legendre Orthogonal Polynomial Expansion:
    phi_{i, j}(x, y) = P_i(x) * P_j(y)
    Orthogonal with respect to unweighted L2 norm over [-1, 1]^2.
    Computed via Bonnet's 3-term recurrence.
    """

    def __init__(
        self,
        deg_x: int = 4,
        deg_y: int = 4,
        truncation: Literal["total_degree", "hyperbolic_cross", "full_tensor", "tensor"] = "total_degree",
        coefficients: Optional[np.ndarray] = None,
    ) -> None:
        if deg_x < 0 or deg_y < 0:
            raise ValueError(f"Polynomial degrees must be non-negative, got ({deg_x}, {deg_y})")

        self.deg_x = deg_x
        self.deg_y = deg_y
        self.truncation = truncation

        self.indices: List[Tuple[int, int]] = []
        max_deg = max(deg_x, deg_y)

        for i in range(deg_x + 1):
            for j in range(deg_y + 1):
                if truncation == "total_degree":
                    if i + j <= max_deg:
                        self.indices.append((i, j))
                elif truncation == "hyperbolic_cross":
                    if (i + 1) * (j + 1) <= max_deg + 1:
                        self.indices.append((i, j))
                elif truncation in ("full_tensor", "tensor"):
                    self.indices.append((i, j))
                else:
                    raise ValueError(f"Unknown truncation mode: {truncation}")

        if coefficients is not None:
            c = np.asarray(coefficients, dtype=np.float64).ravel()
            if len(c) != len(self.indices):
                raise ValueError(
                    f"Coefficients length {len(c)} does not match number of basis terms {len(self.indices)}"
                )
            self._coefficients = c
        else:
            self._coefficients = np.ones(len(self.indices), dtype=np.float64)

    @property
    def num_terms(self) -> int:
        return len(self.indices)

    def _eval_1d_legendre(self, coords: np.ndarray, max_deg: int) -> np.ndarray:
        """
        Dynamically compute 1D Legendre polynomials P_0(x) .. P_max_deg(x)
        via Bonnet's recurrence: (n+1) P_{n+1} = (2n+1) x P_n - n P_{n-1}.
        """
        flat = coords.ravel()
        n_pts = flat.size
        P = np.zeros((n_pts, max_deg + 1), dtype=np.float64)
        P[:, 0] = 1.0
        if max_deg >= 1:
            P[:, 1] = flat
            for n in range(1, max_deg):
                P[:, n + 1] = ((2 * n + 1) * flat * P[:, n] - n * P[:, n - 1]) / (n + 1)
        return P

    def basis_matrix(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        Px = self._eval_1d_legendre(X, self.deg_x)
        Py = self._eval_1d_legendre(Y, self.deg_y)

        N = X.size
        Phi = np.empty((N, self.num_terms), dtype=np.float64)
        for k, (i, j) in enumerate(self.indices):
            Phi[:, k] = Px[:, i] * Py[:, j]
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

    @staticmethod
    def _compute_legendre_poly_coeffs(order: int) -> np.ndarray:
        """Dynamically compute polynomial coefficient vector for P_n(x)."""
        if order == 0:
            return np.array([1.0], dtype=np.float64)
        if order == 1:
            return np.array([0.0, 1.0], dtype=np.float64)

        c_prev2 = np.zeros(order + 1, dtype=np.float64)
        c_prev2[0] = 1.0
        c_prev1 = np.zeros(order + 1, dtype=np.float64)
        c_prev1[1] = 1.0

        for n in range(1, order):
            c_curr = np.zeros(order + 1, dtype=np.float64)
            # ((2n + 1) * x * P_n(x) - n * P_{n-1}(x)) / (n + 1)
            c_curr[1:] = ((2 * n + 1) / (n + 1)) * c_prev1[:-1]
            c_curr -= (n / (n + 1)) * c_prev2
            c_prev2 = c_prev1
            c_prev1 = c_curr

        return c_curr

    def to_ast(
        self,
        coefficients: Optional[np.ndarray] = None,
        param_offset: int = 0,
        symbolic_params: bool = False,
        prune_threshold: float = 1e-12,
    ) -> ASTNode:
        c = self._coefficients if coefficients is None else np.asarray(coefficients, dtype=np.float64).ravel()
        if len(c) < self.num_terms:
            raise ValueError(f"Coefficients length {len(c)} is less than required basis terms {self.num_terms}")

        term_nodes: List[ASTNode] = []
        for k, (i, j) in enumerate(self.indices):
            coeff_val = float(c[k])
            if abs(coeff_val) < prune_threshold and not symbolic_params:
                continue

            param_node: ASTNode
            if symbolic_params:
                param_node = ParameterNode(f"c{param_offset + k}", param_offset + k)
            else:
                param_node = ConstantNode(coeff_val)

            poly_i = self._compute_legendre_poly_coeffs(i)
            px_ast = _build_horner_poly(poly_i, "x")

            poly_j = self._compute_legendre_poly_coeffs(j)
            py_ast = _build_horner_poly(poly_j, "y")

            term_ast = BinaryOpNode("*", param_node, BinaryOpNode("*", px_ast, py_ast))
            term_nodes.append(term_ast)

        if not term_nodes:
            return ConstantNode(0.0)
        return _build_balanced_addition(term_nodes)

    def get_complexity(self, coefficients: Optional[np.ndarray] = None) -> int:
        c = self._coefficients if coefficients is None else np.asarray(coefficients, dtype=np.float64).ravel()
        active = sum(1 for val in c if abs(val) > 1e-12)
        return active * 4
