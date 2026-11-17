"""
mathfield/core/basis/rbf.py

Radial Basis Function (RBF) 2D Expansions:
- WendlandC2RBF (compactly supported C2 Wendland psi_3,1 kernel)
- GaussianRBF (infinitely smooth Gaussian kernel)
- MultiquadricRBF (smooth Hardy multiquadric kernel)
- Deterministic center placement algorithms (uniform grid, Halton sequence, residual peak placement)
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple, Union

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


def _build_euclidean_distance_ast(cx: float, cy: float) -> ASTNode:
    """Build sqrt((x - cx)^2 + (y - cy)^2) AST."""
    var_x = VariableNode("x")
    var_y = VariableNode("y")
    dx = BinaryOpNode("-", var_x, ConstantNode(cx))
    dy = BinaryOpNode("-", var_y, ConstantNode(cy))
    dx2 = BinaryOpNode("^", dx, ConstantNode(2.0))
    dy2 = BinaryOpNode("^", dy, ConstantNode(2.0))
    dist_sq = BinaryOpNode("+", dx2, dy2)
    return FunctionCallNode("sqrt", (dist_sq,))


class WendlandC2RBF(BasisFunction2D):
    """
    Compactly Supported C2 Wendland psi_{3, 1} RBF:
    phi(r) = (1 - r/R)_+^4 * (4 * r/R + 1)
    Strictly zero for r >= R, C2 smooth at r = R.
    """

    def __init__(
        self,
        centers: np.ndarray,
        radii: Union[float, np.ndarray] = 1.0,
        coefficients: Optional[np.ndarray] = None,
    ) -> None:
        self.centers = np.asarray(centers, dtype=np.float64)
        if self.centers.ndim == 1:
            self.centers = self.centers.reshape(1, -1)
        if self.centers.shape[1] != 2:
            raise ValueError(f"Centers must have shape (K, 2), got {self.centers.shape}")

        k = self.centers.shape[0]
        if np.isscalar(radii):
            self.radii = np.full(k, float(radii), dtype=np.float64)
        else:
            r_arr = np.asarray(radii, dtype=np.float64).ravel()
            if len(r_arr) != k:
                raise ValueError(f"Radii length {len(r_arr)} does not match centers count {k}")
            self.radii = r_arr

        if np.any(self.radii <= 0.0):
            raise ValueError("All RBF radii must be strictly positive")

        if coefficients is not None:
            c = np.asarray(coefficients, dtype=np.float64).ravel()
            if len(c) != k:
                raise ValueError(f"Coefficients length {len(c)} does not match centers count {k}")
            self._coefficients = c
        else:
            self._coefficients = np.ones(k, dtype=np.float64)

    @property
    def num_terms(self) -> int:
        return self.centers.shape[0]

    def basis_matrix(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        x_flat = X.ravel()
        y_flat = Y.ravel()
        N = x_flat.size
        K = self.num_terms

        Phi = np.zeros((N, K), dtype=np.float64)
        for k in range(K):
            cx, cy = self.centers[k]
            R = self.radii[k]
            dx = x_flat - cx
            dy = y_flat - cy
            dist = np.sqrt(dx * dx + dy * dy)
            scaled_r = dist / R
            mask = scaled_r < 1.0
            if np.any(mask):
                sr = scaled_r[mask]
                Phi[mask, k] = ((1.0 - sr) ** 4.0) * (4.0 * sr + 1.0)

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
        c = self._coefficients if coefficients is None else np.asarray(coefficients, dtype=np.float64).ravel()
        if len(c) < self.num_terms:
            raise ValueError(f"Coefficients length {len(c)} is less than required basis terms {self.num_terms}")
        term_nodes: List[ASTNode] = []

        for k in range(self.num_terms):
            coeff_val = float(c[k])
            if abs(coeff_val) < prune_threshold and not symbolic_params:
                continue

            cx, cy = float(self.centers[k, 0]), float(self.centers[k, 1])
            R = float(self.radii[k])

            p_node: ASTNode = (
                ParameterNode(f"c{param_offset + k}", param_offset + k) if symbolic_params else ConstantNode(coeff_val)
            )
            dist_ast = _build_euclidean_distance_ast(cx, cy)
            rbf_ast = FunctionCallNode("rbf_wendland", (dist_ast, ConstantNode(R)))
            term_nodes.append(BinaryOpNode("*", p_node, rbf_ast))

        if not term_nodes:
            return ConstantNode(0.0)
        return _build_balanced_addition(term_nodes)

    def get_complexity(self, coefficients: Optional[np.ndarray] = None) -> int:
        c = self._coefficients if coefficients is None else np.asarray(coefficients, dtype=np.float64).ravel()
        active = sum(1 for val in c if abs(val) > 1e-12)
        return active * 8


class GaussianRBF(BasisFunction2D):
    """
    Gaussian Radial Basis Function:
    phi(r) = exp(-(epsilon * r)^2)
    """

    def __init__(
        self,
        centers: np.ndarray,
        epsilons: Union[float, np.ndarray] = 1.0,
        coefficients: Optional[np.ndarray] = None,
    ) -> None:
        self.centers = np.asarray(centers, dtype=np.float64)
        if self.centers.ndim == 1:
            self.centers = self.centers.reshape(1, -1)
        if self.centers.shape[1] != 2:
            raise ValueError(f"Centers must have shape (K, 2), got {self.centers.shape}")

        k = self.centers.shape[0]
        if np.isscalar(epsilons):
            self.epsilons = np.full(k, float(epsilons), dtype=np.float64)
        else:
            e_arr = np.asarray(epsilons, dtype=np.float64).ravel()
            if len(e_arr) != k:
                raise ValueError(f"Epsilons length {len(e_arr)} does not match centers count {k}")
            self.epsilons = e_arr

        if np.any(self.epsilons <= 0.0):
            raise ValueError("All RBF epsilons must be strictly positive")

        if coefficients is not None:
            c = np.asarray(coefficients, dtype=np.float64).ravel()
            if len(c) != k:
                raise ValueError(f"Coefficients length {len(c)} does not match centers count {k}")
            self._coefficients = c
        else:
            self._coefficients = np.ones(k, dtype=np.float64)

    @property
    def num_terms(self) -> int:
        return self.centers.shape[0]

    def basis_matrix(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        x_flat = X.ravel()
        y_flat = Y.ravel()
        N = x_flat.size
        K = self.num_terms

        Phi = np.empty((N, K), dtype=np.float64)
        for k in range(K):
            cx, cy = self.centers[k]
            eps = self.epsilons[k]
            dx = x_flat - cx
            dy = y_flat - cy
            dist_sq = dx * dx + dy * dy
            Phi[:, k] = np.exp(-((eps * eps) * dist_sq))

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
        c = self._coefficients if coefficients is None else np.asarray(coefficients, dtype=np.float64).ravel()
        if len(c) < self.num_terms:
            raise ValueError(f"Coefficients length {len(c)} is less than required basis terms {self.num_terms}")
        term_nodes: List[ASTNode] = []

        for k in range(self.num_terms):
            coeff_val = float(c[k])
            if abs(coeff_val) < prune_threshold and not symbolic_params:
                continue

            cx, cy = float(self.centers[k, 0]), float(self.centers[k, 1])
            eps = float(self.epsilons[k])

            p_node: ASTNode = (
                ParameterNode(f"c{param_offset + k}", param_offset + k) if symbolic_params else ConstantNode(coeff_val)
            )
            dist_ast = _build_euclidean_distance_ast(cx, cy)
            rbf_ast = FunctionCallNode("rbf_gaussian", (dist_ast, ConstantNode(eps)))
            term_nodes.append(BinaryOpNode("*", p_node, rbf_ast))

        if not term_nodes:
            return ConstantNode(0.0)
        return _build_balanced_addition(term_nodes)

    def get_complexity(self, coefficients: Optional[np.ndarray] = None) -> int:
        c = self._coefficients if coefficients is None else np.asarray(coefficients, dtype=np.float64).ravel()
        active = sum(1 for val in c if abs(val) > 1e-12)
        return active * 8


class MultiquadricRBF(BasisFunction2D):
    """
    Multiquadric Radial Basis Function:
    phi(r) = sqrt(1 + (epsilon * r)^2)
    """

    def __init__(
        self,
        centers: np.ndarray,
        epsilons: Union[float, np.ndarray] = 1.0,
        coefficients: Optional[np.ndarray] = None,
    ) -> None:
        self.centers = np.asarray(centers, dtype=np.float64)
        if self.centers.ndim == 1:
            self.centers = self.centers.reshape(1, -1)
        if self.centers.shape[1] != 2:
            raise ValueError(f"Centers must have shape (K, 2), got {self.centers.shape}")

        k = self.centers.shape[0]
        if np.isscalar(epsilons):
            self.epsilons = np.full(k, float(epsilons), dtype=np.float64)
        else:
            e_arr = np.asarray(epsilons, dtype=np.float64).ravel()
            if len(e_arr) != k:
                raise ValueError(f"Epsilons length {len(e_arr)} does not match centers count {k}")
            self.epsilons = e_arr

        if np.any(self.epsilons <= 0.0):
            raise ValueError("All RBF epsilons must be strictly positive")

        if coefficients is not None:
            c = np.asarray(coefficients, dtype=np.float64).ravel()
            if len(c) != k:
                raise ValueError(f"Coefficients length {len(c)} does not match centers count {k}")
            self._coefficients = c
        else:
            self._coefficients = np.ones(k, dtype=np.float64)

    @property
    def num_terms(self) -> int:
        return self.centers.shape[0]

    def basis_matrix(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        x_flat = X.ravel()
        y_flat = Y.ravel()
        N = x_flat.size
        K = self.num_terms

        Phi = np.empty((N, K), dtype=np.float64)
        for k in range(K):
            cx, cy = self.centers[k]
            eps = self.epsilons[k]
            dx = x_flat - cx
            dy = y_flat - cy
            dist_sq = dx * dx + dy * dy
            Phi[:, k] = np.sqrt(1.0 + (eps * eps) * dist_sq)

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
        c = self._coefficients if coefficients is None else np.asarray(coefficients, dtype=np.float64).ravel()
        if len(c) < self.num_terms:
            raise ValueError(f"Coefficients length {len(c)} is less than required basis terms {self.num_terms}")
        term_nodes: List[ASTNode] = []

        for k in range(self.num_terms):
            coeff_val = float(c[k])
            if abs(coeff_val) < prune_threshold and not symbolic_params:
                continue

            cx, cy = float(self.centers[k, 0]), float(self.centers[k, 1])
            eps = float(self.epsilons[k])

            p_node: ASTNode = (
                ParameterNode(f"c{param_offset + k}", param_offset + k) if symbolic_params else ConstantNode(coeff_val)
            )
            dist_ast = _build_euclidean_distance_ast(cx, cy)
            rbf_ast = FunctionCallNode("rbf_multiquadric", (dist_ast, ConstantNode(eps)))
            term_nodes.append(BinaryOpNode("*", p_node, rbf_ast))

        if not term_nodes:
            return ConstantNode(0.0)
        return _build_balanced_addition(term_nodes)

    def get_complexity(self, coefficients: Optional[np.ndarray] = None) -> int:
        c = self._coefficients if coefficients is None else np.asarray(coefficients, dtype=np.float64).ravel()
        active = sum(1 for val in c if abs(val) > 1e-12)
        return active * 8


# --- Deterministic Center Placement Strategies ---


def generate_uniform_centers(
    nx: int,
    ny: int,
    x_bounds: Tuple[float, float] = (-1.0, 1.0),
    y_bounds: Tuple[float, float] = (-1.0, 1.0),
) -> np.ndarray:
    """Generate regular grid of centers across domain."""
    xs = np.linspace(x_bounds[0], x_bounds[1], nx)
    ys = np.linspace(y_bounds[0], y_bounds[1], ny)
    X, Y = np.meshgrid(xs, ys, indexing="xy")
    return np.column_stack([X.ravel(), Y.ravel()])


def _van_der_corput(n: int, base: int) -> np.ndarray:
    """Generate n terms of van der Corput sequence for given base."""
    seq = np.empty(n, dtype=np.float64)
    for i in range(n):
        val = 0.0
        f = 1.0 / base
        idx = i + 1
        while idx > 0:
            val += (idx % base) * f
            idx //= base
            f /= base
        seq[i] = val
    return seq


def generate_halton_centers(
    n: int,
    x_bounds: Tuple[float, float] = (-1.0, 1.0),
    y_bounds: Tuple[float, float] = (-1.0, 1.0),
) -> np.ndarray:
    """Generate 2D Halton quasi-random low-discrepancy centers (bases 2 and 3)."""
    h_x = _van_der_corput(n, 2)
    h_y = _van_der_corput(n, 3)
    x = x_bounds[0] + h_x * (x_bounds[1] - x_bounds[0])
    y = y_bounds[0] + h_y * (y_bounds[1] - y_bounds[0])
    return np.column_stack([x, y])


def generate_residual_peak_centers(
    residual: np.ndarray,
    n_centers: int,
    x_bounds: Tuple[float, float] = (-1.0, 1.0),
    y_bounds: Tuple[float, float] = (-1.0, 1.0),
    min_dist: float = 0.1,
) -> np.ndarray:
    """Greedy residual peak center selection with exclusion radius."""
    res_abs = np.abs(np.asarray(residual, dtype=np.float64))
    h, w = res_abs.shape
    centers: List[Tuple[float, float]] = []

    flat_order = np.argsort(res_abs.ravel())[::-1]
    for idx in flat_order:
        if len(centers) >= n_centers:
            break
        r_idx, c_idx = np.unravel_index(idx, (h, w))
        # Map to continuous coordinates
        x_pt = x_bounds[0] + (c_idx / max(w - 1, 1)) * (x_bounds[1] - x_bounds[0])
        y_pt = y_bounds[0] + (r_idx / max(h - 1, 1)) * (y_bounds[1] - y_bounds[0])

        # Check exclusion distance against already placed centers
        too_close = False
        for cx, cy in centers:
            d = math.hypot(x_pt - cx, y_pt - cy)
            if d < min_dist:
                too_close = True
                break

        if not too_close:
            centers.append((x_pt, y_pt))

    if not centers:
        return np.array([[0.0, 0.0]], dtype=np.float64)
    return np.array(centers, dtype=np.float64)
