"""
mathfield/core/basis/fourier.py

2D Harmonic and Discrete Cosine Transform Basis Expansions:
- Fourier2D (real harmonics: cos/sin wave combinations, isotropic circular cutoff)
- DCT2Basis (Type-II 2D DCT with even boundary reflection eliminating Gibbs phenomena)
"""

from __future__ import annotations

import math
from typing import List, Literal, Optional, Tuple

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


class Fourier2D(BasisFunction2D):
    """
    2D Real Harmonic Fourier Basis:
    phi_0(x, y) = 1.0 (DC)
    phi_{k, cos}(x, y) = cos(kx * pi * x + ky * pi * y)
    phi_{k, sin}(x, y) = sin(kx * pi * x + ky * pi * y)
    """

    def __init__(
        self,
        frequencies: Optional[List[Tuple[float, float]]] = None,
        freq_x: int = 4,
        freq_y: int = 4,
        truncation: Literal["circular", "rectangular"] = "circular",
        coefficients: Optional[np.ndarray] = None,
    ) -> None:
        self.freq_x = freq_x
        self.freq_y = freq_y
        self.truncation = truncation

        if frequencies is not None:
            self.freq_pairs = [(float(fx), float(fy)) for fx, fy in frequencies]
        else:
            self.freq_pairs = []
            max_r_sq = max(freq_x, freq_y) ** 2
            for kx in range(freq_x + 1):
                for ky in range(-freq_y, freq_y + 1):
                    if kx == 0 and ky <= 0:
                        continue  # avoid negative half-plane duplicates
                    if truncation == "circular":
                        if kx * kx + ky * ky <= max_r_sq:
                            self.freq_pairs.append((float(kx), float(ky)))
                    else:
                        self.freq_pairs.append((float(kx), float(ky)))

        # num_terms = 1 (DC) + 2 * len(self.freq_pairs)
        self._num_terms = 1 + 2 * len(self.freq_pairs)

        if coefficients is not None:
            c = np.asarray(coefficients, dtype=np.float64).ravel()
            if len(c) != self._num_terms:
                raise ValueError(f"Coefficients length {len(c)} does not match number of basis terms {self._num_terms}")
            self._coefficients = c
        else:
            self._coefficients = np.ones(self._num_terms, dtype=np.float64) / math.sqrt(self._num_terms)

    @property
    def num_terms(self) -> int:
        return self._num_terms

    def basis_matrix(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        x_flat = X.ravel()
        y_flat = Y.ravel()
        N = x_flat.size

        Phi = np.empty((N, self._num_terms), dtype=np.float64)
        Phi[:, 0] = 1.0  # DC component

        col = 1
        for kx, ky in self.freq_pairs:
            arg = (kx * np.pi * x_flat) + (ky * np.pi * y_flat)
            Phi[:, col] = np.cos(arg)
            col += 1
            Phi[:, col] = np.sin(arg)
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
        c = self._coefficients if coefficients is None else np.asarray(coefficients, dtype=np.float64).ravel()
        if len(c) < self._num_terms:
            raise ValueError(f"Coefficients length {len(c)} is less than required basis terms {self._num_terms}")
        term_nodes: List[ASTNode] = []

        # DC component
        if symbolic_params or abs(float(c[0])) >= prune_threshold:
            dc_node: ASTNode = (
                ParameterNode(f"c{param_offset}", param_offset) if symbolic_params else ConstantNode(float(c[0]))
            )
            term_nodes.append(dc_node)

        col = 1
        var_x = VariableNode("x")
        var_y = VariableNode("y")

        for kx, ky in self.freq_pairs:
            # Phase arg: (kx * pi * x + ky * pi * y)
            kx_pi = kx * math.pi
            ky_pi = ky * math.pi
            kx_term = BinaryOpNode("*", ConstantNode(kx_pi), var_x)
            ky_term = BinaryOpNode("*", ConstantNode(ky_pi), var_y)
            arg_node = BinaryOpNode("+", kx_term, ky_term)

            # Cosine term
            c_cos = float(c[col])
            if symbolic_params or abs(c_cos) >= prune_threshold:
                p_cos = (
                    ParameterNode(f"c{param_offset + col}", param_offset + col)
                    if symbolic_params
                    else ConstantNode(c_cos)
                )
                cos_node = FunctionCallNode("cos", (arg_node,))
                term_nodes.append(BinaryOpNode("*", p_cos, cos_node))
            col += 1

            # Sine term
            c_sin = float(c[col])
            if symbolic_params or abs(c_sin) >= prune_threshold:
                p_sin = (
                    ParameterNode(f"c{param_offset + col}", param_offset + col)
                    if symbolic_params
                    else ConstantNode(c_sin)
                )
                sin_node = FunctionCallNode("sin", (arg_node,))
                term_nodes.append(BinaryOpNode("*", p_sin, sin_node))
            col += 1

        if not term_nodes:
            return ConstantNode(0.0)
        return _build_balanced_addition(term_nodes)

    def get_complexity(self, coefficients: Optional[np.ndarray] = None) -> int:
        c = self._coefficients if coefficients is None else np.asarray(coefficients, dtype=np.float64).ravel()
        active = sum(1 for val in c if abs(val) > 1e-12)
        return active * 5


class DCT2Basis(BasisFunction2D):
    """
    2D Discrete Cosine Transform (Type-II) Continuous Basis:
    phi_{p, q}(x, y) = alpha_p * alpha_q * cos(p * pi * x_norm) * cos(q * pi * y_norm)
    Guarantees zero normal derivative on boundaries, eliminating Gibbs oscillations.
    """

    def __init__(
        self,
        freq_x: int = 4,
        freq_y: int = 4,
        x_bounds: Tuple[float, float] = (-1.0, 1.0),
        y_bounds: Tuple[float, float] = (-1.0, 1.0),
        coefficients: Optional[np.ndarray] = None,
    ) -> None:
        if freq_x < 0 or freq_y < 0:
            raise ValueError(f"Frequencies must be non-negative, got ({freq_x}, {freq_y})")

        self.freq_x = freq_x
        self.freq_y = freq_y
        self.x_bounds = x_bounds
        self.y_bounds = y_bounds

        self.indices: List[Tuple[int, int]] = []
        for p in range(freq_x + 1):
            for q in range(freq_y + 1):
                self.indices.append((p, q))

        self._num_terms = len(self.indices)

        if coefficients is not None:
            c = np.asarray(coefficients, dtype=np.float64).ravel()
            if len(c) != self._num_terms:
                raise ValueError(f"Coefficients length {len(c)} does not match number of basis terms {self._num_terms}")
            self._coefficients = c
        else:
            self._coefficients = np.ones(self._num_terms, dtype=np.float64) / math.sqrt(self._num_terms)

    @property
    def num_terms(self) -> int:
        return self._num_terms

    def _normalize(self, X: np.ndarray, Y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Normalize coordinates into [0, 1]."""
        x_min, x_max = self.x_bounds
        y_min, y_max = self.y_bounds
        x_norm = (X - x_min) / (x_max - x_min)
        y_norm = (Y - y_min) / (y_max - y_min)
        return x_norm, y_norm

    def basis_matrix(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        xn, yn = self._normalize(X, Y)
        x_flat = xn.ravel()
        y_flat = yn.ravel()
        N = x_flat.size

        Phi = np.empty((N, self._num_terms), dtype=np.float64)
        sqrt_2 = math.sqrt(2.0)
        cos_x = np.ones((self.freq_x + 1, N), dtype=np.float64)
        cos_y = np.ones((self.freq_y + 1, N), dtype=np.float64)
        for p in range(1, self.freq_x + 1):
            cos_x[p] = np.cos(p * np.pi * x_flat)
        for q in range(1, self.freq_y + 1):
            cos_y[q] = np.cos(q * np.pi * y_flat)

        for k, (p, q) in enumerate(self.indices):
            alpha_p = 1.0 if p == 0 else sqrt_2
            alpha_q = 1.0 if q == 0 else sqrt_2
            scale = alpha_p * alpha_q
            Phi[:, k] = scale * (cos_x[p] * cos_y[q])

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
        if len(c) < self._num_terms:
            raise ValueError(f"Coefficients length {len(c)} is less than required basis terms {self._num_terms}")
        term_nodes: List[ASTNode] = []
        sqrt_2 = math.sqrt(2.0)

        x_min, x_max = self.x_bounds
        y_min, y_max = self.y_bounds
        inv_x_span = 1.0 / (x_max - x_min)
        inv_y_span = 1.0 / (y_max - y_min)

        # Normalized coordinates: (x - x_min) * inv_x_span
        var_x = VariableNode("x")
        var_y = VariableNode("y")
        xn_ast = BinaryOpNode("*", BinaryOpNode("-", var_x, ConstantNode(x_min)), ConstantNode(inv_x_span))
        yn_ast = BinaryOpNode("*", BinaryOpNode("-", var_y, ConstantNode(y_min)), ConstantNode(inv_y_span))

        for k, (p, q) in enumerate(self.indices):
            coeff_val = float(c[k])
            if abs(coeff_val) < prune_threshold and not symbolic_params:
                continue

            alpha_p = 1.0 if p == 0 else sqrt_2
            alpha_q = 1.0 if q == 0 else sqrt_2
            effective_scale = coeff_val * alpha_p * alpha_q

            p_node: ASTNode = (
                ParameterNode(f"c{param_offset + k}", param_offset + k)
                if symbolic_params
                else ConstantNode(effective_scale)
            )

            # Cosine factors
            cos_p_ast: ASTNode
            if p == 0:
                cos_p_ast = ConstantNode(1.0)
            else:
                arg_p = BinaryOpNode("*", ConstantNode(p * math.pi), xn_ast)
                cos_p_ast = FunctionCallNode("cos", (arg_p,))

            cos_q_ast: ASTNode
            if q == 0:
                cos_q_ast = ConstantNode(1.0)
            else:
                arg_q = BinaryOpNode("*", ConstantNode(q * math.pi), yn_ast)
                cos_q_ast = FunctionCallNode("cos", (arg_q,))

            if p == 0 and q == 0:
                term_ast = p_node
            elif p == 0:
                term_ast = BinaryOpNode("*", p_node, cos_q_ast)
            elif q == 0:
                term_ast = BinaryOpNode("*", p_node, cos_p_ast)
            else:
                term_ast = BinaryOpNode("*", p_node, BinaryOpNode("*", cos_p_ast, cos_q_ast))

            term_nodes.append(term_ast)

        if not term_nodes:
            return ConstantNode(0.0)
        return _build_balanced_addition(term_nodes)

    def get_complexity(self, coefficients: Optional[np.ndarray] = None) -> int:
        c = self._coefficients if coefficients is None else np.asarray(coefficients, dtype=np.float64).ravel()
        active = sum(1 for val in c if abs(val) > 1e-12)
        return active * 5
