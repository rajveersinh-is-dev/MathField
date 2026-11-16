"""
mathfield/core/basis/base.py

Abstract Base Class for 2D Global and Local Basis Functions.
Defines matrix generation, numerical evaluation, AST serialization,
and structural complexity quantification.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np

from mathfield.core.ast.nodes import ASTNode, ConstantNode, ParameterNode


class BasisFunction2D(ABC):
    """Abstract 2D basis function family phi_k(x, y)."""

    @property
    @abstractmethod
    def num_terms(self) -> int:
        """Total number of basis components K in this expansion."""
        pass

    @abstractmethod
    def basis_matrix(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """
        Evaluate all K basis functions across 2D coordinate matrices X, Y.
        Returns design matrix Phi of shape (X.size, num_terms).
        """
        pass

    @abstractmethod
    def evaluate(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        coefficients: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Evaluate linear combination sum_{k=0}^{K-1} c_k * phi_k(X, Y).
        If coefficients is None, uses internal coefficients or default ones.
        Returns array matching X.shape.
        """
        pass

    @abstractmethod
    def to_ast(
        self,
        coefficients: Optional[np.ndarray] = None,
        param_offset: int = 0,
        symbolic_params: bool = False,
        prune_threshold: float = 1e-12,
    ) -> ASTNode:
        """
        Convert basis expansion into an immutable symbolic ASTNode.
        If symbolic_params is True, emits ParameterNode for active terms.
        If symbolic_params is False, emits ConstantNode with coefficient values.
        Prunes terms with |c_k| < prune_threshold.
        """
        pass

    @abstractmethod
    def get_complexity(self, coefficients: Optional[np.ndarray] = None) -> int:
        """
        Structural complexity C(M) of active terms for objective:
        J(M) = E(M) + lambda * C(M)
        """
        pass
