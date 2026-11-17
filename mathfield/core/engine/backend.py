"""
Unified Multi-Tier Evaluator Dispatcher.
Implements automatic graceful fallback cascade:
  CuPy CUDA (Tier 1) -> MSVC OpenMP C++ DLL (Tier 2) -> NumPy CPU SIMD (Tier 3).
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

import numpy as np

from mathfield.core.ast.nodes import ASTNode
from mathfield.core.engine.cpu_native import (
    NativeKernelNotAvailableError,
    compute_residuals_native,
    compute_sse_native,
    evaluate_native,
    is_native_available,
)
from mathfield.core.engine.cpu_numpy import (
    compute_residuals_numpy,
    compute_sse_numpy,
    evaluate_numpy,
)
from mathfield.core.engine.gpu_cupy import (
    CUDANotAvailableError,
    CUDAOutOfMemoryError,
    compute_residuals_cupy,
    compute_sse_cupy,
    evaluate_cupy,
    is_cuda_available,
)


class MultiTierEvaluator:
    """
    High-performance expression evaluator with automatic tier selection and fallback.
    """

    def __init__(self, preferred_backend: str = "auto", default_tolerance: float = 1e-5):
        valid_backends = {"auto", "cuda", "cupy", "native", "openmp", "numpy", "cpu"}
        if preferred_backend.lower() not in valid_backends:
            raise ValueError(f"Unknown backend {preferred_backend!r}. Must be one of {valid_backends}")

        self.preferred_backend = self._normalize_backend_name(preferred_backend)
        self.default_tolerance = default_tolerance

        self._last_backend_used: Optional[str] = None
        self._last_eval_time_ms: float = 0.0
        self._telemetry: Dict[str, Any] = {
            "eval_count": 0,
            "fallback_count": 0,
            "tier_counts": {"cuda": 0, "native": 0, "numpy": 0},
        }

    @staticmethod
    def _normalize_backend_name(name: str) -> str:
        n = name.lower()
        if n in ("cuda", "cupy"):
            return "cuda"
        elif n in ("native", "openmp", "c++", "cpp"):
            return "native"
        elif n in ("numpy", "cpu", "simd"):
            return "numpy"
        return "auto"

    @property
    def last_backend_used(self) -> Optional[str]:
        return self._last_backend_used

    @property
    def last_eval_time_ms(self) -> float:
        return self._last_eval_time_ms

    @property
    def telemetry(self) -> Dict[str, Any]:
        return dict(self._telemetry)

    def is_cuda_available(self) -> bool:
        return is_cuda_available()

    def is_native_available(self) -> bool:
        return is_native_available()

    def get_available_backends(self) -> List[str]:
        """Returns list of operational backends ordered by performance tier."""
        backends: List[str] = []
        if self.is_cuda_available():
            backends.append("cuda")
        if self.is_native_available():
            backends.append("native")
        backends.append("numpy")  # Always available
        return backends

    def get_active_backend(self) -> str:
        """Determines which backend is selected under current configuration."""
        pref = self.preferred_backend
        if pref == "auto":
            if self.is_cuda_available():
                return "cuda"
            if self.is_native_available():
                return "native"
            return "numpy"
        return pref

    def evaluate(
        self,
        ast: ASTNode,
        X: np.ndarray,
        Y: np.ndarray,
        params: np.ndarray,
        preferred_backend: Optional[str] = None,
    ) -> np.ndarray:
        """
        Evaluates ASTNode expression over coordinate arrays X, Y with params.
        Automatically cascades down tiers upon resource exhaustion or failure.
        """
        if hasattr(X, "shape") and hasattr(Y, "shape") and X.shape != Y.shape:
            raise ValueError(f"Coordinate arrays X and Y must have identical shapes, got {X.shape} and {Y.shape}")

        from mathfield.core.ast.nodes import validate_parameter_bounds

        validate_parameter_bounds(ast, params)

        backend_choice = self._normalize_backend_name(preferred_backend or self.preferred_backend)
        start_t = time.perf_counter()

        self._telemetry["eval_count"] += 1

        # 1. Direct request for specific tier
        if backend_choice == "numpy":
            res = evaluate_numpy(ast, X, Y, params)
            self._record_eval("numpy", start_t)
            return res

        elif backend_choice == "native":
            if not self.is_native_available():
                raise NativeKernelNotAvailableError("Explicitly requested 'native' backend, but DLL is not loaded.")
            res = evaluate_native(ast, X, Y, params)
            self._record_eval("native", start_t)
            return res

        elif backend_choice == "cuda":
            if not self.is_cuda_available():
                raise CUDANotAvailableError("Explicitly requested 'cuda' backend, but CUDA/CuPy is unavailable.")
            res = evaluate_cupy(ast, X, Y, params)
            self._record_eval("cuda", start_t)
            return res

        # 2. "auto" hierarchy: CUDA -> Native -> NumPy
        # Try Tier 1: CUDA
        if self.is_cuda_available():
            try:
                res = evaluate_cupy(ast, X, Y, params)
                self._record_eval("cuda", start_t)
                return res
            except (CUDAOutOfMemoryError, Exception):
                self._telemetry["fallback_count"] += 1

        # Try Tier 2: Native OpenMP C++ DLL
        if self.is_native_available():
            try:
                res = evaluate_native(ast, X, Y, params)
                self._record_eval("native", start_t)
                return res
            except Exception:
                self._telemetry["fallback_count"] += 1

        # Tier 3: NumPy Reference Baseline
        res = evaluate_numpy(ast, X, Y, params)
        self._record_eval("numpy", start_t)
        return res

    def compute_residuals(
        self,
        pred: np.ndarray,
        target: np.ndarray,
        preferred_backend: Optional[str] = None,
    ) -> np.ndarray:
        """Computes point-wise residual array (pred - target)."""
        backend = self._normalize_backend_name(preferred_backend or self.preferred_backend)

        if backend == "cuda" and self.is_cuda_available():
            try:
                return compute_residuals_cupy(pred, target)
            except Exception:
                pass

        if (backend in ("native", "auto")) and self.is_native_available():
            try:
                return compute_residuals_native(pred, target)
            except Exception:
                pass

        return compute_residuals_numpy(pred, target)

    def compute_sse(
        self,
        pred: np.ndarray,
        target: np.ndarray,
        preferred_backend: Optional[str] = None,
    ) -> float:
        """Computes sum of squared errors."""
        backend = self._normalize_backend_name(preferred_backend or self.preferred_backend)

        if backend == "cuda" and self.is_cuda_available():
            try:
                return compute_sse_cupy(pred, target)
            except Exception:
                pass

        if (backend in ("native", "auto")) and self.is_native_available():
            try:
                return compute_sse_native(pred, target)
            except Exception:
                pass

        return compute_sse_numpy(pred, target)

    def verify_tier_consistency(
        self,
        ast: ASTNode,
        X: np.ndarray,
        Y: np.ndarray,
        params: np.ndarray,
        tolerance: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Evaluates the same AST expression across all active backends and confirms
        cross-tier numerical parity within the specified tolerance.
        """
        tol = tolerance or self.default_tolerance
        results: Dict[str, np.ndarray] = {}
        diffs: Dict[str, float] = {}

        # Reference baseline: NumPy
        z_numpy = evaluate_numpy(ast, X, Y, params)
        results["numpy"] = z_numpy

        passed_all = True
        if self.is_native_available():
            z_native = evaluate_native(ast, X, Y, params)
            results["native"] = z_native
            diffs["native_vs_numpy"] = float(np.max(np.abs(z_native - z_numpy)))
            if not np.allclose(z_native, z_numpy, rtol=1e-5, atol=tol):
                passed_all = False

        if self.is_cuda_available():
            z_cuda = evaluate_cupy(ast, X, Y, params)
            results["cuda"] = z_cuda
            diffs["cuda_vs_numpy"] = float(np.max(np.abs(z_cuda - z_numpy)))
            if not np.allclose(z_cuda, z_numpy, rtol=1e-5, atol=tol):
                passed_all = False

        return {
            "passed": passed_all,
            "tolerance": tol,
            "max_differences": diffs,
            "active_tiers": list(results.keys()),
        }

    def _record_eval(self, backend: str, start_time: float) -> None:
        self._last_backend_used = backend
        self._last_eval_time_ms = (time.perf_counter() - start_time) * 1000.0
        self._telemetry["tier_counts"][backend] = self._telemetry["tier_counts"].get(backend, 0) + 1
