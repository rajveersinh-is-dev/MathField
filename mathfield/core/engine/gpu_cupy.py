"""
Tier 1 Accelerated Evaluator: CuPy Fused CUDA ElementwiseKernel Engine.
Compiles AST expressions into register-resident CUDA C++ elementwise kernels on NVIDIA GPU.
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, Optional, Tuple, Union
import numpy as np

# Global flag and cache
_CUPY_AVAILABLE: Optional[bool] = None
_KERNEL_CACHE: Dict[int, Any] = {}


class CUDAOutOfMemoryError(RuntimeError):
    """Raised when GPU VRAM allocation fails."""
    pass


class CUDANotAvailableError(RuntimeError):
    """Raised when CUDA/CuPy is requested but unavailable."""
    pass


def is_cuda_available() -> bool:
    """Checks whether CuPy and a compatible NVIDIA CUDA GPU are available."""
    global _CUPY_AVAILABLE
    if _CUPY_AVAILABLE is not None:
        return _CUPY_AVAILABLE
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            import cupy as cp
        count = cp.cuda.runtime.getDeviceCount()
        if count > 0:
            # Perform a test kernel run to ensure CUDA runtime initializes properly
            _ = cp.array([1.0], dtype=cp.float32) + cp.array([1.0], dtype=cp.float32)
            _CUPY_AVAILABLE = True
        else:
            _CUPY_AVAILABLE = False
    except Exception:
        _CUPY_AVAILABLE = False
    return _CUPY_AVAILABLE


CUDA_PREAMBLE = r"""
#ifndef M_PI
#define M_PI 3.1415926535897932384626433832795028841971
#endif
#ifndef M_E
#define M_E 2.7182818284590452353602874713526624977
#endif

template <typename T>
__device__ inline T safe_div(T u, T v, T eps) {
    T s = (v >= (T)0.0) ? (T)1.0 : (T)-1.0;
    return u / (v + eps * s);
}

template <typename T>
__device__ inline T safe_log(T x, T eps) {
    return log((x > eps) ? x : eps);
}

template <typename T>
__device__ inline T safe_exp(T x) {
    if (x > (T)80.0) x = (T)80.0;
    if (x < (T)-80.0) x = (T)-80.0;
    return exp(x);
}

template <typename T>
__device__ inline T safe_sqrt(T x) {
    if (isnan(x)) return (T)0.0;
    return sqrt((x > (T)0.0) ? x : (T)0.0);
}

template <typename T>
__device__ inline T safe_cbrt(T x) {
    return cbrt(x);
}

template <typename T>
__device__ inline T safe_pow(T u, T v) {
    if (isnan(u) || isnan(v)) return (T)0.0;
    if (fabs(u) < (T)1e-15 && fabs(v) < (T)1e-15) return (T)1.0;
    T abs_u = (u >= (T)0.0) ? u : -u;
    if (abs_u < (T)1e-15) abs_u = (T)1e-15;
    T clamped_v = v;
    if (clamped_v > (T)80.0) clamped_v = (T)80.0;
    if (clamped_v < (T)-80.0) clamped_v = (T)-80.0;

    T s = (T)1.0;
    if (u < (T)0.0) {
        T r = round(clamped_v);
        if (fabs(clamped_v - r) < (T)1e-7) {
            long long int_v = (long long)fabs(r);
            if (int_v % 2 != 0) {
                s = (T)-1.0;
            }
        }
    }
    T log_abs_u = log(abs_u);
    T exp_val = clamped_v * log_abs_u;
    if (exp_val > (T)80.0) exp_val = (T)80.0;
    if (exp_val < (T)-80.0) exp_val = (T)-80.0;
    return s * exp(exp_val);
}

template <typename T>
__device__ inline T safe_sinc(T x) {
    if (fabs(x) < (T)1e-12) return (T)1.0;
    T px = (T)M_PI * x;
    return sin(px) / px;
}

template <typename T>
__device__ inline T safe_tan(T x) {
    T val = tan(x);
    if (val > (T)1e6) val = (T)1e6;
    if (val < (T)-1e6) val = (T)-1e6;
    return val;
}

template <typename T>
__device__ inline T safe_asin(T x) {
    if (x > (T)1.0) x = (T)1.0;
    if (x < (T)-1.0) x = (T)-1.0;
    return asin(x);
}

template <typename T>
__device__ inline T safe_acos(T x) {
    if (x > (T)1.0) x = (T)1.0;
    if (x < (T)-1.0) x = (T)-1.0;
    return acos(x);
}

template <typename T>
__device__ inline T rbf_wendland(T r, T rad) {
    T safe_rad = (rad >= (T)0.0 ? rad : -rad);
    if (safe_rad < (T)1e-12) safe_rad = (T)1e-12;
    T r_pos = (r < (T)0.0) ? (T)0.0 : r;
    T norm_r = r_pos / safe_rad;
    if (norm_r >= (T)1.0) return (T)0.0;
    T term = (T)1.0 - norm_r;
    T term4 = term * term * term * term;
    return term4 * ((T)4.0 * norm_r + (T)1.0);
}

template <typename T>
__device__ inline T rbf_gaussian(T r, T eps) {
    T scaled = eps * r;
    T v = scaled * scaled;
    if (v > (T)80.0) v = (T)80.0;
    return exp(-v);
}

template <typename T>
__device__ inline T rbf_multiquadric(T r, T eps) {
    T scaled = eps * r;
    if (scaled > (T)1e80) scaled = (T)1e80;
    if (scaled < (T)-1e80) scaled = (T)-1e80;
    T sc2 = scaled * scaled;
    if (sc2 > (T)1e160) sc2 = (T)1e160;
    return sqrt((T)1.0 + sc2);
}

template <typename T>
__device__ inline T sigmoid_fn(T x) {
    if (x > (T)80.0) x = (T)80.0;
    if (x < (T)-80.0) x = (T)-80.0;
    return (T)1.0 / ((T)1.0 + exp(-x));
}

template <typename T>
__device__ inline T softplus_fn(T x) {
    T ax = (x >= (T)0.0) ? x : -x;
    T max_x = (x > (T)0.0) ? x : (T)0.0;
    return log1p(exp(-ax)) + max_x;
}

template <typename T>
__device__ inline T clamp_fn(T x, T min_val, T max_val) {
    if (x < min_val) return min_val;
    if (x > max_val) return max_val;
    return x;
}

template <typename T>
__device__ inline T safe_sinh(T x) {
    if (x > (T)80.0) x = (T)80.0;
    if (x < (T)-80.0) x = (T)-80.0;
    return sinh(x);
}

template <typename T>
__device__ inline T safe_cosh(T x) {
    if (x > (T)80.0) x = (T)80.0;
    if (x < (T)-80.0) x = (T)-80.0;
    return cosh(x);
}
"""


def ast_to_cuda_snippet(node: Any) -> str:
    """Emits C++/CUDA expression string from ASTNode."""
    from mathfield.core.ast.nodes import (
        BinaryOpNode,
        ConstantNode,
        FunctionCallNode,
        ParameterNode,
        UnaryOpNode,
        VariableNode,
    )

    if isinstance(node, ConstantNode):
        if node.symbol == "pi":
            return "(T)M_PI"
        elif node.symbol == "e":
            return "(T)M_E"
        return f"((T){node.value:.17g})"

    elif isinstance(node, VariableNode):
        name = node.name.lower()
        if name in ("x", "u"):
            return "x"
        elif name in ("y", "v"):
            return "y"
        return "x"

    elif isinstance(node, ParameterNode):
        return f"params[{node.index}]"

    elif isinstance(node, UnaryOpNode):
        child = ast_to_cuda_snippet(node.operand)
        if node.op == "-":
            return f"(-({child}))"
        return child

    elif isinstance(node, BinaryOpNode):
        l = ast_to_cuda_snippet(node.left)
        r = ast_to_cuda_snippet(node.right)
        op = node.op

        if op == "+": return f"(({l}) + ({r}))"
        elif op == "-": return f"(({l}) - ({r}))"
        elif op == "*": return f"(({l}) * ({r}))"
        elif op == "/": return f"safe_div(({l}), ({r}), (T)1e-12)"
        elif op in ("^", "**"): return f"safe_pow(({l}), ({r}))"
        elif op == "%": return f"fmod(({l}), (({r}) != (T)0.0 ? ({r}) : (T)1e-12))"

    elif isinstance(node, FunctionCallNode):
        args = [ast_to_cuda_snippet(arg) for arg in node.args]
        f = node.func

        if f == "sin": return f"sin({args[0]})"
        elif f == "cos": return f"cos({args[0]})"
        elif f == "tan": return f"safe_tan({args[0]})"
        elif f == "sinc": return f"safe_sinc({args[0]})"
        elif f == "asin": return f"safe_asin({args[0]})"
        elif f == "acos": return f"safe_acos({args[0]})"
        elif f == "atan": return f"atan({args[0]})"
        elif f == "sinh": return f"safe_sinh({args[0]})"
        elif f == "cosh": return f"safe_cosh({args[0]})"
        elif f == "tanh": return f"tanh({args[0]})"
        elif f == "exp": return f"safe_exp({args[0]})"
        elif f == "log": return f"safe_log({args[0]}, (T)1e-15)"
        elif f == "sqrt": return f"safe_sqrt({args[0]})"
        elif f == "cbrt": return f"safe_cbrt({args[0]})"
        elif f == "square": return f"(({args[0]}) * ({args[0]}))"
        elif f == "pow": return f"safe_pow(({args[0]}), ({args[1]}))"
        elif f == "abs": return f"fabs({args[0]})"
        elif f == "sign": return f"((({args[0]}) > (T)0.0) ? (T)1.0 : ((({args[0]}) < (T)0.0) ? (T)-1.0 : (T)0.0))"
        elif f == "sigmoid": return f"sigmoid_fn({args[0]})"
        elif f == "softplus": return f"softplus_fn({args[0]})"
        elif f == "clamp": return f"clamp_fn(({args[0]}), ({args[1]}), ({args[2]}))"
        elif f == "rbf_gaussian": return f"rbf_gaussian(({args[0]}), ({args[1]}))"
        elif f == "rbf_multiquadric": return f"rbf_multiquadric(({args[0]}), ({args[1]}))"
        elif f == "rbf_wendland": return f"rbf_wendland(({args[0]}), ({args[1]}))"

    return f"({node})"


def compile_cupy_kernel(node: Any) -> Any:
    """Compiles ASTNode into a CuPy ElementwiseKernel."""
    if not is_cuda_available():
        raise CUDANotAvailableError("CUDA/CuPy is not available.")

    import cupy as cp

    node_hash = hash(node)
    if node_hash in _KERNEL_CACHE:
        return _KERNEL_CACHE[node_hash]

    snippet = ast_to_cuda_snippet(node)
    kernel_name = f"mathfield_kernel_{abs(node_hash)}"
    kernel = cp.ElementwiseKernel(
        in_params="T x, T y, raw T params",
        out_params="T z",
        operation=f"z = {snippet};",
        name=kernel_name,
        preamble=CUDA_PREAMBLE,
    )
    _KERNEL_CACHE[node_hash] = kernel
    return kernel


def evaluate_cupy(
    node: Any,
    X: Union[np.ndarray, Any],
    Y: Union[np.ndarray, Any],
    params: Union[np.ndarray, Any],
    return_cupy: bool = False,
) -> Union[np.ndarray, Any]:
    """
    Evaluates ASTNode using NVIDIA GPU with CuPy.
    Supports resident GPU arrays or automatic NumPy<->CuPy transfers.
    """
    if not is_cuda_available():
        raise CUDANotAvailableError("CUDA/CuPy is not available.")

    if hasattr(X, "shape") and hasattr(Y, "shape") and X.shape != Y.shape:
        raise ValueError(f"Coordinate arrays X and Y must have identical shapes, got {X.shape} and {Y.shape}")

    from mathfield.core.ast.nodes import validate_parameter_bounds
    validate_parameter_bounds(node, params)

    import cupy as cp

    try:
        kernel = compile_cupy_kernel(node)

        # Handle input transfers
        is_cp_x = isinstance(X, cp.ndarray)
        is_cp_y = isinstance(Y, cp.ndarray)
        is_cp_p = isinstance(params, cp.ndarray)

        gpu_x = X if is_cp_x else cp.asarray(X)
        gpu_y = Y if is_cp_y else cp.asarray(Y)
        gpu_params = params if is_cp_p else cp.asarray(params)

        # Allocate output buffer on GPU
        out = cp.empty_like(gpu_x)

        # Launch kernel
        kernel(gpu_x, gpu_y, gpu_params, out)

        if return_cupy:
            return out
        return cp.asnumpy(out) if not is_cp_x else out

    except cp.cuda.memory.OutOfMemoryError as e:
        cp.get_default_memory_pool().free_all_blocks()
        raise CUDAOutOfMemoryError(f"GPU Out Of Memory in evaluate_cupy: {e}") from e


def compute_residuals_cupy(
    pred: Union[np.ndarray, Any],
    target: Union[np.ndarray, Any],
    return_cupy: bool = False,
) -> Union[np.ndarray, Any]:
    if not is_cuda_available():
        raise CUDANotAvailableError("CUDA/CuPy is not available.")

    import cupy as cp
    try:
        gpu_p = pred if isinstance(pred, cp.ndarray) else cp.asarray(pred)
        gpu_t = target if isinstance(target, cp.ndarray) else cp.asarray(target)
        res = gpu_p - gpu_t
        if return_cupy:
            return res
        return cp.asnumpy(res) if not isinstance(pred, cp.ndarray) else res
    except cp.cuda.memory.OutOfMemoryError as e:
        cp.get_default_memory_pool().free_all_blocks()
        raise CUDAOutOfMemoryError(f"GPU OOM in compute_residuals_cupy: {e}") from e


def compute_sse_cupy(
    pred: Union[np.ndarray, Any],
    target: Union[np.ndarray, Any],
) -> float:
    if not is_cuda_available():
        raise CUDANotAvailableError("CUDA/CuPy is not available.")

    import cupy as cp
    try:
        gpu_p = pred if isinstance(pred, cp.ndarray) else cp.asarray(pred)
        gpu_t = target if isinstance(target, cp.ndarray) else cp.asarray(target)
        diff = gpu_p - gpu_t
        sse = float(cp.sum(diff * diff))
        return sse
    except cp.cuda.memory.OutOfMemoryError as e:
        cp.get_default_memory_pool().free_all_blocks()
        raise CUDAOutOfMemoryError(f"GPU OOM in compute_sse_cupy: {e}") from e
