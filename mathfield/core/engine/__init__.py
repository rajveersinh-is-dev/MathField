"""
Multi-tier accelerated evaluation engine for MathField.
"""

from mathfield.core.engine.cpu_numpy import (
    compile_numpy,
    compute_residuals_numpy,
    compute_sse_numpy,
    evaluate_numpy,
)
from mathfield.core.engine.cpu_native import (
    BytecodeProgram,
    NativeKernelManager,
    NativeKernelNotAvailableError,
    build_native_dll,
    compile_bytecode,
    compute_residuals_native,
    compute_sse_native,
    evaluate_native,
    get_openmp_thread_count,
    is_native_available,
)
from mathfield.core.engine.gpu_cupy import (
    CUDANotAvailableError,
    CUDAOutOfMemoryError,
    compile_cupy_kernel,
    compute_residuals_cupy,
    compute_sse_cupy,
    evaluate_cupy,
    is_cuda_available,
)
from mathfield.core.engine.backend import MultiTierEvaluator

__all__ = [
    # NumPy CPU
    "compile_numpy",
    "evaluate_numpy",
    "compute_residuals_numpy",
    "compute_sse_numpy",
    # Native C++ OpenMP
    "NativeKernelManager",
    "NativeKernelNotAvailableError",
    "BytecodeProgram",
    "compile_bytecode",
    "evaluate_native",
    "compute_residuals_native",
    "compute_sse_native",
    "get_openmp_thread_count",
    "is_native_available",
    "build_native_dll",
    # CuPy GPU
    "is_cuda_available",
    "compile_cupy_kernel",
    "evaluate_cupy",
    "compute_residuals_cupy",
    "compute_sse_cupy",
    "CUDAOutOfMemoryError",
    "CUDANotAvailableError",
    # Unified Dispatcher
    "MultiTierEvaluator",
]
