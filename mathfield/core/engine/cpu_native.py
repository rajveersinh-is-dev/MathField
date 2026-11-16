"""
Tier 2 Accelerated Evaluator: MSVC OpenMP C++ Dynamic Link Library.
Zero-allocation Bytecode VM and SIMD residual kernels via ctypes.
"""

from __future__ import annotations

import ctypes
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Tuple
import numpy as np

from mathfield.core.ast.nodes import (
    ASTNode,
    BinaryOpNode,
    ConstantNode,
    FunctionCallNode,
    ParameterNode,
    UnaryOpNode,
    VariableNode,
)


class NativeKernelNotAvailableError(RuntimeError):
    """Raised when native C++ kernel DLL is requested but unavailable."""
    pass


class BytecodeProgram(NamedTuple):
    opcodes: np.ndarray      # uint32
    operands: np.ndarray     # int32
    const_pool: np.ndarray   # float64
    max_stack_depth: int = 0


def compile_bytecode(node: ASTNode) -> BytecodeProgram:
    """
    Compiles an ASTNode expression tree into flat bytecode for the C++ Virtual Machine.
    """
    opcodes: List[int] = []
    operands: List[int] = []
    const_pool: List[float] = []
    const_map: Dict[float, int] = {}
    current_sp = 0
    max_sp = 0

    def get_const_idx(val: float) -> int:
        if val in const_map:
            return const_map[val]
        idx = len(const_pool)
        const_pool.append(val)
        const_map[val] = idx
        return idx

    def emit(op: int, arg: int = 0) -> None:
        nonlocal current_sp, max_sp
        opcodes.append(op)
        operands.append(arg)
        if op in (1, 2, 3, 4):  # LOAD_X, LOAD_Y, LOAD_CONST, LOAD_PARAM
            current_sp += 1
            if current_sp > max_sp:
                max_sp = current_sp
        elif op in (5, 6, 7, 8, 9, 10, 28, 29, 30):  # Binary / 2-arg RBF
            current_sp -= 1
        elif op == 32:  # CLAMP: pop 3, push 1
            current_sp -= 2

    def traverse(curr: ASTNode) -> None:
        if isinstance(curr, VariableNode):
            name = curr.name.lower()
            if name in ("x", "u"):
                emit(1, 0)  # LOAD_X
            elif name in ("y", "v"):
                emit(2, 0)  # LOAD_Y
            else:
                emit(1, 0)

        elif isinstance(curr, ConstantNode):
            idx = get_const_idx(float(curr.value))
            emit(3, idx)  # LOAD_CONST

        elif isinstance(curr, ParameterNode):
            emit(4, curr.index)  # LOAD_PARAM

        elif isinstance(curr, UnaryOpNode):
            traverse(curr.operand)
            if curr.op == "-":
                emit(11, 0)  # NEG

        elif isinstance(curr, BinaryOpNode):
            traverse(curr.left)
            traverse(curr.right)
            op = curr.op
            if op == "+": emit(5, 0)
            elif op == "-": emit(6, 0)
            elif op == "*": emit(7, 0)
            elif op == "/": emit(8, 0)
            elif op in ("^", "**"): emit(9, 0)
            elif op == "%": emit(10, 0)

        elif isinstance(curr, FunctionCallNode):
            for arg in curr.args:
                traverse(arg)
            f = curr.func
            if f == "sin": emit(13, 0)
            elif f == "cos": emit(14, 0)
            elif f == "tan": emit(15, 0)
            elif f == "sinc": emit(16, 0)
            elif f == "asin": emit(17, 0)
            elif f == "acos": emit(18, 0)
            elif f == "atan": emit(19, 0)
            elif f == "exp": emit(20, 0)
            elif f == "log": emit(21, 0)
            elif f == "sqrt": emit(22, 0)
            elif f == "cbrt": emit(23, 0)
            elif f == "square": emit(24, 0)
            elif f == "abs": emit(25, 0)
            elif f == "sign": emit(26, 0)
            elif f == "sigmoid": emit(27, 0)
            elif f == "rbf_wendland": emit(28, 0)
            elif f == "rbf_gaussian": emit(29, 0)
            elif f == "rbf_multiquadric": emit(30, 0)
            elif f == "softplus": emit(31, 0)
            elif f == "clamp": emit(32, 0)
            elif f == "sinh": emit(33, 0)
            elif f == "cosh": emit(34, 0)
            elif f == "tanh": emit(35, 0)
            elif f == "pow": emit(9, 0)

    traverse(node)
    emit(36, 0)  # RET

    return BytecodeProgram(
        opcodes=np.ascontiguousarray(opcodes, dtype=np.uint32),
        operands=np.ascontiguousarray(operands, dtype=np.int32),
        const_pool=np.ascontiguousarray(const_pool if const_pool else [0.0], dtype=np.float64),
        max_stack_depth=max_sp,
    )


class NativeKernelManager:
    """
    Singleton manager providing ctypes bindings to mathfield_kernels.dll.
    """
    _instance: Optional[NativeKernelManager] = None

    def __init__(self) -> None:
        self.dll: Optional[ctypes.CDLL] = None
        self.is_available: bool = False
        self.dll_path: Optional[str] = None
        self._load_dll()

    @classmethod
    def get_instance(cls) -> NativeKernelManager:
        if cls._instance is None:
            cls._instance = NativeKernelManager()
        return cls._instance

    def _load_dll(self) -> None:
        candidate_paths: List[Path] = []

        # 1. Environment variable override
        env_path = os.environ.get("MATHFIELD_NATIVE_DLL")
        if env_path:
            candidate_paths.append(Path(env_path))

        # 2. Package engine directory
        current_dir = Path(__file__).parent.resolve()
        candidate_paths.append(current_dir / "mathfield_kernels.dll")
        candidate_paths.append(current_dir / "libmathfield_kernels.so")

        # 3. native/build/Release directory
        root_dir = current_dir.parent.parent.parent.resolve()
        candidate_paths.append(root_dir / "native" / "build" / "Release" / "mathfield_kernels.dll")
        candidate_paths.append(root_dir / "native" / "build" / "mathfield_kernels.dll")

        for p in candidate_paths:
            if p.is_file():
                try:
                    loaded = ctypes.CDLL(str(p))
                    self._bind_symbols(loaded)
                    self.dll = loaded
                    self.dll_path = str(p)
                    self.is_available = True
                    return
                except Exception:
                    continue

        self.is_available = False
        self.dll = None

    def _bind_symbols(self, dll: ctypes.CDLL) -> None:
        dll.get_openmp_thread_count.restype = ctypes.c_int
        dll.get_openmp_thread_count.argtypes = []

        dll.set_openmp_thread_count.restype = None
        dll.set_openmp_thread_count.argtypes = [ctypes.c_int]

        dll.compute_residuals_f64.restype = None
        dll.compute_residuals_f64.argtypes = [
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_size_t,
        ]

        dll.compute_sse_f64.restype = ctypes.c_double
        dll.compute_sse_f64.argtypes = [
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_size_t,
        ]

        dll.evaluate_bytecode_f64.restype = ctypes.c_int
        dll.evaluate_bytecode_f64.argtypes = [
            ctypes.POINTER(ctypes.c_uint32),
            ctypes.POINTER(ctypes.c_int32),
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_double),
        ]


def build_native_dll(force: bool = False) -> bool:
    """
    Compiles mathfield_kernels.cpp into a native shared library using MSVC.
    Places the resulting DLL into mathfield/core/engine/mathfield_kernels.dll.
    """
    engine_dir = Path(__file__).parent.resolve()
    target_dll = engine_dir / "mathfield_kernels.dll"
    if target_dll.is_file() and not force:
        mgr = NativeKernelManager.get_instance()
        mgr._load_dll()
        return mgr.is_available

    root_dir = engine_dir.parent.parent.parent.resolve()
    cpp_source = root_dir / "native" / "src" / "mathfield_kernels.cpp"

    if not cpp_source.is_file():
        return False

    vcvars_path = Path(r"C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvars64.bat")
    if not vcvars_path.is_file():
        # Search VS 2022 default paths
        vcvars_alt = Path(r"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat")
        if vcvars_alt.is_file():
            vcvars_path = vcvars_alt

    if vcvars_path.is_file():
        cmd = (
            f'call "{vcvars_path}" && '
            f'cl /std:c++17 /O2 /fp:precise /W4 /MD /openmp /LD "{cpp_source}" /Fe"{target_dll}"'
        )
        try:
            res = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd=str(engine_dir))
            if res.returncode == 0 and target_dll.is_file():
                mgr = NativeKernelManager.get_instance()
                mgr._load_dll()
                return mgr.is_available
        except Exception:
            pass

    return False


def is_native_available() -> bool:
    return NativeKernelManager.get_instance().is_available


def get_openmp_thread_count() -> int:
    mgr = NativeKernelManager.get_instance()
    if not mgr.is_available or mgr.dll is None:
        raise NativeKernelNotAvailableError("MSVC OpenMP DLL is not loaded.")
    return int(mgr.dll.get_openmp_thread_count())


def compute_residuals_native(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    mgr = NativeKernelManager.get_instance()
    if not mgr.is_available or mgr.dll is None:
        raise NativeKernelNotAvailableError("Native C++ OpenMP DLL is not available.")

    p_c = np.ascontiguousarray(pred, dtype=np.float64)
    t_c = np.ascontiguousarray(target, dtype=np.float64)
    out = np.empty_like(p_c)

    p_ptr = p_c.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    t_ptr = t_c.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    out_ptr = out.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

    mgr.dll.compute_residuals_f64(p_ptr, t_ptr, out_ptr, p_c.size)
    return out


def compute_sse_native(pred: np.ndarray, target: np.ndarray) -> float:
    mgr = NativeKernelManager.get_instance()
    if not mgr.is_available or mgr.dll is None:
        raise NativeKernelNotAvailableError("Native C++ OpenMP DLL is not available.")

    p_c = np.ascontiguousarray(pred, dtype=np.float64)
    t_c = np.ascontiguousarray(target, dtype=np.float64)

    p_ptr = p_c.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    t_ptr = t_c.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

    return float(mgr.dll.compute_sse_f64(p_ptr, t_ptr, p_c.size))


def evaluate_native(
    node: ASTNode,
    X: np.ndarray,
    Y: np.ndarray,
    params: np.ndarray,
) -> np.ndarray:
    """
    Evaluates ASTNode using C++ OpenMP Bytecode VM.
    """
    if X.shape != Y.shape:
        raise ValueError(f"Coordinate arrays X and Y must have identical shapes, got {X.shape} and {Y.shape}")

    from mathfield.core.ast.nodes import validate_parameter_bounds
    validate_parameter_bounds(node, params)

    from mathfield.core.engine.cpu_numpy import evaluate_numpy

    prog = compile_bytecode(node)
    if prog.max_stack_depth > 256:
        # Fallback cleanly to NumPy if expression stack depth exceeds 256
        return evaluate_numpy(node, X, Y, params)

    mgr = NativeKernelManager.get_instance()
    if not mgr.is_available or mgr.dll is None:
        raise NativeKernelNotAvailableError("Native C++ OpenMP DLL is not available.")

    x_c = np.ascontiguousarray(X, dtype=np.float64)
    y_c = np.ascontiguousarray(Y, dtype=np.float64)
    p_c = np.ascontiguousarray(params, dtype=np.float64)

    if x_c.size == 0:
        return np.empty_like(x_c).reshape(X.shape)

    out = np.empty_like(x_c)

    op_ptr = prog.opcodes.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32))
    arg_ptr = prog.operands.ctypes.data_as(ctypes.POINTER(ctypes.c_int32))
    const_ptr = prog.const_pool.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    x_ptr = x_c.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    y_ptr = y_c.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    params_ptr = p_c.ctypes.data_as(ctypes.POINTER(ctypes.c_double)) if p_c.size > 0 else None
    out_ptr = out.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

    ret = mgr.dll.evaluate_bytecode_f64(
        op_ptr,
        arg_ptr,
        len(prog.opcodes),
        const_ptr,
        len(prog.const_pool),
        x_ptr,
        y_ptr,
        x_c.size,
        params_ptr,
        p_c.size,
        out_ptr,
    )
    if ret != 0:
        return evaluate_numpy(node, X, Y, params)

    return out.reshape(X.shape)
