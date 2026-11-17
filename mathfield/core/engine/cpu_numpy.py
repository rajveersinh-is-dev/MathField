"""
Tier 3 Accelerated Evaluator: Vectorized CPU evaluation using NumPy ufuncs.
Uses recursive closure composition with zero use of eval() or exec().
"""

from __future__ import annotations

import functools
from typing import Callable, Dict

import numpy as np

from mathfield.core.ast.domain import (
    clamp_fn,
    protected_acos,
    protected_asin,
    protected_cbrt,
    protected_exp,
    protected_log,
    protected_pow,
    protected_sinc,
    protected_sqrt,
    protected_tan,
    rbf_gaussian,
    rbf_multiquadric,
    rbf_wendland,
    regularized_div,
    sigmoid_fn,
    softplus_fn,
)
from mathfield.core.ast.nodes import (
    ASTNode,
    BinaryOpNode,
    ConstantNode,
    FunctionCallNode,
    ParameterNode,
    UnaryOpNode,
    VariableNode,
)

# Global closure compilation cache keyed by AST hash
_COMPILATION_CACHE: Dict[int, Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]] = {}


def compile_numpy(node: ASTNode) -> Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]:
    """
    Compiles an ASTNode into a vectorized closure over (X, Y, params).
    Zero use of eval or exec.
    """
    node_hash = hash(node)
    if node_hash in _COMPILATION_CACHE:
        return _COMPILATION_CACHE[node_hash]

    raw_fn = _build_closure(node)

    @functools.wraps(raw_fn)
    def vector_eval(X: np.ndarray, Y: np.ndarray, params: np.ndarray) -> np.ndarray:
        res = raw_fn(X, Y, params)
        if np.isscalar(res):
            return np.full_like(X, res, dtype=X.dtype)
        # In case a 0-d array is returned
        if isinstance(res, np.ndarray) and res.ndim == 0:
            return np.full_like(X, float(res), dtype=X.dtype)
        return res

    _COMPILATION_CACHE[node_hash] = vector_eval
    return vector_eval


def _build_closure(node: ASTNode) -> Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]:
    """Recursively constructs composed closures from AST nodes."""
    if isinstance(node, ConstantNode):
        val = float(node.value)
        return lambda X, Y, p: val

    elif isinstance(node, VariableNode):
        name = node.name.lower()
        if name in ("x", "u"):
            return lambda X, Y, p: X
        elif name in ("y", "v"):
            return lambda X, Y, p: Y
        return lambda X, Y, p: X

    elif isinstance(node, ParameterNode):
        idx = node.index
        return lambda X, Y, p: p[idx]

    elif isinstance(node, UnaryOpNode):
        child_fn = _build_closure(node.operand)
        if node.op == "-":
            return lambda X, Y, p: np.negative(child_fn(X, Y, p))
        elif node.op == "+":
            return child_fn

    elif isinstance(node, BinaryOpNode):
        l_fn = _build_closure(node.left)
        r_fn = _build_closure(node.right)
        op = node.op

        if op == "+":
            return lambda X, Y, p: np.add(l_fn(X, Y, p), r_fn(X, Y, p))
        elif op == "-":
            return lambda X, Y, p: np.subtract(l_fn(X, Y, p), r_fn(X, Y, p))
        elif op == "*":
            return lambda X, Y, p: np.multiply(l_fn(X, Y, p), r_fn(X, Y, p))
        elif op == "/":
            return lambda X, Y, p: regularized_div(l_fn(X, Y, p), r_fn(X, Y, p))
        elif op in ("^", "**"):
            return lambda X, Y, p: protected_pow(l_fn(X, Y, p), r_fn(X, Y, p))
        elif op == "%":
            return lambda X, Y, p: np.fmod(l_fn(X, Y, p), np.maximum(np.abs(r_fn(X, Y, p)), 1e-12))

    elif isinstance(node, FunctionCallNode):
        arg_fns = [_build_closure(arg) for arg in node.args]
        f = node.func

        if f == "sin":
            fn0 = arg_fns[0]
            return lambda X, Y, p: np.sin(fn0(X, Y, p))
        elif f == "cos":
            fn0 = arg_fns[0]
            return lambda X, Y, p: np.cos(fn0(X, Y, p))
        elif f == "tan":
            fn0 = arg_fns[0]
            return lambda X, Y, p: protected_tan(fn0(X, Y, p))
        elif f == "sinc":
            fn0 = arg_fns[0]
            return lambda X, Y, p: protected_sinc(fn0(X, Y, p))
        elif f == "asin":
            fn0 = arg_fns[0]
            return lambda X, Y, p: protected_asin(fn0(X, Y, p))
        elif f == "acos":
            fn0 = arg_fns[0]
            return lambda X, Y, p: protected_acos(fn0(X, Y, p))
        elif f == "atan":
            fn0 = arg_fns[0]
            return lambda X, Y, p: np.arctan(fn0(X, Y, p))
        elif f == "sinh":
            fn0 = arg_fns[0]
            return lambda X, Y, p: np.sinh(np.clip(fn0(X, Y, p), -80.0, 80.0))
        elif f == "cosh":
            fn0 = arg_fns[0]
            return lambda X, Y, p: np.cosh(np.clip(fn0(X, Y, p), -80.0, 80.0))
        elif f == "tanh":
            fn0 = arg_fns[0]
            return lambda X, Y, p: np.tanh(fn0(X, Y, p))
        elif f == "exp":
            fn0 = arg_fns[0]
            return lambda X, Y, p: protected_exp(fn0(X, Y, p))
        elif f == "log":
            fn0 = arg_fns[0]
            return lambda X, Y, p: protected_log(fn0(X, Y, p))
        elif f == "sqrt":
            fn0 = arg_fns[0]
            return lambda X, Y, p: protected_sqrt(fn0(X, Y, p))
        elif f == "cbrt":
            fn0 = arg_fns[0]
            return lambda X, Y, p: protected_cbrt(fn0(X, Y, p))
        elif f == "square":
            fn0 = arg_fns[0]
            return lambda X, Y, p: np.square(fn0(X, Y, p))
        elif f == "pow":
            fn0, fn1 = arg_fns[0], arg_fns[1]
            return lambda X, Y, p: protected_pow(fn0(X, Y, p), fn1(X, Y, p))
        elif f == "abs":
            fn0 = arg_fns[0]
            return lambda X, Y, p: np.abs(fn0(X, Y, p))
        elif f == "sign":
            fn0 = arg_fns[0]
            return lambda X, Y, p: np.sign(fn0(X, Y, p))
        elif f == "sigmoid":
            fn0 = arg_fns[0]
            return lambda X, Y, p: sigmoid_fn(fn0(X, Y, p))
        elif f == "softplus":
            fn0 = arg_fns[0]
            return lambda X, Y, p: softplus_fn(fn0(X, Y, p))
        elif f == "clamp":
            fn0, fn1, fn2 = arg_fns[0], arg_fns[1], arg_fns[2]
            return lambda X, Y, p: clamp_fn(fn0(X, Y, p), fn1(X, Y, p), fn2(X, Y, p))
        elif f == "rbf_gaussian":
            fn0, fn1 = arg_fns[0], arg_fns[1]
            return lambda X, Y, p: rbf_gaussian(fn0(X, Y, p), fn1(X, Y, p))
        elif f == "rbf_multiquadric":
            fn0, fn1 = arg_fns[0], arg_fns[1]
            return lambda X, Y, p: rbf_multiquadric(fn0(X, Y, p), fn1(X, Y, p))
        elif f == "rbf_wendland":
            fn0, fn1 = arg_fns[0], arg_fns[1]
            return lambda X, Y, p: rbf_wendland(fn0(X, Y, p), fn1(X, Y, p))

    raise ValueError(f"Unknown or unsupported AST node: {type(node)}")


def evaluate_numpy(
    node: ASTNode,
    X: np.ndarray,
    Y: np.ndarray,
    params: np.ndarray,
) -> np.ndarray:
    """
    Evaluates an AST expression on coordinate grids X, Y with parameters using NumPy.
    """
    from mathfield.core.ast.nodes import validate_parameter_bounds

    validate_parameter_bounds(node, params)
    fn = compile_numpy(node)
    return fn(X, Y, params)


def compute_residuals_numpy(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    """
    Computes elementwise residuals: pred - target.
    """
    return pred - target


def compute_sse_numpy(pred: np.ndarray, target: np.ndarray) -> float:
    """
    Computes sum of squared errors: sum((pred - target)^2).
    """
    diff = pred - target
    return float(np.sum(diff * diff))
