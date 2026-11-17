"""
Immutable Abstract Syntax Tree (AST) node definitions for MathField.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Sequence, Set, Tuple

import numpy as np

from mathfield.core.ast.domain import (
    safe_scalar_acos,
    safe_scalar_asin,
    safe_scalar_cbrt,
    safe_scalar_div,
    safe_scalar_exp,
    safe_scalar_log,
    safe_scalar_pow,
    safe_scalar_rbf_wendland,
    safe_scalar_sigmoid,
    safe_scalar_sinc,
    safe_scalar_softplus,
    safe_scalar_sqrt,
    safe_scalar_tan,
)

# Exact 24 whitelisted functions and required arities
FUNCTION_ARITIES: Dict[str, int] = {
    # Trigonometric / Hyperbolic
    "sin": 1,
    "cos": 1,
    "tan": 1,
    "sinc": 1,
    "asin": 1,
    "acos": 1,
    "atan": 1,
    "sinh": 1,
    "cosh": 1,
    "tanh": 1,
    # Exponential / Logarithmic / Roots
    "exp": 1,
    "log": 1,
    "sqrt": 1,
    "cbrt": 1,
    "square": 1,
    "pow": 2,
    # Special / Activation
    "abs": 1,
    "sign": 1,
    "sigmoid": 1,
    "softplus": 1,
    "clamp": 3,
    # Radial Basis Functions
    "rbf_gaussian": 2,
    "rbf_multiquadric": 2,
    "rbf_wendland": 2,
}

VALID_VARIABLES: Set[str] = {"x", "y", "u", "v"}


class ASTNode(ABC):
    """
    Abstract base class for all immutable AST expressions in MathField.
    """

    @abstractmethod
    def children(self) -> Tuple[ASTNode, ...]:
        """Return immediate child nodes."""

    @abstractmethod
    def complexity(self) -> int:
        """Structural complexity score C(M) for regularized loss J(M) = E(M) + lambda * C(M)."""

    @abstractmethod
    def free_variables(self) -> Set[str]:
        """Set of coordinate variable names used (e.g. {'x', 'y'})."""

    @abstractmethod
    def parameters(self) -> Set[str]:
        """Set of parameter names used (e.g. {'c0', 'theta1'})."""

    @abstractmethod
    def eval_scalar(self, x: float, y: float, params: Sequence[float]) -> float:
        """Pure-Python reference scalar evaluation for unit checks and debugging."""

    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        """Serialize AST node to JSON-compatible dictionary."""

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ASTNode:
        """Reconstruct ASTNode from serialized dictionary."""
        node_type = data.get("type")
        if node_type == "ConstantNode":
            return ConstantNode(value=float(data["value"]), symbol=data.get("symbol"))
        elif node_type == "VariableNode":
            return VariableNode(name=data["name"])
        elif node_type == "ParameterNode":
            return ParameterNode(name=data["name"], index=int(data["index"]))
        elif node_type == "UnaryOpNode":
            return UnaryOpNode(op=data["op"], operand=cls.from_dict(data["operand"]))
        elif node_type == "BinaryOpNode":
            return BinaryOpNode(
                op=data["op"],
                left=cls.from_dict(data["left"]),
                right=cls.from_dict(data["right"]),
            )
        elif node_type == "FunctionCallNode":
            args = tuple(cls.from_dict(arg) for arg in data["args"])
            return FunctionCallNode(func=data["func"], args=args)
        else:
            raise ValueError(f"Unknown AST node type: {node_type}")

    def compile_numpy(self) -> Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]:
        """Compile AST into a vectorized callable closure over (X, Y, params)."""
        from mathfield.core.engine.cpu_numpy import compile_numpy

        return compile_numpy(self)

    def compile_cuda(self) -> str:
        """Emit C++/CUDA elementwise kernel string snippet."""
        from mathfield.core.engine.gpu_cupy import ast_to_cuda_snippet

        return ast_to_cuda_snippet(self)

    def diff(self, param_name: str) -> ASTNode:
        """Compute analytical symbolic derivative with respect to param_name or variable."""
        from mathfield.core.ast.transforms import diff

        return diff(self, param_name)

    def to_latex(self) -> str:
        """Convert AST into AMS-LaTeX string."""
        from mathfield.core.codegen.latex import to_latex

        return to_latex(self)

    def to_python_code(self) -> str:
        """Convert AST into standalone Python function body using NumPy."""
        from mathfield.core.codegen.numpy_gen import to_python_code

        return to_python_code(self)


@dataclass(frozen=True, slots=True)
class ConstantNode(ASTNode):
    """Represents a numerical constant (e.g. 3.14159, or symbolic 'pi', 'e')."""

    value: float
    symbol: Optional[str] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "value", float(self.value))

    def children(self) -> Tuple[ASTNode, ...]:
        return ()

    def complexity(self) -> int:
        return 1

    def free_variables(self) -> Set[str]:
        return set()

    def parameters(self) -> Set[str]:
        return set()

    def eval_scalar(self, x: float, y: float, params: Sequence[float]) -> float:
        return self.value

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {"type": "ConstantNode", "value": self.value}
        if self.symbol:
            d["symbol"] = self.symbol
        return d

    def __repr__(self) -> str:
        if self.symbol:
            return self.symbol
        if self.value.is_integer():
            return f"{int(self.value)}"
        return f"{self.value:g}"


@dataclass(frozen=True, slots=True)
class VariableNode(ASTNode):
    """Represents a coordinate variable (x, y, u, v)."""

    name: str

    def __post_init__(self) -> None:
        norm_name = self.name.lower()
        if norm_name not in VALID_VARIABLES:
            raise ValueError(f"Invalid coordinate variable {self.name!r}. Must be one of {VALID_VARIABLES}")
        object.__setattr__(self, "name", norm_name)

    def children(self) -> Tuple[ASTNode, ...]:
        return ()

    def complexity(self) -> int:
        return 1

    def free_variables(self) -> Set[str]:
        return {self.name}

    def parameters(self) -> Set[str]:
        return set()

    def eval_scalar(self, x: float, y: float, params: Sequence[float]) -> float:
        if self.name in ("x", "u"):
            return x
        elif self.name in ("y", "v"):
            return y
        raise ValueError(f"Unknown variable name {self.name}")

    def to_dict(self) -> Dict[str, Any]:
        return {"type": "VariableNode", "name": self.name}

    def __repr__(self) -> str:
        return self.name


@dataclass(frozen=True, slots=True)
class ParameterNode(ASTNode):
    """Represents a continuously optimized parameter (e.g. c0, theta1, p2)."""

    name: str
    index: int

    def __post_init__(self) -> None:
        if self.index < 0:
            raise ValueError(f"Parameter index must be non-negative, got {self.index}")
        object.__setattr__(self, "name", self.name.lower())

    def children(self) -> Tuple[ASTNode, ...]:
        return ()

    def complexity(self) -> int:
        return 1

    def free_variables(self) -> Set[str]:
        return set()

    def parameters(self) -> Set[str]:
        return {self.name}

    def eval_scalar(self, x: float, y: float, params: Sequence[float]) -> float:
        if self.index >= len(params):
            raise IndexError(f"Parameter index {self.index} out of bounds for params of length {len(params)}")
        return float(params[self.index])

    def to_dict(self) -> Dict[str, Any]:
        return {"type": "ParameterNode", "name": self.name, "index": self.index}

    def __repr__(self) -> str:
        return self.name


@dataclass(frozen=True, slots=True)
class UnaryOpNode(ASTNode):
    """Represents a unary operation (+ or -)."""

    op: str
    operand: ASTNode

    def __post_init__(self) -> None:
        if self.op not in ("+", "-"):
            raise ValueError(f"Invalid unary operator {self.op!r}. Must be '+' or '-'")

    def children(self) -> Tuple[ASTNode, ...]:
        return (self.operand,)

    def complexity(self) -> int:
        return 1 + self.operand.complexity()

    def free_variables(self) -> Set[str]:
        return self.operand.free_variables()

    def parameters(self) -> Set[str]:
        return self.operand.parameters()

    def eval_scalar(self, x: float, y: float, params: Sequence[float]) -> float:
        val = self.operand.eval_scalar(x, y, params)
        return -val if self.op == "-" else val

    def to_dict(self) -> Dict[str, Any]:
        return {"type": "UnaryOpNode", "op": self.op, "operand": self.operand.to_dict()}

    def __repr__(self) -> str:
        return f"({self.op}{self.operand})"


@dataclass(frozen=True, slots=True)
class BinaryOpNode(ASTNode):
    """Represents a binary operation (+, -, *, /, ^, %)."""

    op: str
    left: ASTNode
    right: ASTNode

    def __post_init__(self) -> None:
        valid_ops = {"+", "-", "*", "/", "^", "%"}
        if self.op not in valid_ops:
            raise ValueError(f"Invalid binary operator {self.op!r}. Must be one of {valid_ops}")

    def children(self) -> Tuple[ASTNode, ...]:
        return (self.left, self.right)

    def complexity(self) -> int:
        return 1 + self.left.complexity() + self.right.complexity()

    def free_variables(self) -> Set[str]:
        return self.left.free_variables() | self.right.free_variables()

    def parameters(self) -> Set[str]:
        return self.left.parameters() | self.right.parameters()

    def eval_scalar(self, x: float, y: float, params: Sequence[float]) -> float:
        l_val = self.left.eval_scalar(x, y, params)
        r_val = self.right.eval_scalar(x, y, params)
        if self.op == "+":
            return l_val + r_val
        elif self.op == "-":
            return l_val - r_val
        elif self.op == "*":
            return l_val * r_val
        elif self.op == "/":
            return safe_scalar_div(l_val, r_val)
        elif self.op == "^":
            return safe_scalar_pow(l_val, r_val)
        elif self.op == "%":
            return math.fmod(l_val, max(abs(r_val), 1e-12))
        raise ValueError(f"Unsupported binary operator: {self.op}")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "BinaryOpNode",
            "op": self.op,
            "left": self.left.to_dict(),
            "right": self.right.to_dict(),
        }

    def __repr__(self) -> str:
        return f"({self.left} {self.op} {self.right})"


@dataclass(frozen=True, slots=True)
class FunctionCallNode(ASTNode):
    """Represents an invocation of one of the 24 whitelisted mathematical functions."""

    func: str
    args: Tuple[ASTNode, ...]

    def __post_init__(self) -> None:
        norm_func = self.func.lower()
        if norm_func not in FUNCTION_ARITIES:
            raise ValueError(f"Disallowed or unknown function {self.func!r}. Must be in {set(FUNCTION_ARITIES.keys())}")
        expected_arity = FUNCTION_ARITIES[norm_func]
        # Ensure args is a tuple for immutability and hashability
        if not isinstance(self.args, tuple):
            object.__setattr__(self, "args", tuple(self.args))
        if len(self.args) != expected_arity:
            raise ValueError(f"Function {norm_func} expects {expected_arity} arguments, but got {len(self.args)}")
        object.__setattr__(self, "func", norm_func)

    def children(self) -> Tuple[ASTNode, ...]:
        return self.args

    def complexity(self) -> int:
        # Base cost 2 to penalize non-linear functions in Mode A discovery
        return 2 + sum(arg.complexity() for arg in self.args)

    def free_variables(self) -> Set[str]:
        vars_set: Set[str] = set()
        for arg in self.args:
            vars_set |= arg.free_variables()
        return vars_set

    def parameters(self) -> Set[str]:
        params_set: Set[str] = set()
        for arg in self.args:
            params_set |= arg.parameters()
        return params_set

    def eval_scalar(self, x: float, y: float, params: Sequence[float]) -> float:
        eval_args = [arg.eval_scalar(x, y, params) for arg in self.args]
        f = self.func
        if f == "sin":
            return math.sin(eval_args[0])
        elif f == "cos":
            return math.cos(eval_args[0])
        elif f == "tan":
            return safe_scalar_tan(eval_args[0])
        elif f == "sinc":
            return safe_scalar_sinc(eval_args[0])
        elif f == "asin":
            return safe_scalar_asin(eval_args[0])
        elif f == "acos":
            return safe_scalar_acos(eval_args[0])
        elif f == "atan":
            return math.atan(eval_args[0])
        elif f == "sinh":
            return math.sinh(max(-80.0, min(80.0, eval_args[0])))
        elif f == "cosh":
            return math.cosh(max(-80.0, min(80.0, eval_args[0])))
        elif f == "tanh":
            return math.tanh(eval_args[0])
        elif f == "exp":
            return safe_scalar_exp(eval_args[0])
        elif f == "log":
            return safe_scalar_log(eval_args[0])
        elif f == "sqrt":
            return safe_scalar_sqrt(eval_args[0])
        elif f == "cbrt":
            return safe_scalar_cbrt(eval_args[0])
        elif f == "square":
            return eval_args[0] * eval_args[0]
        elif f == "pow":
            return safe_scalar_pow(eval_args[0], eval_args[1])
        elif f == "abs":
            return abs(eval_args[0])
        elif f == "sign":
            return 1.0 if eval_args[0] > 0.0 else (-1.0 if eval_args[0] < 0.0 else 0.0)
        elif f == "sigmoid":
            return safe_scalar_sigmoid(eval_args[0])
        elif f == "softplus":
            return safe_scalar_softplus(eval_args[0])
        elif f == "clamp":
            return max(eval_args[1], min(eval_args[2], eval_args[0]))
        elif f == "rbf_gaussian":
            scaled = eval_args[1] * eval_args[0]
            return math.exp(-min(scaled * scaled, 80.0))
        elif f == "rbf_multiquadric":
            scaled = eval_args[1] * eval_args[0]
            return math.sqrt(1.0 + min(scaled * scaled, 1e160))
        elif f == "rbf_wendland":
            return safe_scalar_rbf_wendland(eval_args[0], eval_args[1])
        raise ValueError(f"Unknown function: {f}")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "FunctionCallNode",
            "func": self.func,
            "args": [arg.to_dict() for arg in self.args],
        }

    def __repr__(self) -> str:
        args_str = ", ".join(repr(arg) for arg in self.args)
        return f"{self.func}({args_str})"


def validate_parameter_bounds(node: ASTNode, params: Any) -> None:
    """
    Validates that the parameter collection is sufficient for all parameter indices
    referenced in the AST expression tree.
    Raises IndexError if any parameter index is out of bounds.
    """
    p_len = len(params) if params is not None else 0
    max_idx = -1

    def _visit(n: ASTNode) -> None:
        nonlocal max_idx
        if isinstance(n, ParameterNode) and n.index > max_idx:
            max_idx = n.index
        for ch in n.children():
            _visit(ch)

    _visit(node)
    if max_idx >= p_len:
        raise IndexError(f"Parameter index {max_idx} is out of bounds for parameter array of length {p_len}.")
