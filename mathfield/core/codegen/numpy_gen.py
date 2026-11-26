"""
Standalone vectorized NumPy code generation from MathField AST.
Emits pure Python expressions and self-contained reproduction scripts.
"""

from __future__ import annotations

from typing import Sequence

from mathfield.core.ast.nodes import (
    ASTNode,
    BinaryOpNode,
    ConstantNode,
    FunctionCallNode,
    ParameterNode,
    UnaryOpNode,
    VariableNode,
)


def to_python_code(node: ASTNode) -> str:
    """
    Converts an AST expression into a standalone vectorized Python expression using NumPy.
    The emitted expression is safe, bounded, and uses zero imports beyond `numpy as np`.

    Parameters:
        node: The root ASTNode.

    Returns:
        A Python expression string operating on arrays X, Y, and params.
    """
    return _node_to_numpy_expr(node)


def _protected_power_expr(base: str, exponent: str) -> str:
    """Emit an expression with the exact clamping and parity rules of protected_pow."""
    # Keep this expression-only so to_python_code remains usable on its own. The
    # nested lambdas evaluate each operand (and the rounded exponent) only once,
    # avoiding repeated subexpressions for complex bases or exponents.
    return (
        "(lambda u, v: (lambda vc: (lambda vr: "
        "np.where("
        "(np.abs(u) < 1e-15) & (np.abs(v) < 1e-15), "
        "1.0, "
        "np.where("
        "(u < 0.0) & np.isfinite(vc) & "
        "np.isclose(vc, vr, atol=1e-7, rtol=0.0) & "
        "(np.mod(np.abs(np.nan_to_num(vr, nan=0.0).astype(np.int64)), 2) == 1), "
        "-1.0, 1.0) * "
        "np.exp(np.clip(vc * np.log(np.maximum(np.abs(u), 1e-15)), -80.0, 80.0))"
        "))(np.round(vc)))(np.clip(v, -80.0, 80.0)))"
        f"({base}, {exponent})"
    )


def _flatten_additive(node: ASTNode, sign: int = 1) -> list[tuple[int, ASTNode]]:
    """Return the signed leaf terms of an associative +/- chain."""
    if isinstance(node, BinaryOpNode) and node.op in ("+", "-"):
        right_sign = sign if node.op == "+" else -sign
        return _flatten_additive(node.left, sign) + _flatten_additive(node.right, right_sign)
    return [(sign, node)]


def _node_to_numpy_expr(node: ASTNode) -> str:
    if isinstance(node, ConstantNode):
        if node.symbol == "pi":
            return "np.pi"
        elif node.symbol == "e":
            return "np.e"
        return f"{node.value:.17g}"

    elif isinstance(node, VariableNode):
        name = node.name.lower()
        if name in ("x", "u"):
            return "X"
        elif name in ("y", "v"):
            return "Y"
        return "X"

    elif isinstance(node, ParameterNode):
        return f"params[{node.index}]"

    elif isinstance(node, UnaryOpNode):
        child_str = _node_to_numpy_expr(node.operand)
        if node.op == "-":
            return f"(-({child_str}))"
        return child_str

    elif isinstance(node, BinaryOpNode):
        l_str = _node_to_numpy_expr(node.left)
        r_str = _node_to_numpy_expr(node.right)
        op = node.op

        if op in ("+", "-"):
            # Flatten associative +/- chains. A coefficient expansion can have
            # thousands of terms; recursively parenthesising them exceeds
            # Python's nested-parenthesis limit when the script is compiled.
            terms = _flatten_additive(node)
            parts = [(sign, _node_to_numpy_expr(child)) for sign, child in terms]
            expression = f"({parts[0][1]})"
            for sign, part in parts[1:]:
                expression += f" {'+ ' if sign > 0 else '- '}({part})"
            return f"({expression})"
        elif op == "*":
            return f"({l_str} * {r_str})"
        elif op == "/":
            return f"({l_str} / ({r_str} + np.copysign(1e-12, {r_str})))"
        elif op in ("^", "**"):
            return _protected_power_expr(l_str, r_str)
        elif op == "%":
            return f"np.fmod({l_str}, np.maximum(np.abs({r_str}), 1e-12))"

    elif isinstance(node, FunctionCallNode):
        f = node.func
        args_str = [_node_to_numpy_expr(arg) for arg in node.args]

        if f == "sin":
            return f"np.sin({args_str[0]})"
        elif f == "cos":
            return f"np.cos({args_str[0]})"
        elif f == "tan":
            return f"np.clip(np.tan({args_str[0]}), -1e6, 1e6)"
        elif f == "sinc":
            return f"np.sinc({args_str[0]})"
        elif f == "asin":
            return f"np.arcsin(np.clip({args_str[0]}, -1.0, 1.0))"
        elif f == "acos":
            return f"np.arccos(np.clip({args_str[0]}, -1.0, 1.0))"
        elif f == "atan":
            return f"np.arctan({args_str[0]})"
        elif f == "sinh":
            return f"np.sinh(np.clip({args_str[0]}, -80.0, 80.0))"
        elif f == "cosh":
            return f"np.cosh(np.clip({args_str[0]}, -80.0, 80.0))"
        elif f == "tanh":
            return f"np.tanh({args_str[0]})"
        elif f == "exp":
            return f"np.exp(np.clip({args_str[0]}, -80.0, 80.0))"
        elif f == "log":
            return f"np.log(np.maximum({args_str[0]}, 1e-15))"
        elif f == "sqrt":
            return f"np.sqrt(np.maximum({args_str[0]}, 0.0))"
        elif f == "cbrt":
            return f"np.cbrt({args_str[0]})"
        elif f == "square":
            return f"np.square({args_str[0]})"
        elif f == "pow":
            return _protected_power_expr(args_str[0], args_str[1])
        elif f == "abs":
            return f"np.abs({args_str[0]})"
        elif f == "sign":
            return f"np.sign({args_str[0]})"
        elif f == "sigmoid":
            return f"(1.0 / (1.0 + np.exp(-np.clip({args_str[0]}, -80.0, 80.0))))"
        elif f == "softplus":
            return f"(np.log1p(np.exp(-np.abs({args_str[0]}))) + np.maximum({args_str[0]}, 0.0))"
        elif f == "clamp":
            return f"np.clip({args_str[0]}, {args_str[1]}, {args_str[2]})"
        elif f == "rbf_gaussian":
            r, eps = args_str[0], args_str[1]
            return f"np.exp(-np.clip(np.square({eps} * {r}), 0.0, 80.0))"
        elif f == "rbf_multiquadric":
            r, eps = args_str[0], args_str[1]
            return f"np.sqrt(1.0 + np.square({eps} * {r}))"
        elif f == "rbf_wendland":
            r, rad = args_str[0], args_str[1]
            return (
                f"((np.clip(1.0 - np.maximum({r}, 0.0) / np.maximum(np.abs({rad}), 1e-12), 0.0, 1.0) ** 4) * "
                f"(4.0 * (np.maximum({r}, 0.0) / np.maximum(np.abs({rad}), 1e-12)) + 1.0))"
            )

    return f"({node})"


def generate_standalone_function(node: ASTNode, func_name: str = "evaluate_field") -> str:
    """
    Emits a complete, self-contained Python function that evaluates the AST.
    """
    expr_code = to_python_code(node)
    return (
        "import numpy as np\n\n"
        f"def {func_name}(X: np.ndarray, Y: np.ndarray, params: np.ndarray) -> np.ndarray:\n"
        '    """\n'
        "    Evaluates discovered mathematical field with domain-protected vectorized NumPy.\n"
        '    """\n'
        f"    return {expr_code}\n"
    )


def generate_standalone_script(
    node: ASTNode,
    params: Sequence[float],
    width: int = 512,
    height: int = 512,
    x_min: float = -1.0,
    x_max: float = 1.0,
    y_min: float = -1.0,
    y_max: float = 1.0,
    output_path: str = "reconstruction.png",
) -> str:
    """
    Generates a full standalone Python script that reproduces the continuous field image
    without requiring MathField to be installed.
    """
    fn_code = generate_standalone_function(node)
    params_repr = "[" + ", ".join(f"{float(p):.17g}" for p in params) + "]"

    script = (
        "#!/usr/bin/env python3\n"
        '"""\n'
        "Standalone reproduction script generated by MathField.\n"
        "Requires only numpy and Pillow.\n"
        '"""\n\n'
        "import numpy as np\n"
        "from PIL import Image\n\n"
        f"{fn_code}\n\n"
        "def main():\n"
        f"    width = {width}\n"
        f"    height = {height}\n"
        f"    x = np.linspace({x_min:.17g}, {x_max:.17g}, width)\n"
        f"    y = np.linspace({y_min:.17g}, {y_max:.17g}, height)\n"
        "    X, Y = np.meshgrid(x, y)\n"
        f"    params = np.array({params_repr}, dtype=np.float64)\n"
        "    \n"
        "    Z = evaluate_field(X, Y, params)\n"
        "    \n"
        "    # Normalize to 0-255 grayscale or image range\n"
        "    z_min, z_max = np.min(Z), np.max(Z)\n"
        "    if z_max > z_min:\n"
        "        norm = ((Z - z_min) / (z_max - z_min) * 255.0).astype(np.uint8)\n"
        "    else:\n"
        "        norm = np.zeros_like(Z, dtype=np.uint8)\n"
        "        \n"
        "    img = Image.fromarray(norm)\n"
        f"    img.save({output_path!r})\n"
        f'    print(f"Reconstruction successfully saved to {{{output_path!r}}}")\n\n'
        'if __name__ == "__main__":\n'
        "    main()\n"
    )
    return script
