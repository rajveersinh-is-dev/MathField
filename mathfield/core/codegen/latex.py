"""
AMS-LaTeX code generation from MathField Abstract Syntax Tree (AST).
Emits clean, mathematically accurate LaTeX parseable by matplotlib.mathtext.
"""

from __future__ import annotations

import re
from mathfield.core.ast.nodes import (
    ASTNode,
    BinaryOpNode,
    ConstantNode,
    FunctionCallNode,
    ParameterNode,
    UnaryOpNode,
    VariableNode,
)

_PARAM_REGEX = re.compile(r"^(c|theta|p)(\d+)$", re.IGNORECASE)

# Operator precedence levels: higher numbers bind more tightly
PREC_ADD_SUB = 10
PREC_MUL_DIV = 20
PREC_UNARY = 25
PREC_POW = 30
PREC_ATOM = 40


def _get_precedence(node: ASTNode) -> int:
    if isinstance(node, (ConstantNode, VariableNode, ParameterNode, FunctionCallNode)):
        return PREC_ATOM
    elif isinstance(node, UnaryOpNode):
        return PREC_UNARY
    elif isinstance(node, BinaryOpNode):
        if node.op in ("+", "-"):
            return PREC_ADD_SUB
        elif node.op in ("*", "/", "%"):
            return PREC_MUL_DIV
        elif node.op in ("^", "**"):
            return PREC_POW
    return PREC_ATOM


def to_latex(node: ASTNode) -> str:
    """
    Converts an AST expression into an AMS-LaTeX string.

    Parameters:
        node: The root ASTNode.

    Returns:
        A LaTeX-formatted mathematical expression string.
    """
    return _node_to_latex(node)


def _node_to_latex(node: ASTNode) -> str:
    if isinstance(node, ConstantNode):
        if node.symbol == "pi":
            return r"\pi"
        elif node.symbol == "e":
            return "e"
        if node.value.is_integer():
            return str(int(node.value))
        return f"{node.value:g}"

    elif isinstance(node, VariableNode):
        return node.name

    elif isinstance(node, ParameterNode):
        m = _PARAM_REGEX.match(node.name)
        if m:
            prefix, idx = m.groups()
            prefix_latex = r"\theta" if prefix.lower() == "theta" else prefix.lower()
            return f"{prefix_latex}_{{{idx}}}"
        return node.name

    elif isinstance(node, UnaryOpNode):
        operand_prec = _get_precedence(node.operand)
        child_str = _node_to_latex(node.operand)
        if operand_prec < PREC_UNARY:
            child_str = rf"\left({child_str}\right)"
        return f"{node.op}{child_str}"

    elif isinstance(node, BinaryOpNode):
        op = node.op
        left, right = node.left, node.right
        l_prec = _get_precedence(left)
        r_prec = _get_precedence(right)
        my_prec = _get_precedence(node)

        # 1. Fractions
        if op == "/":
            l_str = _node_to_latex(left)
            r_str = _node_to_latex(right)
            return rf"\frac{{{l_str}}}{{{r_str}}}"

        # 2. Exponentiation
        elif op in ("^", "**"):
            # If base has lower precedence, parenthesize base
            # Note: fractions as base must also be parenthesized
            if l_prec < PREC_POW or (isinstance(left, BinaryOpNode) and left.op == "/"):
                l_str = rf"\left({_node_to_latex(left)}\right)"
            else:
                l_str = _node_to_latex(left)

            # Exponent can be formatted cleanly inside braces
            r_str = _node_to_latex(right)
            return f"{l_str}^{{{r_str}}}"

        # 3. Multiplication
        elif op == "*":
            l_str = _node_to_latex(left)
            r_str = _node_to_latex(right)
            if l_prec < PREC_MUL_DIV:
                l_str = rf"\left({l_str}\right)"
            if r_prec < PREC_MUL_DIV:
                r_str = rf"\left({r_str}\right)"
            return rf"{l_str} \cdot {r_str}"

        # 4. Modulo
        elif op == "%":
            l_str = _node_to_latex(left)
            r_str = _node_to_latex(right)
            if l_prec < PREC_MUL_DIV:
                l_str = rf"\left({l_str}\right)"
            if r_prec < PREC_MUL_DIV:
                r_str = rf"\left({r_str}\right)"
            return rf"{l_str} \bmod {r_str}"

        # 5. Addition
        elif op == "+":
            l_str = _node_to_latex(left)
            r_str = _node_to_latex(right)
            return f"{l_str} + {r_str}"

        # 6. Subtraction
        elif op == "-":
            l_str = _node_to_latex(left)
            r_str = _node_to_latex(right)
            # Subtraction is left-associative: a - (b + c) or a - (b - c) needs parens on right
            if r_prec <= PREC_ADD_SUB:
                r_str = rf"\left({r_str}\right)"
            return f"{l_str} - {r_str}"

    elif isinstance(node, FunctionCallNode):
        f = node.func
        args = node.args

        if f == "sin":
            return rf"\sin\left({_node_to_latex(args[0])}\right)"
        elif f == "cos":
            return rf"\cos\left({_node_to_latex(args[0])}\right)"
        elif f == "tan":
            return rf"\tan\left({_node_to_latex(args[0])}\right)"
        elif f == "sinc":
            return rf"\operatorname{{sinc}}\left({_node_to_latex(args[0])}\right)"
        elif f == "asin":
            return rf"\arcsin\left({_node_to_latex(args[0])}\right)"
        elif f == "acos":
            return rf"\arccos\left({_node_to_latex(args[0])}\right)"
        elif f == "atan":
            return rf"\arctan\left({_node_to_latex(args[0])}\right)"
        elif f == "sinh":
            return rf"\sinh\left({_node_to_latex(args[0])}\right)"
        elif f == "cosh":
            return rf"\cosh\left({_node_to_latex(args[0])}\right)"
        elif f == "tanh":
            return rf"\tanh\left({_node_to_latex(args[0])}\right)"
        elif f == "exp":
            return rf"\exp\left({_node_to_latex(args[0])}\right)"
        elif f == "log":
            return rf"\ln\left({_node_to_latex(args[0])}\right)"
        elif f == "sqrt":
            return rf"\sqrt{{{_node_to_latex(args[0])}}}"
        elif f == "cbrt":
            return rf"\sqrt[3]{{{_node_to_latex(args[0])}}}"
        elif f == "square":
            inner_prec = _get_precedence(args[0])
            inner_str = _node_to_latex(args[0])
            if inner_prec < PREC_POW or (isinstance(args[0], BinaryOpNode) and args[0].op == "/"):
                return rf"\left({inner_str}\right)^{{2}}"
            return f"{inner_str}^{{2}}"
        elif f == "pow":
            return _node_to_latex(BinaryOpNode("^", args[0], args[1]))
        elif f == "abs":
            return rf"\left|{_node_to_latex(args[0])}\right|"
        elif f == "sign":
            return rf"\operatorname{{sgn}}\left({_node_to_latex(args[0])}\right)"
        elif f == "sigmoid":
            return rf"\sigma\left({_node_to_latex(args[0])}\right)"
        elif f == "softplus":
            return rf"\operatorname{{softplus}}\left({_node_to_latex(args[0])}\right)"
        elif f == "clamp":
            a, lo, hi = [_node_to_latex(arg) for arg in args]
            return rf"\operatorname{{clamp}}\left({a}, {lo}, {hi}\right)"
        elif f == "rbf_gaussian":
            r, eps = [_node_to_latex(arg) for arg in args]
            return rf"\operatorname{{RBF}}_{{\text{{gauss}}}}\left({r}, {eps}\right)"
        elif f == "rbf_multiquadric":
            r, eps = [_node_to_latex(arg) for arg in args]
            return rf"\operatorname{{RBF}}_{{\text{{mq}}}}\left({r}, {eps}\right)"
        elif f == "rbf_wendland":
            r, rad = [_node_to_latex(arg) for arg in args]
            return rf"\operatorname{{Wendland}}_{{C^2}}\left({r}, {rad}\right)"

    return str(node)
