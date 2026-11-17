"""
Unit tests for the O(N) Pratt parser, precedence climbing, and sandbox security.
"""

import math

import pytest

from mathfield.core.ast.nodes import (
    BinaryOpNode,
    ConstantNode,
    FunctionCallNode,
    UnaryOpNode,
    VariableNode,
)
from mathfield.core.ast.parser import ParseError, parse
from mathfield.core.ast.tokens import LexerError


def test_operator_precedence():
    # Multiplicative over Additive: 2 + 3 * 4 -> 2 + (3 * 4)
    tree1 = parse("2 + 3 * 4")
    assert isinstance(tree1, BinaryOpNode)
    assert tree1.op == "+"
    assert tree1.left == ConstantNode(2.0)
    assert tree1.right == BinaryOpNode("*", ConstantNode(3.0), ConstantNode(4.0))

    # Power over Multiplicative: 2 * 3 ^ 2 -> 2 * (3 ^ 2)
    tree2 = parse("2 * 3 ^ 2")
    assert isinstance(tree2, BinaryOpNode)
    assert tree2.op == "*"
    assert tree2.left == ConstantNode(2.0)
    assert tree2.right == BinaryOpNode("^", ConstantNode(3.0), ConstantNode(2.0))


def test_operator_associativity():
    # Left-associativity of Subtraction: 10 - 5 - 2 -> (10 - 5) - 2
    tree1 = parse("10 - 5 - 2")
    assert isinstance(tree1, BinaryOpNode)
    assert tree1.op == "-"
    assert tree1.left == BinaryOpNode("-", ConstantNode(10.0), ConstantNode(5.0))
    assert tree1.right == ConstantNode(2.0)
    assert tree1.eval_scalar(0.0, 0.0, []) == 3.0

    # Left-associativity of Division: 100 / 10 / 2 -> (100 / 10) / 2
    tree2 = parse("100 / 10 / 2")
    assert isinstance(tree2, BinaryOpNode)
    assert tree2.op == "/"
    assert tree2.left == BinaryOpNode("/", ConstantNode(100.0), ConstantNode(10.0))
    assert tree2.right == ConstantNode(2.0)
    assert math.isclose(tree2.eval_scalar(0.0, 0.0, []), 5.0, abs_tol=1e-7)

    # Right-associativity of Power: 2 ^ 3 ^ 2 -> 2 ^ (3 ^ 2) = 2^9 = 512
    tree3 = parse("2 ^ 3 ^ 2")
    assert isinstance(tree3, BinaryOpNode)
    assert tree3.op == "^"
    assert tree3.left == ConstantNode(2.0)
    assert tree3.right == BinaryOpNode("^", ConstantNode(3.0), ConstantNode(2.0))
    assert math.isclose(tree3.eval_scalar(0.0, 0.0, []), 512.0, abs_tol=1e-7)

    # Python-style ** exponentiation
    tree4 = parse("2 ** 3 ** 2")
    assert tree4 == tree3


def test_unary_minus_precedence():
    # -x ^ 2 should be parsed as -(x ^ 2)
    tree = parse("-x ^ 2")
    assert isinstance(tree, UnaryOpNode)
    assert tree.op == "-"
    assert isinstance(tree.operand, BinaryOpNode)
    assert tree.operand.op == "^"
    assert tree.operand.left == VariableNode("x")
    assert tree.operand.right == ConstantNode(2.0)


def test_parentheses_grouping():
    tree = parse("(2 + 3) * 4")
    assert isinstance(tree, BinaryOpNode)
    assert tree.op == "*"
    assert tree.left == BinaryOpNode("+", ConstantNode(2.0), ConstantNode(3.0))
    assert tree.right == ConstantNode(4.0)
    assert tree.eval_scalar(0.0, 0.0, []) == 20.0


def test_implicit_multiplication():
    # 2x + 3y -> (2 * x) + (3 * y)
    t1 = parse("2x + 3y")
    t2 = parse("2 * x + 3 * y")
    assert t1 == t2

    # c0 sin(x) -> c0 * sin(x)
    t3 = parse("c0 sin(x)")
    t4 = parse("c0 * sin(x)")
    assert t3 == t4

    # (x + 1)(y - 1) -> (x + 1) * (y - 1)
    t5 = parse("(x + 1)(y - 1)")
    t6 = parse("(x + 1) * (y - 1)")
    assert t5 == t6


def test_scientific_notation_and_constants():
    tree = parse("1.5e-3 + pi * e")
    assert isinstance(tree, BinaryOpNode)
    assert tree.op == "+"
    assert isinstance(tree.left, ConstantNode)
    assert tree.left.value == 0.0015
    assert tree.right.left.symbol == "pi"
    assert tree.right.right.symbol == "e"


def test_syntax_error_diagnostics():
    # Unmatched parenthesis
    with pytest.raises(ParseError) as exc_info:
        parse("(x + 1")
    assert "Unmatched" in str(exc_info.value) or "missing closing ')'" in str(exc_info.value)
    assert exc_info.value.line == 1

    # Missing operand
    with pytest.raises(ParseError) as exc_info:
        parse("x +")
    assert "Unexpected end of expression" in str(exc_info.value)

    # Invalid character
    with pytest.raises(LexerError) as exc_info:
        parse("x + $y")
    assert "Invalid character '$'" in str(exc_info.value)
    assert exc_info.value.column == 5


def test_security_sandbox_rejection():
    # Malicious injection attempts must raise ParseError or LexerError
    malicious_inputs = [
        "__import__('os').system('dir')",
        "eval('2 + 2')",
        "exec('import sys')",
        "open('secret.txt')",
        "(1).__class__.__bases__[0]",
        "lambda x: x",
        "def f(): pass",
        "import os",
        "subprocess.Popen(['notepad'])",
    ]

    for attack in malicious_inputs:
        with pytest.raises((ParseError, LexerError)):
            parse(attack)


def test_modulo_operator_and_multi_arg_functions():
    tree1 = parse("x % 2")
    assert isinstance(tree1, BinaryOpNode)
    assert tree1.op == "%"
    assert tree1.eval_scalar(5.0, 0.0, []) == 1.0

    tree2 = parse("rbf_wendland(x, 2.0)")
    assert isinstance(tree2, FunctionCallNode)
    assert tree2.func == "rbf_wendland"
    assert len(tree2.args) == 2

    tree3 = parse("clamp(sin(x), -0.5, 0.5)")
    assert isinstance(tree3, FunctionCallNode)
    assert tree3.func == "clamp"
    assert len(tree3.args) == 3


def test_implicit_multiplication_edge_cases():
    # 3(x + 1) -> 3 * (x + 1)
    t1 = parse("3(x + 1)")
    assert t1 == parse("3 * (x + 1)")

    # 2 pi -> 2 * pi
    t2 = parse("2 pi")
    assert t2 == parse("2 * pi")

    # x y -> x * y
    t3 = parse("x y")
    assert t3 == parse("x * y")

    # p0 theta1 -> p0 * theta1
    t4 = parse("p0 theta1")
    assert t4 == parse("p0 * theta1")
