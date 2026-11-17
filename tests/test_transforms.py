"""
Unit tests for algebraic simplification and analytical symbolic differentiation.
Verifies analytical derivatives against central finite differences.
"""

import math

import numpy as np

from mathfield.core.ast.nodes import (
    ConstantNode,
    VariableNode,
)
from mathfield.core.ast.parser import parse
from mathfield.core.ast.transforms import diff, simplify


def test_constant_folding():
    ast1 = parse("1 + 2 * 3")
    simp1 = simplify(ast1)
    assert isinstance(simp1, ConstantNode)
    assert simp1.value == 7.0

    ast2 = parse("sin(0) + cos(0) + exp(0)")
    simp2 = simplify(ast2)
    assert isinstance(simp2, ConstantNode)
    assert simp2.value == 2.0

    ast3 = parse("sqrt(4) * cbrt(8)")
    simp3 = simplify(ast3)
    assert isinstance(simp3, ConstantNode)
    assert simp3.value == 4.0


def test_algebraic_identities():
    # x * 0 + y * 1 -> y
    t1 = simplify(parse("x * 0 + y * 1"))
    assert t1 == VariableNode("y")

    # x - x -> 0
    t2 = simplify(parse("x - x"))
    assert t2 == ConstantNode(0.0)

    # x / x -> 1
    t3 = simplify(parse("x / x"))
    assert t3 == ConstantNode(1.0)

    # x ^ 1 -> x
    t4 = simplify(parse("x ^ 1"))
    assert t4 == VariableNode("x")

    # x ^ 0 -> 1
    t5 = simplify(parse("x ^ 0"))
    assert t5 == ConstantNode(1.0)

    # -(-x) -> x
    t6 = simplify(parse("-(-x)"))
    assert t6 == VariableNode("x")

    # -(x - y) -> y - x
    t7 = simplify(parse("-(x - y)"))
    assert t7 == parse("y - x")


def test_analytical_vs_finite_difference():
    """
    Validates exact analytical derivatives against central finite differences
    across diverse composite expressions and parameter variations.
    """
    expressions = [
        ("c0 * x^3 + c1 * y^2", "c0", [1.5, -2.0]),
        ("c0 * x^3 + c1 * y^2", "c1", [1.5, -2.0]),
        ("sin(c0 * x) * cos(c1 * y)", "c0", [2.5, 1.8]),
        ("sin(c0 * x) * cos(c1 * y)", "c1", [2.5, 1.8]),
        ("exp(c0 * x) + log(c1 * y^2 + 1.0)", "c0", [0.5, 1.2]),
        ("exp(c0 * x) + log(c1 * y^2 + 1.0)", "c1", [0.5, 1.2]),
        ("(c0 * x + 1.0) / (c1 * y^2 + 2.0)", "c0", [1.1, 0.8]),
        ("(c0 * x + 1.0) / (c1 * y^2 + 2.0)", "c1", [1.1, 0.8]),
        ("sigmoid(c0 * x) + softplus(c1 * y)", "c0", [-1.2, 0.9]),
        ("sigmoid(c0 * x) + softplus(c1 * y)", "c1", [-1.2, 0.9]),
        ("rbf_gaussian(x^2 + y^2, c0)", "c0", [1.5]),
        ("rbf_multiquadric(x^2 + y^2, c0)", "c0", [1.2]),
    ]

    # Coordinate test grid in [-0.8, 0.8] to avoid boundary discontinuities
    x_vals = np.linspace(-0.8, 0.8, 15)
    y_vals = np.linspace(-0.8, 0.8, 15)
    X, Y = np.meshgrid(x_vals, y_vals)

    h = 1e-6
    tolerance = 1e-5

    for expr_str, target_param, params_list in expressions:
        ast = parse(expr_str)
        d_ast = diff(ast, target_param)

        eval_fn = ast.compile_numpy()
        d_eval_fn = d_ast.compile_numpy()

        params = np.array(params_list, dtype=np.float64)

        # 1. Analytical derivative field
        analytical = d_eval_fn(X, Y, params)

        # 2. Central finite difference field
        param_idx = int(target_param.replace("c", "").replace("theta", "").replace("p", ""))
        p_plus = params.copy()
        p_minus = params.copy()
        p_plus[param_idx] += h
        p_minus[param_idx] -= h

        val_plus = eval_fn(X, Y, p_plus)
        val_minus = eval_fn(X, Y, p_minus)

        finite_diff = (val_plus - val_minus) / (2.0 * h)

        max_err = float(np.max(np.abs(analytical - finite_diff)))
        assert max_err < tolerance, (
            f"Derivative mismatch for {expr_str!r} wrt {target_param}: "
            f"max error {max_err:.2e} exceeds tolerance {tolerance:.2e}"
        )


def test_coordinate_differentiation_and_anti_blowup():
    # d/dx (x^4) -> 4 * x^3
    d1 = diff(parse("x^4"), "x")
    assert math.isclose(d1.eval_scalar(2.0, 0.0, []), 32.0, abs_tol=1e-9)

    # d/dx (sin(x) * y) -> cos(x) * y
    d2 = diff(parse("sin(x) * y"), "x")
    assert math.isclose(d2.eval_scalar(0.0, 5.0, []), 5.0, abs_tol=1e-12)

    # d/dy (sin(x) * y) -> sin(x)
    d3 = diff(parse("sin(x) * y"), "y")
    assert math.isclose(d3.eval_scalar(0.0, 5.0, []), 0.0, abs_tol=1e-12)

    # Repeated differentiation anti-blowup: d^4/dx^4 (x^4) = 24
    curr = parse("x^4")
    for _ in range(4):
        curr = diff(curr, "x")
    assert isinstance(curr, ConstantNode)
    assert curr.value == 24.0
