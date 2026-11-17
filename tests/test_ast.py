"""
Unit tests for Abstract Syntax Tree (AST) node architecture and immutability.
"""

import math
from dataclasses import FrozenInstanceError

import pytest

from mathfield.core.ast.nodes import (
    FUNCTION_ARITIES,
    ASTNode,
    BinaryOpNode,
    ConstantNode,
    FunctionCallNode,
    ParameterNode,
    UnaryOpNode,
    VariableNode,
)


def test_constant_node_immutability():
    node = ConstantNode(3.14159, symbol="pi")
    assert node.value == 3.14159
    assert node.symbol == "pi"
    assert node.complexity() == 1
    assert node.free_variables() == set()
    assert node.parameters() == set()
    assert node.eval_scalar(0.0, 0.0, []) == 3.14159

    with pytest.raises((FrozenInstanceError, AttributeError)):
        node.value = 2.0  # type: ignore


def test_variable_node_validation():
    x_node = VariableNode("x")
    assert x_node.name == "x"
    assert x_node.free_variables() == {"x"}
    assert x_node.parameters() == set()
    assert x_node.complexity() == 1
    assert x_node.eval_scalar(1.5, -2.0, []) == 1.5

    y_node = VariableNode("Y")
    assert y_node.name == "y"
    assert y_node.eval_scalar(1.5, -2.0, []) == -2.0

    with pytest.raises(ValueError, match="Invalid coordinate variable"):
        VariableNode("z")


def test_parameter_node_validation():
    p_node = ParameterNode("c0", 0)
    assert p_node.name == "c0"
    assert p_node.index == 0
    assert p_node.free_variables() == set()
    assert p_node.parameters() == {"c0"}
    assert p_node.complexity() == 1
    assert p_node.eval_scalar(0.0, 0.0, [4.2]) == 4.2

    with pytest.raises(ValueError, match="Parameter index must be non-negative"):
        ParameterNode("c1", -1)

    with pytest.raises(IndexError):
        p_node.eval_scalar(0.0, 0.0, [])


def test_unary_op_node():
    x = VariableNode("x")
    neg_x = UnaryOpNode("-", x)
    assert neg_x.op == "-"
    assert neg_x.operand == x
    assert neg_x.complexity() == 2
    assert neg_x.free_variables() == {"x"}
    assert neg_x.eval_scalar(3.0, 0.0, []) == -3.0

    pos_x = UnaryOpNode("+", x)
    assert pos_x.eval_scalar(3.0, 0.0, []) == 3.0

    with pytest.raises(ValueError, match="Invalid unary operator"):
        UnaryOpNode("*", x)


def test_binary_op_node():
    x = VariableNode("x")
    y = VariableNode("y")
    add = BinaryOpNode("+", x, y)
    assert add.complexity() == 3
    assert add.free_variables() == {"x", "y"}
    assert add.eval_scalar(2.0, 3.5, []) == 5.5

    sub = BinaryOpNode("-", x, y)
    assert sub.eval_scalar(2.0, 3.5, []) == -1.5

    mul = BinaryOpNode("*", x, y)
    assert mul.eval_scalar(2.0, 3.5, []) == 7.0

    div = BinaryOpNode("/", x, y)
    assert math.isclose(div.eval_scalar(7.0, 2.0, []), 3.5, rel_tol=1e-7)

    power = BinaryOpNode("^", x, ConstantNode(3.0))
    assert math.isclose(power.eval_scalar(2.0, 0.0, []), 8.0, rel_tol=1e-7)

    with pytest.raises(ValueError, match="Invalid binary operator"):
        BinaryOpNode("&", x, y)


def test_function_call_node_arities():
    x = VariableNode("x")
    y = VariableNode("y")
    c = ConstantNode(1.0)

    # All 24 functions check
    assert len(FUNCTION_ARITIES) == 24

    # 1-arg functions
    sin_node = FunctionCallNode("sin", (x,))
    assert sin_node.complexity() == 3
    assert math.isclose(sin_node.eval_scalar(0.0, 0.0, []), 0.0, abs_tol=1e-12)

    # 2-arg functions
    rbf_node = FunctionCallNode("rbf_gaussian", (x, c))
    assert rbf_node.complexity() == 4
    assert math.isclose(rbf_node.eval_scalar(0.0, 0.0, []), 1.0, abs_tol=1e-12)

    # 3-arg functions
    clamp_node = FunctionCallNode("clamp", (x, ConstantNode(-1.0), ConstantNode(1.0)))
    assert clamp_node.complexity() == 5
    assert clamp_node.eval_scalar(5.0, 0.0, []) == 1.0
    assert clamp_node.eval_scalar(-5.0, 0.0, []) == -1.0
    assert clamp_node.eval_scalar(0.5, 0.0, []) == 0.5

    # Invalid function
    with pytest.raises(ValueError, match="Disallowed or unknown function"):
        FunctionCallNode("system", (x,))

    # Wrong arity
    with pytest.raises(ValueError, match="expects 1 arguments, but got 2"):
        FunctionCallNode("sin", (x, y))


def test_node_hashability_and_set_membership():
    n1 = BinaryOpNode("+", VariableNode("x"), ConstantNode(1.0))
    n2 = BinaryOpNode("+", VariableNode("x"), ConstantNode(1.0))
    n3 = BinaryOpNode("-", VariableNode("x"), ConstantNode(1.0))

    assert n1 == n2
    assert hash(n1) == hash(n2)
    s = {n1, n2, n3}
    assert len(s) == 2


def test_serialization_roundtrip():
    tree = BinaryOpNode(
        "+",
        FunctionCallNode("sin", (VariableNode("x"),)),
        BinaryOpNode(
            "*",
            ParameterNode("theta0", 0),
            FunctionCallNode("rbf_gaussian", (VariableNode("y"), ConstantNode(2.5))),
        ),
    )

    data = tree.to_dict()
    assert isinstance(data, dict)
    reconstructed = ASTNode.from_dict(data)
    assert reconstructed == tree
    assert hash(reconstructed) == hash(tree)


def test_uv_coordinates_and_parameter_prefixes():
    u = VariableNode("u")
    v = VariableNode("v")
    assert u.name == "u"
    assert v.name == "v"
    assert u.eval_scalar(3.0, 4.0, []) == 3.0
    assert v.eval_scalar(3.0, 4.0, []) == 4.0

    p_param = ParameterNode("p0", 0)
    theta_param = ParameterNode("theta1", 1)
    c_param = ParameterNode("c2", 2)
    assert p_param.parameters() == {"p0"}
    assert theta_param.parameters() == {"theta1"}
    assert c_param.parameters() == {"c2"}

    params = [10.0, 20.0, 30.0]
    assert p_param.eval_scalar(0.0, 0.0, params) == 10.0
    assert theta_param.eval_scalar(0.0, 0.0, params) == 20.0
    assert c_param.eval_scalar(0.0, 0.0, params) == 30.0


def test_ast_node_delegates():
    node = BinaryOpNode(
        "+",
        FunctionCallNode("sin", (VariableNode("x"),)),
        ParameterNode("c0", 0),
    )
    # to_latex() delegate
    latex_str = node.to_latex()
    assert r"\sin" in latex_str
    assert r"c_{0}" in latex_str

    # to_python_code() delegate
    py_code = node.to_python_code()
    assert "np.sin(X)" in py_code
    assert "params[0]" in py_code

    # compile_cuda() delegate
    cuda_str = node.compile_cuda()
    assert "sin(x)" in cuda_str
    assert "params[0]" in cuda_str

    # diff() delegate
    d_node = node.diff("c0")
    assert isinstance(d_node, ConstantNode)
    assert d_node.value == 1.0


def test_protected_pow_boundary_and_nan_safety() -> None:
    import warnings

    import numpy as np

    from mathfield.core.ast.domain import protected_log, protected_pow, protected_sqrt, regularized_div

    # Verify zero RuntimeWarning under any trapped warnings
    with warnings.catch_warnings():
        warnings.simplefilter("error")

        # 1. NaN exponents and bases
        u_nan = np.array([float("nan"), 2.0, -2.0, 0.0])
        v_nan = np.array([2.0, float("nan"), 3.0, 0.0])
        res = protected_pow(u_nan, v_nan)
        assert np.isnan(res[0])
        assert np.isnan(res[1])
        assert math.isclose(float(res[2]), -8.0, abs_tol=1e-10)  # (-2)^3 = -8
        assert math.isclose(float(res[3]), 1.0, abs_tol=1e-10)   # 0^0 = 1

        # 2. Extreme and infinite exponents
        u_inf = np.array([2.0, -2.0, 0.5])
        v_inf = np.array([1e10, 3.0, -1e10])
        res_inf = protected_pow(u_inf, v_inf)
        assert np.all(np.isfinite(res_inf))

        # 3. Regularized division
        div_res = regularized_div(np.array([1.0, -1.0, 0.0]), np.array([0.0, -0.0, 0.0]))
        assert np.all(np.isfinite(div_res))

        # 4. Protected log and sqrt
        log_res = protected_log(np.array([-5.0, 0.0, 1.0]))
        assert np.all(np.isfinite(log_res))
        sqrt_res = protected_sqrt(np.array([-10.0, 0.0, 4.0]))
        assert np.all(np.isfinite(sqrt_res))


def test_scalar_math_nan_propagation():
    """Validates that pure-Python scalar math helpers propagate NaN instead of coercing to finite numbers."""
    from mathfield.core.ast.domain import (
        safe_scalar_acos,
        safe_scalar_asin,
        safe_scalar_div,
        safe_scalar_exp,
        safe_scalar_log,
        safe_scalar_pow,
        safe_scalar_sigmoid,
        safe_scalar_sqrt,
        safe_scalar_tan,
    )

    nan = float("nan")
    assert math.isnan(safe_scalar_asin(nan))
    assert math.isnan(safe_scalar_acos(nan))
    assert math.isnan(safe_scalar_exp(nan))
    assert math.isnan(safe_scalar_tan(nan))
    assert math.isnan(safe_scalar_sigmoid(nan))
    assert math.isnan(safe_scalar_log(nan))
    assert math.isnan(safe_scalar_sqrt(nan))
    assert math.isnan(safe_scalar_div(nan, 1.0))
    assert math.isnan(safe_scalar_div(1.0, nan))
    assert math.isnan(safe_scalar_pow(nan, 2.0))
    assert math.isnan(safe_scalar_pow(2.0, nan))

    # Also check finite boundary values
    assert math.isclose(safe_scalar_asin(2.0), math.pi / 2.0, rel_tol=1e-10)
    assert math.isclose(safe_scalar_asin(-2.0), -math.pi / 2.0, rel_tol=1e-10)
    assert math.isclose(safe_scalar_exp(1000.0), math.exp(80.0), rel_tol=1e-10)
    assert safe_scalar_div(1.0, 0.0) == 1.0 / 1e-12

