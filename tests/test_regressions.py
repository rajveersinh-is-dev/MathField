"""Regression tests for correctness bugs found in the post-overhaul audit."""

from __future__ import annotations

import json

import numpy as np
import pytest

from mathfield.core.ast.domain import protected_pow, safe_scalar_pow
from mathfield.core.ast.parser import parse
from mathfield.core.ast.transforms import diff, simplify
from mathfield.core.basis.rational import Rational2D
from mathfield.core.basis.splines import BSplineSurface2D
from mathfield.core.checkpoint.serializer import CheckpointSerializer
from mathfield.core.codegen.numpy_gen import generate_standalone_function
from mathfield.core.domain import CoordinateDomain2D, DomainMode
from mathfield.core.engine.cpu_numpy import evaluate_numpy
from mathfield.core.local.partition_of_unity import WendlandPartitionOfUnity
from mathfield.core.local.quadtree import QuadtreeNode


def _assert_generated_power_parity(
    expression: str,
    X: np.ndarray,
    Y: np.ndarray,
    params: np.ndarray,
) -> np.ndarray:
    tree = parse(expression)
    source = generate_standalone_function(tree)
    assert "eval(" not in source
    assert "exec(" not in source
    assert "mathfield" not in source

    sandbox = {"np": np}
    exec(source, sandbox)
    with np.errstate(all="ignore"):
        expected = evaluate_numpy(tree, X, Y, params)
        actual = sandbox["evaluate_field"](X, Y, params)
    # Constant-only expressions are scalar in generated code but broadcast by
    # evaluate_numpy; compare their values over the runtime result's shape.
    np.testing.assert_array_equal(np.broadcast_to(actual, expected.shape), expected)
    return np.asarray(actual)


def test_scalar_constant_evaluation_does_not_inherit_integer_grid_dtype() -> None:
    X, Y = np.meshgrid(np.arange(4), np.arange(4))
    params = np.array([0.5])
    np.testing.assert_allclose(evaluate_numpy(parse("0.5"), X, Y, params), np.full((4, 4), 0.5))
    np.testing.assert_allclose(evaluate_numpy(parse("c0"), X, Y, params), np.full((4, 4), 0.5))


def test_power_reassociation_respects_sign_of_negative_base() -> None:
    tree = parse("(x^2)^0.5")
    assert simplify(tree) == tree  # |x| cannot be simplified to x
    X = np.array([[-2.0, -0.5], [0.5, 2.0]])
    Y = np.zeros_like(X)
    np.testing.assert_allclose(evaluate_numpy(tree, X, Y, np.array([])), np.abs(X))

    # Integer exponents may still be reassociated.
    assert simplify(parse("(x^2)^3")).__class__.__name__ == "BinaryOpNode"


def test_scalar_and_vector_power_parity_for_near_integer_exponents() -> None:
    bases = np.array([-2.0, -1.0, 0.5, 2.0])
    exponents = np.array([3.00001, 3.0, 2.00001, 2.0])
    vector = protected_pow(bases[:, None], exponents[None, :])
    scalar = np.array([[safe_scalar_pow(float(u), float(v)) for v in exponents] for u in bases])
    np.testing.assert_allclose(vector, scalar, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("expression", ["x ^ y", "pow(x, y)"])
def test_generated_power_matches_runtime_for_negative_fractional_base(expression: str) -> None:
    actual = _assert_generated_power_parity(expression, np.array([[-2.0]]), np.array([[0.6]]), np.array([]))
    assert actual.item() > 0.0
    assert actual.item() == pytest.approx(1.51571657, rel=1e-8)


@pytest.mark.parametrize("expression", ["x ^ y", "pow(x, y)"])
def test_generated_power_matches_runtime_at_numerical_boundaries(expression: str) -> None:
    bases = np.array(
        [
            -np.inf,
            -2.0,
            -1.0,
            -1e-15,
            -1e-16,
            -0.0,
            0.0,
            1e-16,
            1e-15,
            0.5,
            1.0,
            2.0,
            1e100,
            1e-100,
            np.inf,
            np.nan,
        ]
    )
    exponents = np.array(
        [
            -np.inf,
            -1e10,
            -80.6,
            -80.0,
            -3.00000011,
            -3.00000009,
            -3.00001,
            -3.0,
            -1e-15,
            -0.0,
            0.0,
            1e-16,
            1e-15,
            0.6,
            2.00000009,
            2.00000011,
            3.00000009,
            3.00000011,
            3.00001,
            3.0,
            79.0,
            80.0,
            80.6,
            1e10,
            np.inf,
            np.nan,
        ]
    )
    _assert_generated_power_parity(expression, bases[:, None], exponents[None, :], np.array([]))


@pytest.mark.parametrize("expression_template", ["x ^ ({exponent})", "pow(x, ({exponent}))"])
@pytest.mark.parametrize(
    "exponent",
    [0.0, 0.6, 2.0, 3.0, 3.00000009, 3.00000011, -3.00000009, -3.00000011, 80.6, -80.6, 1e10, -1e10],
)
def test_generated_power_matches_runtime_for_constant_exponents(expression_template: str, exponent: float) -> None:
    X = np.array([-2.0, -1e-16, -0.0, 0.0, 1e-16, 2.0, 1e100, 1e-100, np.inf, np.nan])
    _assert_generated_power_parity(expression_template.format(exponent=exponent), X, np.zeros_like(X), np.array([]))


@pytest.mark.parametrize("power_syntax", ["operator", "function"])
@pytest.mark.parametrize("base_source", ["x", "-2", "c0"])
@pytest.mark.parametrize("exponent_source", ["y", "0.6", "c1"])
def test_generated_power_matches_runtime_for_scalar_and_array_operands(
    power_syntax: str,
    base_source: str,
    exponent_source: str,
) -> None:
    if power_syntax == "operator":
        expression = f"({base_source}) ^ ({exponent_source})"
    else:
        expression = f"pow({base_source}, {exponent_source})"
    X = np.array([-2.0, -0.0, 0.0, 2.0, np.nan])
    Y = np.array([0.6, 3.00000009, 3.00000011, 0.0, np.nan])
    _assert_generated_power_parity(expression, X, Y, np.array([-2.0, 0.6]))


@pytest.mark.parametrize("expression", ["x ^ y", "pow(x, y)"])
def test_generated_power_matches_runtime_exponent_and_magnitude_clamps(expression: str) -> None:
    X = np.array([2.0, 2.0, 0.5, 0.5, 1e100, 1e-100, -1e100, -1e-100])
    Y = np.array([1e10, -1e10, 1e10, -1e10, 2.0, 3.0, 3.0, -3.0])
    expected = np.array(
        [
            np.exp(80.0 * np.log(2.0)),
            np.exp(-80.0 * np.log(2.0)),
            np.exp(80.0 * np.log(0.5)),
            np.exp(-80.0 * np.log(0.5)),
            np.exp(80.0),
            np.exp(-80.0),
            -np.exp(80.0),
            -np.exp(80.0),
        ]
    )
    actual = _assert_generated_power_parity(expression, X, Y, np.array([]))
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize(
    ("expression", "params"),
    [
        ("c0 ^ y", [np.nan]),
        ("pow(c0, y)", [np.nan]),
        ("x ^ c0", [np.nan]),
        ("pow(x, c0)", [np.nan]),
        ("c0 ^ c1", [np.nan, 3.0]),
        ("pow(c0, c1)", [np.nan, 3.0]),
        ("c0 ^ c1", [-2.0, np.nan]),
        ("pow(c0, c1)", [-2.0, np.nan]),
    ],
)
def test_generated_power_propagates_nan_from_parameters(expression: str, params: list[float]) -> None:
    X = np.array([[-2.0, 0.0, 2.0]])
    Y = np.array([[0.6, 3.0, 0.0]])
    actual = _assert_generated_power_parity(expression, X, Y, np.array(params))
    assert np.all(np.isnan(actual))


def test_generated_power_expression_does_not_duplicate_complex_operands() -> None:
    source = generate_standalone_function(parse("pow(sin(x) + cos(y), exp(x) + exp(y))"))
    assert len(source) < 1200
    for fragment in ("np.sin(X)", "np.cos(Y)", "np.clip(X, -80.0, 80.0)", "np.clip(Y, -80.0, 80.0)"):
        assert source.count(fragment) == 1


def test_bspline_to_ast_is_exactly_the_exported_surface() -> None:
    knots = np.linspace(-1.0, 1.0, 8)
    control = np.linspace(0.1, 0.9, 25).reshape(5, 5)
    surface = BSplineSurface2D(knots, knots, control, degree=3)
    X, Y = np.meshgrid(np.linspace(-1, 1, 23), np.linspace(-1, 1, 19))
    expected = surface.evaluate(X, Y)
    actual = evaluate_numpy(surface.to_ast(), X, Y, np.array([]))
    np.testing.assert_allclose(actual, expected, atol=1e-8)


def test_rational_fit_evaluate_and_ast_are_mutually_consistent() -> None:
    x = np.linspace(-1, 1, 40)
    X, Y = np.meshgrid(x, x)
    target = 1.0 / (1.0 + 0.3 * X**2 + 0.2 * Y**2)
    rational = Rational2D(deg_x=2, deg_y=2, positive_denominator=True)
    rational.fit_sanathanan_koerner(X, Y, target, max_iter=15, tol=1e-11)
    fitted = rational.evaluate(X, Y)
    np.testing.assert_allclose(evaluate_numpy(rational.to_ast(), X, Y, np.array([])), fitted, atol=1e-10)
    assert np.sqrt(np.mean((fitted - target) ** 2)) < 0.05


def test_named_checkpoint_parameters_round_trip_with_names() -> None:
    original = {"alpha": 1.25, "beta": -3.5}
    encoded = json.loads(json.dumps(CheckpointSerializer.serialize_parameters(original)))
    restored = CheckpointSerializer.deserialize_parameters(encoded)
    assert isinstance(restored, dict)
    assert list(restored) == ["alpha", "beta"]
    assert restored == pytest.approx(original)


def test_partition_of_unity_has_no_discontinuous_fallback() -> None:
    leaves = [QuadtreeNode((-1.0, -1.0, 0.0, 1.0)), QuadtreeNode((0.0, -1.0, 1.0, 1.0))]
    pou = WendlandPartitionOfUnity(overlap_ratio=0.0)
    X, Y = np.meshgrid(np.linspace(-1.02, 1.02, 9), np.linspace(-1.02, 1.02, 9))
    W = pou.compute_weights(leaves, X, Y)
    np.testing.assert_allclose(W.sum(axis=0), 1.0, atol=1e-10)
    # Weights vary smoothly across the seam at x=0 rather than jumping.
    seam = W[:, :, 4]
    assert np.max(np.abs(seam[:, 4] - seam[:, 3])) < 0.5


def test_node_spacing_matches_endpoint_inclusive_grid() -> None:
    domain = CoordinateDomain2D(width=64, height=32, mode=DomainMode.ISOTROPIC)
    dx, dy = domain.node_spacing()
    X, _ = domain.to_discrete_grid()
    np.testing.assert_allclose(dx, X[0, 1] - X[0, 0])
    np.testing.assert_allclose(dy, 2.0 / 31.0)


def test_wendland_derivative_vanishes_outside_support() -> None:
    expr = diff(parse("rbf_wendland(x, 1.0)"), "x")
    X = np.array([[2.0, 3.0], [-2.0, 0.5]])
    Y = np.zeros_like(X)
    values = evaluate_numpy(expr, X, Y, np.array([]))
    np.testing.assert_allclose(values[0, :], 0.0, atol=1e-12)
    np.testing.assert_allclose(values[1, 0], 0.0, atol=1e-12)
