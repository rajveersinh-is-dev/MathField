"""
Unit tests for the multi-tier acceleration engine:
NumPy CPU SIMD, MSVC OpenMP C++ DLL, and CuPy GPU.
Validates numerical parity, fallback hierarchy, and domain safety edge cases.
"""

import numpy as np
import pytest

from mathfield.core.ast.parser import parse
from mathfield.core.engine.backend import MultiTierEvaluator
from mathfield.core.engine.cpu_native import (
    compute_residuals_native,
    compute_sse_native,
    evaluate_native,
    get_openmp_thread_count,
    is_native_available,
)
from mathfield.core.engine.cpu_numpy import (
    compute_residuals_numpy,
    compute_sse_numpy,
    evaluate_numpy,
)
from mathfield.core.engine.gpu_cupy import (
    compute_residuals_cupy,
    compute_sse_cupy,
    evaluate_cupy,
    is_cuda_available,
)


def test_numpy_vector_safety_and_shapes():
    ast = parse("sin(x) + c0 * cos(y)")
    X = np.linspace(-1, 1, 64).reshape(8, 8)
    Y = np.linspace(-1, 1, 64).reshape(8, 8)
    params = np.array([2.5])

    out = evaluate_numpy(ast, X, Y, params)
    assert out.shape == X.shape
    assert out.dtype == X.dtype
    assert np.all(np.isfinite(out))


def test_native_openmp_dll():
    if not is_native_available():
        pytest.skip("Native OpenMP DLL is not compiled or available in this environment.")

    threads = get_openmp_thread_count()
    assert threads >= 1

    ast = parse("sin(x) + c0 * cos(y)")
    X = np.linspace(-1, 1, 32).reshape(4, 8)
    Y = np.linspace(-1, 1, 32).reshape(4, 8)
    params = np.array([1.8])

    out_native = evaluate_native(ast, X, Y, params)
    out_numpy = evaluate_numpy(ast, X, Y, params)

    assert np.all(np.isfinite(out_native))
    max_diff = float(np.max(np.abs(out_native - out_numpy)))
    assert max_diff < 1e-6

    # Test residual and SSE
    pred = np.array([1.0, 2.0, 3.0, 4.0])
    target = np.array([1.1, 1.9, 3.2, 3.8])

    res_native = compute_residuals_native(pred, target)
    res_numpy = compute_residuals_numpy(pred, target)
    assert np.allclose(res_native, res_numpy, atol=1e-12)

    sse_native = compute_sse_native(pred, target)
    sse_numpy = compute_sse_numpy(pred, target)
    assert np.isclose(sse_native, sse_numpy, atol=1e-12)


def test_cupy_gpu_evaluation():
    if not is_cuda_available():
        pytest.skip("CUDA or CuPy is not available.")

    ast = parse("sin(x) + c0 * cos(y)")
    X = np.linspace(-1, 1, 32).reshape(4, 8)
    Y = np.linspace(-1, 1, 32).reshape(4, 8)
    params = np.array([1.8])

    out_cupy = evaluate_cupy(ast, X, Y, params)
    out_numpy = evaluate_numpy(ast, X, Y, params)

    assert np.all(np.isfinite(out_cupy))
    max_diff = float(np.max(np.abs(out_cupy - out_numpy)))
    assert max_diff < 1e-5

    pred = np.array([1.0, 2.0, 3.0, 4.0])
    target = np.array([1.1, 1.9, 3.2, 3.8])

    res_cupy = compute_residuals_cupy(pred, target)
    res_numpy = compute_residuals_numpy(pred, target)
    assert np.allclose(res_cupy, res_numpy, atol=1e-12)

    sse_cupy = compute_sse_cupy(pred, target)
    sse_numpy = compute_sse_numpy(pred, target)
    assert np.isclose(sse_cupy, sse_numpy, atol=1e-12)


def test_multi_tier_evaluator_consistency_and_fallback():
    evaluator = MultiTierEvaluator(preferred_backend="auto")
    available = evaluator.get_available_backends()
    assert "numpy" in available

    ast = parse("c0 * exp(-c1 * (x^2 + y^2)) + sin(c2 * x)")
    X, Y = np.meshgrid(np.linspace(-1, 1, 20), np.linspace(-1, 1, 20))
    params = np.array([3.0, 1.5, 2.0])

    # Check tier consistency
    report = evaluator.verify_tier_consistency(ast, X, Y, params, tolerance=1e-5)
    assert report["passed"] is True, f"Consistency failure: {report}"

    # Forced backend tests
    out_np = evaluator.evaluate(ast, X, Y, params, preferred_backend="numpy")
    assert evaluator.last_backend_used == "numpy"

    if is_native_available():
        out_nat = evaluator.evaluate(ast, X, Y, params, preferred_backend="native")
        assert evaluator.last_backend_used == "native"
        assert np.max(np.abs(out_nat - out_np)) < 1e-5

    if is_cuda_available():
        out_cu = evaluator.evaluate(ast, X, Y, params, preferred_backend="cuda")
        assert evaluator.last_backend_used == "cuda"
        assert np.max(np.abs(out_cu - out_np)) < 1e-5


def test_domain_protection_edge_cases():
    """
    Evaluates extreme singular cases across backends to verify that
    zero NaNs or Infs are produced.
    """
    evaluator = MultiTierEvaluator(preferred_backend="auto")
    X = np.array([[-1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
    Y = np.array([[0.0, -1.0], [1.0, 0.0]], dtype=np.float64)
    params = np.array([0.0, 1000.0, -100.0, -5.0])

    edge_cases = [
        "1 / 0",  # Division by constant zero
        "x / y",  # Division with zeros in coordinate grid
        "0 / 0",  # Indeterminate division
        "log(0)",  # Log of zero
        "log(y)",  # Log of negative/zero values
        "exp(c1)",  # Exponential of 1000 (overflow prevention)
        "exp(c2)",  # Exponential of -100 (underflow prevention)
        "sqrt(c3)",  # Square root of -5 (negative square root)
        "sqrt(x)",  # Square root of negative coordinates
        "tan(pi / 2)",  # Tangent pole
        "asin(2.0)",  # Asin out of [-1, 1]
        "acos(-2.0)",  # Acos out of [-1, 1]
        "c3 ^ 0.5",  # Negative base with fractional power
        "0 ^ 0",  # Zero to the zero power
        "0 ^ (-2)",  # Zero to negative power
    ]

    for expr in edge_cases:
        ast = parse(expr)
        for backend in evaluator.get_available_backends():
            res = evaluator.evaluate(ast, X, Y, params, preferred_backend=backend)
            assert np.all(np.isfinite(res)), (
                f"Domain protection failure for expression {expr!r} on backend {backend!r}: "
                f"result contains NaNs or Infs:\n{res}"
            )


def test_large_grid_evaluation_and_telemetry():
    evaluator = MultiTierEvaluator(preferred_backend="auto")
    ast = parse("c0 * sin(x) + c1 * cos(y)")
    # 512 x 512 = 262,144 points
    x = np.linspace(-1, 1, 512)
    y = np.linspace(-1, 1, 512)
    X, Y = np.meshgrid(x, y)
    params = np.array([1.5, 2.5])

    initial_eval_count = evaluator.telemetry["eval_count"]
    res = evaluator.evaluate(ast, X, Y, params)

    assert res.shape == (512, 512)
    assert np.all(np.isfinite(res))
    assert evaluator.telemetry["eval_count"] == initial_eval_count + 1
    assert evaluator.last_eval_time_ms > 0.0


def test_cupy_modulo_operator():
    """Validates that the modulo operator (%) compiles and executes on CuPy GPU."""
    if not is_cuda_available():
        pytest.skip("CUDA or CuPy is not available.")

    ast = parse("x % y")
    X, Y = np.meshgrid(np.linspace(1, 10, 8), np.linspace(1, 5, 8))
    params = np.array([])

    res_cupy = evaluate_cupy(ast, X, Y, params)
    res_numpy = evaluate_numpy(ast, X, Y, params)

    assert res_cupy.shape == X.shape
    assert np.all(np.isfinite(res_cupy))
    assert np.allclose(res_cupy, res_numpy, atol=1e-12)


def test_deep_ast_stack_depth():
    """
    Validates that deep AST nesting (depth 65+) does not trigger
    MSVC stack buffer overrun crash (STATUS_STACK_BUFFER_OVERRUN) in Native C++ VM.
    """
    if not is_native_available():
        pytest.skip("Native DLL is not available.")

    # Test right-nested exponent tower of depth 65
    expr_65 = " ^ ".join(["x"] * 65)
    ast_65 = parse(expr_65)
    X = np.ones((4, 4), dtype=np.float64)
    Y = np.ones((4, 4), dtype=np.float64)
    params = np.array([], dtype=np.float64)

    res_native_65 = evaluate_native(ast_65, X, Y, params)
    res_numpy_65 = evaluate_numpy(ast_65, X, Y, params)
    assert res_native_65.shape == X.shape
    assert np.allclose(res_native_65, res_numpy_65, atol=1e-12)

    # Test wide addition chain of 80 terms
    expr_80 = " + ".join(["x"] * 80)
    ast_80 = parse(expr_80)
    res_native_80 = evaluate_native(ast_80, X, Y, params)
    assert np.allclose(res_native_80, 80.0, atol=1e-12)

    # Test extreme depth (300 terms) which exceeds 256 stack limit: must fallback cleanly
    expr_300 = " ^ ".join(["x"] * 280)
    ast_300 = parse(expr_300)
    res_fallback = evaluate_native(ast_300, X, Y, params)
    assert res_fallback.shape == X.shape
    assert np.allclose(res_fallback, 1.0, atol=1e-12)


def test_large_power_overflow_and_nan_protection():
    """
    Validates cross-backend harmonization of safe_pow and zero NaN escape.
    Exponentiation magnitude must be clamped to exp(80) to prevent +inf.
    """
    ast = parse("x ^ y")
    X = np.array([[10000.0]])
    Y = np.array([[80.0]])
    params = np.array([])

    res_np = evaluate_numpy(ast, X, Y, params)
    assert np.all(np.isfinite(res_np))
    assert res_np[0, 0] <= 1e35

    if is_native_available():
        res_nat = evaluate_native(ast, X, Y, params)
        assert np.all(np.isfinite(res_nat))
        assert np.allclose(res_nat, res_np, rtol=1e-7)

    if is_cuda_available():
        res_cu = evaluate_cupy(ast, X, Y, params)
        assert np.all(np.isfinite(res_cu))
        assert np.allclose(res_cu, res_np, rtol=1e-7)

    # Verify sqrt NaN guard on indeterminate differences
    sqrt_ast = parse("sqrt(x^3 - y^3)")
    X_large = np.array([[1e200]])
    Y_large = np.array([[1e200]])

    res_sqrt_np = evaluate_numpy(sqrt_ast, X_large, Y_large, params)
    assert not np.isnan(res_sqrt_np[0, 0])

    if is_native_available():
        res_sqrt_nat = evaluate_native(sqrt_ast, X_large, Y_large, params)
        assert not np.isnan(res_sqrt_nat[0, 0])
        assert res_sqrt_nat[0, 0] == 0.0

    if is_cuda_available():
        res_sqrt_cu = evaluate_cupy(sqrt_ast, X_large, Y_large, params)
        assert not np.isnan(res_sqrt_cu[0, 0])
        assert res_sqrt_cu[0, 0] == 0.0


def test_coordinate_shape_mismatch_raises_value_error():
    """Validates that mismatched coordinate shapes raise ValueError instead of segfaulting."""
    ast = parse("x + y")
    X = np.zeros((10, 10))
    Y = np.zeros((5, 5))
    params = np.array([])

    evaluator = MultiTierEvaluator(preferred_backend="auto")

    with pytest.raises(ValueError, match="Coordinate arrays X and Y must have identical shapes"):
        evaluator.evaluate(ast, X, Y, params)

    if is_native_available():
        with pytest.raises(ValueError, match="Coordinate arrays X and Y must have identical shapes"):
            evaluate_native(ast, X, Y, params)

    if is_cuda_available():
        with pytest.raises(ValueError, match="Coordinate arrays X and Y must have identical shapes"):
            evaluate_cupy(ast, X, Y, params)


def test_parameter_length_mismatch_raises_index_error():
    """Validates that insufficient params array raises IndexError consistently."""
    ast = parse("c5 * x + y")
    X = np.ones((2, 2))
    Y = np.ones((2, 2))
    short_params = np.array([1.0, 2.0])  # Length 2, but c5 requires at least 6

    evaluator = MultiTierEvaluator(preferred_backend="auto")

    with pytest.raises(IndexError, match="Parameter index 5 is out of bounds"):
        evaluator.evaluate(ast, X, Y, short_params)

    with pytest.raises(IndexError, match="Parameter index 5 is out of bounds"):
        evaluate_numpy(ast, X, Y, short_params)

    if is_native_available():
        with pytest.raises(IndexError, match="Parameter index 5 is out of bounds"):
            evaluate_native(ast, X, Y, short_params)

    if is_cuda_available():
        with pytest.raises(IndexError, match="Parameter index 5 is out of bounds"):
            evaluate_cupy(ast, X, Y, short_params)


def test_scalar_0d_shape_preservation():
    """Validates that 0D scalar inputs return 0D array outputs across backends."""
    ast = parse("sin(x) + cos(y)")
    X = np.array(0.5)
    Y = np.array(0.5)
    params = np.array([])

    res_np = evaluate_numpy(ast, X, Y, params)
    assert res_np.shape == ()

    if is_native_available():
        res_nat = evaluate_native(ast, X, Y, params)
        assert res_nat.shape == ()
        assert np.isclose(res_nat, res_np, atol=1e-12)


def test_rbf_multiquadric_overflow_protection():
    """Validates that rbf_multiquadric clamps large inputs and does not overflow to inf."""
    ast = parse("rbf_multiquadric(x, y)")
    X = np.array([[1e160]])
    Y = np.array([[1e10]])
    params = np.array([])

    res_np = evaluate_numpy(ast, X, Y, params)
    assert np.all(np.isfinite(res_np))

    if is_native_available():
        res_nat = evaluate_native(ast, X, Y, params)
        assert np.all(np.isfinite(res_nat))
        assert np.allclose(res_nat, res_np, atol=1e-12)

    if is_cuda_available():
        res_cu = evaluate_cupy(ast, X, Y, params)
        assert np.all(np.isfinite(res_cu))
        assert np.allclose(res_cu, res_np, atol=1e-12)


def test_reference_renderer_type_hints_and_rendering():
    """Validates that ReferenceRenderer has valid typing annotations and renders cleanly."""
    import typing

    from mathfield.core.domain import CoordinateDomain2D, DomainMode
    from mathfield.core.verify.renderer import ReferenceRenderer

    hints = typing.get_type_hints(ReferenceRenderer.render_ast)
    assert "color_space" in hints

    renderer = ReferenceRenderer()
    ast = parse("sin(x) + cos(y)")
    domain = CoordinateDomain2D(width=16, height=16, mode=DomainMode.UNIT_SQUARE)
    rendered = renderer.render_ast(ast, domain=domain, width=16, height=16)
    assert rendered.shape == (16, 16)
    assert np.all(np.isfinite(rendered))
