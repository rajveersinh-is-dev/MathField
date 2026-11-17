"""
tests/test_basis.py

Unit test suite for mathfield.core.basis:
- Chebyshev2D (Clenshaw recurrence and Horner polynomial generation)
- Legendre2D (Bonnet 3-term recurrence)
- Fourier2D (harmonics, circular/rectangular truncation)
- DCT2Basis (Type-II discrete cosine transform, zero boundary derivative)
- WendlandC2RBF (compact support, C2 smoothness)
- BSplineSurface2D (Cox-de Boor recurrence, partition of unity)
- Rational2D (positive denominator, pole safety)
- Basis to AST parity
"""

import numpy as np
import pytest

from mathfield.core.basis.fourier import DCT2Basis, Fourier2D
from mathfield.core.basis.polynomial import Chebyshev2D, Legendre2D
from mathfield.core.basis.rational import Rational2D
from mathfield.core.basis.rbf import (
    GaussianRBF,
    MultiquadricRBF,
    WendlandC2RBF,
    generate_halton_centers,
    generate_residual_peak_centers,
    generate_uniform_centers,
)
from mathfield.core.basis.splines import BSplineSurface2D
from mathfield.core.engine.cpu_numpy import evaluate_numpy


def test_chebyshev_clenshaw_recurrence() -> None:
    # Test T_n(x) == cos(n * arccos(x)) for n up to 12
    x = np.linspace(-1.0, 1.0, 100)
    for n in range(13):
        basis = Chebyshev2D(deg_x=n, deg_y=0, truncation="tensor")
        Y = np.zeros_like(x)
        c = np.zeros(basis.num_terms)
        # Find index for (n, 0)
        idx = basis.indices.index((n, 0))
        c[idx] = 1.0
        val = basis.evaluate(x, Y, coefficients=c)

        # Reference analytical formula
        ref = np.cos(n * np.arccos(x))
        np.testing.assert_allclose(val, ref, atol=1e-12)


def test_legendre_bonnet_recurrence() -> None:
    # Test P_0, P_1, P_2, P_3, P_4 analytical formulas
    x = np.linspace(-1.0, 1.0, 100)
    Y = np.zeros_like(x)

    # P_0 = 1
    # P_1 = x
    # P_2 = 0.5 * (3*x^2 - 1)
    # P_3 = 0.5 * (5*x^3 - 3*x)
    # P_4 = (35*x^4 - 30*x^2 + 3) / 8
    refs = [
        np.ones_like(x),
        x,
        0.5 * (3.0 * x**2 - 1.0),
        0.5 * (5.0 * x**3 - 3.0 * x),
        (35.0 * x**4 - 30.0 * x**2 + 3.0) / 8.0,
    ]

    basis = Legendre2D(deg_x=4, deg_y=0, truncation="tensor")
    for n in range(5):
        idx = basis.indices.index((n, 0))
        c = np.zeros(basis.num_terms)
        c[idx] = 1.0
        val = basis.evaluate(x, Y, coefficients=c)
        np.testing.assert_allclose(val, refs[n], atol=1e-12)


def test_hyperbolic_cross_truncation() -> None:
    # For degree 4x4, full tensor has (4+1)*(4+1) = 25 terms
    # Total degree has (5*6)/2 = 15 terms
    # Hyperbolic cross has (i+1)*(j+1) <= 5:
    # (0,0)[1], (1,0)[2], (2,0)[3], (3,0)[4], (4,0)[5]
    # (0,1)[2], (1,1)[4],
    # (0,2)[3], (0,3)[4], (0,4)[5] -> count: 10 terms
    tensor = Chebyshev2D(deg_x=4, deg_y=4, truncation="tensor")
    tot_deg = Chebyshev2D(deg_x=4, deg_y=4, truncation="total_degree")
    hyp_cross = Chebyshev2D(deg_x=4, deg_y=4, truncation="hyperbolic_cross")

    assert tensor.num_terms == 25
    assert tot_deg.num_terms == 15
    assert hyp_cross.num_terms == 10
    assert hyp_cross.num_terms < tot_deg.num_terms < tensor.num_terms


def test_fourier_orthogonality() -> None:
    # 2D Fourier harmonics evaluated on uniform fine grid
    basis = Fourier2D(freq_x=2, freq_y=2, truncation="rectangular")
    x = np.linspace(-1.0, 1.0, 64, endpoint=False)
    X, Y = np.meshgrid(x, x)

    Phi = basis.basis_matrix(X, Y)
    # Gram matrix G = Phi^T Phi / N
    G = (Phi.T @ Phi) / X.size

    # Off-diagonal should be approximately zero
    np.fill_diagonal(G, 0.0)
    assert np.max(np.abs(G)) < 1e-10


def test_dct2_boundary_reflection() -> None:
    dct = DCT2Basis(freq_x=3, freq_y=3)
    # Evaluate numerical derivative at x = -1 using fine step
    dx = 1e-5
    x0 = np.array([-1.0])
    x1 = np.array([-1.0 + dx])
    y = np.array([0.0])
    Phi0 = dct.basis_matrix(x0, y)
    Phi1 = dct.basis_matrix(x1, y)
    dPhi_dx = (Phi1 - Phi0) / dx
    np.testing.assert_allclose(dPhi_dx, 0.0, atol=1e-3)


def test_wendland_c2_rbf_compact_support() -> None:
    centers = np.array([[0.0, 0.0]])
    radii = np.array([1.0])
    rbf = WendlandC2RBF(centers=centers, radii=radii)

    # Evaluate at points inside and outside radius 1.0
    x_in = np.array([0.0, 0.5])
    y_in = np.array([0.0, 0.0])
    v_in = rbf.evaluate(x_in, y_in)
    assert v_in[0] == 1.0  # (1-0)^4 * (4*0 + 1) = 1.0
    assert v_in[1] > 0.0

    x_out = np.array([1.0, 1.5, 2.0])
    y_out = np.array([0.0, 0.0, 0.0])
    v_out = rbf.evaluate(x_out, y_out)
    np.testing.assert_array_equal(v_out, [0.0, 0.0, 0.0])


def test_cox_de_boor_bspline_partition_of_unity() -> None:
    knots = np.array([0.0, 0.0, 0.0, 0.0, 0.25, 0.5, 0.75, 1.0, 1.0, 1.0, 1.0])
    ctrl = np.ones((7, 7))
    bspline = BSplineSurface2D(knots_x=knots, knots_y=knots, control_points=ctrl, degree=3)

    x = np.linspace(0.0, 1.0, 40)
    X, Y = np.meshgrid(x, x)
    vals = bspline.evaluate(X, Y)

    # When all control points are 1.0, B-spline surface must be identically 1.0
    np.testing.assert_allclose(vals, 1.0, atol=1e-12)


def test_rational_denominator_positivity() -> None:
    # Denominator coefficients zero -> Q(x, y) = 1.0
    rat = Rational2D(num_coeffs=np.array([1.0, 0.5]), den_coeffs=np.array([0.0, 0.0]), deg_x=1, deg_y=0)
    x = np.linspace(-5.0, 5.0, 50)
    X, Y = np.meshgrid(x, x)
    vals = rat.evaluate(X, Y)
    assert np.all(np.isfinite(vals))


def test_basis_to_ast_evaluation_parity() -> None:
    basis = Chebyshev2D(deg_x=2, deg_y=2, truncation="total_degree")
    c = np.array([1.5, -0.8, 0.3, 0.5, -0.2, 0.1])
    ast = basis.to_ast(coefficients=c)

    x = np.linspace(-1.0, 1.0, 16)
    X, Y = np.meshgrid(x, x)

    val_basis = basis.evaluate(X, Y, coefficients=c)
    val_ast = evaluate_numpy(ast, X, Y, np.array([]))

    np.testing.assert_allclose(val_basis, val_ast, atol=1e-10)


def test_all_basis_to_ast_parity_and_guards() -> None:
    x = np.linspace(-0.8, 0.8, 8)
    X, Y = np.meshgrid(x, x)

    # 1. Legendre 2D
    leg = Legendre2D(2, 2)
    c_leg = np.ones(leg.num_terms) * 0.4
    ast_leg = leg.to_ast(coefficients=c_leg)
    np.testing.assert_allclose(leg.evaluate(X, Y, coefficients=c_leg), evaluate_numpy(ast_leg, X, Y, np.array([])), atol=1e-10)

    # 2. Fourier 2D
    fou = Fourier2D([(1.0, 0.0), (0.0, 1.0)])
    c_fou = np.array([0.2, 0.3, 0.4, 0.1, -0.2])
    ast_fou = fou.to_ast(coefficients=c_fou)
    np.testing.assert_allclose(fou.evaluate(X, Y, coefficients=c_fou), evaluate_numpy(ast_fou, X, Y, np.array([])), atol=1e-10)

    # Fourier length guard
    with pytest.raises(ValueError, match="less than required basis terms"):
        fou.to_ast(coefficients=np.array([0.1, 0.2]))

    # Chebyshev length guard
    cheb = Chebyshev2D(2, 2)
    with pytest.raises(ValueError, match="less than required basis terms"):
        cheb.to_ast(coefficients=np.array([0.1]))

    # Legendre length guard
    with pytest.raises(ValueError, match="less than required basis terms"):
        leg.to_ast(coefficients=np.array([0.1]))

    # 3. DCT2
    dct = DCT2Basis(2, 2)
    c_dct = np.ones(dct.num_terms) * 0.1
    ast_dct = dct.to_ast(coefficients=c_dct)
    np.testing.assert_allclose(dct.evaluate(X, Y, coefficients=c_dct), evaluate_numpy(ast_dct, X, Y, np.array([])), atol=1e-10)

    # DCT2 length guard
    with pytest.raises(ValueError, match="less than required basis terms"):
        dct.to_ast(coefficients=np.array([0.1, 0.2]))

    # 4. Wendland C2 RBF
    w = WendlandC2RBF(np.array([[0.0, 0.0], [0.5, -0.5]]), np.array([1.5, 1.5]))
    c_w = np.array([1.0, -0.5])
    ast_w = w.to_ast(coefficients=c_w)
    np.testing.assert_allclose(w.evaluate(X, Y, coefficients=c_w), evaluate_numpy(ast_w, X, Y, np.array([])), atol=1e-10)
    with pytest.raises(ValueError, match="less than required basis terms"):
        w.to_ast(coefficients=np.array([0.1]))

    # 5. Gaussian RBF
    g = GaussianRBF(np.array([[0.0, 0.0]]), np.array([1.0]))
    c_g = np.array([0.8])
    ast_g = g.to_ast(coefficients=c_g)
    np.testing.assert_allclose(g.evaluate(X, Y, coefficients=c_g), evaluate_numpy(ast_g, X, Y, np.array([])), atol=1e-10)
    with pytest.raises(ValueError, match="less than required basis terms"):
        g.to_ast(coefficients=np.array([]))

    # 6. Multiquadric RBF
    mq = MultiquadricRBF(np.array([[0.0, 0.0]]), np.array([1.0]))
    c_mq = np.array([0.4])
    ast_mq = mq.to_ast(coefficients=c_mq)
    np.testing.assert_allclose(mq.evaluate(X, Y, coefficients=c_mq), evaluate_numpy(ast_mq, X, Y, np.array([])), atol=1e-10)
    with pytest.raises(ValueError, match="less than required basis terms"):
        mq.to_ast(coefficients=np.array([]))

    # 7. Rational Surface
    rat = Rational2D(deg_x=1, deg_y=1)
    ast_rat = rat.to_ast()
    np.testing.assert_allclose(rat.evaluate(X, Y), evaluate_numpy(ast_rat, X, Y, np.array([])), atol=1e-10)

    # 8. B-Spline Surface (verifies FunctionCallNode fix and length guard)
    bsp = BSplineSurface2D(np.linspace(0, 1, 8), np.linspace(0, 1, 8))
    ast_bsp = bsp.to_ast()
    val_bsp = evaluate_numpy(ast_bsp, X, Y, np.array([]))
    assert np.all(np.isfinite(val_bsp))
    with pytest.raises(ValueError, match="less than required basis terms"):
        bsp.to_ast(coefficients=np.array([0.1]))


def test_center_generators() -> None:
    uniform = generate_uniform_centers(3, 3)
    assert uniform.shape == (9, 2)

    halton = generate_halton_centers(10)
    assert halton.shape == (10, 2)
    assert np.all(halton >= -1.0) and np.all(halton <= 1.0)

    res = np.zeros((20, 20))
    res[5, 5] = 10.0
    res[15, 15] = 8.0
    peaks = generate_residual_peak_centers(res, n_centers=2)
    assert peaks.shape[0] == 2
