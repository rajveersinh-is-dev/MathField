"""
tests/test_domain.py

Unit test suite for mathfield.core.domain (CoordinateDomain2D, DomainMode).
"""

import math
import numpy as np
import pytest

from mathfield.core.domain import CoordinateDomain2D, DomainMode


def test_unit_square_domain() -> None:
    domain = CoordinateDomain2D(width=64, height=64, mode=DomainMode.UNIT_SQUARE)
    assert domain.mode == DomainMode.UNIT_SQUARE
    assert domain.x_bounds == (-1.0, 1.0)
    assert domain.y_bounds == (-1.0, 1.0)
    assert math.isclose(domain.aspect_ratio, 1.0)

    X, Y = domain.to_discrete_grid()
    assert X.shape == (64, 64)
    assert Y.shape == (64, 64)
    assert math.isclose(float(np.min(X)), -1.0, abs_tol=1e-12)
    assert math.isclose(float(np.max(X)), 1.0, abs_tol=1e-12)
    assert math.isclose(float(np.min(Y)), -1.0, abs_tol=1e-12)
    assert math.isclose(float(np.max(Y)), 1.0, abs_tol=1e-12)


def test_aspect_preserved_domain() -> None:
    # 64x32 -> aspect ratio 2.0 -> x in [-2, 2], y in [-1, 1]
    domain = CoordinateDomain2D(width=64, height=32, mode=DomainMode.ISOTROPIC)
    assert math.isclose(domain.aspect_ratio, 2.0, rel_tol=1e-5)
    assert domain.x_bounds[0] <= -1.9
    assert domain.x_bounds[1] >= 1.9
    assert domain.y_bounds == (-1.0, 1.0)

    dx, dy = domain.cell_size()
    assert math.isclose(dx, dy, rel_tol=1e-5)
    assert math.isclose(dx, 2.0 / 32.0, rel_tol=1e-5)


def test_coordinate_transforms_roundtrip() -> None:
    domain = CoordinateDomain2D(width=128, height=64, mode=DomainMode.ASPECT_PRESERVED)
    i_px = np.arange(128)
    j_px = np.arange(64)

    x_cont, y_cont = domain.to_continuous(i_px, j_px)
    i_rec, j_rec = domain.to_discrete(x_cont, y_cont)

    np.testing.assert_array_equal(i_px, i_rec)
    np.testing.assert_array_equal(j_px, j_rec)

    # Scalar checks
    x_s, y_s = domain.to_continuous(0, 0)
    assert math.isclose(x_s, domain.x_min, abs_tol=1e-12)
    assert math.isclose(y_s, domain.y_min, abs_tol=1e-12)
    i_s, j_s = domain.to_discrete(x_s, y_s)
    assert i_s == 0
    assert j_s == 0


def test_domain_continuous_resampling() -> None:
    domain = CoordinateDomain2D(width=64, height=64)
    x_pts = np.array([0.12345, -0.6789, 0.999])
    y_pts = np.array([-0.5432, 0.3210, -0.999])
    i_px, j_px = domain.to_discrete(x_pts, y_pts)
    assert np.all(i_px >= 0) and np.all(i_px < 64)
    assert np.all(j_px >= 0) and np.all(j_px < 64)


def test_domain_corner_symmetry_bounds() -> None:
    domain = CoordinateDomain2D(width=100, height=100, mode=DomainMode.SQUARE)
    X, Y = domain.to_discrete_grid()
    center_x = float((X[0, 0] + X[-1, -1]) * 0.5)
    center_y = float((Y[0, 0] + Y[-1, -1]) * 0.5)
    assert math.isclose(center_x, 0.0, abs_tol=1e-10)
    assert math.isclose(center_y, 0.0, abs_tol=1e-10)


def test_domain_serialization() -> None:
    domain = CoordinateDomain2D(width=80, height=40, mode=DomainMode.ISOTROPIC)
    d = domain.to_dict()
    assert d["width"] == 80
    assert d["height"] == 40
    assert d["mode"] == "aspect_preserved"

    recovered = CoordinateDomain2D.from_dict(d)
    assert recovered.width == domain.width
    assert recovered.height == domain.height
    assert recovered.x_bounds == domain.x_bounds
    assert recovered.y_bounds == domain.y_bounds


def test_domain_invalid_args() -> None:
    with pytest.raises(ValueError):
        CoordinateDomain2D(width=0, height=10)
    with pytest.raises(ValueError):
        CoordinateDomain2D(width=10, height=-5)
    with pytest.raises(ValueError):
        CoordinateDomain2D(width=10, height=10, x_min=1.0, x_max=0.0)
    with pytest.raises(ValueError):
        DomainMode.from_str("invalid_mode")
