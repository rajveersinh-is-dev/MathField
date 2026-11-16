"""
tests/test_color.py

Unit test suite for mathfield.core.color (ColorSpaceConverter, ColorSpace).
"""

import math
import numpy as np
import pytest

from mathfield.core.color import ColorSpaceConverter, ColorSpace


def test_srgb_linear_roundtrip() -> None:
    # Test across dynamic range [0, 1] including near toe threshold 0.04045
    srgb = np.linspace(0.0, 1.0, 1000)
    linear = ColorSpaceConverter.srgb_to_linear(srgb)
    srgb_rec = ColorSpaceConverter.linear_to_srgb(linear)
    np.testing.assert_allclose(srgb, srgb_rec, atol=1e-12)


def test_linear_to_cartesian_hsv_continuity() -> None:
    # Test near red hue wrap-around H = 0 <=> 2pi
    rgb_left = np.array([1.0, 0.0, 0.05])
    rgb_right = np.array([1.0, 0.05, 0.0])

    uvV_left = ColorSpaceConverter.rgb_to_cartesian_hsv(rgb_left)
    uvV_right = ColorSpaceConverter.rgb_to_cartesian_hsv(rgb_right)

    # In polar HSV, hue angle jumps from ~0.99 to ~0.01 (diff = 0.98 in [0, 1] scale)
    # In Cartesian (u, v), distance must be very small (< 0.2)
    dist = float(np.linalg.norm(uvV_left[:2] - uvV_right[:2]))
    assert dist < 0.2


def test_cartesian_hsv_roundtrip() -> None:
    rng = np.random.default_rng(42)
    rgb = rng.uniform(0.0, 1.0, size=(50, 3))
    uvV = ColorSpaceConverter.rgb_to_cartesian_hsv(rgb)
    rgb_rec = ColorSpaceConverter.cartesian_hsv_to_rgb(uvV)
    np.testing.assert_allclose(rgb, rgb_rec, atol=1e-10)


def test_cielab_roundtrip_and_dark_derivatives() -> None:
    srgb = np.array([
        [0.0, 0.0, 0.0],
        [0.5, 0.25, 0.75],
        [1.0, 1.0, 1.0],
        [0.01, 0.02, 0.005],  # in linear toe region
    ])
    lab = ColorSpaceConverter.srgb_to_cielab(srgb)
    rec = ColorSpaceConverter.cielab_to_srgb(lab)
    np.testing.assert_allclose(srgb, rec, atol=1e-3)

    # Check finite derivative at dark threshold
    eps = 1e-6
    dark1 = np.array([[eps, eps, eps]])
    dark2 = np.array([[2 * eps, 2 * eps, 2 * eps]])
    lab1 = ColorSpaceConverter.srgb_to_cielab(dark1)
    lab2 = ColorSpaceConverter.srgb_to_cielab(dark2)
    grad = (lab2[0, 0] - lab1[0, 0]) / eps
    assert np.isfinite(grad)
    assert grad > 0.0


def test_alpha_weighted_loss() -> None:
    alpha = np.array([0.0, 0.0, 1.0, 1.0])
    pred = np.array([1000.0, 5000.0, 1.0, 2.0])
    target = np.array([0.0, 0.0, 0.0, 0.0])

    loss = ColorSpaceConverter.alpha_weighted_loss(pred, target, alpha)
    # Only the last two entries (1^2 + 2^2)/2 = 5/2 = 2.5 should contribute
    assert math.isclose(loss, 2.5, rel_tol=1e-5)


def test_smooth_soft_clipping() -> None:
    unbounded = np.linspace(-20.0, 20.0, 500)
    clipped = ColorSpaceConverter.soft_clamp(unbounded)
    assert np.all(clipped > 0.0)
    assert np.all(clipped < 1.0)
    assert math.isclose(float(ColorSpaceConverter.soft_clamp(np.array([0.5]))[0]), 0.5)

    # Numerical derivative must be strictly positive everywhere
    d_clipped = np.diff(clipped) / np.diff(unbounded)
    assert np.all(d_clipped > 0.0)
    # At center 0.5 derivative should be 1.0
    eps = 1e-5
    c_plus = ColorSpaceConverter.soft_clamp(np.array([0.5 + eps]))[0]
    c_minus = ColorSpaceConverter.soft_clamp(np.array([0.5 - eps]))[0]
    deriv_at_half = (c_plus - c_minus) / (2 * eps)
    assert math.isclose(float(deriv_at_half), 1.0, rel_tol=1e-4)


def test_color_space_converter_hub() -> None:
    rgb = np.array([[[0.2, 0.4, 0.6]]])
    lab = ColorSpaceConverter.convert(rgb, ColorSpace.SRGB, ColorSpace.CIELAB)
    hsv = ColorSpaceConverter.convert(lab, ColorSpace.CIELAB, ColorSpace.HSV)
    rgb_rec = ColorSpaceConverter.convert(hsv, ColorSpace.HSV, ColorSpace.SRGB)
    np.testing.assert_allclose(rgb, rgb_rec, atol=1e-2)
