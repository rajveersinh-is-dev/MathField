"""
tests/test_metrics.py

Unit test suite for mathfield.core.verify.metrics and mathfield.core.verify.pruning:
- MSE, MAE, L_infinity
- PSNR with zero-MSE protection capped at 100.0 dB
- SSIM with 11x11 Gaussian window
- CIEDE2000 standard validation
- BIC pruning backward elimination
"""

import math
import warnings

import numpy as np
import pytest

from mathfield.core.verify.metrics import (
    _convolve_2d_reflect,
    _gaussian_kernel_2d,
    compute_ciede2000,
    compute_linf,
    compute_mae,
    compute_metrics,
    compute_mse,
    compute_psnr,
    compute_ssim,
)
from mathfield.core.verify.pruning import BICPruner


def test_mse_mae_linf() -> None:
    a = np.ones((10, 10)) * 0.5
    b = np.ones((10, 10)) * 0.5
    assert compute_mse(a, b) == 0.0
    assert compute_mae(a, b) == 0.0
    assert compute_linf(a, b) == 0.0

    b_shifted = a + 0.1
    assert math.isclose(compute_mse(a, b_shifted), 0.01, rel_tol=1e-5)
    assert math.isclose(compute_mae(a, b_shifted), 0.1, rel_tol=1e-5)
    assert math.isclose(compute_linf(a, b_shifted), 0.1, rel_tol=1e-5)


def test_psnr_exact_and_cap() -> None:
    a = np.ones((10, 10)) * 0.5
    # Identical images must return capped 100.0 dB
    assert compute_psnr(a, a) == 100.0

    # With MSE = 0.01 and MAX = 1.0 -> 20 * log10(1 / 0.1) = 20 * 1 = 20 dB
    b = a + 0.1
    psnr = compute_psnr(a, b, max_val=1.0)
    assert math.isclose(psnr, 20.0, rel_tol=1e-5)


def test_ssim_reference() -> None:
    # Identical image -> SSIM = 1.0
    rng = np.random.default_rng(123)
    img = rng.uniform(0.0, 1.0, size=(32, 32))
    assert math.isclose(compute_ssim(img, img), 1.0, abs_tol=1e-5)

    # Symmetry: SSIM(A, B) == SSIM(B, A)
    img2 = rng.uniform(0.0, 1.0, size=(32, 32))
    s12 = compute_ssim(img, img2)
    s21 = compute_ssim(img2, img)
    assert math.isclose(s12, s21, abs_tol=1e-10)

    # Inverted image has low SSIM
    inverted = 1.0 - img
    s_inv = compute_ssim(img, inverted)
    assert s_inv < 0.2


def _reference_ssim(a: np.ndarray, b: np.ndarray, max_val: float, window_size: int, sigma: float) -> float:
    if a.ndim == 3:
        return float(np.mean([_reference_ssim(a[:, :, c], b[:, :, c], max_val, window_size, sigma) for c in range(a.shape[2])]))

    kernel = _gaussian_kernel_2d(window_size, sigma)
    c1 = (0.01 * max_val) ** 2
    c2 = (0.03 * max_val) ** 2
    mu1 = _convolve_2d_reflect(a, kernel)
    mu2 = _convolve_2d_reflect(b, kernel)
    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = _convolve_2d_reflect(a * a, kernel) - mu1_sq
    sigma2_sq = _convolve_2d_reflect(b * b, kernel) - mu2_sq
    sigma12 = _convolve_2d_reflect(a * b, kernel) - mu1_mu2
    num = (2.0 * mu1_mu2 + c1) * (2.0 * sigma12 + c2)
    den = (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
    return float(np.mean(num / den))


@pytest.mark.parametrize("window_size", [5, 11, 12])
@pytest.mark.parametrize("shape", [(19, 23), (32, 32)])
def test_ssim_separable_matches_naive_convolution(window_size: int, shape: tuple[int, int]) -> None:
    rng = np.random.default_rng(123)
    a = rng.uniform(0.0, 1.0, size=shape)
    b = rng.uniform(0.0, 1.0, size=shape)
    expected = _reference_ssim(a, b, max_val=1.0, window_size=window_size, sigma=1.5)
    assert math.isclose(compute_ssim(a, b, window_size=window_size), expected, abs_tol=2e-14)


def test_ssim_separable_matches_rgb_impulses_and_near_constants() -> None:
    rng = np.random.default_rng(321)
    rgb_a = rng.uniform(0.0, 1.0, size=(24, 27, 3))
    rgb_b = rng.uniform(0.0, 1.0, size=(24, 27, 3))
    expected_rgb = _reference_ssim(rgb_a, rgb_b, max_val=1.0, window_size=11, sigma=1.5)
    assert math.isclose(compute_ssim(rgb_a, rgb_b), expected_rgb, abs_tol=2e-14)

    impulse_a = np.zeros((23, 27))
    impulse_b = np.zeros((23, 27))
    impulse_a[0, 13] = 1.0
    impulse_b[22, 13] = 0.8
    expected_impulse = _reference_ssim(impulse_a, impulse_b, max_val=1.0, window_size=11, sigma=1.5)
    assert math.isclose(compute_ssim(impulse_a, impulse_b), expected_impulse, abs_tol=2e-14)

    constant_a = np.full((23, 27), 0.5) + rng.normal(0.0, 1e-9, size=(23, 27))
    constant_b = np.full((23, 27), 0.5) + rng.normal(0.0, 1e-9, size=(23, 27))
    expected_constant = _reference_ssim(constant_a, constant_b, max_val=1.0, window_size=11, sigma=1.5)
    assert math.isclose(compute_ssim(constant_a, constant_b), expected_constant, abs_tol=2e-14)


def test_rgba_metrics_ignore_fully_transparent_hidden_rgb() -> None:
    rng = np.random.default_rng(44)
    target = np.zeros((16, 18, 4), dtype=np.float64)
    target[:8, :, :3] = rng.random((8, 18, 3))
    target[8:, :, 3] = 1.0
    first = target.copy()
    second = target.copy()
    second[:8, :, :3] = rng.random((8, 18, 3))

    first_metrics = compute_metrics(first, target)
    second_metrics = compute_metrics(second, target)
    for key, value in first_metrics.items():
        if not key.startswith("raw_"):
            assert second_metrics[key] == value
    assert first_metrics["raw_mse"] != second_metrics["raw_mse"]
    assert first_metrics["mse"] == 0.0
    assert first_metrics["psnr"] == 100.0
    assert first_metrics["ssim"] == 1.0


def test_rgba_metrics_include_transparency_error() -> None:
    target = np.ones((16, 16, 4), dtype=np.float64)
    target[:, :, :3] = 0.0
    prediction = target.copy()
    prediction[:, :, 3] = 0.0
    metrics = compute_metrics(prediction, target)
    assert metrics["color_mse"] == 0.0
    assert metrics["alpha_mse"] == 1.0
    assert metrics["mse"] == 0.25
    assert metrics["psnr"] < 100.0


def test_metrics_reject_mismatched_shapes() -> None:
    with pytest.raises(ValueError, match="shapes must match"):
        compute_metrics(np.zeros((4, 4, 3)), np.zeros((4, 4)))


def test_ciede2000_sharma_pairs() -> None:
    # Identical colors -> Delta E_00 = 0.0
    lab1 = np.array([50.0, 2.6772, -79.7751])
    assert compute_ciede2000(lab1, lab1) == 0.0

    # Test standard reference pair from Sharma et al. (2005)
    # Pair 1: L1=50.0, a1=2.6772, b1=-79.7751, L2=50.0, a2=0.0, b2=-82.7485
    # Standard published Delta E_00 = 2.0425
    lab2 = np.array([50.0, 0.0, -82.7485])
    de = compute_ciede2000(lab1, lab2)
    assert math.isclose(de, 2.0425, abs_tol=1e-3)


def test_bic_pruning() -> None:
    n = 500
    x = np.linspace(-1, 1, n)
    # True signal: 2.0 * x + 1.0
    # Augmented design matrix with 2 true terms (1, x) and 3 random noise columns
    rng = np.random.default_rng(42)
    col1 = np.ones(n)
    col2 = x
    noise_cols = rng.normal(0, 0.01, size=(n, 3))
    Phi = np.column_stack([col1, col2, noise_cols])

    target = 1.0 * col1 + 2.0 * col2 + rng.normal(0, 0.001, size=n)

    pruner = BICPruner()
    res = pruner.prune_matrix(Phi, target, reg_lambda=1e-6)

    # The first two columns should be retained
    assert 0 in res.retained_indices
    assert 1 in res.retained_indices
    assert res.final_bic <= res.initial_bic


def test_metrics_empty_inputs_zero_warnings() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        m = compute_metrics(np.array([]), np.array([]))
        assert m["mse"] == 0.0
        assert m["mae"] == 0.0
        assert m["linf"] == 0.0
        assert m["psnr"] == 100.0
        assert m["ssim"] == 1.0

        assert compute_ciede2000(np.array([]), np.array([])) == 0.0


def test_bic_pruner_empty_matrix() -> None:
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        pruner = BICPruner()
        res = pruner.prune_matrix(np.zeros((0, 3)), np.zeros(0))
        assert res.initial_mse == 0.0
        assert res.final_mse == 0.0
        assert len(res.coefficients) == 3
