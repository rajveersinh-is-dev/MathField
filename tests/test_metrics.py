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

from mathfield.core.optimize.solvers import solve_linear_lstsq
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
from mathfield.core.verify.pruning import BICPruner, PruningResult


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


def _dense_bic_pruning_reference(
    Phi: np.ndarray,
    target: np.ndarray,
    reg_lambda: float = 1e-6,
    weights: np.ndarray | None = None,
) -> PruningResult:
    """Original backward-elimination algorithm, always solving dense SVDs."""
    phi_mat = np.asarray(Phi, dtype=np.float64)
    y = np.asarray(target, dtype=np.float64).ravel()
    n_rows, n_terms = phi_mat.shape
    weight_vec = np.ones(n_rows) if weights is None else np.asarray(weights, dtype=np.float64).ravel()
    weight_sum = float(np.sum(weight_vec))
    n_samples = int(np.count_nonzero(weight_vec > 0.0))
    if weight_sum <= 0.0:
        return PruningResult(
            list(range(n_terms)),
            [],
            0.0,
            0.0,
            0.0,
            0.0,
            np.zeros(n_terms, dtype=np.float64),
        )

    sqrt_weights = np.sqrt(weight_vec)
    phi_fit = phi_mat * sqrt_weights[:, np.newaxis]
    y_fit = y * sqrt_weights
    active_indices = list(range(n_terms))
    pruner = BICPruner()

    curr_c, _ = solve_linear_lstsq(phi_fit, y_fit, reg_lambda=reg_lambda, method="svd")
    res_init = y_fit - phi_fit @ curr_c
    rss_init = float(np.sum(res_init**2))
    mse_init = rss_init / weight_sum
    bic_init = pruner.compute_bic(rss_init, n_samples, len(active_indices))
    curr_bic = bic_init

    while len(active_indices) > 1:
        best_bic = float("inf")
        best_idx = -1
        best_c: np.ndarray | None = None
        for term_idx in active_indices:
            candidate_indices = [idx for idx in active_indices if idx != term_idx]
            phi_sub = phi_fit[:, candidate_indices]
            candidate_c, _ = solve_linear_lstsq(phi_sub, y_fit, reg_lambda=reg_lambda, method="svd")
            residual = y_fit - phi_sub @ candidate_c
            candidate_bic = pruner.compute_bic(float(np.sum(residual**2)), n_samples, len(candidate_indices))
            if candidate_bic < best_bic:
                best_bic = candidate_bic
                best_idx = term_idx
                best_c = candidate_c

        if best_bic < curr_bic and best_idx >= 0:
            active_indices.remove(best_idx)
            curr_bic = best_bic
            curr_c = best_c
        else:
            break

    assert curr_c is not None
    eliminated_indices = [idx for idx in range(n_terms) if idx not in active_indices]
    full_c = np.zeros(n_terms, dtype=np.float64)
    full_c[active_indices] = curr_c
    res_final = y_fit - phi_fit @ full_c
    mse_final = float(np.sum(res_final**2) / weight_sum)
    return PruningResult(
        retained_indices=active_indices,
        eliminated_indices=eliminated_indices,
        initial_bic=bic_init,
        final_bic=curr_bic,
        initial_mse=mse_init,
        final_mse=mse_final,
        coefficients=full_c,
    )


def _assert_pruning_matches_dense(
    Phi: np.ndarray,
    target: np.ndarray,
    reg_lambda: float = 1e-6,
    weights: np.ndarray | None = None,
) -> PruningResult:
    # The shared solver evaluates 0 / 0 before zeroing filtered singular
    # factors for exactly rank-deficient, unregularized systems.
    with np.errstate(invalid="ignore"):
        expected = _dense_bic_pruning_reference(Phi, target, reg_lambda=reg_lambda, weights=weights)
        actual = BICPruner().prune_matrix(Phi, target, reg_lambda=reg_lambda, weights=weights)

    assert actual.retained_indices == expected.retained_indices
    assert actual.eliminated_indices == expected.eliminated_indices
    # QR/SVD roundoff is far below these tolerances for the conditioned cases;
    # BIC gets an absolute floor because near-exact fits use an RSS floor.
    np.testing.assert_allclose(actual.coefficients, expected.coefficients, rtol=1e-9, atol=1e-10)
    np.testing.assert_allclose(
        [actual.initial_mse, actual.final_mse],
        [expected.initial_mse, expected.final_mse],
        rtol=1e-9,
        atol=1e-12,
    )
    assert actual.initial_bic == pytest.approx(expected.initial_bic, rel=1e-11, abs=1e-8)
    assert actual.final_bic == pytest.approx(expected.final_bic, rel=1e-11, abs=1e-8)
    return actual


def test_bic_pruning_weighted_ridge_matches_dense_svd() -> None:
    rng = np.random.default_rng(2026)
    n_rows = 240
    x = np.linspace(-1.0, 1.0, n_rows)
    noise_columns = rng.normal(size=(n_rows, 3))
    phi = np.column_stack([np.ones(n_rows), x, noise_columns])
    target = 0.4 - 0.8 * x + rng.normal(0.0, 0.02, size=n_rows)
    weights = rng.uniform(0.2, 1.8, size=n_rows)
    weights[::17] = 0.0

    # Neither the target nor the design values on zero-weight rows may matter.
    target[weights == 0.0] = 1e8
    phi[weights == 0.0, 2] = 1e6
    result = _assert_pruning_matches_dense(phi, target, reg_lambda=0.05, weights=weights)
    assert len(result.retained_indices) < phi.shape[1]


@pytest.mark.parametrize("reg_lambda", [0.0, 0.2])
def test_bic_pruning_rank_deficient_matches_dense_svd(reg_lambda: float) -> None:
    rng = np.random.default_rng(81)
    n_rows = 120
    base = rng.normal(size=(n_rows, 3))
    phi = np.column_stack(
        [
            np.ones(n_rows),
            base[:, 0],
            base[:, 1],
            base[:, 0] + 2.0 * base[:, 1],
            np.zeros(n_rows),
            base[:, 2],
        ]
    )
    target = 0.7 + 1.1 * base[:, 0] - 0.4 * base[:, 1] + 0.3 * base[:, 2]
    target += rng.normal(0.0, 0.03, size=n_rows)
    _assert_pruning_matches_dense(phi, target, reg_lambda=reg_lambda)


@pytest.mark.parametrize("reg_lambda", [0.0, 0.3])
def test_bic_pruning_underdetermined_matches_dense_svd(reg_lambda: float) -> None:
    rng = np.random.default_rng(92)
    phi = rng.normal(size=(5, 11))
    target = rng.normal(size=5)
    result = _assert_pruning_matches_dense(phi, target, reg_lambda=reg_lambda)
    assert result.coefficients.shape == (11,)


@pytest.mark.parametrize("reg_lambda", [0.0, 1.0])
def test_bic_pruning_collinear_matches_dense_svd(reg_lambda: float) -> None:
    rng = np.random.default_rng(63)
    n_rows = 300
    x = np.linspace(-1.0, 1.0, n_rows)
    phi = np.column_stack([np.ones(n_rows), x, x, x**2])
    target = 0.3 + 1.2 * x + 0.2 * x**2 + rng.normal(0.0, 0.01, size=n_rows)
    result = _assert_pruning_matches_dense(phi, target, reg_lambda=reg_lambda)

    if reg_lambda > 0.0:
        # Ridge damping can make both duplicate terms worth retaining; the
        # symmetric minimum-norm solution must still split their coefficient.
        assert 1 in result.retained_indices
        assert 2 in result.retained_indices
        assert result.coefficients[1] == pytest.approx(result.coefficients[2], rel=1e-9, abs=1e-10)


def test_bic_pruning_rank_ambiguity_uses_dense_svd_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    rng = np.random.default_rng(314)
    n_rows, n_terms = 12, 4
    left, _ = np.linalg.qr(rng.normal(size=(n_rows, n_terms)), mode="reduced")
    right, _ = np.linalg.qr(rng.normal(size=(n_terms, n_terms)), mode="reduced")
    singular_values = np.array([1.0, 1e-12, 1e-13, 0.5])
    phi = (left * singular_values) @ right.T
    target = left @ np.array([0.3, 1e-12, 2e-13, 0.4])

    calls: list[tuple[int, int]] = []
    original_solver = solve_linear_lstsq

    def recording_solver(
        design: np.ndarray,
        response: np.ndarray,
        reg_lambda: float = 0.0,
        method: str = "svd",
        rcond: float = 1e-12,
    ) -> tuple[np.ndarray, dict]:
        calls.append(design.shape)
        return original_solver(design, response, reg_lambda=reg_lambda, method=method, rcond=rcond)

    monkeypatch.setattr("mathfield.core.verify.pruning.solve_linear_lstsq", recording_solver)
    _assert_pruning_matches_dense(phi, target, reg_lambda=0.0)
    assert (n_rows, n_terms - 1) in calls


def test_bic_pruning_near_tie_uses_dense_svd_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    rng = np.random.default_rng(2718)
    n_rows = 200
    correlation = 0.8
    x = np.linspace(-1.0, 1.0, n_rows)
    x -= np.mean(x)
    x /= np.linalg.norm(x)
    orthogonal = rng.normal(size=n_rows)
    orthogonal -= x * (x @ orthogonal)
    orthogonal -= np.mean(orthogonal)
    orthogonal /= np.linalg.norm(orthogonal)
    second = correlation * x + math.sqrt(1.0 - correlation**2) * orthogonal
    phi = np.column_stack([np.ones(n_rows), x, second])
    q_mat, _ = np.linalg.qr(phi, mode="reduced")
    noise = rng.normal(size=n_rows)
    noise -= q_mat @ (q_mat.T @ noise)
    noise *= math.sqrt(2.0 * n_rows * (1.0 - correlation**2)) / np.linalg.norm(noise)

    # The two first-deletion BICs differ by only 2e-9, but deleting the weaker
    # term is accepted; the intercept then drops and the other term remains.
    desired_gap = 2e-9
    coefficient = 1.0
    fixed_rss = 2.0 * n_rows * (1.0 - correlation**2) * coefficient**2
    ratio = 1.0 + coefficient**2 * (1.0 - correlation**2) / fixed_rss
    second_coefficient_sq = (
        fixed_rss * (ratio * math.exp(-desired_gap / n_rows) - 1.0) / (1.0 - correlation**2)
    )
    target = noise + coefficient * x + math.sqrt(second_coefficient_sq) * second

    calls: list[tuple[int, int]] = []
    original_solver = solve_linear_lstsq

    def recording_solver(
        design: np.ndarray,
        response: np.ndarray,
        reg_lambda: float = 0.0,
        method: str = "svd",
        rcond: float = 1e-12,
    ) -> tuple[np.ndarray, dict]:
        calls.append(design.shape)
        return original_solver(design, response, reg_lambda=reg_lambda, method=method, rcond=rcond)

    monkeypatch.setattr("mathfield.core.verify.pruning.solve_linear_lstsq", recording_solver)
    result = _assert_pruning_matches_dense(phi, target, reg_lambda=0.0)
    assert result.retained_indices == [1]
    assert (n_rows, 2) in calls


def test_bic_pruning_uses_reduced_candidate_systems(monkeypatch: pytest.MonkeyPatch) -> None:
    rng = np.random.default_rng(1234)
    n_rows, n_terms = 400, 7
    phi = rng.normal(size=(n_rows, n_terms))
    target = 2.0 * phi[:, 0] - phi[:, 1] + rng.normal(0.0, 0.1, size=n_rows)
    calls: list[tuple[int, int]] = []
    original_solver = solve_linear_lstsq

    def recording_solver(
        design: np.ndarray,
        response: np.ndarray,
        reg_lambda: float = 0.0,
        method: str = "svd",
        rcond: float = 1e-12,
    ) -> tuple[np.ndarray, dict]:
        calls.append(design.shape)
        return original_solver(design, response, reg_lambda=reg_lambda, method=method, rcond=rcond)

    monkeypatch.setattr("mathfield.core.verify.pruning.solve_linear_lstsq", recording_solver)
    result = BICPruner().prune_matrix(phi, target, reg_lambda=0.0)
    assert len(calls) > n_terms
    assert any(columns == n_terms - 1 for _, columns in calls)
    # Candidate solves stay reduced; only the final retained-model refit uses
    # the original row count.
    assert all(rows <= n_terms for rows, _ in calls[:-1])
    assert calls[-1] == (n_rows, len(result.retained_indices))


def test_bic_pruning_all_zero_weights() -> None:
    phi = np.arange(30.0).reshape(10, 3)
    target = np.full(10, 1e6)
    result = BICPruner().prune_matrix(phi, target, weights=np.zeros(10))
    assert result.retained_indices == [0, 1, 2]
    assert result.eliminated_indices == []
    assert result.initial_bic == result.final_bic == 0.0
    assert result.initial_mse == result.final_mse == 0.0
    np.testing.assert_array_equal(result.coefficients, np.zeros(3))


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


def test_bic_pruning_ignores_zero_weight_outliers() -> None:
    x = np.linspace(-1.0, 1.0, 20)
    phi = np.column_stack([np.ones(x.size), x])
    target = 0.4 + 0.7 * x
    contaminated = target.copy()
    contaminated[-1] = 1e6
    weights = np.ones(x.size)
    weights[-1] = 0.0
    pruner = BICPruner()
    clean = pruner.prune_matrix(phi, target, reg_lambda=1e-6)
    weighted = pruner.prune_matrix(phi, contaminated, reg_lambda=1e-6, weights=weights)
    np.testing.assert_allclose(weighted.coefficients, clean.coefficients, atol=1e-12)


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
