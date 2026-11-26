from __future__ import annotations

import numpy as np
import pytest

from mathfield.core.optimize.solvers import blocked_qr, solve_blocked_qr


def _stream_blocks(a: np.ndarray, b: np.ndarray, block_size: int):
    for start in range(0, a.shape[0], block_size):
        stop = min(start + block_size, a.shape[0])
        yield a[start:stop], b[start:stop]


@pytest.mark.parametrize("block_size", [1, 3, 8, 64])
@pytest.mark.parametrize("case", ["well_conditioned", "collinear", "underdetermined", "zero"])
def test_blocked_qr_matches_dense_least_squares(block_size: int, case: str) -> None:
    rng = np.random.default_rng(17)
    if case == "well_conditioned":
        a = rng.normal(size=(80, 7))
        b = rng.normal(size=(80, 3))
    elif case == "collinear":
        base = rng.normal(size=(60, 3))
        a = np.column_stack([base, base[:, 0], base[:, 1] + base[:, 2]])
        b = rng.normal(size=(60, 2))
    elif case == "underdetermined":
        a = rng.normal(size=(5, 9))
        b = rng.normal(size=(5, 2))
    else:
        a = np.zeros((20, 5))
        b = rng.normal(size=(20, 2))

    r_matrix, projected_targets = blocked_qr(_stream_blocks(a, b, block_size), a.shape[1])
    actual, diagnostics = solve_blocked_qr(r_matrix, projected_targets, enable_regularization=False)
    expected = np.linalg.lstsq(a, b, rcond=1e-10)[0]

    np.testing.assert_allclose(actual, expected, atol=2e-10)
    np.testing.assert_allclose(a @ actual, a @ expected, atol=2e-10)
    assert diagnostics["effective_rank"] <= a.shape[1]
    assert diagnostics["raw_condition_number"] is None or np.isfinite(diagnostics["raw_condition_number"])


@pytest.mark.parametrize("reg_lambda", [1e-8, 1e-4, 1.0])
def test_blocked_qr_tikhonov_matches_dense_augmented_predictions(reg_lambda: float) -> None:
    rng = np.random.default_rng(23)
    base = rng.normal(size=(70, 5))
    a = np.column_stack([base, base[:, 0]])
    b = rng.normal(size=(70, 3))
    r_matrix, projected_targets = blocked_qr(_stream_blocks(a, b, 11), a.shape[1])
    actual, diagnostics = solve_blocked_qr(
        r_matrix,
        projected_targets,
        reg_lambda=reg_lambda,
        max_condition_number=1.0,
    )
    design = np.vstack([a, np.sqrt(reg_lambda) * np.eye(a.shape[1])])
    targets = np.vstack([b, np.zeros((a.shape[1], b.shape[1]))])
    expected = np.linalg.lstsq(design, targets, rcond=1e-10)[0]

    assert diagnostics["is_regularized"] is True
    np.testing.assert_allclose(a @ actual, a @ expected, atol=2e-10)


def test_blocked_qr_rejects_malformed_blocks() -> None:
    with pytest.raises(ValueError, match="positive"):
        blocked_qr([], 0)
    with pytest.raises(ValueError, match="At least one"):
        blocked_qr([], 3)
    with pytest.raises(ValueError, match="shape"):
        blocked_qr([(np.ones((4, 2)), np.ones((4, 1)))], 3)
    with pytest.raises(ValueError, match="rows"):
        blocked_qr([(np.ones((4, 3)), np.ones((3, 1)))], 3)
