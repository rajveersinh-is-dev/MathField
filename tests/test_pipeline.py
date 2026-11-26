"""
tests/test_pipeline.py

Unit and integration test suite for Milestone 2 reconstruction pipeline:
- Stage 1: Image analysis (TV, entropy, spectral roll-off)
- Stage 3: Quadtree adaptive spatial subdivision & Partition of Unity
- Stage 4: VarPro, Levenberg-Marquardt, deterministic evolutionary seeding, SVD conditioning
- Stage 5: Modes A & B end-to-end execution and ReconstructionController lifecycle
"""

import math

import numpy as np
import pytest

from mathfield.core.basis.fourier import DCT2Basis, Fourier2D
from mathfield.core.basis.polynomial import Chebyshev2D
from mathfield.core.domain import CoordinateDomain2D
from mathfield.core.image_analysis import ImageAnalyzer
from mathfield.core.local.local_patch import LocalPatch
from mathfield.core.local.partition_of_unity import WendlandPartitionOfUnity
from mathfield.core.local.quadtree import QuadtreeDecomposition, QuadtreeNode
from mathfield.core.optimize.conditioning import ConditionMonitor
from mathfield.core.optimize.evolutionary import EvolutionaryOptimizer
from mathfield.core.optimize.solvers import LevenbergMarquardtOptimizer
from mathfield.core.optimize.varpro import VarProOptimizer
from mathfield.core.pipeline.controller import ReconstructionController
from mathfield.core.pipeline.mode_a import ModeAPipeline
from mathfield.core.pipeline.mode_b import ModeBPipeline
from mathfield.core.pipeline.stage import PipelineContext, StageStatus


def test_stage1_image_analysis() -> None:
    analyzer = ImageAnalyzer()
    # Flat image
    flat = np.full((32, 32, 3), 100, dtype=np.uint8)
    res_flat = analyzer.analyze(flat)
    assert res_flat.width == 32
    assert res_flat.height == 32
    assert res_flat.channels == 3
    assert res_flat.has_alpha is False
    assert res_flat.total_variation == 0.0
    assert math.isclose(res_flat.entropy, 0.0, abs_tol=1e-10)

    # Checkerboard image with high frequency
    checker = np.zeros((32, 32), dtype=np.uint8)
    checker[::2, ::2] = 255
    checker[1::2, 1::2] = 255
    res_checker = analyzer.analyze(checker)
    assert res_checker.total_variation > 0.0
    assert res_checker.entropy > 0.0

    # Empty array edge cases
    empty_arr = np.array([])
    assert analyzer.compute_total_variation(empty_arr) == 0.0
    assert analyzer.compute_shannon_entropy(empty_arr) == 0.0
    np.testing.assert_allclose(analyzer.compute_channel_correlations(empty_arr), np.eye(1))




def test_stage3_quadtree_subdivision() -> None:
    decomp = QuadtreeDecomposition(max_depth=3, split_threshold=0.05)
    residual = np.zeros((32, 32))
    # High error on top-left quadrant only
    residual[:16, :16] = 1.0

    root = decomp.decompose(residual)
    leaves = decomp.get_leaf_nodes(root)

    # Should have more than 4 leaves because top-left quadrant split deeper
    assert len(leaves) >= 4
    assert any(leaf.depth > 1 for leaf in leaves)


def test_stage3_partition_of_unity_completeness() -> None:
    pou = WendlandPartitionOfUnity(overlap_ratio=0.25)
    leaves = [
        QuadtreeNode(bounds=(-1.0, -1.0, 0.0, 0.0), depth=1, is_leaf=True),
        QuadtreeNode(bounds=(0.0, -1.0, 1.0, 0.0), depth=1, is_leaf=True),
        QuadtreeNode(bounds=(-1.0, 0.0, 0.0, 1.0), depth=1, is_leaf=True),
        QuadtreeNode(bounds=(0.0, 0.0, 1.0, 1.0), depth=1, is_leaf=True),
    ]
    x = np.linspace(-1.0, 1.0, 32)
    X, Y = np.meshgrid(x, x)

    weights = pou.compute_weights(leaves, X, Y)
    sum_weights = np.sum(weights, axis=0)
    # Weights must sum to 1.0 everywhere within 1e-5
    np.testing.assert_allclose(sum_weights, 1.0, atol=1e-5)


def _dense_partition_reference(pou, leaves, X, Y):
    if len(leaves) == 1:
        return np.ones((1, *X.shape), dtype=np.float64)
    x_flat = np.asarray(X, dtype=np.float64).ravel()
    y_flat = np.asarray(Y, dtype=np.float64).ravel()
    centers = np.array([leaf.center for leaf in leaves])
    half_widths = np.array([leaf.half_widths for leaf in leaves])

    def build(factor):
        rx = factor * np.maximum(half_widths[:, 0], 1e-6)
        ry = factor * np.maximum(half_widths[:, 1], 1e-6)
        dx = (x_flat[None, :] - centers[:, 0:1]) / rx[:, None]
        dy = (y_flat[None, :] - centers[:, 1:2]) / ry[:, None]
        if pou.separable:
            adx = np.minimum(np.abs(dx), 1.0)
            ady = np.minimum(np.abs(dy), 1.0)
            inside = (np.abs(dx) <= 1.0 + 1e-12) & (np.abs(dy) <= 1.0 + 1e-12)
            wx = ((1.0 - adx) ** 4.0) * (4.0 * adx + 1.0)
            wy = ((1.0 - ady) ** 4.0) * (4.0 * ady + 1.0)
            return np.where(inside, wx * wy, 0.0)
        radius = np.sqrt(dx * dx + dy * dy)
        sr = np.minimum(radius, 1.0)
        return np.where(radius <= 1.0 + 1e-12, ((1.0 - sr) ** 4.0) * (4.0 * sr + 1.0), 0.0)

    factor = 1.0 + max(pou.overlap_ratio, 1e-6)
    weights = build(factor)
    for _ in range(12):
        if np.all(np.sum(weights, axis=0) > 1e-12):
            break
        factor *= 1.25
        weights = build(factor)
    zero_mask = np.sum(weights, axis=0) <= 1e-12
    if np.any(zero_mask):
        zx, zy = x_flat[zero_mask], y_flat[zero_mask]
        d2 = (centers[:, 0:1] - zx[None, :]) ** 2 + (centers[:, 1:2] - zy[None, :]) ** 2
        weights[:, zero_mask] = np.exp(-d2 / max(float(np.max(d2)), 1e-12))
    return (weights / np.maximum(np.sum(weights, axis=0), 1e-300)).reshape(len(leaves), *X.shape)


@pytest.mark.parametrize("separable", [True, False])
@pytest.mark.parametrize("overlap_ratio", [0.0, 0.25])
@pytest.mark.parametrize("outside", [False, True])
def test_partition_of_unity_sparse_iteration_matches_dense_reference(
    separable: bool,
    overlap_ratio: float,
    outside: bool,
) -> None:
    leaves = [
        QuadtreeNode(bounds=(-1.0, -1.0, 0.0, 0.0), depth=1, is_leaf=True),
        QuadtreeNode(bounds=(0.0, -1.0, 1.0, 0.0), depth=1, is_leaf=True),
        QuadtreeNode(bounds=(-1.0, 0.0, 0.0, 1.0), depth=1, is_leaf=True),
        QuadtreeNode(bounds=(0.0, 0.0, 1.0, 1.0), depth=1, is_leaf=True),
    ]
    extent = 1.3 if outside else 1.0
    axis = np.linspace(-extent, extent, 29)
    X, Y = np.meshgrid(axis, axis)
    pou = WendlandPartitionOfUnity(overlap_ratio=overlap_ratio, separable=separable)
    expected = _dense_partition_reference(pou, leaves, X, Y)
    actual = pou.compute_weights(leaves, X, Y)
    np.testing.assert_allclose(actual, expected, atol=2e-14)
    np.testing.assert_allclose(np.sum(actual, axis=0), 1.0, atol=2e-14)

    streamed = np.zeros_like(actual)
    flat = streamed.reshape(len(leaves), -1)
    for k, indices, values in pou.iter_weights(leaves, X, Y):
        flat[k, indices] = values
    np.testing.assert_array_equal(streamed, actual)


def test_partition_of_unity_sparse_storage_is_proportional_to_support() -> None:
    leaves = []
    for row in range(4):
        for col in range(4):
            leaves.append(
                QuadtreeNode(
                    bounds=(-1.0 + col * 0.5, -1.0 + row * 0.5, -0.5 + col * 0.5, -0.5 + row * 0.5),
                    depth=2,
                    is_leaf=True,
                )
            )
    axis = np.linspace(-1.0, 1.0, 128)
    X, Y = np.meshgrid(axis, axis)
    pou = WendlandPartitionOfUnity(overlap_ratio=0.25)
    nonzero = sum(indices.size for _, indices, _ in pou.iter_weights(leaves, X, Y))
    assert nonzero < 0.2 * len(leaves) * X.size


@pytest.mark.parametrize("reg_lambda", [0.0, 1e-5])
def test_local_patch_rgb_joint_solve_matches_channel_reference(reg_lambda: float) -> None:
    rng = np.random.default_rng(91)
    height, width = 12, 14
    x = np.linspace(-0.9, 0.8, width)
    y = np.linspace(-0.7, 1.0, height)
    X, Y = np.meshgrid(x, y)
    residual = rng.normal(size=(height, width, 3))
    weights = rng.uniform(0.2, 1.0, size=(height, width))
    basis = DCT2Basis(freq_x=3, freq_y=3)
    patch = LocalPatch(bounds=(-0.8, -0.6, 0.9, 1.1), basis=basis)

    patch.fit(residual, X, Y, weights=weights, reg_lambda=reg_lambda)

    u, v = patch.normalize_coordinates(X, Y)
    active = weights.ravel() > 1e-12
    phi = basis.basis_matrix(u.ravel()[active], v.ravel()[active])
    sqrt_w = np.sqrt(weights.ravel()[active])
    design = phi * sqrt_w[:, np.newaxis]
    targets = residual.reshape(-1, 3)[active] * sqrt_w[:, np.newaxis]
    if reg_lambda > 0.0:
        design = np.vstack([design, np.sqrt(reg_lambda) * np.eye(basis.num_terms)])
        targets = np.vstack([targets, np.zeros((basis.num_terms, 3))])
    expected = np.column_stack([np.linalg.lstsq(design, targets[:, ch], rcond=1e-10)[0] for ch in range(3)])
    np.testing.assert_allclose(patch.coefficients, expected, atol=2e-12)

    full_phi = basis.basis_matrix(u, v)
    expected_eval = (full_phi @ expected).reshape(height, width, 3)
    np.testing.assert_allclose(patch.evaluate(X, Y), expected_eval, atol=2e-12)


def test_local_patch_rgba_sample_weights_match_channel_reference() -> None:
    rng = np.random.default_rng(92)
    height, width = 12, 14
    x = np.linspace(-0.9, 0.8, width)
    y = np.linspace(-0.7, 1.0, height)
    X, Y = np.meshgrid(x, y)
    residual = rng.normal(size=(height, width, 4))
    partition_weights = rng.uniform(0.2, 1.0, size=(height, width))
    alpha = rng.uniform(0.0, 1.0, size=(height, width))
    sample_weights = np.stack([alpha, alpha, alpha, np.ones_like(alpha)], axis=-1)
    basis = DCT2Basis(freq_x=2, freq_y=2)
    patch = LocalPatch(bounds=(-0.8, -0.6, 0.9, 1.1), basis=basis)
    patch.fit(residual, X, Y, weights=partition_weights, reg_lambda=1e-5, sample_weights=sample_weights)

    u, v = patch.normalize_coordinates(X, Y)
    phi = basis.basis_matrix(u, v)
    expected_channels = []
    for channel in range(4):
        row_weights = np.sqrt(partition_weights * sample_weights[:, :, channel])
        design = phi * row_weights.ravel()[:, np.newaxis]
        targets = residual[:, :, channel].ravel() * row_weights.ravel()
        design = np.vstack([design, np.sqrt(1e-5) * np.eye(basis.num_terms)])
        targets = np.concatenate([targets, np.zeros(basis.num_terms)])
        expected_channels.append(np.linalg.lstsq(design, targets, rcond=1e-10)[0])
    np.testing.assert_allclose(patch.coefficients, np.column_stack(expected_channels), atol=2e-12)


def test_stage4_varpro_convergence() -> None:
    # Model: y = c0 * exp(-theta0 * x^2) + c1 * cos(theta1 * x)
    # True parameters: theta = [2.0, 3.0], c = [1.5, -0.8]
    x = np.linspace(-1.0, 1.0, 50)
    Y = np.zeros_like(x)
    y_true = 1.5 * np.exp(-2.0 * x**2) - 0.8 * np.cos(3.0 * x)

    def basis_fn(theta: np.ndarray, X_coord: np.ndarray, Y_coord: np.ndarray) -> np.ndarray:
        th0, th1 = theta[0], theta[1]
        x_flat = X_coord.ravel()
        col0 = np.exp(-th0 * (x_flat**2))
        col1 = np.cos(th1 * x_flat)
        return np.column_stack([col0, col1])

    solver = VarProOptimizer(max_iter=30, tol_g=1e-6)
    res = solver.fit(basis_fn, x, Y, y_true, theta_init=np.array([1.5, 2.5]))

    assert res.cost < 1e-4
    np.testing.assert_allclose(res.theta_opt, [2.0, 3.0], atol=1e-2)
    np.testing.assert_allclose(res.c_opt, [1.5, -0.8], atol=1e-2)

    # Pure linear subproblem: empty theta_init
    def linear_only_basis(theta: np.ndarray, X_coord: np.ndarray, Y_coord: np.ndarray) -> np.ndarray:
        x_flat = X_coord.ravel()
        return np.column_stack([np.ones_like(x_flat), x_flat])

    y_linear = 2.0 + 3.0 * x
    res_linear = solver.fit(linear_only_basis, x, Y, y_linear, theta_init=np.array([]))
    assert res_linear.converged is True
    assert res_linear.n_iter == 0
    np.testing.assert_allclose(res_linear.c_opt, [2.0, 3.0], atol=1e-5)


def test_stage4_levenberg_marquardt_adaptive_damping() -> None:
    # Minimize Rosenbrock-like residual: r(x) = [10*(x1 - x0^2), 1 - x0]
    def res_fn(x: np.ndarray) -> np.ndarray:
        return np.array([10.0 * (x[1] - x[0] ** 2), 1.0 - x[0]])

    def jac_fn(x: np.ndarray) -> np.ndarray:
        return np.array(
            [
                [-20.0 * x[0], 10.0],
                [-1.0, 0.0],
            ]
        )

    lm = LevenbergMarquardtOptimizer(initial_lambda=1e-3, max_iter=50)
    res = lm.minimize(res_fn, jac_fn, x0=np.array([0.0, 0.0]))
    assert res["cost"] < 1e-6
    np.testing.assert_allclose(res["x"], [1.0, 1.0], atol=1e-3)

    # Edge cases: max_iter <= 0 and empty x0
    lm_zero = LevenbergMarquardtOptimizer(max_iter=0)
    res_zero = lm_zero.minimize(res_fn, jac_fn, x0=np.array([0.5, 0.5]))
    assert res_zero["n_iter"] == 0
    assert not res_zero["converged"]

    res_empty = lm.minimize(lambda x: np.array([]), lambda x: np.zeros((0, 0)), x0=np.array([]))
    assert res_empty["converged"] is True
    assert res_empty["n_iter"] == 0



def test_stage4_deterministic_evolutionary_seeding() -> None:
    x = np.linspace(-1.0, 1.0, 16)
    X, Y = np.meshgrid(x, x)
    target = np.sin(np.pi * X) + np.cos(np.pi * Y)

    opt1 = EvolutionaryOptimizer(random_seed=123, population_size=10, generations=3)
    best1, diag1 = opt1.evolve(X, Y, target)

    opt2 = EvolutionaryOptimizer(random_seed=123, population_size=10, generations=3)
    best2, diag2 = opt2.evolve(X, Y, target)

    # Identical seed must yield identical best fitness and formula string
    assert math.isclose(diag1["best_fitness"], diag2["best_fitness"], abs_tol=1e-12)
    assert str(best1) == str(best2)


def test_evolutionary_tolerance_stops_after_stagnation() -> None:
    axis = np.linspace(-1.0, 1.0, 12)
    X, Y = np.meshgrid(axis, axis)
    target = np.sin(X) + np.cos(Y)
    optimizer = EvolutionaryOptimizer(random_seed=5, population_size=8, generations=20, tolerance=1e9)
    _, diagnostics = optimizer.evolve(X, Y, target)
    assert diagnostics["converged"] is True
    assert diagnostics["generations_run"] == 5

    exhaustive = EvolutionaryOptimizer(random_seed=5, population_size=8, generations=7, tolerance=0.0)
    _, diagnostics = exhaustive.evolve(X, Y, target)
    assert diagnostics["converged"] is False
    assert diagnostics["generations_run"] == 7

    with pytest.raises(ValueError, match="tolerance"):
        EvolutionaryOptimizer(tolerance=-1.0)
    with pytest.raises(ValueError, match="tolerance"):
        EvolutionaryOptimizer(tolerance=float("nan"))


def test_stage4_conditioning_stabilization() -> None:
    monitor = ConditionMonitor(max_condition_number=1e10, tikhonov_alpha=1e-4)
    # Collinear matrix with tiny perturbation
    A = np.array([[1.0, 1.0], [1.0, 1.0 + 1e-15]])
    report = monitor.analyze_matrix(A)
    assert report["is_ill_conditioned"] is True

    A_reg = monitor.regularize(A)
    assert np.linalg.cond(A_reg) < 1e8


def test_mode_a_discovery() -> None:
    domain = CoordinateDomain2D(width=24, height=24)
    X, Y = domain.to_discrete_grid()
    synth_image = (np.sin(np.pi * X) * 0.5 + 0.5).astype(np.float64)

    context = PipelineContext(image=synth_image, domain=domain)
    pipeline = ModeAPipeline(config={"lambda_reg": 0.05, "max_iter": 5, "seed": 42})
    status = pipeline.execute(context)

    assert status == StageStatus.COMPLETED
    assert context.global_model is not None
    assert "formula" in context.global_model
    assert context.metrics["psnr"] > 15.0
    assert context.metrics["converged"] in (0.0, 1.0)
    assert context.metrics["generations_run"] >= 1.0


def test_mode_a_rgba_fit_ignores_hidden_rgb() -> None:
    rng = np.random.default_rng(101)
    height, width = 12, 14
    domain = CoordinateDomain2D(width=width, height=height)
    first = np.zeros((height, width, 4), dtype=np.float64)
    second = first.copy()
    first[:6, :, :3] = rng.random((6, width, 3))
    second[:6, :, :3] = rng.random((6, width, 3))
    yy, xx = np.mgrid[0:height, 0:width] / np.array([height - 1, width - 1])[:, None, None]
    first[6:, :, 0] = 0.2 + 0.6 * xx[6:]
    first[6:, :, 1] = 0.7 - 0.4 * yy[6:]
    first[6:, :, 2] = 0.1 + 0.7 * xx[6:] * yy[6:]
    first[:6, :, 3] = 0.0
    first[6:, :, 3] = 0.3 + 0.5 * xx[6:]
    second[6:] = first[6:]
    contexts = []
    for image in (first, second):
        context = PipelineContext(image=image, domain=domain)
        assert ModeAPipeline(config={"max_iter": 1, "seed": 42}).execute(context) == StageStatus.COMPLETED
        contexts.append(context)
    for actual, expected in zip(contexts[0].global_model["channel_coefficients"], contexts[1].global_model["channel_coefficients"], strict=True):
        np.testing.assert_allclose(actual, expected, atol=1e-12)
    np.testing.assert_allclose(contexts[0].rendered_image, contexts[1].rendered_image, atol=1e-12)
    assert contexts[0].metrics["mse"] == pytest.approx(contexts[1].metrics["mse"], abs=1e-15)


def test_mode_b_fidelity() -> None:
    domain = CoordinateDomain2D(width=24, height=24)
    X, Y = domain.to_discrete_grid()
    synth_image = (0.5 + 0.3 * X + 0.2 * Y).astype(np.float64)

    context = PipelineContext(image=synth_image, domain=domain)
    pipeline = ModeBPipeline(config={"global_degree": 3, "max_depth": 2})
    status = pipeline.execute(context)

    assert status == StageStatus.COMPLETED
    assert context.condition_report is not None
    assert context.metrics["psnr"] > 25.0


def test_mode_b_rgba_fit_ignores_hidden_rgb() -> None:
    rng = np.random.default_rng(102)
    height, width = 12, 14
    domain = CoordinateDomain2D(width=width, height=height)
    first = np.zeros((height, width, 4), dtype=np.float64)
    second = first.copy()
    first[:6, :, :3] = rng.random((6, width, 3))
    second[:6, :, :3] = rng.random((6, width, 3))
    yy, xx = np.mgrid[0:height, 0:width] / np.array([height - 1, width - 1])[:, None, None]
    first[6:, :, 0] = 0.2 + 0.6 * xx[6:]
    first[6:, :, 1] = 0.7 - 0.4 * yy[6:]
    first[6:, :, 2] = 0.1 + 0.7 * xx[6:] * yy[6:]
    first[:6, :, 3] = 0.0
    first[6:, :, 3] = 0.3 + 0.5 * xx[6:]
    second[6:] = first[6:]
    config = {
        "global_degree": 1,
        "fourier_freq": 1,
        "max_depth": 1,
        "split_threshold": 0.0,
        "local_freq": 1,
        "enable_varpro": False,
    }
    contexts = []
    for image in (first, second):
        context = PipelineContext(image=image, domain=domain, config=config)
        assert ModeBPipeline(config=config).execute(context) == StageStatus.COMPLETED
        contexts.append(context)
    np.testing.assert_allclose(contexts[0].global_model["coefficients"], contexts[1].global_model["coefficients"], atol=1e-12)
    for actual, expected in zip(contexts[0].local_decomposition["patches"], contexts[1].local_decomposition["patches"], strict=True):
        np.testing.assert_allclose(actual.coefficients, expected.coefficients, atol=1e-12)
    np.testing.assert_allclose(contexts[0].rendered_image, contexts[1].rendered_image, atol=1e-12)
    assert contexts[0].metrics["mse"] == pytest.approx(contexts[1].metrics["mse"], abs=1e-15)


def test_mode_b_joint_global_solve_matches_channel_reference() -> None:
    rng = np.random.default_rng(72)
    height, width = 10, 12
    domain = CoordinateDomain2D(width=width, height=height)
    image = rng.uniform(0.0, 1.0, size=(height, width, 3))
    context = PipelineContext(image=image, domain=domain)
    pipeline = ModeBPipeline(
        config={
            "global_degree": 2,
            "fourier_freq": 2,
            "max_depth": 0,
            "enable_varpro": False,
            "decouple_color": True,
        }
    )

    assert pipeline.execute(context) == StageStatus.COMPLETED
    model = context.global_model
    X, Y = domain.to_discrete_grid()
    phi = np.hstack([model["chebyshev_basis"].basis_matrix(X, Y), model["fourier_basis"].basis_matrix(X, Y)[:, 1:]])
    targets = image.reshape(-1, 3)
    expected = np.column_stack([np.linalg.lstsq(phi, targets[:, ch], rcond=1e-10)[0] for ch in range(3)])
    np.testing.assert_allclose(model["coefficients"], expected, atol=2e-11)


@pytest.mark.parametrize("decouple_color", [True, False])
def test_mode_b_streaming_blocks_match_single_block_reconstruction(decouple_color: bool) -> None:
    rng = np.random.default_rng(81)
    height, width = 18, 22
    domain = CoordinateDomain2D(width=width, height=height)
    image = rng.uniform(0.0, 1.0, size=(height, width, 3))
    contexts = []
    for block_size in (50, 1000):
        context = PipelineContext(image=image, domain=domain)
        pipeline = ModeBPipeline(
            config={
                "global_degree": 2,
                "fourier_freq": 2,
                "max_depth": 0,
                "enable_varpro": False,
                "decouple_color": decouple_color,
                "fit_block_size": block_size,
            }
        )
        assert pipeline.execute(context) == StageStatus.COMPLETED
        contexts.append(context)

    np.testing.assert_allclose(contexts[0].global_model["coefficients"], contexts[1].global_model["coefficients"], atol=2e-11)
    np.testing.assert_allclose(contexts[0].rendered_image, contexts[1].rendered_image, atol=2e-11)
    assert contexts[0].condition_report["effective_rank"] == contexts[1].condition_report["effective_rank"]


def test_mode_b_global_basis_is_materialized_in_bounded_blocks(monkeypatch: pytest.MonkeyPatch) -> None:
    height, width = 40, 50
    domain = CoordinateDomain2D(width=width, height=height)
    image = np.zeros((height, width, 3), dtype=np.float64)
    observed_rows: list[int] = []
    original_chebyshev = Chebyshev2D.basis_matrix
    original_fourier = Fourier2D.basis_matrix

    def tracked_chebyshev(self, X, Y):
        observed_rows.append(X.size)
        return original_chebyshev(self, X, Y)

    def tracked_fourier(self, X, Y):
        observed_rows.append(X.size)
        return original_fourier(self, X, Y)

    monkeypatch.setattr(Chebyshev2D, "basis_matrix", tracked_chebyshev)
    monkeypatch.setattr(Fourier2D, "basis_matrix", tracked_fourier)
    context = PipelineContext(image=image, domain=domain)
    pipeline = ModeBPipeline(
        config={
            "global_degree": 2,
            "fourier_freq": 2,
            "max_depth": 0,
            "enable_varpro": False,
            "fit_block_size": 128,
        }
    )

    assert pipeline.execute(context) == StageStatus.COMPLETED
    assert observed_rows
    assert max(observed_rows) <= 128


def test_controller_lifecycle() -> None:
    domain = CoordinateDomain2D(width=16, height=16)
    X, Y = domain.to_discrete_grid()
    img = np.zeros((16, 16))

    controller = ReconstructionController()
    stages_visited = []

    def on_stage(name: str, idx: int, total: int) -> None:
        stages_visited.append(name)

    ctx = controller.start(
        image=img,
        mode="discovery",
        domain=domain,
        config={"max_iter": 2, "lambda_reg": 0.1},
        on_stage_changed=on_stage,
    )

    assert len(stages_visited) >= 3
    assert ctx.metrics["mse"] <= 1e-4

    # Cancellation check
    controller.cancel()
    assert controller.context.is_cancelled() is True
