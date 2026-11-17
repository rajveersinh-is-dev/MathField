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

from mathfield.core.domain import CoordinateDomain2D
from mathfield.core.image_analysis import ImageAnalyzer
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
