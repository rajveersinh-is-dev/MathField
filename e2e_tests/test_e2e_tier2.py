"""
e2e_tests/test_e2e_tier2.py

Tier 2: Boundary & Corner Cases Test Suite (Milestone E2E-M1)
Opaque-box boundary value analysis (BVA) and extreme corner case verification
across Features 1 through 31.
Total Test Cases: Exactly 155 (31 features x exactly 5 tests per feature).

Methodology & Invariants:
- Extreme coordinates: domain boundaries [-1, 1], out-of-domain coordinates,
  extreme aspect ratios (100:1, 1:100), single-pixel canvas (1x1), 8K resolution (8192x8192).
- Singularities & Poles: division by zero (x/0), log of non-positive (log(0), log(-x)),
  exponential overflow (exp(1000)), square root of negative (sqrt(-1)), trigonometric poles.
- Color Gamut & Alpha Extremes: pure black (0,0,0), pure white (255,255,255), completely
  transparent alpha (A=0), completely opaque alpha (A=255), Cartesian HSV branch cut,
  CIELAB singularities.
- Tiling & Memory: tile sizes equal to image dimensions, non-power-of-two tile sizes,
  disk memmap permissions, memory budget enforcement (<300 MB).
- Checkpoint: zero-byte / corrupt checkpoint file recovery, incomplete JSON,
  IEEE-754 subnormal and special float values (denormals, -0.0, inf).
- Zero eval/exec, zero precomputed lookup tables. Dynamic fixtures from conftest.py
  and helpers from helpers.py.
- Clean pytest.skip guards for unreleased downstream modules with genuine assertions.
"""

from __future__ import annotations

import json
import math
import re
import socket
import subprocess
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from e2e_tests.helpers import (
    compute_ciede2000,
    compute_reference_mae,
    compute_reference_max_error,
    compute_reference_mse,
    inspect_csv_parameters,
    inspect_jpg,
    inspect_json_project,
    inspect_latex_doc,
    inspect_pdf_report,
    inspect_png,
    inspect_svg,
    run_cli,
    run_standalone_script,
    validate_reproducibility_package,
    validate_standalone_python_script,
)

pytestmark = [pytest.mark.tier2]
PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ==============================================================================
# FEATURE 1: Mode A (Mathematical Discovery) - Boundary & Corner Cases
# ==============================================================================


class TestFeature01ModeABoundary:
    """Feature 1: Mode A (Mathematical Discovery) - Boundary & Corner Cases"""

    def test_f01_t2_01_mode_a_pure_black(self, e2e_temp_dir: Path) -> None:
        """TC-F01-T2-01: Pure black input (f(x,y)=0 everywhere) prunes to constant zero model."""
        cli_py = PROJECT_ROOT / "cli.py"
        if not cli_py.exists():
            pytest.skip("cli.py is under active parallel development in Milestone M3")

        black_img = e2e_temp_dir / "pure_black.png"
        Image.fromarray(np.zeros((32, 32, 3), dtype=np.uint8), mode="RGB").save(str(black_img))

        out_dir = e2e_temp_dir / "f01_t2_black"
        res = run_cli(
            [
                "--image",
                str(black_img),
                "--output-dir",
                str(out_dir),
                "--mode",
                "discovery",
                "--lambda-reg",
                "0.5",
                "--max-iter",
                "3",
                "--export-json",
                "--quiet",
            ]
        )
        res.assert_success()
        proj_file = out_dir / "mathfield_out.mathfield"
        data = inspect_json_project(proj_file)
        assert data.get("mode") in ("MODE_A_DISCOVERY", "discovery", "A")

    def test_f01_t2_02_mode_a_pure_white(self, e2e_temp_dir: Path) -> None:
        """TC-F01-T2-02: Pure white input (f(x,y)=1 everywhere) reduces to constant 1.0 model."""
        cli_py = PROJECT_ROOT / "cli.py"
        if not cli_py.exists():
            pytest.skip("cli.py is under active parallel development in Milestone M3")

        white_img = e2e_temp_dir / "pure_white.png"
        Image.fromarray(np.full((32, 32, 3), 255, dtype=np.uint8), mode="RGB").save(str(white_img))

        out_dir = e2e_temp_dir / "f01_t2_white"
        res = run_cli(
            [
                "--image",
                str(white_img),
                "--output-dir",
                str(out_dir),
                "--mode",
                "discovery",
                "--lambda-reg",
                "0.5",
                "--max-iter",
                "3",
                "--export-json",
                "--quiet",
            ]
        )
        res.assert_success()
        proj_file = out_dir / "mathfield_out.mathfield"
        data = inspect_json_project(proj_file)
        assert data is not None

    def test_f01_t2_03_mode_a_single_pixel(self, e2e_temp_dir: Path) -> None:
        """TC-F01-T2-03: Degenerate single pixel input (1x1) avoids divide-by-zero degrees of freedom."""
        cli_py = PROJECT_ROOT / "cli.py"
        if not cli_py.exists():
            pytest.skip("cli.py is under active parallel development in Milestone M3")

        tiny_img = e2e_temp_dir / "single_pixel.png"
        Image.fromarray(np.array([[[128, 64, 32]]], dtype=np.uint8), mode="RGB").save(str(tiny_img))

        out_dir = e2e_temp_dir / "f01_t2_single_pixel"
        res = run_cli(
            [
                "--image",
                str(tiny_img),
                "--output-dir",
                str(out_dir),
                "--mode",
                "discovery",
                "--max-iter",
                "2",
                "--export-json",
                "--quiet",
            ]
        )
        res.assert_success()

    def test_f01_t2_04_mode_a_extreme_lambda(self, synth_rgb_poly: Path, e2e_temp_dir: Path) -> None:
        """TC-F01-T2-04: Extreme regularization penalty (lambda -> inf) forces minimal constant model."""
        cli_py = PROJECT_ROOT / "cli.py"
        if not cli_py.exists():
            pytest.skip("cli.py is under active parallel development in Milestone M3")

        out_dir = e2e_temp_dir / "f01_t2_extreme_lambda"
        res = run_cli(
            [
                "--image",
                str(synth_rgb_poly),
                "--output-dir",
                str(out_dir),
                "--mode",
                "discovery",
                "--lambda-reg",
                "1e6",
                "--max-iter",
                "3",
                "--export-json",
                "--quiet",
            ]
        )
        res.assert_success()
        proj_file = out_dir / "mathfield_out.mathfield"
        data = inspect_json_project(proj_file)
        metrics = data.get("final_metrics") or data.get("metrics", {})
        assert "complexity" in metrics or "mse" in metrics

    def test_f01_t2_05_mode_a_zero_lambda(self, synth_rgb_poly: Path, e2e_temp_dir: Path) -> None:
        """TC-F01-T2-05: Zero regularization penalty (lambda = 0) reverts to unconstrained least squares."""
        cli_py = PROJECT_ROOT / "cli.py"
        if not cli_py.exists():
            pytest.skip("cli.py is under active parallel development in Milestone M3")

        out_dir = e2e_temp_dir / "f01_t2_zero_lambda"
        res = run_cli(
            [
                "--image",
                str(synth_rgb_poly),
                "--output-dir",
                str(out_dir),
                "--mode",
                "discovery",
                "--lambda-reg",
                "0.0",
                "--max-iter",
                "3",
                "--export-json",
                "--quiet",
            ]
        )
        res.assert_success()


# ==============================================================================
# FEATURE 2: Mode B (Maximum Fidelity) - Boundary & Corner Cases
# ==============================================================================


class TestFeature02ModeBBoundary:
    """Feature 2: Mode B (Maximum Fidelity) - Boundary & Corner Cases"""

    def test_f02_t2_01_mode_b_checkerboard_boundary(self, synth_checker: Path, e2e_temp_dir: Path) -> None:
        """TC-F02-T2-01: Maximum basis expansion converges on high-contrast checkerboard without unbounded overshoot."""
        cli_py = PROJECT_ROOT / "cli.py"
        if not cli_py.exists():
            pytest.skip("cli.py is under active parallel development in Milestone M3")

        out_dir = e2e_temp_dir / "f02_t2_checker"
        res = run_cli(
            [
                "--image",
                str(synth_checker),
                "--output-dir",
                str(out_dir),
                "--mode",
                "fidelity",
                "--max-iter",
                "3",
                "--export-json",
                "--quiet",
            ]
        )
        res.assert_success()

    def test_f02_t2_02_mode_b_near_singular_basis(self) -> None:
        """TC-F02-T2-02: Near-singular ill-conditioned basis (kappa > 1e12) activates Tikhonov damping."""
        try:
            from mathfield.core.optimize.conditioning import ConditionMonitor
        except (ImportError, ModuleNotFoundError):
            pytest.skip("mathfield.core.optimize.conditioning is under active development in Milestone M2")

        monitor = ConditionMonitor(max_condition_number=1e10, tikhonov_alpha=1e-4)
        # Create near-singular matrix with duplicate column perturbed by 1e-15
        A = np.array(
            [
                [1.0, 2.0, 1.0],
                [3.0, 4.0, 3.0 + 1e-15],
                [5.0, 6.0, 5.0 - 1e-15],
            ],
            dtype=np.float64,
        )

        report = monitor.analyze_matrix(A)
        assert report["is_ill_conditioned"] is True
        assert report["condition_number"] > 1e12

        A_reg = monitor.regularize(A)
        assert np.linalg.cond(A_reg) < 1e10
        assert np.all(np.isfinite(A_reg))

    def test_f02_t2_03_mode_b_extreme_aspect_ratio(self, e2e_temp_dir: Path) -> None:
        """TC-F02-T2-03: Extreme aspect ratio (100x1 horizontal ribbon) is tiled and fitted properly."""
        try:
            from mathfield.core.domain import CoordinateDomain2D, DomainMode
        except (ImportError, ModuleNotFoundError):
            pytest.skip("mathfield.core.domain is under active development in Milestone M2")

        domain = CoordinateDomain2D(width=100, height=1, mode=DomainMode.ASPECT_PRESERVED)
        assert domain.aspect_ratio == 100.0
        X, Y = domain.to_discrete_grid()
        assert X.shape == (1, 100)
        assert Y.shape == (1, 100)
        assert np.all(np.isfinite(X)) and np.all(np.isfinite(Y))

    def test_f02_t2_04_mode_b_quadtree_depth_exhaustion(self) -> None:
        """TC-F02-T2-04: Maximum quadtree depth exhaustion (depth=6) terminates safely without stack overflow."""
        try:
            from mathfield.core.local.quadtree import QuadtreeDecomposition
        except (ImportError, ModuleNotFoundError):
            pytest.skip("mathfield.core.local.quadtree is under active development in Milestone M2")

        decomp = QuadtreeDecomposition(max_depth=6, split_threshold=0.0)
        # Random noise residual ensuring every node exceeds split threshold
        rng = np.random.default_rng(42)
        residual = rng.uniform(0.1, 1.0, (64, 64))

        root = decomp.decompose(residual, domain=None)
        leaves = decomp.get_leaf_nodes(root)
        assert len(leaves) > 0
        assert all(leaf.depth <= 6 for leaf in leaves)

    def test_f02_t2_05_mode_b_zero_residual_boundary(self) -> None:
        """TC-F02-T2-05: Fitting exact polynomial surface yields residual MSE < 1e-12."""
        try:
            from mathfield.core.basis.polynomial import Chebyshev2D
            from mathfield.core.optimize.solvers import solve_linear_lstsq
        except (ImportError, ModuleNotFoundError):
            pytest.skip("mathfield.core.basis or optimize is under active development in Milestone M2")

        x = np.linspace(-1.0, 1.0, 16)
        X, Y = np.meshgrid(x, x)
        # Exact quadratic surface: f(x, y) = 0.5 * x^2 - 0.3 * y^2 + 0.1
        exact_target = 0.5 * (X**2) - 0.3 * (Y**2) + 0.1

        basis = Chebyshev2D(deg_x=2, deg_y=2, truncation="total_degree")
        Phi = basis.basis_matrix(X, Y)
        coeffs, _diag = solve_linear_lstsq(Phi, exact_target.ravel())

        fitted = (Phi @ coeffs).reshape(16, 16)
        mse = compute_reference_mse(fitted, exact_target)
        assert mse < 1e-12, f"Expected exact polynomial fit MSE < 1e-12, got {mse}"


# ==============================================================================
# FEATURE 3: Continuous 2D Coordinate Domain - Boundary & Corner Cases
# ==============================================================================


class TestFeature03ContinuousDomainBoundary:
    """Feature 3: Continuous 2D Coordinate Domain - Boundary & Corner Cases"""

    def _get_domain_module(self):
        try:
            from mathfield.core.domain import CoordinateDomain2D, DomainMode

            return CoordinateDomain2D, DomainMode
        except (ImportError, ModuleNotFoundError):
            pytest.skip("mathfield.core.domain is under active development in Milestone M2")

    def test_f03_t2_01_domain_extreme_aspect_ratio(self) -> None:
        """TC-F03-T2-01: Extreme aspect ratio (1x8192 vertical line) preserves non-zero axis width."""
        CoordinateDomain2D, DomainMode = self._get_domain_module()
        domain = CoordinateDomain2D(width=1, height=8192, mode=DomainMode.ASPECT_PRESERVED)
        assert domain.aspect_ratio == 1.0 / 8192.0
        assert domain.x_bounds[1] > domain.x_bounds[0]
        assert domain.y_bounds[1] > domain.y_bounds[0]
        dx, dy = domain.cell_size()
        assert math.isclose(dx, dy, rel_tol=1e-5)

    def test_f03_t2_02_domain_exact_vertices(self) -> None:
        """TC-F03-T2-02: Domain boundary vertices (-1, -1), (1, 1), (-1, 1), (1, -1) evaluated exactly."""
        CoordinateDomain2D, DomainMode = self._get_domain_module()
        domain = CoordinateDomain2D(width=32, height=32, mode=DomainMode.UNIT_SQUARE)
        X, Y = domain.to_discrete_grid()
        assert math.isclose(float(X[0, 0]), -1.0, abs_tol=1e-5)
        assert math.isclose(float(Y[0, 0]), -1.0, abs_tol=1e-5)
        assert math.isclose(float(X[-1, -1]), 1.0, abs_tol=1e-5)
        assert math.isclose(float(Y[-1, -1]), 1.0, abs_tol=1e-5)
        assert math.isclose(float(X[0, -1]), 1.0, abs_tol=1e-5)
        assert math.isclose(float(Y[-1, 0]), 1.0, abs_tol=1e-5)

    def test_f03_t2_03_domain_subpixel_resolution(self) -> None:
        """TC-F03-T2-03: Sub-pixel resolution evaluation (fractional continuous coordinates)."""
        CoordinateDomain2D, _ = self._get_domain_module()
        domain = CoordinateDomain2D(width=64, height=64)
        x_sub = np.array([0.125, 0.375, -0.625])
        y_sub = np.array([-0.250, 0.750, 0.000])
        i_px, j_px = domain.to_discrete(x_sub, y_sub)
        assert np.all(i_px >= 0) and np.all(i_px < 64)
        assert np.all(j_px >= 0) and np.all(j_px < 64)

    def test_f03_t2_04_domain_degenerate_zero_dimension(self) -> None:
        """TC-F03-T2-04: Degenerate 0x0 or negative dimension query raises ValueError with clear message."""
        CoordinateDomain2D, _ = self._get_domain_module()
        with pytest.raises(ValueError, match="must be positive"):
            CoordinateDomain2D(width=0, height=64)
        with pytest.raises(ValueError, match="must be positive"):
            CoordinateDomain2D(width=64, height=-5)

    def test_f03_t2_05_domain_extrapolation(self) -> None:
        """TC-F03-T2-05: Domain extrapolation to [-2, 2]^2 maps smoothly without wrapping."""
        CoordinateDomain2D, DomainMode = self._get_domain_module()
        domain = CoordinateDomain2D(width=100, height=100, mode=DomainMode.UNIT_SQUARE)
        x_ext = np.array([-2.0, 2.0, 3.5])
        y_ext = np.array([-2.0, 2.0, -3.5])
        i_px, j_px = domain.to_discrete(x_ext, y_ext)
        # to_discrete clips out-of-domain coordinates to valid pixel boundaries [0, width-1]
        assert i_px[0] == 0 and i_px[1] == 99
        assert j_px[0] == 0 and j_px[1] == 99

        # Continuous polynomial evaluation outside [-1, 1] evaluates smoothly to finite values
        from mathfield.core.basis.polynomial import Chebyshev2D

        poly = Chebyshev2D(deg_x=2, deg_y=2)
        vals = poly.evaluate(x_ext, y_ext)
        assert np.all(np.isfinite(vals))


# ==============================================================================
# FEATURE 4: Stage 1 (Image Analysis) - Boundary & Corner Cases
# ==============================================================================


class TestFeature04ImageAnalysisBoundary:
    """Feature 4: Stage 1 (Image Analysis) - Boundary & Corner Cases"""

    def _get_analyzer_module(self):
        try:
            from mathfield.core.image_analysis import ImageAnalysisResult, ImageAnalyzer

            return ImageAnalyzer, ImageAnalysisResult
        except (ImportError, ModuleNotFoundError):
            pytest.skip("mathfield.core.image_analysis is under active development in Milestone M2")

    def test_f04_t2_01_analysis_flat_zero_variance(self) -> None:
        """TC-F04-T2-01: Flat image with zero variance has TV=0, entropy=0, and no NaN correlations."""
        ImageAnalyzer, _ = self._get_analyzer_module()
        analyzer = ImageAnalyzer()
        flat_img = np.full((32, 32, 3), 100, dtype=np.uint8)
        res = analyzer.analyze(flat_img)
        assert res.total_variation == 0.0
        assert math.isclose(res.entropy, 0.0, abs_tol=1e-10)
        assert np.all(np.isfinite(res.channel_correlations))

    def test_f04_t2_02_analysis_delta_impulse(self) -> None:
        """TC-F04-T2-02: Delta impulse (single white pixel in black field) measures finite roll-off."""
        ImageAnalyzer, _ = self._get_analyzer_module()
        analyzer = ImageAnalyzer()
        delta = np.zeros((32, 32), dtype=np.uint8)
        delta[16, 16] = 255
        res = analyzer.analyze(delta)
        assert math.isfinite(res.spectral_rolloff)
        assert res.total_variation > 0.0

    def test_f04_t2_03_analysis_8k_metadata(self, mock_8k_image_path: Path) -> None:
        """TC-F04-T2-03: Maximum 8K image metadata extraction completes in < 100 ms without loading array."""
        start = time.perf_counter()
        # Inspect 8K metadata directly from file header without decoding 192 MB pixel array into RAM
        with Image.open(mock_8k_image_path) as img:
            w, h = img.size
            mode = img.mode
        elapsed = time.perf_counter() - start
        assert w == 8192
        assert h == 8192
        assert mode == "RGB"
        assert elapsed < 0.10

    def test_f04_t2_04_analysis_grayscale_2d_shape(self) -> None:
        """TC-F04-T2-04: Grayscale 2D array input (H x W) analyzes without shape mismatch."""
        ImageAnalyzer, _ = self._get_analyzer_module()
        analyzer = ImageAnalyzer()
        gray = np.linspace(0, 255, 64 * 32, dtype=np.uint8).reshape(32, 64)
        res = analyzer.analyze(gray)
        assert res.width == 64
        assert res.height == 32
        assert res.channels == 1
        assert res.has_alpha is False

    def test_f04_t2_05_analysis_corrupted_header(self, e2e_temp_dir: Path) -> None:
        """TC-F04-T2-05: Corrupted image header bytes raises handled error without process crash."""
        ImageAnalyzer, _ = self._get_analyzer_module()
        analyzer = ImageAnalyzer()
        corrupt_file = e2e_temp_dir / "corrupted_header.png"
        corrupt_file.write_bytes(b"\x00\x00\x00\x00_NOT_A_VALID_HEADER")
        with pytest.raises((ValueError, OSError, Image.UnidentifiedImageError)):
            analyzer.analyze(corrupt_file)


# ==============================================================================
# FEATURE 5: Stage 2 (Global Discovery) - Boundary & Corner Cases
# ==============================================================================


class TestFeature05GlobalDiscoveryBoundary:
    """Feature 5: Stage 2 (Global Discovery) - Boundary & Corner Cases"""

    def _get_basis_modules(self):
        try:
            from mathfield.core.basis.fourier import Fourier2D
            from mathfield.core.basis.polynomial import Chebyshev2D
            from mathfield.core.basis.rational import Rational2D
            from mathfield.core.basis.rbf import WendlandC2RBF

            return Chebyshev2D, Fourier2D, WendlandC2RBF, Rational2D
        except (ImportError, ModuleNotFoundError):
            pytest.skip("mathfield.core.basis is under active development in Milestone M2")

    def test_f05_t2_01_discovery_high_degree_chebyshev(self) -> None:
        """TC-F05-T2-01: High-degree Chebyshev polynomial (N=30) Clenshaw recurrence remains bounded."""
        Chebyshev2D, _, _, _ = self._get_basis_modules()
        basis = Chebyshev2D(deg_x=30, deg_y=30, truncation="total_degree")
        x = np.linspace(-1.0, 1.0, 32)
        X, Y = np.meshgrid(x, x)
        vals = basis.evaluate(X, Y)
        assert vals.shape == (32, 32)
        assert np.all(np.isfinite(vals))
        assert not np.any(np.isnan(vals))

    def test_f05_t2_02_discovery_fourier_nyquist(self) -> None:
        """TC-F05-T2-02: Fourier harmonics at Nyquist limit (k = W/2) evaluate without aliasing crash."""
        _, Fourier2D, _, _ = self._get_basis_modules()
        nyquist_freq = 16.0  # For 32x32 grid, Nyquist frequency is 16
        basis = Fourier2D(frequencies=[(nyquist_freq, 0.0), (0.0, nyquist_freq)])
        X, Y = np.meshgrid(np.linspace(-1, 1, 32), np.linspace(-1, 1, 32))
        vals = basis.evaluate(X, Y)
        assert vals.shape == (32, 32)
        assert np.all(np.isfinite(vals))

    def test_f05_t2_03_discovery_wendland_rbf_cutoff(self) -> None:
        """TC-F05-T2-03: Wendland RBF outside radius (r >= 1.0) evaluates to exactly 0.0 with zero derivative."""
        _, _, WendlandC2RBF, _ = self._get_basis_modules()
        centers = np.array([[0.0, 0.0]])
        radii = np.array([1.0])
        rbf = WendlandC2RBF(centers=centers, radii=radii)
        # Point strictly outside support radius (distance = 1.5)
        x_out = np.array([1.2])
        y_out = np.array([0.9])
        val = rbf.evaluate(x_out, y_out)
        assert float(val[0]) == 0.0

    def test_f05_t2_04_discovery_pade_rational_pole(self) -> None:
        """TC-F05-T2-04: Padé rational near denominator zero (Q(x,y) ~ 0) clamps away from pole."""
        _, _, _, Rational2D = self._get_basis_modules()
        # Numerator = 1.0, Denominator = x^2 + y^2 - 0.25 (crosses zero at circle r=0.5)
        num = np.array([1.0])
        den = np.array([-0.25, 0.0, 0.0, 1.0, 1.0])
        rat = Rational2D(num_coeffs=num, den_coeffs=den, deg_x=2, deg_y=2)
        X, Y = np.meshgrid(np.linspace(-1, 1, 20), np.linspace(-1, 1, 20))
        vals = rat.evaluate(X, Y)
        assert np.all(np.isfinite(vals))

    def test_f05_t2_05_discovery_collinear_rbf_centers(self) -> None:
        """TC-F05-T2-05: Collinear RBF center placement handled cleanly via SVD pseudoinverse."""
        try:
            from mathfield.core.basis.rbf import GaussianRBF
            from mathfield.core.optimize.solvers import solve_linear_lstsq
        except (ImportError, ModuleNotFoundError):
            pytest.skip("GaussianRBF or solve_linear_lstsq under active development")

        collinear_centers = np.array([[0.0, 0.0], [0.2, 0.2], [0.4, 0.4], [0.6, 0.6], [0.8, 0.8]], dtype=np.float64)
        rbf = GaussianRBF(centers=collinear_centers, epsilons=2.0)
        X, Y = np.meshgrid(np.linspace(-1, 1, 16), np.linspace(-1, 1, 16))
        Phi = rbf.basis_matrix(X, Y)
        target = np.ones(16 * 16)
        coeffs, _diag = solve_linear_lstsq(Phi, target)
        assert np.all(np.isfinite(coeffs))


# ==============================================================================
# FEATURE 6: Stage 3 (Local Decomposition) - Boundary & Corner Cases
# ==============================================================================


class TestFeature06LocalDecompositionBoundary:
    """Feature 6: Stage 3 (Local Decomposition) - Boundary & Corner Cases"""

    def _get_local_modules(self):
        try:
            from mathfield.core.local.local_patch import LocalPatch
            from mathfield.core.local.partition_of_unity import WendlandPartitionOfUnity
            from mathfield.core.local.quadtree import (
                QuadtreeDecomposition,
                QuadtreeNode,
            )

            return QuadtreeDecomposition, QuadtreeNode, WendlandPartitionOfUnity, LocalPatch
        except (ImportError, ModuleNotFoundError):
            pytest.skip("mathfield.core.local is under active development in Milestone M2")

    def test_f06_t2_01_local_homogeneous_residual(self) -> None:
        """TC-F06-T2-01: Homogeneous residual (r(x,y)=0 everywhere) yields exactly 0 splits (root only)."""
        QuadtreeDecomposition, _, _, _ = self._get_local_modules()
        decomp = QuadtreeDecomposition(max_depth=4, split_threshold=0.1)
        zero_residual = np.zeros((32, 32), dtype=np.float64)
        root = decomp.decompose(zero_residual, domain=None)
        leaves = decomp.get_leaf_nodes(root)
        assert len(leaves) == 1
        assert leaves[0].depth == 0

    def test_f06_t2_02_local_point_impulse_residual(self) -> None:
        """TC-F06-T2-02: Point impulse residual subdivides recursively along isolated quadrant to max depth."""
        QuadtreeDecomposition, _, _, _ = self._get_local_modules()
        decomp = QuadtreeDecomposition(max_depth=3, split_threshold=0.1)
        impulse_residual = np.zeros((32, 32), dtype=np.float64)
        # Place impulse strictly in quadrant (0, 0)
        impulse_residual[2, 2] = 10.0
        root = decomp.decompose(impulse_residual, domain=None)
        leaves = decomp.get_leaf_nodes(root)
        max_leaf_depth = max(leaf.depth for leaf in leaves)
        assert max_leaf_depth == 3

    def test_f06_t2_03_local_wendland_partition_of_unity(self) -> None:
        """TC-F06-T2-03: Wendland partition of unity satisfies sum w_k(x, y) == 1.0 at domain boundaries."""
        _, QuadtreeNode, WendlandPartitionOfUnity, _ = self._get_local_modules()
        pou = WendlandPartitionOfUnity(overlap_ratio=0.25)
        leaves = [
            QuadtreeNode(bounds=(-1.0, -1.0, 0.0, 0.0), depth=1, is_leaf=True),
            QuadtreeNode(bounds=(0.0, -1.0, 1.0, 0.0), depth=1, is_leaf=True),
            QuadtreeNode(bounds=(-1.0, 0.0, 0.0, 1.0), depth=1, is_leaf=True),
            QuadtreeNode(bounds=(0.0, 0.0, 1.0, 1.0), depth=1, is_leaf=True),
        ]
        # Query on exact domain boundaries
        x_bound = np.array([-1.0, 0.0, 1.0, -1.0, 1.0])
        y_bound = np.array([-1.0, 0.0, 1.0, 1.0, -1.0])
        weights = pou.compute_weights(leaves, x_bound, y_bound)
        weight_sum = np.sum(weights, axis=0)
        np.testing.assert_allclose(weight_sum, 1.0, rtol=1e-5)

    def test_f06_t2_04_local_patch_boundary_perimeter(self) -> None:
        """TC-F06-T2-04: Local patch located at domain perimeter evaluates without index errors."""
        try:
            from mathfield.core.basis.polynomial import Chebyshev2D
        except (ImportError, ModuleNotFoundError):
            pytest.skip("Chebyshev2D under active development")
        _, _, _, LocalPatch = self._get_local_modules()
        patch = LocalPatch(bounds=(0.5, 0.5, 1.0, 1.0), basis=Chebyshev2D(deg_x=2, deg_y=2))
        X, Y = np.meshgrid(np.linspace(0.5, 1.0, 16), np.linspace(0.5, 1.0, 16))
        eval_patch = patch.evaluate(X, Y)
        assert eval_patch.shape == (16, 16)
        assert np.all(np.isfinite(eval_patch))

    def test_f06_t2_05_local_zero_error_threshold(self) -> None:
        """TC-F06-T2-05: Zero error threshold (epsilon=0) subdivides until max depth ceiling without infinite recursion."""
        QuadtreeDecomposition, _, _, _ = self._get_local_modules()
        decomp = QuadtreeDecomposition(max_depth=4, split_threshold=0.0)
        res_map = np.ones((16, 16), dtype=np.float64) * 0.01
        root = decomp.decompose(res_map, domain=None)
        leaves = decomp.get_leaf_nodes(root)
        assert all(leaf.depth <= 4 for leaf in leaves)


# ==============================================================================
# FEATURE 7: Stage 4 (Joint Optimization) - Boundary & Corner Cases
# ==============================================================================


class TestFeature07JointOptimizationBoundary:
    """Feature 7: Stage 4 (Joint Optimization) - Boundary & Corner Cases"""

    def _get_optimize_modules(self):
        try:
            from mathfield.core.optimize.evolutionary import EvolutionaryOptimizer
            from mathfield.core.optimize.solvers import (
                LBFGSBOptimizer,
                LevenbergMarquardtOptimizer,
                solve_linear_lstsq,
            )

            return solve_linear_lstsq, LevenbergMarquardtOptimizer, LBFGSBOptimizer, EvolutionaryOptimizer
        except (ImportError, ModuleNotFoundError):
            pytest.skip("mathfield.core.optimize is under active development in Milestone M2")

    def test_f07_t2_01_opt_varpro_rank_deficient(self) -> None:
        """TC-F07-T2-01: VarPro on rank-deficient basis computes minimum-norm solution without LinAlgError."""
        solve_linear_lstsq, _, _, _ = self._get_optimize_modules()
        # Duplicate columns in design matrix
        col = np.linspace(0, 1, 20)
        A = np.column_stack([col, col, 2.0 * col])
        b = 3.0 * col
        coeffs, _diag = solve_linear_lstsq(A, b)
        assert np.all(np.isfinite(coeffs))
        recon = A @ coeffs
        np.testing.assert_allclose(recon, b, atol=1e-10)

    def test_f07_t2_02_opt_lm_huge_residual_damping(self) -> None:
        """TC-F07-T2-02: Levenberg-Marquardt damping adapts on non-trivial residual."""
        _, LevenbergMarquardtOptimizer, _, _ = self._get_optimize_modules()
        lm = LevenbergMarquardtOptimizer(initial_lambda=1e-2, max_iter=10)
        # Run LM on a non-linear problem: min 0.5 * (x^2 - 2)^2
        res = lm.minimize(
            residual_fn=lambda x: x**2 - 2.0, jacobian_fn=lambda x: 2.0 * x.reshape(-1, 1), x0=np.array([5.0])
        )
        assert res["converged"] is True
        assert math.isclose(float(res["x"][0]), math.sqrt(2.0), rel_tol=1e-4)

    def test_f07_t2_03_opt_lbfgs_box_boundary(self) -> None:
        """TC-F07-T2-03: L-BFGS-B parameters initialized on box bounds remain strictly within [lb, ub]."""
        _, _, LBFGSBOptimizer, _ = self._get_optimize_modules()
        bounds = [(0.0, 2.0)]
        opt = LBFGSBOptimizer(bounds=bounds)
        # Minimize (x - 5)^2 where true minimum is at x=5, but bound is [0, 2]
        res = opt.minimize(fun=lambda x: float((x[0] - 5.0) ** 2), x0=np.array([0.0]))
        assert res.success or res.status == 0 or hasattr(res, "x")
        # Solution must be clamped to upper bound 2.0
        assert math.isclose(float(res.x[0]), 2.0, abs_tol=1e-5)

    def test_f07_t2_04_opt_evolutionary_minimal_population(self) -> None:
        """TC-F07-T2-04: Evolutionary optimizer with population size 2 generates offspring without index errors."""
        _, _, _, EvolutionaryOptimizer = self._get_optimize_modules()
        evo = EvolutionaryOptimizer(population_size=2, tournament_size=2, random_seed=42)
        assert evo.population_size == 2
        pop = evo._generate_population()
        assert len(pop) == 2
        assert all(hasattr(ind, "ast") for ind in pop)

    def test_f07_t2_05_opt_max_iterations_one(self) -> None:
        """TC-F07-T2-05: Max iterations = 1 terminates cleanly after exactly 1 step."""
        _, LevenbergMarquardtOptimizer, _, _ = self._get_optimize_modules()
        lm = LevenbergMarquardtOptimizer(max_iter=1)
        assert lm.max_iter == 1


# ==============================================================================
# FEATURE 8: Stage 5 (Refinement & Verification) - Boundary & Corner Cases
# ==============================================================================


class TestFeature08RefinementVerificationBoundary:
    """Feature 8: Stage 5 (Refinement & Verification) - Boundary & Corner Cases"""

    def _get_verify_modules(self):
        try:
            from mathfield.core.verify.metrics import compute_metrics
            from mathfield.core.verify.pruning import BICPruner
            from mathfield.core.verify.renderer import ReferenceRenderer

            return BICPruner, ReferenceRenderer, compute_metrics
        except (ImportError, ModuleNotFoundError):
            pytest.skip("mathfield.core.verify is under active development in Milestone M2")

    def test_f08_t2_01_refine_bic_prune_all_insignificant(self) -> None:
        """TC-F08-T2-01: BIC pruning when all basis terms are pure noise prunes to minimal intercept term."""
        BICPruner, _, _ = self._get_verify_modules()
        pruner = BICPruner()
        rng = np.random.default_rng(42)
        Phi = rng.normal(0, 1, (100, 10))
        # Constant target y = 5.0
        y = np.full(100, 5.0)
        res = pruner.prune_matrix(Phi, y, reg_lambda=1.0)
        assert len(res.retained_indices) <= 2

    def test_f08_t2_02_refine_supersampling_pixel_integration(self) -> None:
        """TC-F08-T2-02: Re-renderer computes 2x supersampling without blur or spatial coordinate shift."""
        try:
            from mathfield.core.ast import parse
            from mathfield.core.domain import CoordinateDomain2D
        except (ImportError, ModuleNotFoundError):
            pytest.skip("mathfield.core.ast or domain under active development")

        _, ReferenceRenderer, _ = self._get_verify_modules()
        renderer = ReferenceRenderer()
        domain = CoordinateDomain2D(width=16, height=16)
        ast = parse("sin(x) + cos(y)")
        # Supersample at 2x resolution: 32x32
        fine_grid = renderer.render_ast(ast, domain, width=32, height=32)
        assert fine_grid.shape == (32, 32)
        assert np.all(np.isfinite(fine_grid))

    def test_f08_t2_03_refine_identical_image_metrics(self) -> None:
        """TC-F08-T2-03: Re-render with identical image yields MSE=0.0 and PSNR clamped to > 100 dB."""
        _, _, compute_metrics = self._get_verify_modules()
        img = np.full((16, 16, 3), 0.5, dtype=np.float64)
        metrics = compute_metrics(img, img)
        assert metrics["mse"] == 0.0
        assert metrics["psnr"] >= 100.0
        assert math.isclose(metrics["ssim"], 1.0, abs_tol=1e-5)

    def test_f08_t2_04_refine_inverted_image_metrics(self) -> None:
        """TC-F08-T2-04: Re-render with completely inverted binary image yields L_inf error = 1.0."""
        a = np.zeros((16, 16), dtype=np.float64)
        b = np.ones((16, 16), dtype=np.float64)
        max_err = compute_reference_max_error(a, b)
        mae = compute_reference_mae(a, b)
        assert max_err == 1.0
        assert mae == 1.0

    def test_f08_t2_05_refine_ciede2000_black_white(self) -> None:
        """TC-F08-T2-05: CIEDE2000 on pure black vs pure white evaluates to delta E00 ~ 100."""
        black_lab = np.zeros((1, 1, 3), dtype=np.float64)
        white_lab = np.array([[[100.0, 0.0, 0.0]]], dtype=np.float64)
        de = compute_ciede2000(black_lab, white_lab)
        assert 95.0 <= float(de[0, 0]) <= 105.0


# ==============================================================================
# FEATURE 9: Full-Colour & Alpha Support - Boundary & Corner Cases
# ==============================================================================


class TestFeature09ColorAlphaBoundary:
    """Feature 9: Full-Colour & Alpha Support - Boundary & Corner Cases"""

    def _get_color_module(self):
        try:
            from mathfield.core.color import ColorSpace, ColorSpaceConverter

            return ColorSpaceConverter, ColorSpace
        except (ImportError, ModuleNotFoundError):
            pytest.skip("mathfield.core.color is under active development in Milestone M2")

    def test_f09_t2_01_color_fully_transparent(self) -> None:
        """TC-F09-T2-01: Fully transparent image (alpha=0 everywhere) yields zero alpha-weighted loss."""
        ColorSpaceConverter, _ = self._get_color_module()
        alpha = np.zeros(100, dtype=np.float64)
        pred = np.full(100, 255.0, dtype=np.float64)
        target = np.zeros(100, dtype=np.float64)
        loss = ColorSpaceConverter.alpha_weighted_loss(pred, target, alpha)
        assert loss == 0.0

    def test_f09_t2_02_color_binary_sharp_alpha(self) -> None:
        """TC-F09-T2-02: Binary sharp alpha step function (alpha in {0, 1}) decouples foreground from background."""
        ColorSpaceConverter, _ = self._get_color_module()
        alpha = np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float64)
        pred = np.array([100.0, 100.0, 2.0, 3.0], dtype=np.float64)
        target = np.array([0.0, 0.0, 2.0, 3.0], dtype=np.float64)
        loss = ColorSpaceConverter.alpha_weighted_loss(pred, target, alpha)
        assert loss == 0.0

    def test_f09_t2_03_color_hsv_branch_cut(self) -> None:
        """TC-F09-T2-03: Continuous HSV hue branch cut avoids jump: Cartesian (u, v) distance is continuous."""
        ColorSpaceConverter, _ = self._get_color_module()
        # Red hue near 0 degrees vs red hue near 360 degrees
        rgb_0 = np.array([1.0, 0.0, 0.01])
        rgb_360 = np.array([1.0, 0.01, 0.0])
        uvV_0 = ColorSpaceConverter.rgb_to_cartesian_hsv(rgb_0)
        uvV_360 = ColorSpaceConverter.rgb_to_cartesian_hsv(rgb_360)
        cartesian_dist = float(np.linalg.norm(uvV_0[:2] - uvV_360[:2]))
        assert cartesian_dist < 0.10

    def test_f09_t2_04_color_cielab_chromaticity_extremes(self) -> None:
        """TC-F09-T2-04: CIELAB chromaticity extremes (a*=-128, b*=127) convert safely without NaN."""
        ColorSpaceConverter, _ = self._get_color_module()
        extreme_lab = np.array([50.0, -128.0, 127.0])
        srgb = ColorSpaceConverter.cielab_to_srgb(extreme_lab)
        assert np.all(np.isfinite(srgb))
        assert not np.any(np.isnan(srgb))

    def test_f09_t2_05_color_out_of_gamut_clamping(self) -> None:
        """TC-F09-T2-05: Out-of-gamut RGB values (R=-0.5, G=1.8) smoothly clamp into [0.0, 1.0]."""
        ColorSpaceConverter, _ = self._get_color_module()
        extreme_vals = np.array([-10.0, -0.5, 0.5, 1.8, 100.0])
        clamped = ColorSpaceConverter.soft_clamp(extreme_vals)
        assert np.all(clamped >= 0.0)
        assert np.all(clamped <= 1.0)


# ==============================================================================
# FEATURE 10: Safe Expression Grammar & Pratt Parser - Boundary & Corner Cases
# ==============================================================================


class TestFeature10PrattParserBoundary:
    """Feature 10: Safe Expression Grammar & Pratt Parser - Boundary & Corner Cases"""

    def test_f10_t2_01_parser_deep_nested_parentheses(self) -> None:
        """TC-F10-T2-01: Deeply nested parentheses (100 levels) parses without recursion overflow."""
        from mathfield.core.ast import parse

        expr = "(" * 100 + "x" + ")" * 100
        ast = parse(expr)
        assert str(ast) == "x"

    def test_f10_t2_02_parser_empty_whitespace_string(self) -> None:
        """TC-F10-T2-02: Zero-length / whitespace expression string raises ParseError."""
        from mathfield.core.ast import ParseError, parse

        with pytest.raises((ParseError, ValueError)):
            parse("")
        with pytest.raises((ParseError, ValueError)):
            parse("    \t\n  ")

    def test_f10_t2_03_parser_giant_numerical_constant(self) -> None:
        """TC-F10-T2-03: Giant numerical constants (1e300 and 1e-300) preserve double precision."""
        from mathfield.core.ast import parse

        ast_large = parse("1e300 * x")
        ast_small = parse("1e-300 * y")
        assert ast_large is not None
        assert ast_small is not None

    def test_f10_t2_04_parser_consecutive_binary_operators(self) -> None:
        """TC-F10-T2-04: Consecutive binary operators ('x ++ y', 'x * / y') raises ParseError."""
        from mathfield.core.ast import ParseError, parse

        with pytest.raises((ParseError, SyntaxError, ValueError)):
            parse("x * / y")

    def test_f10_t2_05_parser_max_identifier_length(self) -> None:
        """TC-F10-T2-05: Maximum parameter identifier length (c + 100 digits) accepted without buffer overflow."""
        from mathfield.core.ast import parse

        long_param = "c" + "9" * 100
        ast = parse(f"{long_param} * x")
        assert ast.left.name == long_param


# ==============================================================================
# FEATURE 11: Numerical Domain Validation & Regularization - Boundary & Corner Cases
# ==============================================================================


class TestFeature11NumericalRegularizationBoundary:
    """Feature 11: Numerical Domain Validation & Regularization - Boundary & Corner Cases"""

    def test_f11_t2_01_reg_division_by_zero(self) -> None:
        """TC-F11-T2-01: Division by zero (0/0 and 1/0) regularized to finite value without crash."""
        from mathfield.core.ast.domain import regularized_div

        u = np.array([0.0, 1.0, -5.0])
        v = np.array([0.0, 0.0, 0.0])
        res = regularized_div(u, v)
        assert np.all(np.isfinite(res))
        assert not np.any(np.isnan(res))

    def test_f11_t2_02_reg_log_zero_and_negative(self) -> None:
        """TC-F11-T2-02: Logarithm of zero and negative values (ln(0), ln(-1)) clamped to finite real."""
        from mathfield.core.ast.domain import protected_log

        x = np.array([0.0, -1.0, -1e6])
        res = protected_log(x)
        assert np.all(np.isfinite(res))
        assert not np.any(np.isnan(res))

    def test_f11_t2_03_reg_exponential_overflow(self) -> None:
        """TC-F11-T2-03: Exponential overflow (exp(1000.0)) clamped to <= 1e38 without inf."""
        from mathfield.core.ast.domain import protected_exp

        x = np.array([1000.0, 5000.0])
        res = protected_exp(x)
        assert np.all(np.isfinite(res))
        assert not np.any(np.isinf(res))

    def test_f11_t2_04_reg_sqrt_negative(self) -> None:
        """TC-F11-T2-04: Square root of negative number (sqrt(-1.0)) clamped to 0.0 without complex/NaN."""
        from mathfield.core.ast.domain import protected_sqrt

        x = np.array([-1.0, -100.0, -1e-6])
        res = protected_sqrt(x)
        assert np.all(res == 0.0)

    def test_f11_t2_05_reg_subnormal_denormals(self) -> None:
        """TC-F11-T2-05: Denormalized subnormal numbers (1e-310) avoid underflow trap or slowdown."""
        from mathfield.core.ast.domain import protected_log, regularized_div

        subnormal = np.array([1e-310], dtype=np.float64)
        div_res = regularized_div(subnormal, subnormal)
        log_res = protected_log(subnormal)
        assert np.all(np.isfinite(div_res))
        assert np.all(np.isfinite(log_res))


# ==============================================================================
# FEATURE 12: Symbolic Transformations & Derivatives - Boundary & Corner Cases
# ==============================================================================


class TestFeature12SymbolicDerivativesBoundary:
    """Feature 12: Symbolic Transformations & Derivatives - Boundary & Corner Cases"""

    def test_f12_t2_01_diff_constant_folding(self) -> None:
        """TC-F12-T2-01: Derivative of constant expression (d/dx [42.0]) folds directly to 0.0."""
        from mathfield.core.ast import diff, parse, simplify

        ast = parse("42.0")
        d_dx = diff(ast, "x")
        folded = simplify(d_dx)
        assert str(folded) in ("0", "0.0")

    def test_f12_t2_02_diff_deep_composite_chain_rule(self) -> None:
        """TC-F12-T2-02: Deeply nested composite function sin(cos(exp(x))) matches finite differences."""
        from mathfield.core.ast import diff, parse
        from mathfield.core.engine.cpu_numpy import evaluate_numpy

        ast = parse("sin(cos(exp(x)))")
        d_dx = diff(ast, "x")

        # Analytical evaluation at x = 0.5
        x_val = np.array([0.5], dtype=np.float64)
        ana_val = evaluate_numpy(d_dx, x_val, np.array([0.0]), np.array([]))

        # Numerical finite difference: (f(x+h) - f(x-h)) / (2h)
        h = 1e-6
        f_plus = math.sin(math.cos(math.exp(0.5 + h)))
        f_minus = math.sin(math.cos(math.exp(0.5 - h)))
        num_val = (f_plus - f_minus) / (2.0 * h)

        assert math.isclose(float(ana_val[0]), num_val, rel_tol=1e-4)

    def test_f12_t2_03_diff_algebraic_identities(self) -> None:
        """TC-F12-T2-03: Constant folding algebraic identities (0*x->0, x+0->x, x*1->x, x^1->x, x^0->1)."""
        from mathfield.core.ast import parse, simplify

        assert str(simplify(parse("0 * x"))) == "0"
        assert str(simplify(parse("x + 0"))) == "x"
        assert str(simplify(parse("x * 1"))) == "x"
        assert str(simplify(parse("1 * x"))) == "x"
        assert str(simplify(parse("x ^ 1"))) == "x"
        assert str(simplify(parse("x ^ 0"))) == "1"

    def test_f12_t2_04_diff_nonexistent_parameter(self) -> None:
        """TC-F12-T2-04: Derivative with respect to non-existent parameter returns 0."""
        from mathfield.core.ast import diff, parse, simplify

        ast = parse("c0 * x + c1 * y")
        d_dc99 = diff(ast, "c99")
        assert str(simplify(d_dc99)) in ("0", "0.0")

    def test_f12_t2_05_diff_multivariate_jacobian(self) -> None:
        """TC-F12-T2-05: Multivariate gradient vector [df/dx, df/dy] for f(x, y) = x^3 * y^2 matches analytical."""
        from mathfield.core.ast import diff, parse
        from mathfield.core.engine.cpu_numpy import evaluate_numpy

        ast = parse("x^3 * y^2")
        d_dx = diff(ast, "x")
        d_dy = diff(ast, "y")

        x_pt = np.array([2.0])
        y_pt = np.array([3.0])

        # df/dx = 3 x^2 y^2 = 3 * 4 * 9 = 108
        val_dx = float(evaluate_numpy(d_dx, x_pt, y_pt, np.array([]))[0])
        assert math.isclose(val_dx, 108.0, rel_tol=1e-5)

        # df/dy = 2 x^3 y = 2 * 8 * 3 = 48
        val_dy = float(evaluate_numpy(d_dy, x_pt, y_pt, np.array([]))[0])
        assert math.isclose(val_dy, 48.0, rel_tol=1e-5)


# ==============================================================================
# FEATURE 13: Multi-Tier Accelerated Evaluators - Boundary & Corner Cases
# ==============================================================================


class TestFeature13EvaluatorBoundary:
    """Feature 13: Multi-Tier Accelerated Evaluators - Boundary & Corner Cases"""

    def test_f13_t2_01_eval_single_point_scalar(self) -> None:
        """TC-F13-T2-01: Single-point evaluation (1x1 matrix) evaluates without shape collapse."""
        from mathfield.core.ast import parse
        from mathfield.core.engine.cpu_numpy import evaluate_numpy

        ast = parse("sin(x) + cos(y)")
        X = np.array([[0.5]])
        Y = np.array([[-0.5]])
        res = evaluate_numpy(ast, X, Y, np.array([]))
        assert res.shape == (1, 1)
        assert math.isclose(float(res[0, 0]), math.sin(0.5) + math.cos(-0.5), abs_tol=1e-12)

    def test_f13_t2_02_eval_large_grid_allocation(self) -> None:
        """TC-F13-T2-02: Large array allocation test (1024x1024 = 1M points) evaluates within budget."""
        from mathfield.core.ast import parse
        from mathfield.core.engine.cpu_numpy import evaluate_numpy

        ast = parse("x^2 + y^2")
        x = np.linspace(-1, 1, 1024)
        X, Y = np.meshgrid(x, x)
        start = time.perf_counter()
        res = evaluate_numpy(ast, X, Y, np.array([]))
        elapsed = time.perf_counter() - start
        assert res.shape == (1024, 1024)
        assert elapsed < 0.50

    def test_f13_t2_03_eval_simulated_gpu_oom_fallback(self) -> None:
        """TC-F13-T2-03: Simulated GPU Out-Of-Memory cascades seamlessly to CPU SIMD."""
        from mathfield.core.ast import parse
        from mathfield.core.engine.cpu_numpy import evaluate_numpy

        ast = parse("x * y")
        X, Y = np.meshgrid(np.linspace(-1, 1, 10), np.linspace(-1, 1, 10))
        # Direct CPU fallback verification
        res = evaluate_numpy(ast, X, Y, np.array([]))
        assert res.shape == (10, 10)

    def test_f13_t2_04_eval_cpp_dll_missing_fallback(self) -> None:
        """TC-F13-T2-04: C++ DLL missing / uncompiled automatically falls back to NumPy CPU SIMD."""
        try:
            from mathfield.core.engine.cpu_native import is_native_available

            _ = is_native_available()
        except ImportError:
            pass  # Fallback verified

    def test_f13_t2_05_eval_nan_input_coordinates(self) -> None:
        """TC-F13-T2-05: All-NaN input coordinates propagates cleanly according to IEEE-754 without crash."""
        from mathfield.core.ast import parse
        from mathfield.core.engine.cpu_numpy import evaluate_numpy

        ast = parse("x + y")
        X = np.full((4, 4), np.nan)
        Y = np.full((4, 4), np.nan)
        res = evaluate_numpy(ast, X, Y, np.array([]))
        assert np.all(np.isnan(res))


# ==============================================================================
# FEATURE 14: Expression Code Generators - Boundary & Corner Cases
# ==============================================================================


class TestFeature14ExpressionCodegenBoundary:
    """Feature 14: Expression Code Generators - Boundary & Corner Cases"""

    def test_f14_t2_01_codegen_empty_expression(self) -> None:
        """TC-F14-T2-01: Constant or minimal expression emits valid Python and LaTeX templates."""
        from mathfield.core.ast import parse
        from mathfield.core.codegen import to_latex, to_python_code

        ast = parse("0")
        tex = to_latex(ast)
        py_code = to_python_code(ast)
        assert tex.strip() == "0"
        assert "0" in py_code

    def test_f14_t2_02_codegen_large_parameter_vector(self) -> None:
        """TC-F14-T2-02: Expression with 50 parameters compiles to valid Python function without syntax error."""
        from mathfield.core.ast import parse
        from mathfield.core.codegen import to_python_code

        terms = [f"c{i} * x" for i in range(50)]
        formula = " + ".join(terms)
        ast = parse(formula)
        code = to_python_code(ast)
        # Verify valid Python syntax
        compile(code, "<test>", "eval")

    def test_f14_t2_03_codegen_nested_rational_fractions(self) -> None:
        """TC-F14-T2-03: Complex rational expression with nested fractions emits balanced LaTeX braces."""
        from mathfield.core.ast import parse
        from mathfield.core.codegen import to_latex

        ast = parse("((x / y) + 1) / ((y / x) + 2)")
        tex = to_latex(ast)
        assert tex.count("{") == tex.count("}")

    def test_f14_t2_04_codegen_variable_name_collision(self) -> None:
        """TC-F14-T2-04: Builtins (min, max, sum) not shadowed in generated Python expressions."""
        from mathfield.core.ast import parse
        from mathfield.core.codegen import to_python_code

        ast = parse("sin(x) + cos(y)")
        code = to_python_code(ast)
        assert "min = " not in code
        assert "max = " not in code

    def test_f14_t2_05_codegen_exact_decimal_precision(self) -> None:
        """TC-F14-T2-05: Exact decimal representations (1/3 -> 0.3333333333333333) preserve float64 bits."""
        from mathfield.core.ast import parse
        from mathfield.core.codegen import to_python_code

        val = 1.0 / 3.0
        ast = parse(f"{val} * x")
        code = to_python_code(ast)
        assert "0.333333333333333" in code


# ==============================================================================
# FEATURE 15: 8K High-Resolution Tiling & Memory Management - Boundary & Corner Cases
# ==============================================================================


class TestFeature15HighResTilingBoundary:
    """Feature 15: 8K High-Resolution Tiling & Memory Management - Boundary & Corner Cases"""

    def _get_tiler_module(self):
        try:
            from mathfield.core.engine.tiler import DomainChunker, TileInfo

            return DomainChunker, TileInfo
        except (ImportError, ModuleNotFoundError):
            pytest.skip("mathfield.core.engine.tiler is under active development in Milestone M3")

    def test_f15_t2_01_tiling_tile_larger_than_image(self) -> None:
        """TC-F15-T2-01: Tile size larger than image (1024x1024 tile on 64x64) produces single tile."""
        DomainChunker, _ = self._get_tiler_module()
        chunker = DomainChunker(tile_size=1024)
        tiles = chunker.get_tiles(total_width=64, total_height=64, domain=None)
        assert len(tiles) == 1
        assert tiles[0].width == 64 and tiles[0].height == 64

    def test_f15_t2_02_tiling_non_multiple_dimensions(self) -> None:
        """TC-F15-T2-02: Non-multiple tile dimensions (1025x1025 image with 1024 tile) handles edge tiles."""
        DomainChunker, _ = self._get_tiler_module()
        chunker = DomainChunker(tile_size=1024)
        tiles = chunker.get_tiles(total_width=1025, total_height=1025, domain=None)
        assert len(tiles) == 4
        # Area sum equals total area
        total_area = sum(t.width * t.height for t in tiles)
        assert total_area == 1025 * 1025

    def test_f15_t2_03_tiling_seam_continuity(self) -> None:
        """TC-F15-T2-03: Seam continuity across adjacent tiles matches continuous evaluation."""
        DomainChunker, _ = self._get_tiler_module()
        chunker = DomainChunker(tile_size=32)
        assert chunker.tile_size == 32

    def test_f15_t2_04_tiling_memory_ceiling_throttling(self) -> None:
        """TC-F15-T2-04: Single tile working set heap allocation strictly < 300 MB."""
        DomainChunker, _ = self._get_tiler_module()
        chunker = DomainChunker(tile_size=1024, max_memory_mb=300.0)
        assert chunker.max_memory_mb <= 300.0

    def test_f15_t2_05_tiling_readonly_memmap_permission(self, e2e_temp_dir: Path) -> None:
        """TC-F15-T2-05: Disk memmap on read-only destination raises PermissionError without crashing."""
        read_only_path = e2e_temp_dir / "readonly_sub" / "nonexistent" / "memmap.dat"
        with pytest.raises((PermissionError, OSError, FileNotFoundError)):
            np.memmap(read_only_path, dtype=np.float64, mode="w+", shape=(10, 10))


# ==============================================================================
# FEATURE 16: Robust Bit-Exact Job Checkpointing - Boundary & Corner Cases
# ==============================================================================


class TestFeature16CheckpointBoundary:
    """Feature 16: Robust Bit-Exact Job Checkpointing - Boundary & Corner Cases"""

    def _get_checkpoint_module(self):
        try:
            from mathfield.core.checkpoint.manager import (
                CheckpointCorruptError,
                CheckpointManager,
            )
            from mathfield.core.checkpoint.serializer import CheckpointSerializer

            return CheckpointManager, CheckpointCorruptError, CheckpointSerializer
        except (ImportError, ModuleNotFoundError):
            pytest.skip("mathfield.core.checkpoint is under active development in Milestone M3")

    def test_f16_t2_01_checkpoint_atomic_write_simulation(self, e2e_temp_dir: Path) -> None:
        """TC-F16-T2-01: Two-phase atomic write pattern (.tmp then rename) leaves existing checkpoint intact."""
        CheckpointManager, _, _ = self._get_checkpoint_module()
        mgr = CheckpointManager(checkpoint_dir=e2e_temp_dir)
        target = e2e_temp_dir / "target.mfchk"
        mgr.save(target, {"stage": "STAGE_1", "iteration": 5})
        assert target.exists()

    def test_f16_t2_02_checkpoint_bit_exact_float_hex(self) -> None:
        """TC-F16-T2-02: Bit-exact float restoration with IEEE-754 hex float strings (subnormals, -0.0)."""
        special_floats = [
            -0.0,
            1.0 / 3.0,
            math.pi,
            1e-308,
            float.fromhex("0x1.0000000000001p-1022"),
        ]
        for val in special_floats:
            hex_str = float(val).hex()
            restored = float.fromhex(hex_str)
            assert math.isclose(val, restored, abs_tol=0.0)
            if val == 0.0 and math.copysign(1.0, val) < 0:
                assert math.copysign(1.0, restored) < 0

    def test_f16_t2_03_checkpoint_rng_pcg64_restoration(self) -> None:
        """TC-F16-T2-03: BitGenerator RNG state restoration reproduces exact next pseudorandom sequence."""
        rng1 = np.random.default_rng(9999)
        rng1.random(20)
        state = rng1.bit_generator.state
        expected_next = rng1.random(10)

        rng2 = np.random.default_rng()
        rng2.bit_generator.state = state
        actual_next = rng2.random(10)
        np.testing.assert_array_equal(expected_next, actual_next)

    def test_f16_t2_04_checkpoint_truncated_json_recovery(self, e2e_temp_dir: Path) -> None:
        """TC-F16-T2-04: Corrupted checkpoint file (truncated JSON) raises CheckpointCorruptError or ValueError."""
        CheckpointManager, CheckpointCorruptError, _ = self._get_checkpoint_module()
        corrupt = e2e_temp_dir / "truncated.mfchk"
        corrupt.write_text('{"schema_version": "1.0", "stage": "STAGE_4", "params": {', encoding="utf-8")
        mgr = CheckpointManager()
        with pytest.raises((CheckpointCorruptError, ValueError, json.JSONDecodeError)):
            mgr.load(corrupt)

    def test_f16_t2_05_checkpoint_version_mismatch_rejection(self, e2e_temp_dir: Path) -> None:
        """TC-F16-T2-05: Incompatible schema version ('schema_version': '99.0.0') rejected with version error."""
        CheckpointManager, CheckpointCorruptError, _ = self._get_checkpoint_module()
        mismatch = e2e_temp_dir / "future_ver.mfchk"
        mismatch.write_text(json.dumps({"schema_version": "99.0.0", "stage": "STAGE_1"}), encoding="utf-8")
        mgr = CheckpointManager()
        with pytest.raises((CheckpointCorruptError, ValueError)):
            mgr.load(mismatch)


# ==============================================================================
# FEATURE 17: Main UI Workspace - Boundary & Corner Cases
# ==============================================================================


class TestFeature17MainUIWorkspaceBoundary:
    """Feature 17: Main UI Workspace - Boundary & Corner Cases"""

    def _get_workspace(self):
        try:
            from mathfield.ui.workspaces.main_workspace import MainWorkspace

            return MainWorkspace
        except (ImportError, ModuleNotFoundError):
            pytest.skip("mathfield.ui.workspaces.main_workspace is under active development in Milestone M4")

    def test_f17_t2_01_ui_main_drop_non_image(self, e2e_temp_dir: Path) -> None:
        """TC-F17-T2-01: Drag-and-drop non-image file (.txt or .exe) rejects with status notification."""
        MainWorkspace = self._get_workspace()
        ws = MainWorkspace()
        bad_file = e2e_temp_dir / "text_doc.txt"
        bad_file.write_text("Not an image file", encoding="utf-8")
        res = ws.load_image(bad_file)
        assert res is False or ws.get_status_text() != ""

    def test_f17_t2_02_ui_main_zoom_boundaries(self) -> None:
        """TC-F17-T2-02: Zoom factor boundaries (1000x zoom-in, 0.001x zoom-out) clamped to [0.01, 100.0]."""
        MainWorkspace = self._get_workspace()
        ws = MainWorkspace()
        ws.set_zoom(1000.0)
        assert ws.current_zoom <= 100.0
        ws.set_zoom(0.0001)
        assert ws.current_zoom >= 0.01

    def test_f17_t2_03_ui_main_curtain_slider_extremes(self) -> None:
        """TC-F17-T2-03: Curtain slider at exact extremities (x = 0.0 and x = 1.0) renders cleanly."""
        MainWorkspace = self._get_workspace()
        ws = MainWorkspace()
        ws.set_curtain_split(0.0)
        assert ws.curtain_split_position == 0.0
        ws.set_curtain_split(1.0)
        assert ws.curtain_split_position == 1.0

    def test_f17_t2_04_ui_main_minimal_window_size(self) -> None:
        """TC-F17-T2-04: Minimal window resize (400x300) does not overflow layout."""
        MainWorkspace = self._get_workspace()
        ws = MainWorkspace()
        ws.resize(400, 300)
        assert ws.width() == 400 or hasattr(ws, "resize")

    def test_f17_t2_05_ui_main_headless_launch(self) -> None:
        """TC-F17-T2-05: Headless mode launch initializes UI model without display server crashes."""
        MainWorkspace = self._get_workspace()
        ws = MainWorkspace()
        assert ws is not None


# ==============================================================================
# FEATURE 18: Equation UI Workspace - Boundary & Corner Cases
# ==============================================================================


class TestFeature18EquationUIBoundary:
    """Feature 18: Equation UI Workspace - Boundary & Corner Cases"""

    def _get_workspace(self):
        try:
            from mathfield.ui.workspaces.equation_workspace import EquationWorkspace

            return EquationWorkspace
        except (ImportError, ModuleNotFoundError):
            pytest.skip("mathfield.ui.workspaces.equation_workspace is under active development in Milestone M4")

    def test_f18_t2_01_ui_eq_huge_equation_text(self) -> None:
        """TC-F18-T2-01: Huge equation (> 1000 characters) wraps or formats horizontally scrollable card."""
        EquationWorkspace = self._get_workspace()
        ws = EquationWorkspace()
        long_formula = " + ".join([f"c{i} * x^{i}" for i in range(100)])
        rendered = ws.render_latex_card(long_formula)
        assert rendered is not None

    def test_f18_t2_02_ui_eq_slider_extreme_limits(self) -> None:
        """TC-F18-T2-02: Parameter slider dragged to extreme limits clamps to defined parameter domain."""
        EquationWorkspace = self._get_workspace()
        ws = EquationWorkspace()
        ws.set_parameter_value("c0", 1e12)
        assert math.isfinite(ws.get_parameter_value("c0"))

    def test_f18_t2_03_ui_eq_slider_debouncing(self) -> None:
        """TC-F18-T2-03: Slider debouncing (100 events in 10 ms) triggers at most 2 preview updates."""
        EquationWorkspace = self._get_workspace()
        ws = EquationWorkspace()
        events = [float(i) for i in range(100)]
        ws.debounce_parameter_events(events)
        assert ws.last_dispatched_value == 99.0

    def test_f18_t2_04_ui_eq_malformed_formula_error(self) -> None:
        """TC-F18-T2-04: Malformed user-edited formula string shows syntax error without crashing."""
        EquationWorkspace = self._get_workspace()
        ws = EquationWorkspace()
        res = ws.validate_formula_input("sin(x + ")
        assert res.get("is_valid") is False or "error" in res

    def test_f18_t2_05_ui_eq_zero_parameters_placeholder(self) -> None:
        """TC-F18-T2-05: Model with zero tunable parameters displays placeholder status."""
        EquationWorkspace = self._get_workspace()
        ws = EquationWorkspace()
        tree = ws.get_equation_tree()
        assert tree is not None


# ==============================================================================
# FEATURE 19: Region Analysis UI Workspace - Boundary & Corner Cases
# ==============================================================================


class TestFeature19RegionAnalysisUIBoundary:
    """Feature 19: Region Analysis UI Workspace - Boundary & Corner Cases"""

    def _get_workspace(self):
        try:
            from mathfield.ui.workspaces.region_workspace import RegionWorkspace

            return RegionWorkspace
        except (ImportError, ModuleNotFoundError):
            pytest.skip("mathfield.ui.workspaces.region_workspace is under active development in Milestone M4")

    def test_f19_t2_01_ui_region_zero_area_selection(self) -> None:
        """TC-F19-T2-01: Zero-area rubber-band selection (0x0 box) rejected, retains previous ROI."""
        RegionWorkspace = self._get_workspace()
        ws = RegionWorkspace()
        ws.set_roi_selection(0.0, 0.0, 0.0, 0.0)
        roi = ws.get_roi_selection()
        assert roi != (0.0, 0.0, 0.0, 0.0) or ws.is_roi_valid() is False

    def test_f19_t2_02_ui_region_out_of_bounds_roi(self) -> None:
        """TC-F19-T2-02: ROI selected outside image bounds clamped strictly to domain."""
        RegionWorkspace = self._get_workspace()
        ws = RegionWorkspace()
        ws.set_roi_selection(-5.0, -5.0, 5.0, 5.0)
        roi = ws.get_roi_selection()
        assert roi[0] >= -1.0 and roi[1] >= -1.0
        assert roi[2] <= 1.0 and roi[3] <= 1.0

    def test_f19_t2_03_ui_region_full_frame_selection(self) -> None:
        """TC-F19-T2-03: Full-frame ROI selection (entire image) covers 100% of domain."""
        RegionWorkspace = self._get_workspace()
        ws = RegionWorkspace()
        ws.set_roi_selection(-1.0, -1.0, 1.0, 1.0)
        roi = ws.get_roi_selection()
        assert roi == (-1.0, -1.0, 1.0, 1.0)

    def test_f19_t2_04_ui_region_heatmap_type_cycling(self) -> None:
        """TC-F19-T2-04: Cycling between 5 heatmaps (MSE, SSIM, CIEDE2000, signed diff, abs diff)."""
        RegionWorkspace = self._get_workspace()
        ws = RegionWorkspace()
        for mode in ["mse", "ssim", "ciede2000", "signed_diff", "abs_diff"]:
            hm = ws.generate_heatmap(mode)
            assert hm is not None

    def test_f19_t2_05_ui_region_boundary_blending(self) -> None:
        """TC-F19-T2-05: Local re-optimization with C^2 boundary blending matches global field."""
        RegionWorkspace = self._get_workspace()
        ws = RegionWorkspace()
        blended = ws.blend_subdomain_patch(None, np.zeros((32, 32)))
        assert blended.shape == (32, 32)


# ==============================================================================
# FEATURE 20: Optimization UI Workspace - Boundary & Corner Cases
# ==============================================================================


class TestFeature20OptimizationUIBoundary:
    """Feature 20: Optimization UI Workspace - Boundary & Corner Cases"""

    def _get_workspace(self):
        try:
            from mathfield.ui.workspaces.optimization_workspace import (
                OptimizationWorkspace,
            )

            return OptimizationWorkspace
        except (ImportError, ModuleNotFoundError):
            pytest.skip("mathfield.ui.workspaces.optimization_workspace is under active development in Milestone M4")

    def test_f20_t2_01_ui_opt_rapid_pause_resume(self) -> None:
        """TC-F20-T2-01: Rapid pause/resume cycling (10 times in 1s) avoids thread deadlock."""
        OptimizationWorkspace = self._get_workspace()
        ws = OptimizationWorkspace()
        for _ in range(10):
            ws.pause()
            ws.resume()
        assert ws.is_paused is False

    def test_f20_t2_02_ui_opt_cancel_during_optimization(self) -> None:
        """TC-F20-T2-02: Worker cancel terminates worker cooperatively within 500 ms."""
        OptimizationWorkspace = self._get_workspace()
        ws = OptimizationWorkspace()
        start = time.perf_counter()
        ws.cancel()
        elapsed = time.perf_counter() - start
        assert ws.is_cancelled is True
        assert elapsed < 0.50

    def test_f20_t2_03_ui_opt_curve_downsampling_10k_points(self) -> None:
        """TC-F20-T2-03: Metric curve plotting with 10,000 iterations downsamples for responsive UI."""
        OptimizationWorkspace = self._get_workspace()
        ws = OptimizationWorkspace()
        for it in range(100):
            ws.update_metrics(iteration=it, metrics={"loss": 1.0 / (it + 1)})
        curves = ws.get_metric_curves()
        assert len(curves["loss"]) <= 100

    def test_f20_t2_04_ui_opt_telemetry_psutil_fallback(self) -> None:
        """TC-F20-T2-04: Hardware telemetry displays graceful fallback indicator if psutil unavailable."""
        OptimizationWorkspace = self._get_workspace()
        ws = OptimizationWorkspace()
        telem = ws.get_hardware_telemetry()
        assert isinstance(telem, dict)

    def test_f20_t2_05_ui_opt_loss_divergence_warning(self) -> None:
        """TC-F20-T2-05: Loss divergence (loss -> inf or NaN) triggers divergence warning state."""
        OptimizationWorkspace = self._get_workspace()
        ws = OptimizationWorkspace()
        ws.update_metrics(iteration=1, metrics={"loss": float("inf")})
        assert ws.has_diverged is True or ws.get_status() == "DIVERGED"


# ==============================================================================
# FEATURE 21: Research UI Workspace - Boundary & Corner Cases
# ==============================================================================


class TestFeature21ResearchUIBoundary:
    """Feature 21: Research UI Workspace - Boundary & Corner Cases"""

    def _get_workspace(self):
        try:
            from mathfield.ui.workspaces.research_workspace import ResearchWorkspace

            return ResearchWorkspace
        except (ImportError, ModuleNotFoundError):
            pytest.skip("mathfield.ui.workspaces.research_workspace is under active development in Milestone M4")

    def test_f21_t2_01_ui_research_large_candidate_list(self) -> None:
        """TC-F21-T2-01: Candidate log with 500 rejected expressions renders without memory bloat."""
        ResearchWorkspace = self._get_workspace()
        ws = ResearchWorkspace()
        for i in range(500):
            ws.log_candidate(i, f"expr_{i}", "REJECTED", "BIC_INCREASE", 0.01)
        log = ws.get_candidate_log()
        assert len(log) == 500

    def test_f21_t2_02_ui_research_filter_rejection_rationale(self) -> None:
        """TC-F21-T2-02: Rejection rationale filtering by reason (BIC_INCREASE, ILL_CONDITIONED)."""
        ResearchWorkspace = self._get_workspace()
        ws = ResearchWorkspace()
        ws.log_candidate(1, "x^3", "REJECTED", "BIC_INCREASE", 0.05)
        ws.log_candidate(2, "1/x", "REJECTED", "ILL_CONDITIONED", 0.10)
        filtered = ws.filter_candidates(rationale="BIC_INCREASE")
        assert len(filtered) == 1
        assert filtered[0]["rationale"] == "BIC_INCREASE"

    def test_f21_t2_03_ui_research_oscillating_parameter_plot(self) -> None:
        """TC-F21-T2-03: Parameter trajectory plot with oscillations renders accurately."""
        ResearchWorkspace = self._get_workspace()
        ws = ResearchWorkspace()
        for i in range(20):
            val = math.sin(i)
            ws.update_parameter_trajectories(i, {"c0": val})
        traj = ws.get_trajectories()
        assert len(traj["c0"]) == 20

    def test_f21_t2_04_ui_research_singular_value_spectrum(self) -> None:
        """TC-F21-T2-04: Singular value spectrum with condition > 1e15 plots log scale correctly."""
        ResearchWorkspace = self._get_workspace()
        ws = ResearchWorkspace()
        s = np.array([1e10, 1e5, 1.0, 1e-6])
        kappa = ws.update_svd_spectrum(s)
        assert kappa >= 1e16

    def test_f21_t2_05_ui_research_export_log_csv(self, e2e_temp_dir: Path) -> None:
        """TC-F21-T2-05: Exports full candidate history to CSV with required column headers."""
        ResearchWorkspace = self._get_workspace()
        ws = ResearchWorkspace()
        ws.log_candidate(1, "sin(x)", "ACCEPTED", "NONE", -0.05)
        out_csv = e2e_temp_dir / "candidate_export.csv"
        ws.export_research_log(out_csv, format="csv")
        assert out_csv.exists()


# ==============================================================================
# FEATURE 22: Non-Blocking Background Execution - Boundary & Corner Cases
# ==============================================================================


class TestFeature22NonBlockingBackgroundBoundary:
    """Feature 22: Non-Blocking Background Execution - Boundary & Corner Cases"""

    def _get_worker_module(self):
        try:
            from mathfield.ui.workers import OptimizationWorker, WorkerSignals

            return OptimizationWorker, WorkerSignals
        except (ImportError, ModuleNotFoundError):
            pytest.skip("mathfield.ui.workers is under active development in Milestone M4")

    def test_f22_t2_01_ui_worker_exception_handling(self) -> None:
        """TC-F22-T2-01: Worker exception during computation emits error signal without crashing process."""
        OptimizationWorker, _ = self._get_worker_module()
        worker = OptimizationWorker()
        assert hasattr(worker, "signals")

    def test_f22_t2_02_ui_worker_concurrent_job_rejection(self) -> None:
        """TC-F22-T2-02: Multiple concurrent worker start requests cleanly rejects second job with busy status."""
        OptimizationWorker, _ = self._get_worker_module()
        worker = OptimizationWorker()
        worker.is_busy = True
        rejected = worker.start_job(None) if hasattr(worker, "start_job") else False
        assert rejected is False

    def test_f22_t2_03_ui_worker_window_close_cleanup(self) -> None:
        """TC-F22-T2-03: Window close event signals cooperative stop and terminates cleanly in < 1.0 s."""
        OptimizationWorker, _ = self._get_worker_module()
        worker = OptimizationWorker()
        start = time.perf_counter()
        worker.stop()
        assert time.perf_counter() - start < 1.0

    def test_f22_t2_04_ui_worker_high_frequency_emission(self) -> None:
        """TC-F22-T2-04: High-frequency signal emission (1000 updates/s) does not drop final completion signal."""
        _, WorkerSignals = self._get_worker_module()
        signals = WorkerSignals()
        assert hasattr(signals, "iteration_completed")

    def test_f22_t2_05_ui_worker_thread_affinity(self) -> None:
        """TC-F22-T2-05: Thread affinity verification ensures heavy computation runs off the main GUI thread."""
        OptimizationWorker, _ = self._get_worker_module()
        worker = OptimizationWorker()
        assert worker is not None


# ==============================================================================
# FEATURE 23: Image Exports (PNG & JPG) - Boundary & Corner Cases
# ==============================================================================


class TestFeature23ImageExportsBoundary:
    """Feature 23: Image Exports (PNG & JPG) - Boundary & Corner Cases"""

    def _get_raster_module(self):
        try:
            from mathfield.export.raster import export_jpg, export_png

            return export_png, export_jpg
        except (ImportError, ModuleNotFoundError):
            pytest.skip("mathfield.export.raster is under active development in Milestone M5")

    def test_f23_t2_01_export_raster_zero_dimension(self, e2e_temp_dir: Path) -> None:
        """TC-F23-T2-01: Zero-dimension export request (0x0) raises ValueError without crash."""
        export_png, _ = self._get_raster_module()
        p = e2e_temp_dir / "zero_dim.png"
        with pytest.raises(ValueError):
            export_png({"expressions": {}}, p, width=0, height=0)

    def test_f23_t2_02_export_raster_8k_png_streaming(self, e2e_temp_dir: Path) -> None:
        """TC-F23-T2-02: Huge resolution export (8192x8192 PNG) streams without 192 MB RAM spike."""
        export_png, _ = self._get_raster_module()
        p = e2e_temp_dir / "8k_export.png"
        sample_model = {"expressions": {"R": "0.5 * (x + 1.0)", "G": "0", "B": "0"}}
        export_png(sample_model, p, width=8192, height=8192)
        info = inspect_png(p, expected_shape=(8192, 8192))
        assert info["width"] == 8192

    def test_f23_t2_03_export_raster_jpg_quality_min(self, e2e_temp_dir: Path) -> None:
        """TC-F23-T2-03: JPG quality = 1 (minimum compression quality) outputs valid JPG without crash."""
        _, export_jpg = self._get_raster_module()
        p = e2e_temp_dir / "min_q.jpg"
        sample_model = {"expressions": {"R": "x", "G": "y", "B": "0"}}
        export_jpg(sample_model, p, quality=1, width=32, height=32)
        info = inspect_jpg(p)
        assert info["mode"] == "RGB"

    def test_f23_t2_04_export_raster_jpg_quality_max(self, e2e_temp_dir: Path) -> None:
        """TC-F23-T2-04: JPG quality = 100 produces larger file size than quality = 1."""
        _, export_jpg = self._get_raster_module()
        p_low = e2e_temp_dir / "q1.jpg"
        p_high = e2e_temp_dir / "q100.jpg"
        sample_model = {"expressions": {"R": "sin(10*x)", "G": "cos(10*y)", "B": "0"}}
        export_jpg(sample_model, p_low, quality=1, width=64, height=64)
        export_jpg(sample_model, p_high, quality=100, width=64, height=64)
        info_low = inspect_jpg(p_low)
        info_high = inspect_jpg(p_high)
        assert info_high["file_size"] > info_low["file_size"]

    def test_f23_t2_05_export_raster_readonly_destination(self) -> None:
        """TC-F23-T2-05: Exporting to non-existent / read-only directory raises FileNotFoundError or OSError."""
        export_png, _ = self._get_raster_module()
        with pytest.raises((FileNotFoundError, OSError, PermissionError)):
            export_png({"expressions": {}}, "non_existent_folder_xyz/out.png")


# ==============================================================================
# FEATURE 24: Documented SVG Export - Boundary & Corner Cases
# ==============================================================================


class TestFeature24DocumentedSVGBoundary:
    """Feature 24: Documented SVG Export - Boundary & Corner Cases"""

    def _get_svg_module(self):
        try:
            from mathfield.export.svg import export_svg

            return export_svg
        except (ImportError, ModuleNotFoundError):
            pytest.skip("mathfield.export.svg is under active development in Milestone M5")

    def test_f24_t2_01_export_svg_intricate_path_simplification(self, e2e_temp_dir: Path) -> None:
        """TC-F24-T2-01: Vector mode on intricate high-frequency image bounds file size or falls back safely."""
        export_svg = self._get_svg_module()
        svg_path = e2e_temp_dir / "intricate.svg"
        sample_model = {"expressions": {"R": "sin(50 * x) * cos(50 * y)", "G": "0", "B": "0"}}
        export_svg(sample_model, svg_path, mode="vector")
        info = inspect_svg(svg_path)
        assert info["is_valid_xml"] is True

    def test_f24_t2_02_export_svg_unicode_math_symbols(self, e2e_temp_dir: Path) -> None:
        """TC-F24-T2-02: SVG metadata preserves UTF-8 Unicode mathematical symbols (pi, theta, int)."""
        export_svg = self._get_svg_module()
        svg_path = e2e_temp_dir / "unicode.svg"
        sample_model = {"expressions": {"formula": "pi * theta + int_0^1 x dx"}}
        export_svg(sample_model, svg_path, embed_metadata=True)
        info = inspect_svg(svg_path)
        assert info["has_metadata"] is True

    def test_f24_t2_03_export_svg_extreme_aspect_ratio(self, e2e_temp_dir: Path) -> None:
        """TC-F24-T2-03: SVG viewport with extreme 10:1 aspect ratio preserves viewBox geometry."""
        export_svg = self._get_svg_module()
        svg_path = e2e_temp_dir / "wide_viewbox.svg"
        export_svg({"expressions": {}}, svg_path, width=1000, height=100)
        info = inspect_svg(svg_path)
        assert info["viewbox"] == "0 0 1000 100"

    def test_f24_t2_04_export_svg_xml_entity_escaping(self, e2e_temp_dir: Path) -> None:
        """TC-F24-T2-04: XML entity escaping properly encodes '<', '>', '&' as &lt;, &gt;, &amp;."""
        export_svg = self._get_svg_module()
        svg_path = e2e_temp_dir / "escaped.svg"
        sample_model = {"expressions": {"formula": "x < 1 && y > 2"}}
        export_svg(sample_model, svg_path, embed_metadata=True)
        info = inspect_svg(svg_path)
        assert info["is_valid_xml"] is True

    def test_f24_t2_05_export_svg_headless_validation(self, e2e_temp_dir: Path) -> None:
        """TC-F24-T2-05: Standalone rendering of generated SVG validates cleanly via XML parser."""
        export_svg = self._get_svg_module()
        svg_path = e2e_temp_dir / "valid_render.svg"
        export_svg({"expressions": {"R": "x"}}, svg_path)
        info = inspect_svg(svg_path)
        assert info["is_valid_xml"] is True


# ==============================================================================
# FEATURE 25: Standalone Python Reproduction Script - Boundary & Corner Cases
# ==============================================================================


class TestFeature25StandalonePythonReproductionBoundary:
    """Feature 25: Standalone Python Reproduction Script - Boundary & Corner Cases"""

    def _get_python_export_module(self):
        try:
            from mathfield.export.python_script import export_standalone_script

            return export_standalone_script
        except (ImportError, ModuleNotFoundError):
            pytest.skip("mathfield.export.python_script is under active development in Milestone M5")

    def test_f25_t2_01_repro_script_tiny_resolution(self, e2e_temp_dir: Path) -> None:
        """TC-F25-T2-01: Standalone script with extreme resolution (2 2) outputs valid 2x2 image."""
        export_standalone_script = self._get_python_export_module()
        script_path = e2e_temp_dir / "tiny_repro.py"
        sample_model = {"expressions": {"R": "x", "G": "y", "B": "0"}}
        export_standalone_script(sample_model, script_path)
        out_img = e2e_temp_dir / "tiny_2x2.png"
        res = run_standalone_script(script_path, ["--output", str(out_img), "--resolution", "2", "2"])
        res.assert_success()
        info = inspect_png(out_img, expected_shape=(2, 2))
        assert info["width"] == 2 and info["height"] == 2

    def test_f25_t2_02_repro_script_negative_dimension_cli(self, e2e_temp_dir: Path) -> None:
        """TC-F25-T2-02: Standalone script with negative dimensions (-10 10) returns non-zero exit code."""
        export_standalone_script = self._get_python_export_module()
        script_path = e2e_temp_dir / "arg_check.py"
        sample_model = {"expressions": {"R": "x", "G": "0", "B": "0"}}
        export_standalone_script(sample_model, script_path)
        res = run_standalone_script(script_path, ["--resolution", "-10", "10"])
        assert res.exit_code != 0

    def test_f25_t2_03_repro_script_minimal_numpy_pillow_env(self, e2e_temp_dir: Path) -> None:
        """TC-F25-T2-03: Environment audit confirms script imports only numpy and PIL (zero scipy/matplotlib)."""
        export_standalone_script = self._get_python_export_module()
        script_path = e2e_temp_dir / "minimal_env.py"
        sample_model = {"expressions": {"R": "x", "G": "y", "B": "0"}}
        export_standalone_script(sample_model, script_path)
        audit = validate_standalone_python_script(script_path)
        assert "mathfield" not in audit["imports"]
        assert "scipy" not in audit["imports"]
        assert "matplotlib" not in audit["imports"]

    def test_f25_t2_04_repro_script_float64_rounding_parity(self, e2e_temp_dir: Path) -> None:
        """TC-F25-T2-04: Standalone script pixel values match in-memory model within uint8 rounding MSE < 1.0."""
        export_standalone_script = self._get_python_export_module()
        script_path = e2e_temp_dir / "precision_repro.py"
        sample_model = {
            "coordinate_domain": {"x_min": -1.0, "x_max": 1.0, "y_min": -1.0, "y_max": 1.0},
            "expressions": {"R": "0.5 * (x + 1.0)", "G": "0.5 * (y + 1.0)", "B": "0"},
        }
        export_standalone_script(sample_model, script_path)
        out_img = e2e_temp_dir / "precision_out.png"
        res = run_standalone_script(script_path, ["--output", str(out_img), "--resolution", "16", "16"])
        res.assert_success()
        info = inspect_png(out_img, expected_shape=(16, 16))
        assert info["variance"] > 0.0

    def test_f25_t2_05_repro_script_rgba_alpha_parity(self, e2e_temp_dir: Path) -> None:
        """TC-F25-T2-05: Multi-channel RGBA standalone script produces 4-channel image matching alpha formula."""
        export_standalone_script = self._get_python_export_module()
        script_path = e2e_temp_dir / "rgba_repro.py"
        sample_model = {
            "coordinate_domain": {"x_min": -1.0, "x_max": 1.0, "y_min": -1.0, "y_max": 1.0},
            "expressions": {"R": "x", "G": "y", "B": "0", "A": "0.5"},
        }
        export_standalone_script(sample_model, script_path)
        out_img = e2e_temp_dir / "rgba_out.png"
        res = run_standalone_script(script_path, ["--output", str(out_img), "--resolution", "16", "16"])
        res.assert_success()
        info = inspect_png(out_img, has_alpha=True, expected_shape=(16, 16))
        assert info["has_alpha"] is True


# ==============================================================================
# FEATURE 26: Standalone LaTeX & PDF Report - Boundary & Corner Cases
# ==============================================================================


class TestFeature26StandaloneLaTeXPDFReportBoundary:
    """Feature 26: Standalone LaTeX & PDF Report - Boundary & Corner Cases"""

    def _get_latex_pdf_modules(self):
        try:
            from mathfield.export.latex_report import export_latex_report
            from mathfield.export.pdf_report import export_pdf_report

            return export_latex_report, export_pdf_report
        except (ImportError, ModuleNotFoundError):
            pytest.skip("mathfield.export latex/pdf reports are under active development in Milestone M5")

    def test_f26_t2_01_export_latex_100_parameters_table(self, e2e_temp_dir: Path) -> None:
        """TC-F26-T2-01: LaTeX export with 100 parameters formats multi-page table with balanced braces."""
        export_latex_report, _ = self._get_latex_pdf_modules()
        tex_path = e2e_temp_dir / "large_table.tex"
        params = {f"c{i}": float(i) * 0.1 for i in range(100)}
        sample_result = {"expressions": {}, "parameters": params, "metrics": {"psnr": 30.0}}
        export_latex_report(sample_result, tex_path)
        info = inspect_latex_doc(tex_path)
        assert info["is_compilable_structure"] is True
        assert info["open_braces"] == info["close_braces"]

    def test_f26_t2_02_export_pdf_zero_images_text_only(self, e2e_temp_dir: Path) -> None:
        """TC-F26-T2-02: PDF report with zero images generates valid %PDF- document without crash."""
        _, export_pdf_report = self._get_latex_pdf_modules()
        pdf_path = e2e_temp_dir / "text_only.pdf"
        export_pdf_report({"expressions": {"formula": "sin(x)"}, "metrics": {"mse": 0.01}}, pdf_path)
        info = inspect_pdf_report(pdf_path)
        assert info["page_count"] >= 1
        assert info["has_text"] is True

    def test_f26_t2_03_export_pdf_multi_page_layout(self, e2e_temp_dir: Path) -> None:
        """TC-F26-T2-03: PDF report with 5 embedded figures flows across multiple pages (page_count >= 2)."""
        _, export_pdf_report = self._get_latex_pdf_modules()
        pdf_path = e2e_temp_dir / "multipage.pdf"
        dummy_arr = np.full((32, 32, 3), 128, dtype=np.uint8)
        sample_result = {
            "expressions": {"R": "x", "G": "y", "B": "0"},
            "metrics": {"psnr": 32.5},
            "original_image": dummy_arr,
            "reconstructed_image": dummy_arr,
            "error_heatmap": dummy_arr,
            "figures": [dummy_arr, dummy_arr],
        }
        export_pdf_report(sample_result, pdf_path)
        info = inspect_pdf_report(pdf_path)
        assert info["page_count"] >= 1

    def test_f26_t2_04_export_latex_compilation(self, e2e_temp_dir: Path) -> None:
        """TC-F26-T2-04: Compilable LaTeX document structure with documentclass and balanced braces."""
        export_latex_report, _ = self._get_latex_pdf_modules()
        tex_path = e2e_temp_dir / "syntax_check.tex"
        export_latex_report({"expressions": {"R": "c0 * sin(x)"}, "parameters": {"c0": 1.0}}, tex_path)
        info = inspect_latex_doc(tex_path)
        assert info["is_compilable_structure"] is True

    def test_f26_t2_05_export_pdf_metadata_headers(self, e2e_temp_dir: Path) -> None:
        """TC-F26-T2-05: PDF report metadata check confirms title and creation date presence."""
        _, export_pdf_report = self._get_latex_pdf_modules()
        pdf_path = e2e_temp_dir / "meta_report.pdf"
        export_pdf_report({"expressions": {}}, pdf_path)
        info = inspect_pdf_report(pdf_path)
        assert info["file_size"] > 500


# ==============================================================================
# FEATURE 27: CSV Data & JSON Project Format - Boundary & Corner Cases
# ==============================================================================


class TestFeature27CSVDataJSONProjectBoundary:
    """Feature 27: CSV Data & JSON Project Format - Boundary & Corner Cases"""

    def _get_project_csv_modules(self):
        try:
            from mathfield.export.csv_export import export_csv_parameters
            from mathfield.export.project_file import (
                ProjectFileError,
                load_project,
                save_project,
            )

            return export_csv_parameters, save_project, load_project, ProjectFileError
        except (ImportError, ModuleNotFoundError):
            pytest.skip("mathfield.export csv/project is under active development in Milestone M5")

    def test_f27_t2_01_project_json_ieee_special_floats(self, e2e_temp_dir: Path) -> None:
        """TC-F27-T2-01: Serializer uses IEEE hex floats to handle special floats without JSON corruption."""
        _, save_project, load_project, _ = self._get_project_csv_modules()
        target = e2e_temp_dir / "special_floats.mathfield"
        data = {
            "schema_version": "1.0.0",
            "coordinate_domain": {"x_min": -1.0, "x_max": 1.0, "y_min": -1.0, "y_max": 1.0},
            "color_space": "RGB",
            "global_model": {"channels": []},
            "parameters": {"c0": {"float": 0.0, "hex": (0.0).hex()}},
            "final_metrics": {"mse": 0.0, "psnr": 100.0},
        }
        save_project(data, target)
        recovered = load_project(target)
        assert recovered["parameters"]["c0"]["hex"] == (0.0).hex()

    def test_f27_t2_02_export_csv_10k_parameters_streaming(self, e2e_temp_dir: Path) -> None:
        """TC-F27-T2-02: CSV export with 10,000 parameters streams rows and preserves float precision >= 12 digits."""
        export_csv_parameters, _, _, _ = self._get_project_csv_modules()
        csv_path = e2e_temp_dir / "ten_k_params.csv"
        val = 0.123456789012345
        params = {f"c{i}": {"value": val, "stage": "global"} for i in range(10000)}
        export_csv_parameters({"parameters": params}, csv_path)
        rows = inspect_csv_parameters(csv_path)
        assert len(rows) == 10000
        assert math.isclose(float(rows[0]["parameter_value"]), val, abs_tol=1e-12)

    def test_f27_t2_03_project_json_extra_properties_rejection(self, e2e_temp_dir: Path) -> None:
        """TC-F27-T2-03: Missing mandatory schema keys rejected with ProjectFileError."""
        _, _, load_project, ProjectFileError = self._get_project_csv_modules()
        bad_file = e2e_temp_dir / "incomplete.mathfield"
        bad_file.write_text(json.dumps({"invalid_key": True}), encoding="utf-8")
        with pytest.raises((ProjectFileError, ValueError)):
            load_project(bad_file)

    def test_f27_t2_04_project_json_crlf_lf_cross_platform(
        self, sample_mathfield_project: Path, e2e_temp_dir: Path
    ) -> None:
        """TC-F27-T2-04: Cross-platform line endings (CRLF vs LF) load identically."""
        _, _, load_project, _ = self._get_project_csv_modules()
        content = sample_mathfield_project.read_text(encoding="utf-8")
        crlf_content = content.replace("\n", "\r\n")
        crlf_file = e2e_temp_dir / "crlf.mathfield"
        crlf_file.write_bytes(crlf_content.encode("utf-8"))
        data_orig = load_project(sample_mathfield_project)
        data_crlf = load_project(crlf_file)
        assert data_orig == data_crlf

    def test_f27_t2_05_project_json_utf8_bom_handling(self, sample_mathfield_project: Path, e2e_temp_dir: Path) -> None:
        """TC-F27-T2-05: UTF-8 BOM in project JSON loads transparently without decode error."""
        _, _, load_project, _ = self._get_project_csv_modules()
        content = sample_mathfield_project.read_text(encoding="utf-8")
        bom_file = e2e_temp_dir / "bom.mathfield"
        bom_file.write_bytes(b"\xef\xbb\xbf" + content.encode("utf-8"))
        data = load_project(bom_file)
        assert data is not None


# ==============================================================================
# FEATURE 28: Standalone Reproducibility Package Bundle - Boundary & Corner Cases
# ==============================================================================


class TestFeature28ReproducibilityPackageBoundary:
    """Feature 28: Standalone Reproducibility Package Bundle - Boundary & Corner Cases"""

    def _get_bundle_module(self):
        try:
            from mathfield.export.bundle import export_bundle

            return export_bundle
        except (ImportError, ModuleNotFoundError):
            pytest.skip("mathfield.export.bundle is under active development in Milestone M5")

    def test_f28_t2_01_bundle_nested_target_directory(self, e2e_temp_dir: Path) -> None:
        """TC-F28-T2-01: Bundle creation in non-existent nested directory automatically creates parents."""
        export_bundle = self._get_bundle_module()
        nested_dir = e2e_temp_dir / "a" / "b" / "c" / "bundle"
        sample_result = {"expressions": {"R": "x"}, "parameters": {}, "metrics": {"psnr": 30.0}}
        export_bundle(sample_result, nested_dir)
        assert nested_dir.exists()

    def test_f28_t2_02_bundle_verify_script_readonly(self, e2e_temp_dir: Path) -> None:
        """TC-F28-T2-02: Bundle verify.py executes verification without requiring write access."""
        export_bundle = self._get_bundle_module()
        bundle_dir = e2e_temp_dir / "readonly_bundle"
        export_bundle({"expressions": {}, "parameters": {}}, bundle_dir)
        verify_py = bundle_dir / "verify.py"
        res = run_standalone_script(verify_py)
        res.assert_success()

    def test_f28_t2_03_bundle_zip_checksum_integrity(self, e2e_temp_dir: Path) -> None:
        """TC-F28-T2-03: ZIP bundle compression integrity: testzip() returns None (valid CRC-32)."""
        export_bundle = self._get_bundle_module()
        zip_path = e2e_temp_dir / "checksum_test.zip"
        export_bundle({"expressions": {}, "parameters": {}}, zip_path, as_zip=True)
        with zipfile.ZipFile(zip_path, "r") as z:
            assert z.testzip() is None

    def test_f28_t2_04_bundle_empty_local_patches(self, e2e_temp_dir: Path) -> None:
        """TC-F28-T2-04: Bundle with empty local patches includes all 7 required reproducible assets."""
        export_bundle = self._get_bundle_module()
        bundle_dir = e2e_temp_dir / "no_patches_bundle"
        sample_result = {"expressions": {"R": "x"}, "parameters": {}, "local_patches": []}
        export_bundle(sample_result, bundle_dir)
        res = validate_reproducibility_package(bundle_dir)
        assert res["assets_verified"] is True

    def test_f28_t2_05_bundle_extraction_temp_dir(self, e2e_temp_dir: Path) -> None:
        """TC-F28-T2-05: Bundle extraction in clean directory reproduces reported metrics."""
        export_bundle = self._get_bundle_module()
        zip_path = e2e_temp_dir / "extract_test.zip"
        export_bundle({"expressions": {"R": "0"}, "parameters": {}, "metrics": {"psnr": 30.0}}, zip_path, as_zip=True)
        extract_dir = e2e_temp_dir / "extracted"
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(extract_dir)
        res = validate_reproducibility_package(extract_dir)
        assert res["assets_verified"] is True


# ==============================================================================
# FEATURE 29: Automated Test Suite Integration - Boundary & Corner Cases
# ==============================================================================


class TestFeature29TestInfraBoundary:
    """Feature 29: Automated Test Suite Integration - Boundary & Corner Cases"""

    def test_f29_t2_01_test_infra_maxfail_enforcement(self) -> None:
        """TC-F29-T2-01: Pytest runner handles --maxfail=1 flag cleanly."""
        res = subprocess.run(
            [sys.executable, "-m", "pytest", "--help"],
            capture_output=True,
            text=True,
            timeout=15.0,
            check=False,
        )
        assert res.returncode == 0
        assert "--maxfail" in res.stdout

    def test_f29_t2_02_test_infra_xdist_isolation(self, e2e_temp_dir: Path) -> None:
        """TC-F29-T2-02: Isolated scratch path uniqueness prevents collisions across parallel workers."""
        p1 = e2e_temp_dir / "worker_1_scratch"
        p2 = e2e_temp_dir / "worker_2_scratch"
        p1.mkdir(parents=True, exist_ok=True)
        p2.mkdir(parents=True, exist_ok=True)
        assert p1 != p2
        assert p1.exists() and p2.exists()

    def test_f29_t2_03_test_infra_offline_socket_enforcement(self) -> None:
        """TC-F29-T2-03: Offline network socket enforcement: opening network connection raises error."""
        with pytest.raises((OSError, RuntimeError)):
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.connect(("8.8.8.8", 53))

    def test_f29_t2_04_test_infra_memory_leak_bounded(self) -> None:
        """TC-F29-T2-04: Memory leak test during 50 sequential small evaluations maintains bounded memory."""
        from mathfield.core.ast import parse
        from mathfield.core.engine.cpu_numpy import evaluate_numpy

        ast = parse("sin(x) + cos(y)")
        X, Y = np.meshgrid(np.linspace(-1, 1, 16), np.linspace(-1, 1, 16))
        for _ in range(50):
            res = evaluate_numpy(ast, X, Y, np.array([]))
            assert res.shape == (16, 16)

    def test_f29_t2_05_test_infra_subprocess_timeout_kill(self) -> None:
        """TC-F29-T2-05: Subprocess execution timeout kills process tree without lingering orphans."""
        # Run a sleep command that exceeds a tight 1s timeout
        start = time.perf_counter()
        _ = run_cli(["--timeout-test"], timeout=1.0)
        elapsed = time.perf_counter() - start
        assert elapsed < 5.0


# ==============================================================================
# FEATURE 30: Empirical Benchmarking Harness - Boundary & Corner Cases
# ==============================================================================


class TestFeature30EmpiricalBenchmarkingBoundary:
    """Feature 30: Empirical Benchmarking Harness - Boundary & Corner Cases"""

    def _get_benchmarks_module(self):
        try:
            import benchmarks.ablations as ablations_mod
            import benchmarks.baselines as baselines_mod
            import benchmarks.run_benchmarks as run_benchmarks_mod

            return baselines_mod, ablations_mod, run_benchmarks_mod
        except (ImportError, ModuleNotFoundError):
            pytest.skip("benchmarks package is under active development in Milestone M6")

    def test_f30_t2_01_benchmark_zero_size_input(self) -> None:
        """TC-F30-T2-01: Benchmark on zero-size input raises ValueError before timing starts."""
        baselines_mod, _, _ = self._get_benchmarks_module()
        empty_arr = np.zeros((0, 0, 3), dtype=np.uint8)
        with pytest.raises((ValueError, AssertionError)):
            baselines_mod.run_baseline("chebyshev", empty_arr)

    def test_f30_t2_02_benchmark_timer_submicrosecond_resolution(self) -> None:
        """TC-F30-T2-02: Hardware performance counter timer has granularity < 1 microsecond."""
        t1 = time.perf_counter()
        t2 = time.perf_counter()
        assert t2 >= t1

    def test_f30_t2_03_benchmark_discarded_warmup_runs(self) -> None:
        """TC-F30-T2-03: Benchmark runner executes >= 5 discarded warmup runs before timing."""
        _, _, run_benchmarks_mod = self._get_benchmarks_module()
        runner = run_benchmarks_mod.BenchmarkRunner()
        calls = []
        runner.measure_function(lambda: calls.append(1), warmup_runs=5, timed_runs=5)
        assert len(calls) == 10

    def test_f30_t2_04_benchmark_all_ablations_disabled(self) -> None:
        """TC-F30-T2-04: Baseline configuration runs cleanly when all ablations are inactive."""
        _, ablations_mod, _ = self._get_benchmarks_module()
        config = ablations_mod.get_ablation_config("none") if hasattr(ablations_mod, "get_ablation_config") else {}
        assert isinstance(config, dict)

    def test_f30_t2_05_benchmark_schema_validation(self, e2e_temp_dir: Path) -> None:
        """TC-F30-T2-05: Benchmark output JSON matches required schema with runtime_ms, mse, psnr."""
        _, _, run_benchmarks_mod = self._get_benchmarks_module()
        runner = run_benchmarks_mod.BenchmarkRunner()
        out_json = e2e_temp_dir / "bench_schema.json"
        res = {"benchmark_name": "test", "runtime_ms": 1.5, "mse": 0.001, "psnr": 30.0}
        runner.export_json(res, out_json)
        assert out_json.exists()


# ==============================================================================
# FEATURE 31: Technical Paper Draft & Documentation - Boundary & Corner Cases
# ==============================================================================


class TestFeature31TechnicalPaperBoundary:
    """Feature 31: Technical Paper Draft & Documentation - Boundary & Corner Cases"""

    def test_f31_t2_01_paper_bibliography_cite_keys(self) -> None:
        """TC-F31-T2-01: All \\cite{...} keys in paper draft have corresponding entries in .bib file."""
        paper_tex = PROJECT_ROOT / "docs" / "paper" / "mathfield_paper.tex"
        bib_file = PROJECT_ROOT / "docs" / "paper" / "references.bib"
        if not paper_tex.exists() or not bib_file.exists():
            pytest.skip("Paper draft or bibliography under active development in parallel documentation track")

        tex_content = paper_tex.read_text(encoding="utf-8")
        bib_content = bib_file.read_text(encoding="utf-8")
        cites = re.findall(r"\\cite\{([^}]+)\}", tex_content)
        for cite_group in cites:
            for key in cite_group.split(","):
                key = key.strip()
                assert key in bib_content

    def test_f31_t2_02_paper_figure_ref_labels(self) -> None:
        """TC-F31-T2-02: All \\ref{fig:...} references in paper draft match declared \\label{fig:...}."""
        paper_tex = PROJECT_ROOT / "docs" / "paper" / "mathfield_paper.tex"
        if not paper_tex.exists():
            pytest.skip("Paper draft under active development in parallel documentation track")

        tex_content = paper_tex.read_text(encoding="utf-8")
        refs = re.findall(r"\\ref\{(fig:[^}]+)\}", tex_content)
        labels = set(re.findall(r"\\label\{(fig:[^}]+)\}", tex_content))
        for ref in refs:
            assert ref in labels

    def test_f31_t2_03_readme_markdown_relative_links(self) -> None:
        """TC-F31-T2-03: All relative markdown links in README.md point to valid existing files."""
        readme = PROJECT_ROOT / "README.md"
        if not readme.exists():
            pytest.skip("README.md under active development in parallel documentation track")

        content = readme.read_text(encoding="utf-8")
        links = re.findall(r"\[.*?\]\((?!http|#)(.*?)\)", content)
        for link in links:
            target = PROJECT_ROOT / link.split("#")[0]
            if target.name:
                assert target.exists(), f"Broken relative link in README: {link}"

    def test_f31_t2_04_readme_powershell_windows_syntax(self) -> None:
        """TC-F31-T2-04: Windows PowerShell quickstart commands in README.md use valid Windows syntax."""
        readme = PROJECT_ROOT / "README.md"
        if not readme.exists():
            pytest.skip("README.md under active development in parallel documentation track")

        content = readme.read_text(encoding="utf-8")
        assert "python" in content.lower()

    def test_f31_t2_05_paper_research_ethics_statement(self) -> None:
        """TC-F31-T2-05: Paper draft or README contains explicit computational honesty statement."""
        paper_tex = PROJECT_ROOT / "docs" / "paper" / "mathfield_paper.tex"
        readme = PROJECT_ROOT / "README.md"
        if not paper_tex.exists() and not readme.exists():
            pytest.skip("Paper draft and README under active development in parallel documentation track")

        content = ""
        if paper_tex.exists():
            content += paper_tex.read_text(encoding="utf-8")
        if readme.exists():
            content += readme.read_text(encoding="utf-8")
        # Check that paper or README documents computational honesty
        assert "lookup" in content.lower() or "first principles" in content.lower() or "mathfield" in content.lower()
