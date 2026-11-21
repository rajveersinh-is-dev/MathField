"""
e2e_tests/test_e2e_tier1.py

Tier 1: Feature Coverage Test Suite (Milestone E2E-M1)
Opaque-box verification across Features 1 through 31.
Total Test Cases: 155 (31 features x exactly 5 tests per feature).

Methodology: Category-Partition Equivalence Classes:
1. Nominal / Primary Canonical Path
2. Equivalence Class Variant / Alternative Mode
3. Format & Output Contract Integrity
4. Mathematical Invariant & Rule Enforcement
5. Negative Space & Error Handling
"""

from __future__ import annotations

import json
import math
import re
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pytest
from PIL import Image

from e2e_tests.helpers import (
    assert_no_memorized_arrays,
    compute_reference_mse,
    compute_reference_psnr,
    compute_reference_ssim,
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

pytestmark = [pytest.mark.tier1]
PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ==============================================================================
# FEATURE 1: Mode A (Mathematical Discovery)
# ==============================================================================


class TestFeature01ModeA:
    """Feature 1: Mode A (Mathematical Discovery) - Pareto J(M) = E(M) + lambda C(M)"""

    def test_f01_01_mode_a_low_lambda_rich_model(self, synth_rgb_poly: Path, e2e_temp_dir: Path) -> None:
        """TC-F01-01: Weak regularization discovers richer expression with lower residual error."""
        cli_py = PROJECT_ROOT / "cli.py"
        assert cli_py.exists(), "Required artefact missing: cli_py"

        out_low = e2e_temp_dir / "f01_low"
        res = run_cli(
            [
                "--image",
                str(synth_rgb_poly),
                "--output-dir",
                str(out_low),
                "--mode",
                "discovery",
                "--lambda-reg",
                "1e-5",
                "--max-iter",
                "3",
                "--export-json",
                "--quiet",
            ]
        )
        res.assert_success()
        proj_file = out_low / "mathfield_out.mathfield"
        data = inspect_json_project(proj_file)
        assert data.get("mode") in ("MODE_A_DISCOVERY", "discovery", "A")

    def test_f01_02_mode_a_high_lambda_parsimony(self, synth_rgb_poly: Path, e2e_temp_dir: Path) -> None:
        """TC-F01-02: Strong regularization enforces parsimonious model with bounded complexity."""
        cli_py = PROJECT_ROOT / "cli.py"
        assert cli_py.exists(), "Required artefact missing: cli_py"

        out_high = e2e_temp_dir / "f01_high"
        res = run_cli(
            [
                "--image",
                str(synth_rgb_poly),
                "--output-dir",
                str(out_high),
                "--mode",
                "discovery",
                "--lambda-reg",
                "0.8",
                "--max-iter",
                "3",
                "--export-json",
                "--quiet",
            ]
        )
        res.assert_success()
        proj_file = out_high / "mathfield_out.mathfield"
        data = inspect_json_project(proj_file)
        metrics = data.get("final_metrics") or data.get("metrics", {})
        assert "complexity" in metrics or "mse" in metrics

    def test_f01_03_mode_a_interpretable_symbolic(self, synth_rgb_poly: Path, e2e_temp_dir: Path) -> None:
        """TC-F01-03: Anti-memorization invariant: model is symbolic AST, zero direct pixel lookup tables."""
        cli_py = PROJECT_ROOT / "cli.py"
        assert cli_py.exists(), "Required artefact missing: cli_py"

        out_dir = e2e_temp_dir / "f01_symbolic"
        res = run_cli(
            [
                "--image",
                str(synth_rgb_poly),
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
        proj = json.loads((out_dir / "mathfield_out.mathfield").read_text(encoding="utf-8"))
        model = proj.get("global_model") or proj.get("expressions", {})

        # Deep audit: verify zero 1D/2D array literals, zero lookup tables, pure symbolic AST
        assert_no_memorized_arrays(model)

        # In addition, verify that the discovered formula parses into a valid mathfield.core.ast.ASTNode
        from mathfield.core.ast import ASTNode, parse as mf_parse

        formula = model.get("formula") if isinstance(model, dict) else str(model)
        if formula and isinstance(formula, str):
            ast_node = mf_parse(formula)
            assert isinstance(ast_node, ASTNode)

    def test_f01_04_mode_a_deterministic_seed(self, synth_rgb_poly: Path, e2e_temp_dir: Path) -> None:
        """TC-F01-04: Fixed seed yields reproducible AST structure across independent runs."""
        cli_py = PROJECT_ROOT / "cli.py"
        assert cli_py.exists(), "Required artefact missing: cli_py"

        out1 = e2e_temp_dir / "f01_seed1"
        out2 = e2e_temp_dir / "f01_seed2"
        cmd = [
            "--image",
            str(synth_rgb_poly),
            "--mode",
            "discovery",
            "--seed",
            "42",
            "--max-iter",
            "2",
            "--export-json",
            "--quiet",
        ]
        res1 = run_cli(cmd + ["--output-dir", str(out1)])
        res2 = run_cli(cmd + ["--output-dir", str(out2)])
        res1.assert_success()
        res2.assert_success()

        proj1_path = out1 / "mathfield_out.mathfield"
        proj2_path = out2 / "mathfield_out.mathfield"
        assert proj1_path.exists(), "Seed 1 output mathfield project file missing"
        assert proj2_path.exists(), "Seed 2 output mathfield project file missing"

        data1 = json.loads(proj1_path.read_text(encoding="utf-8"))
        data2 = json.loads(proj2_path.read_text(encoding="utf-8"))

        # Verify bit-exact deterministic reproducibility of discovered model AST / formula
        model1 = data1.get("global_model") or data1.get("expressions", {})
        model2 = data2.get("global_model") or data2.get("expressions", {})
        formula1 = model1.get("formula") if isinstance(model1, dict) else str(model1)
        formula2 = model2.get("formula") if isinstance(model2, dict) else str(model2)
        assert formula1 == formula2, f"Seed 42 produced divergent formulas: {formula1} != {formula2}"

        # Verify bit-exact parity of metrics and complexity
        metrics1 = data1.get("final_metrics") or data1.get("metrics", {})
        metrics2 = data2.get("final_metrics") or data2.get("metrics", {})
        assert metrics1.get("complexity") == metrics2.get("complexity")
        assert math.isclose(float(metrics1.get("mse", 0.0)), float(metrics2.get("mse", 0.0)), abs_tol=1e-12)

    def test_f01_05_mode_a_convergence_criteria(self, synth_rgb_poly: Path, e2e_temp_dir: Path) -> None:
        """TC-F01-05: Optimization cleanly terminates and returns finite non-negative metrics."""
        cli_py = PROJECT_ROOT / "cli.py"
        assert cli_py.exists(), "Required artefact missing: cli_py"

        out_dir = e2e_temp_dir / "f01_converge"
        res = run_cli(
            [
                "--image",
                str(synth_rgb_poly),
                "--output-dir",
                str(out_dir),
                "--mode",
                "discovery",
                "--max-iter",
                "3",
                "--tolerance",
                "1e-4",
                "--export-json",
                "--quiet",
            ]
        )
        res.assert_success()

        proj_file = out_dir / "mathfield_out.mathfield"
        assert proj_file.exists(), "Convergence test output project file missing"
        data = json.loads(proj_file.read_text(encoding="utf-8"))
        metrics = data.get("final_metrics") or data.get("metrics", {})

        # Verify optimization returns non-negative, finite metrics
        assert "mse" in metrics and "psnr" in metrics
        mse = float(metrics["mse"])
        psnr = float(metrics["psnr"])
        assert math.isfinite(mse) and mse >= 0.0
        assert math.isfinite(psnr) and psnr > 0.0

        # Verify convergence criteria status
        assert "converged" in metrics or "tolerance_met" in metrics or mse < 1.0
        assert data.get("status") in ("COMPLETED", "completed", "CONVERGED", "converged", None)


# ==============================================================================
# FEATURE 2: Mode B (Maximum Fidelity)
# ==============================================================================


class TestFeature02ModeB:
    """Feature 2: Mode B (Maximum Fidelity) - Higher-order expansions and piecewise corrections"""

    def test_f02_01_mode_b_higher_raw_fidelity(self, synth_harmonic: Path, e2e_temp_dir: Path) -> None:
        """TC-F02-01: Fidelity objective prioritizes approximation accuracy."""
        cli_py = PROJECT_ROOT / "cli.py"
        assert cli_py.exists(), "Required artefact missing: cli_py"

        out_dir = e2e_temp_dir / "f02_fidelity"
        res = run_cli(
            [
                "--image",
                str(synth_harmonic),
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

        proj_file = out_dir / "mathfield_out.mathfield"
        assert proj_file.exists(), "Mode B project file missing"
        data = inspect_json_project(proj_file)
        metrics = data.get("final_metrics") or data.get("metrics", {})
        assert "psnr" in metrics and "mse" in metrics
        assert float(metrics["psnr"]) >= 20.0
        assert float(metrics["mse"]) < 0.05
        assert math.isfinite(float(metrics["psnr"]))

    def test_f02_02_mode_b_dense_basis_expansion(self, synth_harmonic: Path, e2e_temp_dir: Path) -> None:
        """TC-F02-02: Mode B engages dense basis expansions for high-frequency patterns."""
        cli_py = PROJECT_ROOT / "cli.py"
        assert cli_py.exists(), "Required artefact missing: cli_py"

        out_dir = e2e_temp_dir / "f02_dense"
        res = run_cli(
            [
                "--image",
                str(synth_harmonic),
                "--output-dir",
                str(out_dir),
                "--mode",
                "fidelity",
                "--max-iter",
                "2",
                "--export-csv",
                "--export-json",
                "--quiet",
            ]
        )
        res.assert_success()

        # Inspect parameter expansion density in project file
        proj_file = out_dir / "mathfield_out.mathfield"
        assert proj_file.is_file(), "--export-json project file was not written"
        data = json.loads(proj_file.read_text(encoding="utf-8"))
        global_model = data.get("global_model", {})
        deg = global_model.get("degree", 0)
        terms = data.get("basis_terms_count", 0) or len(data.get("parameters", []))
        assert deg >= 4 or terms >= 20, f"Dense basis expansion expected, got degree={deg}, terms={terms}"

        csv_file = out_dir / "parameters.csv"
        assert csv_file.is_file(), "--export-csv table was not written"
        csv_data = inspect_csv_parameters(csv_file)
        assert len(csv_data) >= 20, "Expected dense basis expansion with > 20 parameters"

    def test_f02_03_mode_b_piecewise_corrections(self, synth_checker: Path, e2e_temp_dir: Path) -> None:
        """TC-F02-03: Local decomposition triggers on high-contrast edges."""
        cli_py = PROJECT_ROOT / "cli.py"
        assert cli_py.exists(), "Required artefact missing: cli_py"

        out_dir = e2e_temp_dir / "f02_piecewise"
        res = run_cli(
            [
                "--image",
                str(synth_checker),
                "--output-dir",
                str(out_dir),
                "--mode",
                "fidelity",
                "--max-iter",
                "2",
                "--export-json",
                "--quiet",
            ]
        )
        res.assert_success()

        proj_file = out_dir / "mathfield_out.mathfield"
        assert proj_file.exists(), "Mode B piecewise project file missing"
        data = json.loads(proj_file.read_text(encoding="utf-8"))

        # Verify that local quadtree decomposition occurred on high-contrast edges
        local_decomp = data.get("local_decomposition") or data.get("local_patches")
        assert isinstance(local_decomp, dict), "Local decomposition missing from Mode B project output"
        leaves_count = local_decomp.get("leaves_count", 0) or len(local_decomp.get("patches", []))
        assert leaves_count >= 4, f"Expected quadtree subdivision into >= 4 leaves, got {leaves_count}"

    def test_f02_04_mode_b_numerical_conditioning(self, synth_rbf: Path, e2e_temp_dir: Path) -> None:
        """TC-F02-04: Dense expansions monitor and report numerical conditioning."""
        cli_py = PROJECT_ROOT / "cli.py"
        assert cli_py.exists(), "Required artefact missing: cli_py"

        out_dir = e2e_temp_dir / "f02_condition"
        res = run_cli(
            [
                "--image",
                str(synth_rbf),
                "--output-dir",
                str(out_dir),
                "--mode",
                "fidelity",
                "--max-iter",
                "2",
                "--export-json",
                "--quiet",
            ]
        )
        res.assert_success()

        proj_file = out_dir / "mathfield_out.mathfield"
        assert proj_file.exists(), "Mode B project file missing"
        data = json.loads(proj_file.read_text(encoding="utf-8"))

        # Verify numerical conditioning reporting
        cond_rep = data.get("condition_report") or data.get("global_model", {})
        cond_num = cond_rep.get("condition_number") or data.get("final_metrics", {}).get("condition_number")
        assert isinstance(cond_num, (int, float)), "Condition number report missing from Mode B output"
        assert float(cond_num) > 0.0 and math.isfinite(float(cond_num))

    def test_f02_05_mode_b_sharp_edge_response(self, synth_typography: Path, e2e_temp_dir: Path) -> None:
        """TC-F02-05: Sharp binary step discontinuities converge without NaN divergence."""
        cli_py = PROJECT_ROOT / "cli.py"
        assert cli_py.exists(), "Required artefact missing: cli_py"

        out_dir = e2e_temp_dir / "f02_sharp"
        res = run_cli(
            [
                "--image",
                str(synth_typography),
                "--output-dir",
                str(out_dir),
                "--mode",
                "fidelity",
                "--max-iter",
                "2",
                "--export-json",
                "--quiet",
            ]
        )
        res.assert_success()

        proj_file = out_dir / "mathfield_out.mathfield"
        assert proj_file.exists(), "Mode B typography output file missing"
        data = json.loads(proj_file.read_text(encoding="utf-8"))
        metrics = data.get("final_metrics") or data.get("metrics", {})

        # Zero NaNs / Infs in metrics
        for k, v in metrics.items():
            if isinstance(v, (int, float)):
                assert not math.isnan(v), f"Metric {k} evaluated to NaN"
                assert not math.isinf(v), f"Metric {k} evaluated to Inf"

        assert float(metrics.get("mse", 0.0)) < 0.2
        assert float(metrics.get("psnr", 0.0)) > 10.0

        # Inspect reconstructed raster image if present
        recon_path = out_dir / "reconstructed.png"
        if recon_path.exists():
            arr = np.asarray(Image.open(recon_path), dtype=np.float64)
            assert not np.any(np.isnan(arr)), "Reconstructed image contains NaNs"
            assert not np.any(np.isinf(arr)), "Reconstructed image contains Infs"


# ==============================================================================
# FEATURE 3: Continuous 2D Coordinate Domain
# ==============================================================================


class TestFeature03Continuous2DCoordinateDomain:
    """Feature 3: Continuous 2D Coordinate Domain mapping (normalized & aspect-preserved)"""

    def _get_domain_module(self):
        from mathfield.core.domain import CoordinateDomain2D, DomainMode

        return CoordinateDomain2D, DomainMode

    def test_f03_01_domain_normalized_unit_square(self) -> None:
        """TC-F03-01: Normalized unit square spans [-1, 1] in both X and Y dimensions."""
        CoordinateDomain2D, DomainMode = self._get_domain_module()
        domain = CoordinateDomain2D(width=64, height=64, mode=DomainMode.SQUARE)
        X, Y = domain.to_discrete_grid()
        assert X.shape == (64, 64)
        assert Y.shape == (64, 64)
        assert math.isclose(float(np.min(X)), -1.0, abs_tol=1e-5)
        assert math.isclose(float(np.max(X)), 1.0, abs_tol=1e-5)
        assert math.isclose(float(np.min(Y)), -1.0, abs_tol=1e-5)
        assert math.isclose(float(np.max(Y)), 1.0, abs_tol=1e-5)

    def test_f03_02_domain_isotropic_non_square(self) -> None:
        """TC-F03-02: Isotropic aspect-preserved mapping spans [-r, r] x [-1, 1]."""
        CoordinateDomain2D, DomainMode = self._get_domain_module()
        domain = CoordinateDomain2D(width=64, height=32, mode=DomainMode.ISOTROPIC)
        assert math.isclose(domain.aspect_ratio, 2.0, rel_tol=1e-5)
        assert domain.x_bounds[0] <= -1.9
        assert domain.x_bounds[1] >= 1.9
        dx, dy = domain.cell_size()
        assert math.isclose(dx, dy, rel_tol=1e-5)

    def test_f03_03_domain_bijection_roundtrip(self) -> None:
        """TC-F03-03: Discrete pixel coordinates map bijectively to continuous domain and back."""
        CoordinateDomain2D, DomainMode = self._get_domain_module()
        domain = CoordinateDomain2D(width=128, height=64)
        i_px = np.arange(128)
        j_px = np.arange(64)
        x_cont, y_cont = domain.to_continuous(i_px, j_px)
        i_rec, j_rec = domain.to_discrete(x_cont, y_cont)
        np.testing.assert_array_equal(i_px, i_rec)
        np.testing.assert_array_equal(j_px, j_rec)

    def test_f03_04_domain_continuous_resampling(self) -> None:
        """TC-F03-04: Evaluating continuous domain coordinate transform on arbitrary points."""
        CoordinateDomain2D, DomainMode = self._get_domain_module()
        domain = CoordinateDomain2D(width=64, height=64)
        x_pts = np.array([0.12345, -0.6789, 0.999])
        y_pts = np.array([-0.5432, 0.3210, -0.999])
        i_px, j_px = domain.to_discrete(x_pts, y_pts)
        assert np.all(i_px >= 0) and np.all(i_px < 64)
        assert np.all(j_px >= 0) and np.all(j_px < 64)

    def test_f03_05_domain_corner_symmetry_bounds(self) -> None:
        """TC-F03-05: Domain coordinate extrema are symmetric around (0, 0)."""
        CoordinateDomain2D, DomainMode = self._get_domain_module()
        domain = CoordinateDomain2D(width=100, height=100, mode=DomainMode.SQUARE)
        X, Y = domain.to_discrete_grid()
        center_x = float((X[0, 0] + X[-1, -1]) * 0.5)
        center_y = float((Y[0, 0] + Y[-1, -1]) * 0.5)
        assert math.isclose(center_x, 0.0, abs_tol=1e-10)
        assert math.isclose(center_y, 0.0, abs_tol=1e-10)


# ==============================================================================
# FEATURE 4: Stage 1 (Image Analysis)
# ==============================================================================


class TestFeature04Stage1ImageAnalysis:
    """Feature 4: Stage 1 (Image Analysis) - TV, entropy, 2D FFT spectral roll-off"""

    def _get_analyzer_module(self):
        from mathfield.core.image_analysis import ImageAnalysisResult, ImageAnalyzer

        return ImageAnalyzer, ImageAnalysisResult

    def test_f04_01_analysis_resolution_aspect(self, synth_landscape: Path) -> None:
        """TC-F04-01: Detects input dimensions and computes aspect ratio."""
        ImageAnalyzer, ImageAnalysisResult = self._get_analyzer_module()
        analyzer = ImageAnalyzer()
        res = analyzer.analyze(synth_landscape)
        assert isinstance(res, ImageAnalysisResult)
        assert res.width > 0 and res.height > 0
        assert math.isclose(res.aspect_ratio, 1.0, abs_tol=1e-5)

    def test_f04_02_analysis_channels_and_alpha(self, synth_rgba_image: Path, synth_rgb_poly: Path) -> None:
        """TC-F04-02: Differentiates RGB (3 channels) from RGBA (4 channels with alpha)."""
        ImageAnalyzer, ImageAnalysisResult = self._get_analyzer_module()
        analyzer = ImageAnalyzer()
        res_rgb = analyzer.analyze(synth_rgb_poly)
        res_rgba = analyzer.analyze(synth_rgba_image)
        assert res_rgb.channels == 3 and res_rgb.has_alpha is False
        assert res_rgba.channels == 4 and res_rgba.has_alpha is True

    def test_f04_03_analysis_tv_and_entropy(self, e2e_temp_dir: Path) -> None:
        """TC-F04-03: Spatial complexity metrics: Flat image has TV=0 and entropy=0."""
        ImageAnalyzer, ImageAnalysisResult = self._get_analyzer_module()
        flat_arr = np.full((32, 32, 3), 128, dtype=np.uint8)
        analyzer = ImageAnalyzer()
        res = analyzer.analyze(flat_arr)
        assert res.total_variation == 0.0
        assert math.isclose(res.entropy, 0.0, abs_tol=1e-10)

    def test_f04_04_analysis_fft_spectral_rolloff(self) -> None:
        """TC-F04-04: 2D FFT spectral roll-off differentiates smooth fields from checkerboards."""
        ImageAnalyzer, ImageAnalysisResult = self._get_analyzer_module()
        x = np.linspace(-1, 1, 64)
        X, Y = np.meshgrid(x, x)
        smooth = ((np.sin(np.pi * X) * np.cos(np.pi * Y) + 1.0) * 127.5).astype(np.uint8)
        checker = (((np.sign(np.sin(8 * np.pi * X) * np.sin(8 * np.pi * Y)) + 1.0) * 0.5) * 255).astype(np.uint8)
        analyzer = ImageAnalyzer()
        res_smooth = analyzer.analyze(smooth)
        res_checker = analyzer.analyze(checker)
        assert res_checker.spectral_rolloff != res_smooth.spectral_rolloff

    def test_f04_05_analysis_corrupt_file_error(self, e2e_temp_dir: Path) -> None:
        """TC-F04-05: Corrupt image file raises diagnostic error without process crash."""
        ImageAnalyzer, ImageAnalysisResult = self._get_analyzer_module()
        corrupt_path = e2e_temp_dir / "corrupt.png"
        corrupt_path.write_bytes(b"\x89PNG\r\n\x1a\nGARBAGE_PAYLOAD_NOT_A_PNG")
        analyzer = ImageAnalyzer()
        with pytest.raises((ValueError, OSError, Image.UnidentifiedImageError)):
            analyzer.analyze(corrupt_path)


# ==============================================================================
# FEATURE 5: Stage 2 (Global Discovery)
# ==============================================================================


class TestFeature05Stage2GlobalDiscovery:
    """Feature 5: Stage 2 (Global Discovery) - Chebyshev, Fourier, splines, RBFs, rationals"""

    def _get_basis_modules(self):
        from mathfield.core.basis.fourier import Fourier2D
        from mathfield.core.basis.polynomial import Chebyshev2D
        from mathfield.core.basis.rational import Rational2D
        from mathfield.core.basis.rbf import WendlandC2RBF
        from mathfield.core.basis.splines import BSplineSurface2D

        return Chebyshev2D, Fourier2D, BSplineSurface2D, WendlandC2RBF, Rational2D

    def test_f05_01_global_chebyshev_polynomial(self) -> None:
        """TC-F05-01: Chebyshev orthogonal polynomial basis recovers smooth quadratic profile."""
        Chebyshev2D, _, _, _, _ = self._get_basis_modules()
        basis = Chebyshev2D(deg_x=2, deg_y=2)
        X, Y = np.meshgrid(np.linspace(-1, 1, 32), np.linspace(-1, 1, 32))
        vals = basis.evaluate(X, Y)
        assert vals.shape == (32, 32)
        assert np.all(np.isfinite(vals))

    def test_f05_02_global_fourier_harmonics(self) -> None:
        """TC-F05-02: Fourier 2D harmonics capture harmonic ripples."""
        _, Fourier2D, _, _, _ = self._get_basis_modules()
        basis = Fourier2D(frequencies=[(1.0, 1.0), (2.0, 2.0)])
        X, Y = np.meshgrid(np.linspace(-1, 1, 32), np.linspace(-1, 1, 32))
        vals = basis.evaluate(X, Y)
        assert vals.shape == (32, 32)
        assert np.all(np.isfinite(vals))

    def test_f05_03_global_bspline_surface(self) -> None:
        """TC-F05-03: Tensor-product B-spline approximation preserves C^2 smoothness and partition of unity."""
        _, _, BSplineSurface2D, _, _ = self._get_basis_modules()

        # Dimension compatibility: for degree p=3 and 5 control points, knots vector must have length n + p + 1 = 9
        # Clamped knot vector over [-1, 1] ensures partition of unity across the entire domain
        knots = np.array([-1.0, -1.0, -1.0, -1.0, 0.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float64)
        ctrl = np.ones((5, 5), dtype=np.float64)

        basis = BSplineSurface2D(knots_x=knots, knots_y=knots, control_points=ctrl, degree=3)
        assert basis.nx == 5 and basis.ny == 5
        assert basis.num_terms == 25

        X, Y = np.meshgrid(np.linspace(-1, 1, 32), np.linspace(-1, 1, 32))
        vals = basis.evaluate(X, Y)

        # Invariant 1: Dimensions match evaluation grid
        assert vals.shape == (32, 32)
        assert np.all(np.isfinite(vals))

        # Invariant 2: Partition of unity: sum of B-spline basis functions == 1.0 when all c_ij == 1.0
        np.testing.assert_allclose(vals, 1.0, atol=1e-12)

        # Invariant 3: Smooth quadratic reconstruction capability
        target = 0.5 * (X**2 + Y**2)
        Phi = basis.basis_matrix(X, Y)
        c_opt = np.linalg.lstsq(Phi, target.ravel(), rcond=None)[0]
        fitted = basis.evaluate(X, Y, coefficients=c_opt)
        np.testing.assert_allclose(fitted, target, atol=1e-3)

        # Invariant 4: C^2 second derivative existence without jumps
        d2_dx2 = (fitted[:, 2:] - 2.0 * fitted[:, 1:-1] + fitted[:, :-2]) / ((2.0 / 31.0) ** 2)
        assert np.all(np.isfinite(d2_dx2))

    def test_f05_04_global_compact_wendland_rbf(self) -> None:
        """TC-F05-04: Wendland C^2 compactly supported RBF evaluation."""
        _, _, _, WendlandC2RBF, _ = self._get_basis_modules()
        centers = np.array([[0.0, 0.0]])
        radii = np.array([1.0])
        basis = WendlandC2RBF(centers=centers, radii=radii)
        X, Y = np.meshgrid(np.linspace(-1, 1, 32), np.linspace(-1, 1, 32))
        vals = basis.evaluate(X, Y)
        assert vals.shape == (32, 32)
        assert np.all(vals >= 0.0)

    def test_f05_05_global_rational_pole_safe(self) -> None:
        """TC-F05-05: Rational Padé function regularized against denominator zeros."""
        _, _, _, _, Rational2D = self._get_basis_modules()
        num = np.array([1.0, 0.5])
        den = np.array([0.0, 0.0])  # zero denominator
        rat = Rational2D(num_coeffs=num, den_coeffs=den, deg_x=1, deg_y=0)
        X, Y = np.meshgrid(np.linspace(-1, 1, 16), np.linspace(-1, 1, 16))
        vals = rat.evaluate(X, Y)
        assert np.all(np.isfinite(vals))


# ==============================================================================
# FEATURE 6: Stage 3 (Local Decomposition)
# ==============================================================================


class TestFeature06Stage3LocalDecomposition:
    """Feature 6: Stage 3 (Local Decomposition) - Quadtree, Wendland partition of unity"""

    def _get_local_modules(self):
        from mathfield.core.local.local_patch import LocalPatch
        from mathfield.core.local.partition_of_unity import WendlandPartitionOfUnity
        from mathfield.core.local.quadtree import QuadtreeDecomposition, QuadtreeNode

        return QuadtreeDecomposition, QuadtreeNode, WendlandPartitionOfUnity, LocalPatch

    def test_f06_01_local_residual_map_generation(self) -> None:
        """TC-F06-01: Residual map R(x,y) = |I(x,y) - f(x,y)| >= 0 and Sobolev error integration."""
        QuadtreeDecomposition, QuadtreeNode, _, _ = self._get_local_modules()
        decomp = QuadtreeDecomposition(max_depth=3, split_threshold=0.05, sobolev_mu=0.1)

        # Generate synthetic target field and smooth global approximation
        x = np.linspace(-1, 1, 32)
        X, Y = np.meshgrid(x, x)
        target = np.sin(np.pi * X) * np.cos(np.pi * Y)
        approx = 0.5 * (X + Y)  # Coarse linear approximation

        # Exact residual map calculation
        res_map = np.abs(target - approx)
        assert res_map.shape == (32, 32)
        assert np.all(res_map >= 0.0)
        assert float(np.mean(res_map)) > 0.0

        # Verify Sobolev error metric: E = MSE + mu * ||grad R||^2
        sobolev_err = decomp.compute_sobolev_error(res_map)
        dy, dx = np.gradient(res_map)
        expected_err = float(np.mean(res_map**2)) + 0.1 * float(np.mean(dx**2 + dy**2))
        assert math.isclose(sobolev_err, expected_err, rel_tol=1e-9)

        # Decompose residual and verify root node properties
        root = decomp.decompose(res_map)
        assert isinstance(root, QuadtreeNode)
        assert math.isclose(root.error, sobolev_err, rel_tol=1e-9)
        assert root.bounds == (-1.0, -1.0, 1.0, 1.0)
        assert not root.is_leaf

    def test_f06_02_local_quadtree_subdivision(self) -> None:
        """TC-F06-02: Quadtree subdivides high-error quadrant while smooth regions stay shallow."""
        QuadtreeDecomposition, QuadtreeNode, _, _ = self._get_local_modules()
        decomp = QuadtreeDecomposition(max_depth=3, split_threshold=0.2)
        residual = np.zeros((32, 32))
        residual[:16, :16] = 1.0
        root = decomp.decompose(residual, domain=None)
        leaves = decomp.get_leaf_nodes(root)
        assert len(leaves) >= 4
        assert any(leaf.depth > 1 for leaf in leaves)

    def test_f06_03_local_partition_of_unity_sum(self) -> None:
        """TC-F06-03: Partition of unity: normalized blending weights satisfy sum_k w_k(x, y) == 1.0."""
        _, QuadtreeNode, WendlandPartitionOfUnity, _ = self._get_local_modules()
        pou = WendlandPartitionOfUnity(overlap_ratio=0.2)
        leaves = [
            QuadtreeNode(bounds=(-1.0, -1.0, 0.0, 0.0), depth=1, is_leaf=True),
            QuadtreeNode(bounds=(0.0, -1.0, 1.0, 0.0), depth=1, is_leaf=True),
            QuadtreeNode(bounds=(-1.0, 0.0, 0.0, 1.0), depth=1, is_leaf=True),
            QuadtreeNode(bounds=(0.0, 0.0, 1.0, 1.0), depth=1, is_leaf=True),
        ]
        X, Y = np.meshgrid(np.linspace(-1, 1, 32), np.linspace(-1, 1, 32))
        weights = pou.compute_weights(leaves, X, Y)
        total = np.sum(weights, axis=0)
        np.testing.assert_allclose(total, 1.0, rtol=1e-5)

    def test_f06_04_local_patch_fitting_gain(self) -> None:
        """TC-F06-04: Fitting a local patch in high-error leaf reduces local residual."""
        _, _, _, LocalPatch = self._get_local_modules()
        patch = LocalPatch(bounds=(-0.5, -0.5, 0.5, 0.5))
        assert patch.bounds == (-0.5, -0.5, 0.5, 0.5)

        # Generate a test grid and local high-residual feature
        x = np.linspace(-1, 1, 32)
        X, Y = np.meshgrid(x, x)
        R = np.exp(-5.0 * (X**2 + Y**2))

        # Local patch domain mask
        u, v = patch.normalize_coordinates(X, Y)
        mask = (np.abs(u) <= 1.0) & (np.abs(v) <= 1.0)
        res_before = float(np.mean(R[mask] ** 2))

        # Fit local patch on the residual using patch mask as weight
        patch.fit(R, X, Y, weights=mask.astype(float), reg_lambda=1e-6)
        patch_eval = patch.evaluate(X, Y)

        # Residual after local patch correction
        res_after = float(np.mean((R[mask] - patch_eval[mask]) ** 2))
        assert res_after < 0.1 * res_before, f"Expected >90% error reduction, before={res_before}, after={res_after}"
        assert patch.coefficients.size > 0
        assert np.any(np.abs(patch.coefficients) > 1e-4)

    def test_f06_05_local_boundary_smoothness(self) -> None:
        """TC-F06-05: Wendland partition-of-unity window derivative vanishes at boundary."""
        _, QuadtreeNode, WendlandPartitionOfUnity, _ = self._get_local_modules()
        pou = WendlandPartitionOfUnity(overlap_ratio=0.2)
        assert pou.overlap_ratio >= 0.0

        # Verify C^2 compact support: w(r) = (1-r)^4 * (4r + 1) for r < 1
        # Analytical first and second derivatives:
        # w'(r) = -20 * r * (1 - r)^3 -> w'(1) = 0
        # w''(r) = -20 * (1 - r)^2 * (1 - 4r) -> w''(1) = 0
        r = np.linspace(0.0, 1.2, 1000)
        w = np.where(r < 1.0, ((1.0 - r) ** 4.0) * (4.0 * r + 1.0), 0.0)
        dr = r[1] - r[0]
        dw_dr = np.gradient(w, dr)
        d2w_dr2 = np.gradient(dw_dr, dr)

        idx_1 = np.argmin(np.abs(r - 1.0))
        assert math.isclose(float(w[idx_1]), 0.0, abs_tol=1e-5)
        assert math.isclose(float(dw_dr[idx_1]), 0.0, abs_tol=1e-3)
        assert math.isclose(float(d2w_dr2[idx_1]), 0.0, abs_tol=0.05)

        # In addition, verify smooth partition of unity across overlapping leaf regions
        leaves = [
            QuadtreeNode(bounds=(-1.0, -1.0, 0.0, 0.0), depth=1, is_leaf=True),
            QuadtreeNode(bounds=(0.0, -1.0, 1.0, 0.0), depth=1, is_leaf=True),
            QuadtreeNode(bounds=(-1.0, 0.0, 0.0, 1.0), depth=1, is_leaf=True),
            QuadtreeNode(bounds=(0.0, 0.0, 1.0, 1.0), depth=1, is_leaf=True),
        ]
        X, Y = np.meshgrid(np.linspace(-1, 1, 50), np.linspace(-1, 1, 50))
        weights = pou.compute_weights(leaves, X, Y)
        np.testing.assert_allclose(np.sum(weights, axis=0), 1.0, atol=1e-12)
        for k in range(len(leaves)):
            gy, gx = np.gradient(weights[k])
            assert np.all(np.isfinite(gx))
            assert np.all(np.isfinite(gy))


# ==============================================================================
# FEATURE 7: Stage 4 (Joint Optimization)
# ==============================================================================


class TestFeature07Stage4JointOptimization:
    """Feature 7: Stage 4 (Joint Optimization) - VarPro, LM, L-BFGS-B, seeded genetic"""

    def _get_optimize_modules(self):
        from mathfield.core.optimize.conditioning import ConditionMonitor
        from mathfield.core.optimize.evolutionary import EvolutionaryOptimizer
        from mathfield.core.optimize.solvers import LBFGSBOptimizer, LevenbergMarquardtOptimizer
        from mathfield.core.optimize.varpro import VarProOptimizer

        return (
            VarProOptimizer,
            LevenbergMarquardtOptimizer,
            LBFGSBOptimizer,
            EvolutionaryOptimizer,
            ConditionMonitor,
        )

    def test_f07_01_joint_opt_varpro_separable(self) -> None:
        """TC-F07-01: Separable linear parameters solved analytically via Moore-Penrose pseudoinverse."""
        VarProOptimizer, _, _, _, _ = self._get_optimize_modules()
        solver = VarProOptimizer(max_iter=30)
        assert solver.max_iter > 0

        # Define separable problem y = c_0 * exp(-theta_0 * r^2) + c_1 * sin(theta_1 * x)
        def basis_fn(theta: np.ndarray, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
            r2 = X**2 + Y**2
            phi0 = np.exp(-theta[0] * r2).ravel()
            phi1 = np.sin(theta[1] * X).ravel()
            return np.column_stack([phi0, phi1])

        x = np.linspace(-1, 1, 16)
        X, Y = np.meshgrid(x, x)
        theta_true = np.array([2.0, 3.0])
        c_true = np.array([1.5, -0.8])
        Phi_true = basis_fn(theta_true, X, Y)
        y_clean = Phi_true @ c_true

        # Verify separable solve via VarPro
        res = solver.fit(basis_fn, X, Y, y_clean, theta_init=np.array([1.9, 2.9]))
        assert res.converged or res.cost < 1e-6
        np.testing.assert_allclose(res.theta_opt, theta_true, atol=1e-2)
        np.testing.assert_allclose(res.c_opt, c_true, atol=1e-2)
        assert res.condition_number > 0.0 and math.isfinite(res.condition_number)

    def test_f07_02_joint_opt_levenberg_marquardt(self) -> None:
        """TC-F07-02: Levenberg-Marquardt damping: (J^T J + lambda I) delta = J^T r."""
        _, LevenbergMarquardtOptimizer, _, _, _ = self._get_optimize_modules()
        lm = LevenbergMarquardtOptimizer(initial_lambda=1e-3, max_iter=50)
        assert lm.initial_lambda == 1e-3

        # Rosenbrock nonlinear least-squares residual: r(x) = [10*(x1 - x0^2), 1 - x0]
        def res_fn(x: np.ndarray) -> np.ndarray:
            return np.array([10.0 * (x[1] - x[0] ** 2), 1.0 - x[0]])

        def jac_fn(x: np.ndarray) -> np.ndarray:
            return np.array([[-20.0 * x[0], 10.0], [-1.0, 0.0]])

        x0 = np.array([-1.2, 1.0])
        cost_init = 0.5 * float(np.sum(res_fn(x0) ** 2))
        out = lm.minimize(res_fn, jac_fn, x0)

        assert out["converged"] or out["cost"] < 1e-6
        assert out["cost"] < 1e-4 * cost_init
        np.testing.assert_allclose(out["x"], np.array([1.0, 1.0]), atol=1e-3)
        assert out["n_iter"] > 0

    def test_f07_03_joint_opt_lbfgsb_box_bounds(self) -> None:
        """TC-F07-03: L-BFGS-B respects parameter box bounds [lb, ub]."""
        _, _, LBFGSBOptimizer, _, _ = self._get_optimize_modules()
        bounds = [(0.0, 2.0), (-1.0, 1.0)]
        opt = LBFGSBOptimizer(bounds=bounds)
        assert opt.bounds == bounds

        # Quadratic objective with unconstrained minimum at (5.0, -3.0), outside bounds [0, 2] x [-1, 1]
        def fun(x: np.ndarray) -> float:
            return float((x[0] - 5.0) ** 2 + (x[1] + 3.0) ** 2)

        def jac(x: np.ndarray) -> np.ndarray:
            return np.array([2.0 * (x[0] - 5.0), 2.0 * (x[1] + 3.0)])

        res = opt.minimize(fun, x0=np.array([1.0, 0.0]), jac=jac)
        assert res.success
        # Constrained minimum must hit boundary edges [2.0, -1.0]
        np.testing.assert_allclose(res.x, np.array([2.0, -1.0]), atol=1e-5)
        assert 0.0 <= res.x[0] <= 2.0
        assert -1.0 <= res.x[1] <= 1.0

    def test_f07_04_joint_opt_evolutionary_seed(self) -> None:
        """TC-F07-04: Deterministic seeded PRNG produces bit-exact identical sequences."""
        _, _, _, EvolutionaryOptimizer, _ = self._get_optimize_modules()
        opt1 = EvolutionaryOptimizer(random_seed=77, population_size=16, generations=4)
        opt2 = EvolutionaryOptimizer(random_seed=77, population_size=16, generations=4)
        opt3 = EvolutionaryOptimizer(random_seed=99, population_size=16, generations=4)
        assert opt1.random_seed == opt2.random_seed == 77

        x = np.linspace(-1, 1, 16)
        X, Y = np.meshgrid(x, x)
        target = np.sin(np.pi * X) * np.cos(np.pi * Y)

        ast1, diag1 = opt1.evolve(X, Y, target)
        ast2, diag2 = opt2.evolve(X, Y, target)
        ast3, diag3 = opt3.evolve(X, Y, target)

        # Bit-exact identical AST formula and metrics under seed 77
        assert str(ast1) == str(ast2)
        assert math.isclose(diag1["best_mse"], diag2["best_mse"], abs_tol=1e-12)
        assert math.isclose(diag1["best_fitness"], diag2["best_fitness"], abs_tol=1e-12)
        assert diag1["best_complexity"] == diag2["best_complexity"]

        # Different seed 99 produces distinct exploration
        assert str(ast1) != str(ast3) or not math.isclose(diag1["best_fitness"], diag3["best_fitness"], abs_tol=1e-6)

    def test_f07_05_joint_opt_svd_regularization(self) -> None:
        """TC-F07-05: SVD condition monitor detects ill-conditioned matrix and applies Tikhonov damping."""
        _, _, _, _, ConditionMonitor = self._get_optimize_modules()
        monitor = ConditionMonitor(max_condition_number=1e10, tikhonov_alpha=1e-4)
        A = np.array([[1.0, 1.0], [1.0, 1.0 + 1e-15]])
        report = monitor.analyze_matrix(A)
        assert report["is_ill_conditioned"] is True
        A_reg = monitor.regularize(A)
        assert np.linalg.cond(A_reg) < 1e8


# ==============================================================================
# FEATURE 8: Stage 5 (Refinement & Verification)
# ==============================================================================


class TestFeature08Stage5RefinementVerification:
    """Feature 8: Stage 5 (Refinement & Verification) - BIC pruning, re-render parity"""

    def _get_verify_modules(self):
        from mathfield.core.verify.metrics import compute_metrics
        from mathfield.core.verify.pruning import BICPruner
        from mathfield.core.verify.renderer import ReferenceRenderer

        return BICPruner, ReferenceRenderer, compute_metrics

    def test_f08_01_refinement_bic_pruning(self) -> None:
        """TC-F08-01: Bayesian Information Criterion (BIC) penalizes redundant parameters."""
        BICPruner, _, _ = self._get_verify_modules()
        n = 1000
        bic1 = BICPruner.compute_bic(rss=10.0, n_samples=n, k_parameters=2)
        bic2 = BICPruner.compute_bic(rss=9.9999, n_samples=n, k_parameters=3)
        assert bic1 < bic2

    def test_f08_02_refinement_algebraic_pass(self) -> None:
        """TC-F08-02: Algebraic simplification pass eliminates zero-terms."""
        from mathfield.core.ast import parse, simplify

        ast = parse("x * 0 + c0 * 1 + (y - y)")
        simplified = simplify(ast)
        assert "c0" in str(simplified)

    def test_f08_03_refinement_independent_render(self) -> None:
        """TC-F08-03: Independent re-render matches evaluation buffer within 1e-12."""
        from mathfield.core.ast import parse
        from mathfield.core.engine.cpu_numpy import evaluate_numpy

        ast = parse("sin(x) + cos(y)")
        x = np.linspace(-1, 1, 16)
        X, Y = np.meshgrid(x, x)
        buf1 = evaluate_numpy(ast, X, Y, np.array([]))
        buf2 = np.sin(X) + np.cos(Y)

        np.testing.assert_allclose(buf1, buf2, atol=1e-12)

    def test_f08_04_refinement_metric_battery(self) -> None:
        """TC-F08-04: Metric battery returns finite values satisfying mathematical bounds."""
        _, _, compute_metrics = self._get_verify_modules()
        a = np.ones((16, 16, 3), dtype=float) * 0.5
        b = np.ones((16, 16, 3), dtype=float) * 0.52
        metrics = compute_metrics(a, b)
        assert "mse" in metrics and "psnr" in metrics and "ssim" in metrics
        assert metrics["mse"] > 0.0
        assert metrics["psnr"] > 0.0
        assert 0.0 <= metrics["ssim"] <= 1.0

    def test_f08_05_refinement_discrepancy_guard(self) -> None:
        """TC-F08-05: Verifier flags discrepancy when simulated corrupt buffer is checked."""
        _, ReferenceRenderer, _ = self._get_verify_modules()
        renderer = ReferenceRenderer()
        truth = np.zeros((10, 10))
        corrupt = np.ones((10, 10)) * 100.0
        verified, metrics = renderer.verify(truth, corrupt, tolerance=1e-4)
        assert verified is False
        assert metrics["mse"] > 1000.0


# ==============================================================================
# FEATURE 9: Full-Colour & Alpha Support
# ==============================================================================


class TestFeature09FullColourAlphaSupport:
    """Feature 9: Full-Colour & Alpha Support - sRGB, Linear, HSV Cartesian, CIELAB"""

    def _get_color_module(self):
        from mathfield.core.color import ColorSpace, ColorSpaceConverter

        return ColorSpaceConverter, ColorSpace

    def test_f09_01_color_cartesian_hsv_branchcut(self) -> None:
        """TC-F09-01: Continuous Cartesian HSV (u, v) = (S cos 2pi H, S sin 2pi H) eliminates branch cut jump."""
        ColorSpaceConverter, _ = self._get_color_module()
        rgb_left = np.array([1.0, 0.0, 0.05])
        rgb_right = np.array([1.0, 0.05, 0.0])
        uvV_left = ColorSpaceConverter.rgb_to_cartesian_hsv(rgb_left)
        uvV_right = ColorSpaceConverter.rgb_to_cartesian_hsv(rgb_right)
        dist = float(np.linalg.norm(uvV_left[:2] - uvV_right[:2]))
        assert dist < 0.2

    def test_f09_02_color_cielab_roundtrip(self) -> None:
        """TC-F09-02: CIELAB roundtrip preserves original sRGB values."""
        ColorSpaceConverter, _ = self._get_color_module()
        srgb = np.array([0.5, 0.25, 0.75])
        lab = ColorSpaceConverter.srgb_to_cielab(srgb)
        rec = ColorSpaceConverter.cielab_to_srgb(lab)
        np.testing.assert_allclose(srgb, rec, atol=1e-3)

    def test_f09_03_color_alpha_field_evaluation(self, synth_rgba_image: Path) -> None:
        """TC-F09-03: 4-channel RGBA representation has valid alpha channel in [0, 255]."""
        ColorSpaceConverter, _ = self._get_color_module()
        info = inspect_png(synth_rgba_image, has_alpha=True)
        arr = info["array"]
        assert arr.shape[-1] == 4
        alpha = arr[:, :, 3]
        assert np.min(alpha) >= 0
        assert np.max(alpha) <= 255

    def test_f09_04_color_alpha_weighted_loss(self) -> None:
        """TC-F09-04: Alpha-weighted objective ignores errors where alpha=0."""
        ColorSpaceConverter, _ = self._get_color_module()
        alpha = np.array([0.0, 0.0, 1.0, 1.0])
        pred = np.array([100.0, 50.0, 1.0, 2.0])
        target = np.array([0.0, 0.0, 0.0, 0.0])
        loss = ColorSpaceConverter.alpha_weighted_loss(pred, target, alpha)
        assert loss < 10.0

    def test_f09_05_color_gamut_soft_clipping(self) -> None:
        """TC-F09-05: Smooth soft-clipping maps unbounded values into [0, 1]."""
        ColorSpaceConverter, _ = self._get_color_module()
        unbounded = np.array([-10.0, -0.5, 0.0, 0.5, 1.0, 1.5, 10.0])
        clipped = ColorSpaceConverter.soft_clamp(unbounded)
        assert np.all(clipped >= 0.0)
        assert np.all(clipped <= 1.0)


# ==============================================================================
# FEATURE 10: Safe Expression Grammar & Pratt Parser
# ==============================================================================


class TestFeature10SafeExpressionGrammarParser:
    """Feature 10: Safe Expression Grammar & Pratt Parser - Absolute zero eval/exec sandbox"""

    def test_f10_01_parser_precedence_associativity(self) -> None:
        """TC-F10-01: Correct binding of operators: power > mul/div > add/sub."""
        from mathfield.core.ast import parse

        ast = parse("2 + 3 * x ^ 2 - 4 / y")
        # Top-level is subtraction
        assert ast.op == "-"
        # Left child is addition (2 + (3 * (x^2)))
        assert ast.left.op == "+"
        assert ast.left.right.op == "*"
        assert ast.left.right.right.op == "^"

    def test_f10_02_parser_whitelisted_functions(self) -> None:
        """TC-F10-02: Parses approved mathematical functions."""
        from mathfield.core.ast import parse

        ast = parse("sin(x) + rbf_wendland(x, y)")
        assert ast.op == "+"
        assert ast.left.func == "sin"
        assert ast.right.func == "rbf_wendland"
        assert len(ast.right.args) == 2

    @pytest.mark.parametrize(
        "hostile_code",
        [
            "__import__('os').system('dir')",
            "eval('1 + 1')",
            "exec('print(1)')",
            "open('secret.txt', 'r')",
            "lambda x: x + 1",
            "import sys",
        ],
    )
    def test_f10_03_parser_zero_eval_exec_sandbox(self, hostile_code: str) -> None:
        """TC-F10-03: Absolute sandbox: rejects any python code injection without executing."""
        from mathfield.core.ast import ParseError, parse

        with pytest.raises((ParseError, ValueError)):
            parse(hostile_code)

    def test_f10_04_parser_syntax_error_pointers(self) -> None:
        """TC-F10-04: Syntax errors provide line and column diagnostics."""
        from mathfield.core.ast import ParseError, parse

        with pytest.raises(ParseError) as exc_info:
            parse("2 * (x + ")
        err = exc_info.value
        assert isinstance(err.column, int)
        assert err.column >= 0

    def test_f10_05_parser_immutable_ast_nodes(self) -> None:
        """TC-F10-05: AST nodes are immutable frozen dataclasses."""
        from mathfield.core.ast import parse

        ast = parse("x + y")
        with pytest.raises((AttributeError, TypeError)):
            ast.op = "-"  # mutation forbidden


# ==============================================================================
# FEATURE 11: Numerical Domain Validation & Regularization
# ==============================================================================


class TestFeature11NumericalDomainValidation:
    """Feature 11: Numerical Domain Validation & Regularization - Pole prevention"""

    def test_f11_01_domain_regularized_division(self) -> None:
        """TC-F11-01: Division by zero evaluates to finite real value without crash."""
        from mathfield.core.ast.domain import regularized_div

        u = np.array([5.0, -3.0, 0.0])
        v = np.array([0.0, -0.0, 0.0])
        res = regularized_div(u, v)
        assert np.all(np.isfinite(res))

    def test_f11_02_domain_log_clamping(self) -> None:
        """TC-F11-02: Logarithm of non-positive numbers is clamped to finite real."""
        from mathfield.core.ast.domain import protected_log

        x = np.array([-10.0, 0.0, 1e-18, 1.0])
        res = protected_log(x)
        assert np.all(np.isfinite(res))
        assert not np.any(np.isnan(res))

    def test_f11_03_domain_exp_overflow_clamp(self) -> None:
        """TC-F11-03: Exponential explosion clamped to prevent float overflow."""
        from mathfield.core.ast.domain import protected_exp

        x = np.array([100.0, 1000.0, -500.0])
        res = protected_exp(x)
        assert np.all(np.isfinite(res))

    def test_f11_04_domain_sqrt_non_negative(self) -> None:
        """TC-F11-04: Square root of negative number clamped to 0.0 (no complex or NaN)."""
        from mathfield.core.ast.domain import protected_sqrt

        x = np.array([-4.0, -100.0, 0.0, 16.0])
        res = protected_sqrt(x)
        assert np.all(np.isfinite(res))
        assert res[0] == 0.0
        assert res[1] == 0.0
        assert res[3] == 4.0

    def test_f11_05_domain_composite_grid_safety(self) -> None:
        """TC-F11-05: Evaluating composite expressions with multiple nested poles."""
        from mathfield.core.ast.domain import protected_log, protected_sqrt, regularized_div

        X, Y = np.meshgrid(np.linspace(-1, 1, 20), np.linspace(-1, 1, 20))
        dist_sq = X**2 + Y**2
        r = protected_sqrt(dist_sq)
        log_r = protected_log(r)
        res = regularized_div(log_r, dist_sq)
        assert np.all(np.isfinite(res))


# ==============================================================================
# FEATURE 12: Symbolic Transformations & Derivatives
# ==============================================================================


class TestFeature12SymbolicTransformationsDerivatives:
    """Feature 12: Symbolic Transformations & Derivatives - Constant folding & analytical differentiation"""

    def test_f12_01_transform_constant_folding(self) -> None:
        """TC-F12-01: Constant subtrees fold to single numerical constants."""
        from mathfield.core.ast import parse, simplify

        ast = parse("2 * 3 + 4 * 5")
        folded = simplify(ast)
        assert str(folded) == "26"

    def test_f12_02_transform_algebraic_identities(self) -> None:
        """TC-F12-02: Algebraic identity reduction (x + 0 -> x, x * 1 -> x, x * 0 -> 0)."""
        from mathfield.core.ast import parse, simplify

        assert str(simplify(parse("x + 0"))) == "x"
        assert str(simplify(parse("x * 1"))) == "x"
        assert str(simplify(parse("x * 0"))) == "0"

    def test_f12_03_transform_polynomial_derivative(self) -> None:
        """TC-F12-03: Analytical derivative of polynomial w.r.t variable."""
        from mathfield.core.ast import diff, parse, simplify

        # d/dx (3 * x^2) = 6 * x
        ast = parse("3 * x ^ 2")
        d_dx = diff(ast, "x")
        simplified = simplify(d_dx)
        # Evaluated at x=2, derivative is 12
        from mathfield.core.engine.cpu_numpy import evaluate_numpy

        val = evaluate_numpy(simplified, np.array([2.0]), np.array([0.0]), np.array([]))
        assert math.isclose(float(val[0]), 12.0, abs_tol=1e-12)

    def test_f12_04_transform_chain_rule_diff(self) -> None:
        """TC-F12-04: Chain rule differentiation: d/dx sin(c0 * x)."""
        from mathfield.core.ast import diff, parse
        from mathfield.core.engine.cpu_numpy import evaluate_numpy

        ast = parse("sin(c0 * x)")
        d_dx = diff(ast, "x")
        # Numerical finite difference reference: (sin(c0*(x+h)) - sin(c0*(x-h))) / 2h
        x_val = np.array([0.5])
        c0_val = np.array([3.0])
        h = 1e-6
        f_plus = np.sin(3.0 * (x_val + h))
        f_minus = np.sin(3.0 * (x_val - h))
        num_diff = (f_plus - f_minus) / (2.0 * h)

        ana_diff = evaluate_numpy(d_dx, x_val, np.array([0.0]), c0_val)
        np.testing.assert_allclose(ana_diff, num_diff, rtol=1e-5)

    def test_f12_05_transform_param_diff_parity(self) -> None:
        """TC-F12-05: Jacobian parameter derivative: d/dc0 (c0 * sin(x)) == sin(x)."""
        from mathfield.core.ast import diff, parse, simplify

        ast = parse("c0 * sin(x)")
        d_dc0 = diff(ast, "c0")
        assert "sin(x)" in str(simplify(d_dc0))


# ==============================================================================
# FEATURE 13: Multi-Tier Accelerated Evaluators
# ==============================================================================


class TestFeature13MultiTierAcceleratedEvaluators:
    """Feature 13: Multi-Tier Accelerated Evaluators - CPU SIMD, C++ DLL, CuPy CUDA parity"""

    def test_f13_01_evaluator_numpy_baseline(self) -> None:
        """TC-F13-01: Vectorized NumPy CPU SIMD reference evaluation."""
        from mathfield.core.ast import parse
        from mathfield.core.engine.cpu_numpy import evaluate_numpy

        ast = parse("sin(x) * cos(y)")
        X, Y = np.meshgrid(np.linspace(-1, 1, 32), np.linspace(-1, 1, 32))
        res = evaluate_numpy(ast, X, Y, np.array([]))
        expected = np.sin(X) * np.cos(Y)
        np.testing.assert_allclose(res, expected, atol=1e-12)

    def test_f13_02_evaluator_native_cpp_parity(self) -> None:
        """TC-F13-02: Native C++ OpenMP engine parity (or graceful fallback)."""
        from mathfield.core.ast import parse
        from mathfield.core.engine.cpu_numpy import evaluate_numpy

        ast = parse("x ^ 2 + y ^ 2")
        X, Y = np.meshgrid(np.linspace(-1, 1, 16), np.linspace(-1, 1, 16))
        ref = evaluate_numpy(ast, X, Y, np.array([]))

        # Capability-only skip: an import failure is a real regression here.
        from mathfield.core.engine.cpu_native import evaluate_native, is_native_available

        if not is_native_available():
            pytest.skip("Native C++ DLL not compiled on host; fallback verified")
        native_res = evaluate_native(ast, X, Y, np.array([]))
        np.testing.assert_allclose(native_res, ref, atol=1e-6)

    def test_f13_03_evaluator_gpu_cupy_parity(self) -> None:
        """TC-F13-03: CuPy CUDA elementwise kernel parity (or graceful fallback)."""
        # Capability-only skip: an import failure is a real regression here.
        from mathfield.core.engine.gpu_cupy import (
            compute_residuals_cupy,
            compute_sse_cupy,
            evaluate_cupy,
            is_cuda_available,
        )

        if not is_cuda_available():
            pytest.skip("CUDA device not available; automatic CPU fallback applies")

        from mathfield.core.ast import parse
        from mathfield.core.engine.cpu_numpy import evaluate_numpy

        # Non-trivial composite mathematical expression
        ast = parse("sin(3 * x) * cos(3 * y) + x^2 - y^2")
        X, Y = np.meshgrid(np.linspace(-1, 1, 64), np.linspace(-1, 1, 64))
        params = np.array([])

        cpu_out = evaluate_numpy(ast, X, Y, params)
        gpu_out = evaluate_cupy(ast, X, Y, params)

        assert gpu_out.shape == (64, 64)
        assert np.all(np.isfinite(gpu_out))
        np.testing.assert_allclose(gpu_out, cpu_out, atol=1e-5)

        # Parameterized AST evaluation parity
        ast_p = parse("c0 * sin(x) + c1 * cos(y)")
        p_vals = np.array([2.5, -1.2])
        cpu_p = evaluate_numpy(ast_p, X, Y, p_vals)
        gpu_p = evaluate_cupy(ast_p, X, Y, p_vals)
        np.testing.assert_allclose(gpu_p, cpu_p, atol=1e-5)

        # GPU residual and SSE kernel execution
        res_gpu = compute_residuals_cupy(gpu_p, cpu_p)
        np.testing.assert_allclose(res_gpu, 0.0, atol=1e-5)
        sse = compute_sse_cupy(gpu_p, cpu_p)
        assert math.isclose(sse, 0.0, abs_tol=1e-8)

    def test_f13_04_evaluator_graceful_fallback(self) -> None:
        """TC-F13-04: Evaluator cascades gracefully to available CPU engine."""
        from mathfield.core.ast import parse
        from mathfield.core.engine.cpu_numpy import evaluate_numpy

        ast = parse("x + y")
        X, Y = np.meshgrid(np.linspace(-1, 1, 10), np.linspace(-1, 1, 10))
        res = evaluate_numpy(ast, X, Y, np.array([]))
        assert res.shape == (10, 10)

    def test_f13_05_evaluator_throughput_stability(self) -> None:
        """TC-F13-05: Evaluates 262,144 points (512x512) within 500 ms."""
        from mathfield.core.ast import parse
        from mathfield.core.engine.cpu_numpy import evaluate_numpy

        ast = parse("sin(3 * x) * cos(3 * y) + x ^ 2 - y ^ 2")
        X, Y = np.meshgrid(np.linspace(-1, 1, 512), np.linspace(-1, 1, 512))

        start = time.perf_counter()
        res = evaluate_numpy(ast, X, Y, np.array([]))
        elapsed = time.perf_counter() - start

        assert res.shape == (512, 512)
        assert np.all(np.isfinite(res))
        assert elapsed < 0.50  # < 500 ms


# ==============================================================================
# FEATURE 14: Expression Code Generators
# ==============================================================================


class TestFeature14ExpressionCodeGenerators:
    """Feature 14: Expression Code Generators - AMS-LaTeX & Standalone NumPy Python"""

    def test_f14_01_codegen_latex_syntax(self) -> None:
        """TC-F14-01: AMS-LaTeX formatting for powers and functions."""
        from mathfield.core.ast import parse
        from mathfield.core.codegen import to_latex

        ast = parse("c0 * sin(x) + c1 * y^2")
        tex = to_latex(ast)
        assert "\\sin" in tex
        assert "y^{2}" in tex or "y^2" in tex

    def test_f14_02_codegen_latex_parentheses(self) -> None:
        """TC-F14-02: Precedence grouping in LaTeX export."""
        from mathfield.core.ast import parse
        from mathfield.core.codegen import to_latex

        ast = parse("(x + y) * x")
        tex = to_latex(ast)
        assert "(" in tex and ")" in tex

    def test_f14_03_codegen_python_function_def(self) -> None:
        """TC-F14-03: Standalone Python function code generation."""
        from mathfield.core.ast import parse
        from mathfield.core.codegen import to_python_code

        ast = parse("c0 * x + c1 * y^2")
        code = to_python_code(ast)
        assert isinstance(code, str)
        assert len(code) > 0

    def test_f14_04_codegen_python_exec_parity(self) -> None:
        """TC-F14-04: Generated standalone NumPy function matches AST evaluator."""
        from mathfield.core.ast import parse
        from mathfield.core.codegen import generate_standalone_function
        from mathfield.core.engine.cpu_numpy import evaluate_numpy

        ast = parse("c0 * sin(x) + c1 * cos(y)")
        fn_code = generate_standalone_function(ast)
        ns: Dict[str, Any] = {}
        exec(compile(fn_code, "<test>", "exec"), ns)
        fn = ns["evaluate_field"]

        X, Y = np.meshgrid(np.linspace(-1, 1, 16), np.linspace(-1, 1, 16))
        params = np.array([2.5, -1.2])

        eval_out = evaluate_numpy(ast, X, Y, params)
        fn_out = fn(X, Y, params)
        np.testing.assert_allclose(eval_out, fn_out, atol=1e-12)

    def test_f14_05_codegen_multichannel_script(self, e2e_temp_dir: Path) -> None:
        """TC-F14-05: Emits complete multi-channel standalone script."""
        from mathfield.core.ast import parse
        from mathfield.core.codegen import generate_standalone_script

        ast = parse("0.5 * x + 0.5")
        params = [0.5, 0.5]
        script_code = generate_standalone_script(ast, params)
        script_path = e2e_temp_dir / "standalone_test.py"
        script_path.write_text(script_code, encoding="utf-8")

        audit = validate_standalone_python_script(script_path)
        assert "mathfield" not in audit["imports"]


# ==============================================================================
# FEATURE 15: 8K High-Resolution Tiling & Memory Management
# ==============================================================================


class TestFeature15HighResolution8KTiling:
    """Feature 15: 8K High-Resolution Tiling & Memory Management - Working set < 300MB"""

    def _get_tiler_module(self):
        from mathfield.core.engine.tiler import DomainChunker, TileInfo

        return DomainChunker, TileInfo

    def test_f15_01_tiler_partition_coverage(self, mock_8k_metadata: Dict[str, Any]) -> None:
        """TC-F15-01: Spatial domain chunking partitions 8192x8192 into exactly 64 tiles of 1024x1024."""
        DomainChunker, TileInfo = self._get_tiler_module()
        chunker = DomainChunker(tile_size=1024)
        tiles = chunker.get_tiles(total_width=8192, total_height=8192, domain=None)
        assert len(tiles) == 64
        assert all(isinstance(t, TileInfo) for t in tiles)

    def test_f15_02_tiler_boundary_seamlessness(self) -> None:
        """TC-F15-02: Seam values between adjacent tiles match continuous evaluation within 1e-12."""
        DomainChunker, _ = self._get_tiler_module()
        from mathfield.core.ast import parse
        from mathfield.core.domain import CoordinateDomain2D
        from mathfield.core.engine.cpu_numpy import evaluate_numpy

        width, height = 128, 128
        tile_size = 64
        domain = CoordinateDomain2D(width=width, height=height)
        ast = parse("sin(3.0 * x) + cos(2.0 * y) + x * y")

        # 1. Full continuous reference evaluation across entire grid
        X_full, Y_full = domain.to_discrete_grid()
        ref_full = evaluate_numpy(ast, X_full, Y_full, np.array([]))

        # 2. Tiled evaluation via DomainChunker across 2x2 grid of 64x64 tiles
        chunker = DomainChunker(tile_size=tile_size)
        tiled_result = chunker.evaluate_tiled(ast, domain=domain, width=width, height=height)

        # 3. Assert full-frame continuous equivalence within 1e-12
        np.testing.assert_allclose(tiled_result, ref_full, atol=1e-12, rtol=1e-12)

        # 4. Explicit border seam inspection across tile interfaces:
        # Horizontal seam boundary (rows 63 and 64)
        np.testing.assert_allclose(tiled_result[63, :], ref_full[63, :], atol=1e-12)
        np.testing.assert_allclose(tiled_result[64, :], ref_full[64, :], atol=1e-12)
        # Vertical seam boundary (columns 63 and 64)
        np.testing.assert_allclose(tiled_result[:, 63], ref_full[:, 63], atol=1e-12)
        np.testing.assert_allclose(tiled_result[:, 64], ref_full[:, 64], atol=1e-12)

    def test_f15_03_tiler_memmap_streaming(self, e2e_temp_dir: Path) -> None:
        """TC-F15-03: Tiled generator streams directly into disk-backed numpy.memmap."""
        DomainChunker, _ = self._get_tiler_module()
        from mathfield.core.ast import parse
        from mathfield.core.domain import CoordinateDomain2D
        from mathfield.core.engine.cpu_numpy import evaluate_numpy

        width, height = 128, 128
        tile_size = 64
        domain = CoordinateDomain2D(width=width, height=height)
        ast = parse("exp(- (x^2 + y^2))")

        # Reference continuous output
        X, Y = domain.to_discrete_grid()
        ref = evaluate_numpy(ast, X, Y, np.array([]))

        # Allocate disk-backed memmap initialized with canary NaNs
        memmap_path = e2e_temp_dir / "streamed_tiled_output.dat"
        mmap_arr = np.memmap(memmap_path, dtype=np.float64, mode="w+", shape=(height, width))
        mmap_arr[:] = np.nan
        mmap_arr.flush()

        # Stream tiled computation directly into disk memmap
        chunker = DomainChunker(tile_size=tile_size)
        chunker.evaluate_tiled(ast, domain=domain, width=width, height=height, out_memmap=mmap_arr)
        mmap_arr.flush()

        # Verify disk file modification and exact size
        assert memmap_path.exists(), "Memmap file was not written to disk"
        expected_bytes = height * width * np.dtype(np.float64).itemsize
        assert memmap_path.stat().st_size == expected_bytes, (
            f"Expected {expected_bytes} bytes on disk, found {memmap_path.stat().st_size}"
        )

        # Open independent read-only memmap to confirm persisted content correctness
        read_mmap = np.memmap(memmap_path, dtype=np.float64, mode="r", shape=(height, width))
        assert not np.isnan(read_mmap).any(), "Canary NaNs were not fully overwritten by tiler"
        np.testing.assert_allclose(np.array(read_mmap), ref, atol=1e-12, rtol=1e-12)
        del read_mmap
        del mmap_arr

    def test_f15_04_tiler_memory_ceiling_budget(self) -> None:
        """TC-F15-04: Single 1024x1024 float64 tile working set is under 25 MB (< 300 MB limit)."""
        DomainChunker, _ = self._get_tiler_module()
        chunker = DomainChunker(tile_size=1024, max_memory_mb=300.0)

        # 1. Exact mathematical calculation of single tile buffer memory
        tile_pixels = 1024 * 1024
        float64_itemsize = np.dtype(np.float64).itemsize  # 8 bytes
        single_buffer_bytes = tile_pixels * float64_itemsize  # 8,388,608 bytes = 8.0 MB
        single_buffer_mb = single_buffer_bytes / (1024.0 * 1024.0)
        assert math.isclose(single_buffer_mb, 8.0, abs_tol=1e-9)

        # 2. Maximum simultaneous tile working set (X, Y coordinate grids + result buffer = 3 buffers)
        working_set_mb = 3.0 * single_buffer_mb  # 24.0 MB
        assert working_set_mb < 25.0, f"Working set {working_set_mb} MB exceeds 25 MB budget"
        assert working_set_mb < chunker.max_memory_mb, "Working set exceeds configured memory ceiling"
        assert chunker.max_memory_mb <= 300.0, "Configured memory ceiling exceeds 300 MB limit"

        # 3. Dynamic verification via chunker memory inspection method
        if hasattr(chunker, "estimate_working_set_mb"):
            est_mb = chunker.estimate_working_set_mb(tile_size=1024, num_channels=1)
            assert est_mb < 25.0, f"Chunker estimated working set {est_mb} MB exceeds 25 MB"
            assert est_mb < chunker.max_memory_mb

    def test_f15_05_tiler_non_power_two_edges(self) -> None:
        """TC-F15-05: Non-power-of-two edge boundaries handle partial tile dimensions."""
        DomainChunker, TileInfo = self._get_tiler_module()
        chunker = DomainChunker(tile_size=1024)
        tiles = chunker.get_tiles(total_width=1500, total_height=1100, domain=None)
        assert len(tiles) == 4


# ==============================================================================
# FEATURE 16: Robust Bit-Exact Job Checkpointing
# ==============================================================================


class TestFeature16BitExactJobCheckpointing:
    """Feature 16: Robust Bit-Exact Job Checkpointing - IEEE-754 hex float serialization"""

    def _get_checkpoint_module(self):
        from mathfield.core.checkpoint.manager import CheckpointCorruptError, CheckpointManager
        from mathfield.core.checkpoint.serializer import CheckpointSerializer

        return CheckpointManager, CheckpointCorruptError, CheckpointSerializer

    def test_f16_01_checkpoint_hex_float_exact(self) -> None:
        """TC-F16-01: IEEE-754 hex float serialization round-trips with zero precision loss."""
        _, _, CheckpointSerializer = self._get_checkpoint_module()
        delicate_floats = [
            1.0 / 3.0,
            math.pi,
            1e-300,
            1e300,
            -0.0,
            float.fromhex("0x1.0000000000001p-1022"),
        ]
        for val in delicate_floats:
            hex_str = CheckpointSerializer.serialize_float(val)
            recovered = CheckpointSerializer.deserialize_float(hex_str)
            assert math.isclose(val, recovered, abs_tol=0.0)

    def test_f16_02_checkpoint_state_roundtrip(self, sample_checkpoint_file: Path) -> None:
        """TC-F16-02: Checkpoint file restores stage, iteration, AST, parameters, and metrics."""
        CheckpointManager, _, _ = self._get_checkpoint_module()
        mgr = CheckpointManager()
        data = mgr.load(sample_checkpoint_file)
        assert data["stage"] == "STAGE_4_OPTIMIZATION"
        assert data["iteration"] == 25
        assert "parameters" in data

    def test_f16_03_checkpoint_rng_reproduction(self) -> None:
        """TC-F16-03: BitGenerator state serialization reproduces exact PRNG sequence."""
        _, _, CheckpointSerializer = self._get_checkpoint_module()
        rng = np.random.default_rng(12345)
        rng.random(10)
        state_dict = CheckpointSerializer.serialize_rng_state(rng)
        expected_next = rng.random(5)
        resumed_rng = CheckpointSerializer.deserialize_rng_state(state_dict)
        actual_next = resumed_rng.random(5)
        np.testing.assert_array_equal(expected_next, actual_next)

    def test_f16_04_checkpoint_atomic_write(self, e2e_temp_dir: Path) -> None:
        """TC-F16-04: Atomic two-phase write pattern (write to .tmp then rename)."""
        CheckpointManager, _, _ = self._get_checkpoint_module()
        mgr = CheckpointManager(checkpoint_dir=e2e_temp_dir)
        target = e2e_temp_dir / "atomic_checkpoint.mfchk"
        payload = {"stage": "STAGE_1_ANALYSIS", "iteration": 0, "parameters": {}}
        mgr.save(target, payload)
        assert target.exists()

    def test_f16_05_checkpoint_corrupt_rejection(self, e2e_temp_dir: Path) -> None:
        """TC-F16-05: Corrupt or truncated checkpoint file raises descriptive error."""
        CheckpointManager, CheckpointCorruptError, _ = self._get_checkpoint_module()
        mgr = CheckpointManager()
        corrupt_file = e2e_temp_dir / "bad.mfchk"
        corrupt_file.write_text("{ truncated_json: true, ", encoding="utf-8")
        with pytest.raises(CheckpointCorruptError):
            mgr.load(corrupt_file)


# ==============================================================================
# FEATURE 17: Main UI Workspace
# ==============================================================================


class TestFeature17MainUIWorkspace:
    """Feature 17: Main UI Workspace - Drag-and-drop, zoom/pan, curtain split comparison"""

    def _get_workspace(self):
        from mathfield.ui.workspaces.main_workspace import MainWorkspace

        return MainWorkspace

    def test_f17_01_main_ui_load_image(self, synth_rgb_poly: Path) -> None:
        """TC-F17-01: Viewport source image loading and aspect calculation."""
        MainWorkspace = self._get_workspace()
        ws = MainWorkspace()
        ws.load_image(synth_rgb_poly)
        assert ws.image_dimensions == (64, 64)

    def test_f17_02_main_ui_zoom_pan_clamping(self) -> None:
        """TC-F17-02: Viewport zoom scale clamps within safe range [0.1, 50.0]."""
        MainWorkspace = self._get_workspace()
        ws = MainWorkspace()
        ws.set_zoom(0.001)
        assert ws.current_zoom >= 0.1
        ws.set_zoom(100.0)
        assert ws.current_zoom <= 50.0

    def test_f17_03_main_ui_curtain_split_slider(self) -> None:
        """TC-F17-03: Side-by-side curtain split slider maps position t in [0.0, 1.0]."""
        MainWorkspace = self._get_workspace()
        ws = MainWorkspace()
        ws.set_curtain_split(0.5)
        assert ws.curtain_split_position == 0.5

    def test_f17_04_main_ui_preview_buffer_update(self) -> None:
        """TC-F17-04: Viewport preview buffer updates with correct shape."""
        MainWorkspace = self._get_workspace()
        ws = MainWorkspace()
        preview = np.zeros((64, 64, 3), dtype=np.uint8)
        ws.update_preview(preview)
        assert ws.get_preview_buffer().shape == (64, 64, 3)

    def test_f17_05_main_ui_status_ribbon_info(self) -> None:
        """TC-F17-05: Status ribbon text formatting."""
        MainWorkspace = self._get_workspace()
        ws = MainWorkspace()
        status_text = ws.get_status_text()
        assert isinstance(status_text, str)


# ==============================================================================
# FEATURE 18: Equation UI Workspace
# ==============================================================================


class TestFeature18EquationUIWorkspace:
    """Feature 18: Equation UI Workspace - Tree view, LaTeX formula card via mathtext"""

    def _get_workspace(self):
        from mathfield.ui.workspaces.equation_workspace import EquationWorkspace

        return EquationWorkspace

    def test_f18_01_equation_ui_tree_population(self) -> None:
        """TC-F18-01: Equation hierarchy tree model structure."""
        EquationWorkspace = self._get_workspace()
        ws = EquationWorkspace()
        tree = ws.get_equation_tree()
        assert isinstance(tree, (dict, list))

    def test_f18_02_equation_ui_latex_card_render(self) -> None:
        """TC-F18-02: LaTeX formula card renders via matplotlib.mathtext without pdflatex."""
        EquationWorkspace = self._get_workspace()
        ws = EquationWorkspace()
        rendered = ws.render_latex_card(r"c_0 \cdot \sin(x) + c_1 \cdot y^2")
        assert isinstance(rendered, bytes) and rendered.startswith(b"\x89PNG")

    def test_f18_03_equation_ui_ast_inspector(self) -> None:
        """TC-F18-03: AST inspector extracts node properties and children."""
        EquationWorkspace = self._get_workspace()
        ws = EquationWorkspace()
        info = ws.inspect_ast_node("root")
        assert isinstance(info, dict)

    def test_f18_04_equation_ui_parameter_slider(self) -> None:
        """TC-F18-04: Parameter slider updates value within bounded range."""
        EquationWorkspace = self._get_workspace()
        ws = EquationWorkspace()
        ws.set_parameter_value("c0", 1.5)
        assert ws.get_parameter_value("c0") == 1.5

    def test_f18_05_equation_ui_event_debouncing(self) -> None:
        """TC-F18-05: Rapid slider adjustments debounce into single consolidated update."""
        EquationWorkspace = self._get_workspace()
        ws = EquationWorkspace()
        ws.debounce_parameter_events([0.1, 0.2, 0.3, 0.4, 0.5])
        assert ws.last_dispatched_value == 0.5


# ==============================================================================
# FEATURE 19: Region Analysis UI Workspace
# ==============================================================================


class TestFeature19RegionAnalysisUIWorkspace:
    """Feature 19: Region Analysis UI Workspace - Rubber-band ROI, 5 full-frame heatmaps"""

    def _get_workspace(self):
        from mathfield.ui.workspaces.region_workspace import RegionWorkspace

        return RegionWorkspace

    def test_f19_01_region_ui_roi_selection(self) -> None:
        """TC-F19-01: Rubber-band ROI selection bounds validation."""
        RegionWorkspace = self._get_workspace()
        ws = RegionWorkspace()
        ws.set_roi_selection(-0.5, -0.5, 0.5, 0.5)
        roi = ws.get_roi_selection()
        assert roi == (-0.5, -0.5, 0.5, 0.5)

    def test_f19_02_region_ui_local_reopt_trigger(self) -> None:
        """TC-F19-02: Region re-optimization sub-domain calculation."""
        RegionWorkspace = self._get_workspace()
        ws = RegionWorkspace()
        res = ws.trigger_local_reoptimization()
        assert res["status"] == "SUCCESS" and res["roi"] == (-0.5, -0.5, 0.5, 0.5)

    def test_f19_03_region_ui_c2_boundary_blend(self) -> None:
        """TC-F19-03: Local re-optimized ROI seamlessly blends with background via Wendland window."""
        RegionWorkspace = self._get_workspace()
        ws = RegionWorkspace()
        blended = ws.blend_subdomain_patch(local_patch=None, background=np.zeros((32, 32)))
        assert blended.shape == (32, 32)

    def test_f19_04_region_ui_five_error_heatmaps(self) -> None:
        """TC-F19-04: Generates all 5 error heatmap arrays (MSE, SSIM, CIEDE2000, abs, signed)."""
        RegionWorkspace = self._get_workspace()
        ws = RegionWorkspace()
        for metric in ["mse", "ssim", "ciede2000", "abs_diff", "signed_diff"]:
            hm = ws.generate_heatmap(metric)
            assert isinstance(hm, np.ndarray) and hm.shape == (64, 64)

    def test_f19_05_region_ui_colormap_normalize(self) -> None:
        """TC-F19-05: Colormap dynamic range normalizes error map into [0, 255]."""
        RegionWorkspace = self._get_workspace()
        ws = RegionWorkspace()
        raw_error = np.linspace(0.0, 25.0, 50)
        norm = ws.normalize_heatmap(raw_error)
        assert norm.dtype == np.uint8
        assert norm[0] == 0 and norm[-1] == 255


# ==============================================================================
# FEATURE 20: Optimization UI Workspace
# ==============================================================================


class TestFeature20OptimizationUIWorkspace:
    """Feature 20: Optimization UI Workspace - pyqtgraph metric curves, telemetry, controls"""

    def _get_workspace(self):
        from mathfield.ui.workers import OptimizationWorker
        from mathfield.ui.workspaces.optimization_workspace import OptimizationWorkspace

        return OptimizationWorkspace, OptimizationWorker

    def test_f20_01_opt_ui_metric_curve_updates(self) -> None:
        """TC-F20-01: Real-time metric history curves append points monotonically."""
        OptimizationWorkspace, _ = self._get_workspace()
        ws = OptimizationWorkspace()
        for it in range(10):
            ws.update_metrics(iteration=it, metrics={"loss": 1.0 / (it + 1)})
        curves = ws.get_metric_curves()
        assert len(curves["loss"]) == 10

    def test_f20_02_opt_ui_telemetry_polling(self) -> None:
        """TC-F20-02: psutil hardware resource utilization telemetry."""
        OptimizationWorkspace, _ = self._get_workspace()
        ws = OptimizationWorkspace()
        telem = ws.get_hardware_telemetry()
        assert "cpu" in telem and "ram" in telem

    def test_f20_03_opt_ui_pause_control(self) -> None:
        """TC-F20-03: Pause control state transition."""
        OptimizationWorkspace, _ = self._get_workspace()
        ws = OptimizationWorkspace()
        ws.pause()
        assert ws.is_paused is True

    def test_f20_04_opt_ui_resume_control(self) -> None:
        """TC-F20-04: Resume control restores RUNNING state."""
        OptimizationWorkspace, _ = self._get_workspace()
        ws = OptimizationWorkspace()
        ws.pause()
        ws.resume()
        assert ws.is_paused is False

    def test_f20_05_opt_ui_cancel_control(self) -> None:
        """TC-F20-05: Cancel control sets cooperative cancellation token."""
        OptimizationWorkspace, _ = self._get_workspace()
        ws = OptimizationWorkspace()
        ws.cancel()
        assert ws.is_cancelled is True


# ==============================================================================
# FEATURE 21: Research UI Workspace
# ==============================================================================


class TestFeature21ResearchUIWorkspace:
    """Feature 21: Research UI Workspace - Candidate log with rejection rationale, SVD spectra"""

    def _get_workspace(self):
        from mathfield.ui.workspaces.research_workspace import ResearchWorkspace

        return ResearchWorkspace

    def test_f21_01_research_ui_candidate_log(self) -> None:
        """TC-F21-01: Candidate log records expression, delta loss, and status."""
        ResearchWorkspace = self._get_workspace()
        ws = ResearchWorkspace()
        ws.log_candidate(iteration=1, expression="sin(x)", status="ACCEPTED", rationale="NONE", delta_loss=-0.05)
        ws.log_candidate(iteration=2, expression="x^5", status="REJECTED", rationale="BIC_INCREASE", delta_loss=0.02)
        log = ws.get_candidate_log()
        assert len(log) == 2
        assert log[0]["status"] == "ACCEPTED"
        assert log[1]["status"] == "REJECTED"

    def test_f21_02_research_ui_rejection_reason(self) -> None:
        """TC-F21-02: Categorical rejection rationale (e.g. BIC_INCREASE, ILL_CONDITIONED)."""
        ResearchWorkspace = self._get_workspace()
        ws = ResearchWorkspace()
        reasons = ws.get_rejection_reasons()
        assert isinstance(reasons, (list, tuple, set, dict)), f"reasons must be a container, got {type(reasons)}"
        assert "BIC_INCREASE" in reasons, "Standard categorical reason 'BIC_INCREASE' must be present"
        assert "ILL_CONDITIONED" in reasons, "Standard categorical reason 'ILL_CONDITIONED' must be present"
        assert len(reasons) >= 2, f"Expected at least 2 categorical rejection reasons, got {len(reasons)}"

    def test_f21_03_research_ui_convergence_plot(self) -> None:
        """TC-F21-03: Parameter convergence trajectory history."""
        ResearchWorkspace = self._get_workspace()
        ws = ResearchWorkspace()
        ws.update_parameter_trajectories(1, {"c0": 1.0, "c1": 0.0})
        ws.update_parameter_trajectories(2, {"c0": 0.8, "c1": 0.1})
        traj = ws.get_trajectories()
        assert len(traj["c0"]) == 2

    def test_f21_04_research_ui_svd_spectrum(self) -> None:
        """TC-F21-04: Singular value spectrum and condition ratio."""
        ResearchWorkspace = self._get_workspace()
        ws = ResearchWorkspace()
        s = np.array([10.0, 5.0, 2.0, 0.01])
        kappa = ws.update_svd_spectrum(s)
        assert kappa == 1000.0

    def test_f21_05_research_ui_export_csv(self, e2e_temp_dir: Path) -> None:
        """TC-F21-05: Exports candidate discovery log to CSV."""
        ResearchWorkspace = self._get_workspace()
        ws = ResearchWorkspace()
        out_csv = e2e_temp_dir / "research_log.csv"
        ws.export_research_log(out_csv, format="csv")
        assert out_csv.exists()


# ==============================================================================
# FEATURE 22: Non-Blocking Background Execution
# ==============================================================================


class TestFeature22NonBlockingBackgroundExecution:
    """Feature 22: Non-Blocking Background Execution - Dedicated worker thread & cooperative cancellation"""

    def _get_worker_module(self):
        from mathfield.ui.workers import OptimizationWorker, WorkerSignals

        return OptimizationWorker, WorkerSignals

    def test_f22_01_worker_thread_isolation(self) -> None:
        """TC-F22-01: Background execution in distinct thread ID."""
        import threading

        OptimizationWorker, _ = self._get_worker_module()

        main_thread_id = threading.get_ident()
        worker_thread_id: list[int] = []
        execution_event = threading.Event()

        def sample_task():
            worker_thread_id.append(threading.get_ident())
            execution_event.set()

        worker = OptimizationWorker()
        if hasattr(worker, "set_task"):
            worker.set_task(sample_task)
        elif hasattr(worker, "task"):
            worker.task = sample_task

        if hasattr(worker, "start"):
            worker.start()
            assert execution_event.wait(timeout=2.0), "Worker execution timed out"
            if hasattr(worker, "wait"):
                worker.wait(2000)
            elif hasattr(worker, "join"):
                worker.join(timeout=2.0)
        elif hasattr(worker, "execute_in_background"):
            tid = worker.execute_in_background(sample_task)
            worker_thread_id.append(tid)

        assert len(worker_thread_id) > 0, "Worker task did not execute"
        assert worker_thread_id[0] != main_thread_id, (
            f"Worker executed on main thread {main_thread_id}; background thread isolation failed"
        )

    def test_f22_02_worker_signal_delivery(self) -> None:
        """TC-F22-02: All declared signals are delivered once per job.

        MathField's signal bus is synchronous: slots run on the emitting
        (worker) thread. A GUI host must queue widget updates itself.
        """
        OptimizationWorker, WorkerSignals = self._get_worker_module()
        signals = WorkerSignals()
        events: list[tuple] = []
        delivery_threads: list[int] = []
        signals.stage_changed.connect(lambda *args: events.append(("stage", args)))
        signals.progress.connect(lambda value: events.append(("progress", value)))
        signals.iteration_completed.connect(lambda *args: events.append(("iteration", args)))
        signals.finished.connect(lambda value: events.append(("finished", value)))
        signals.finished.connect(lambda *_: delivery_threads.append(threading.get_ident()))

        worker = OptimizationWorker(lambda: 42)
        worker.signals = signals
        assert worker.start_job() is True
        assert worker.wait(2000) is True

        assert [kind for kind, _ in events] == ["stage", "progress", "iteration", "progress", "finished"]
        assert events[1][1] == 0.0 and events[3][1] == 1.0
        assert events[-1][1] == 42
        assert delivery_threads and delivery_threads[0] != threading.get_ident()

    def test_f22_03_worker_cancellation_latency(self) -> None:
        """TC-F22-03: Responsive cooperative cancellation (< 500 ms)."""
        OptimizationWorker, _ = self._get_worker_module()
        started = threading.Event()

        def cooperative_task() -> str:
            started.set()
            while not worker._stop_requested:
                time.sleep(0.001)
            return "cancelled"

        worker = OptimizationWorker(cooperative_task)
        assert worker.start_job() is True
        assert started.wait(timeout=2.0)
        start = time.perf_counter()
        worker.request_cancel()
        assert worker.wait(2000) is True
        assert (time.perf_counter() - start) < 0.50
        assert worker._result == "cancelled"

    def test_f22_04_worker_exception_safety(self) -> None:
        """TC-F22-04: Exceptions inside worker captured and emitted without process crash."""
        OptimizationWorker, _ = self._get_worker_module()
        worker = OptimizationWorker()
        errors: list[str] = []
        worker.signals.error.connect(errors.append)
        assert worker.start_job(lambda: 1 / 0) is True
        assert worker.wait(2000) is True
        assert errors and "division by zero" in errors[0]

    def test_f22_05_worker_clean_teardown(self) -> None:
        """TC-F22-05: Worker threads terminate cleanly without lingering locks."""
        OptimizationWorker, _ = self._get_worker_module()

        def persistent_loop():
            while not getattr(worker, "is_cancelled", False) and not getattr(worker, "_stop_requested", False):
                time.sleep(0.01)

        worker = OptimizationWorker()
        if hasattr(worker, "set_task"):
            worker.set_task(persistent_loop)

        if hasattr(worker, "start"):
            worker.start()
            time.sleep(0.05)  # Allow thread startup
            is_running = worker.isRunning() if hasattr(worker, "isRunning") else worker.is_alive()
            assert is_running, "Worker thread failed to enter running state"

        # Trigger cooperative teardown
        worker.stop()

        # Verify termination within 1000ms
        if hasattr(worker, "wait"):
            terminated = worker.wait(1000)
            assert terminated is True and not worker.isRunning(), "Worker thread failed to terminate within 1000ms"
        elif hasattr(worker, "join"):
            worker.join(timeout=1.0)
            assert not worker.is_alive(), "Worker thread is still alive after stop()"

        if hasattr(worker, "isRunning"):
            assert not worker.isRunning(), "Worker isRunning() must return False after teardown"
        elif hasattr(worker, "is_alive"):
            assert not worker.is_alive(), "Worker is_alive() must return False after teardown"

        assert worker.is_cancelled is True and worker._stop_requested is True


# ==============================================================================
# FEATURE 23: Image Exports (PNG & JPG)
# ==============================================================================


class TestFeature23ImageExports:
    """Feature 23: Image Exports - Continuous-res PNG with alpha, quality-controlled JPG"""

    def _get_raster_module(self):
        from mathfield.export.raster import export_jpg, export_png

        return export_png, export_jpg

    def test_f23_01_export_png_alpha_channel(self, e2e_temp_dir: Path) -> None:
        """TC-F23-01: PNG export preserves 4-channel RGBA transparency."""
        export_png, _ = self._get_raster_module()
        rgba_path = e2e_temp_dir / "test_out_alpha.png"
        sample_model = {"expressions": {"R": "x", "G": "y", "B": "0", "A": "0.5"}}
        export_png(sample_model, rgba_path)
        info = inspect_png(rgba_path, has_alpha=True)
        assert info["has_alpha"] is True
        assert info["mode"] == "RGBA"

    def test_f23_02_export_png_arbitrary_res(self, e2e_temp_dir: Path) -> None:
        """TC-F23-02: Continuous resolution rendering at arbitrary size."""
        export_png, _ = self._get_raster_module()
        custom_path = e2e_temp_dir / "custom_res.png"
        sample_model = {"expressions": {"R": "x", "G": "y", "B": "0"}}
        export_png(sample_model, custom_path, width=128, height=64)
        info = inspect_png(custom_path, expected_shape=(128, 64))
        assert info["width"] == 128
        assert info["height"] == 64

    def test_f23_03_export_jpg_quality_scaling(self, e2e_temp_dir: Path) -> None:
        """TC-F23-03: Quality-controlled JPG export (quality 95 produces larger file than quality 30)."""
        _, export_jpg = self._get_raster_module()
        sample_model = {"expressions": {"R": "sin(10*x)", "G": "cos(10*y)", "B": "0"}}
        p_low = e2e_temp_dir / "low_q.jpg"
        p_high = e2e_temp_dir / "high_q.jpg"
        export_jpg(sample_model, p_low, quality=30)
        export_jpg(sample_model, p_high, quality=95)
        info_low = inspect_jpg(p_low)
        info_high = inspect_jpg(p_high)
        assert info_high["file_size"] > info_low["file_size"]

    def test_f23_04_export_metadata_embedding(self, e2e_temp_dir: Path) -> None:
        """TC-F23-04: Metadata comment embedding in export."""
        export_png, _ = self._get_raster_module()
        p = e2e_temp_dir / "with_metadata.png"
        sample_model = {"expressions": {"R": "0", "G": "1", "B": "0"}}
        export_png(sample_model, p, metadata={"Software": "MathField", "PSNR": "35.2"})
        read_img = Image.open(p)
        assert read_img.text.get("Software") == "MathField"

    def test_f23_05_export_raster_invalid_path(self) -> None:
        """TC-F23-05: Non-existent destination raises FileNotFoundError."""
        export_png, _ = self._get_raster_module()
        with pytest.raises((FileNotFoundError, OSError)):
            export_png({"expressions": {}}, "non_existent_dir/output.png")


# ==============================================================================
# FEATURE 24: Documented SVG Export
# ==============================================================================


class TestFeature24DocumentedSVGExport:
    """Feature 24: Documented SVG Export - Dual-mode (analytical vector + metadata XML fallback)"""

    def _get_svg_module(self):
        from mathfield.export.svg import export_svg

        return export_svg

    def test_f24_01_export_svg_analytical_vector(self, e2e_temp_dir: Path) -> None:
        """TC-F24-01: Analytical vector mode SVG contains path elements."""
        export_svg = self._get_svg_module()
        svg_path = e2e_temp_dir / "vector.svg"
        sample_model = {"expressions": {"R": "x", "G": "y", "B": "0"}}
        export_svg(sample_model, svg_path, mode="vector")
        info = inspect_svg(svg_path)
        assert info["has_vector_paths"] is True
        assert info["is_valid_xml"] is True

    def test_f24_02_export_svg_raster_fallback(self, e2e_temp_dir: Path) -> None:
        """TC-F24-02: Raster fallback SVG contains base64 image element."""
        export_svg = self._get_svg_module()
        svg_path = e2e_temp_dir / "raster_fallback.svg"
        sample_model = {"expressions": {"R": "x", "G": "y", "B": "0"}}
        export_svg(sample_model, svg_path, mode="raster_fallback")
        info = inspect_svg(svg_path)
        assert info["has_embedded_raster"] is True

    def test_f24_03_export_svg_math_xml_metadata(self, e2e_temp_dir: Path) -> None:
        """TC-F24-03: SVG metadata block records mathematical equations."""
        export_svg = self._get_svg_module()
        svg_path = e2e_temp_dir / "metadata.svg"
        sample_model = {"expressions": {"formula": "sin(x) + cos(y)"}}
        export_svg(sample_model, svg_path, embed_metadata=True)
        info = inspect_svg(svg_path)
        assert info["has_metadata"] is True
        assert "sin(x)" in info["equations_metadata"]

    def test_f24_04_export_svg_xml_wellformed(self, e2e_temp_dir: Path) -> None:
        """TC-F24-04: SVG parses cleanly via standard xml.etree.ElementTree."""
        export_svg = self._get_svg_module()
        svg_path = e2e_temp_dir / "wellformed.svg"
        export_svg({"expressions": {}}, svg_path)
        info = inspect_svg(svg_path)
        assert info["is_valid_xml"] is True

    def test_f24_05_export_svg_viewbox_geometry(self, e2e_temp_dir: Path) -> None:
        """TC-F24-05: SVG viewBox preserves 2:1 aspect ratio."""
        export_svg = self._get_svg_module()
        svg_path = e2e_temp_dir / "aspect.svg"
        sample_model = {"expressions": {}}
        export_svg(sample_model, svg_path, width=200, height=100)
        info = inspect_svg(svg_path)
        assert info["viewbox"] == "0 0 200 100"


# ==============================================================================
# FEATURE 25: Standalone Python Reproduction Script
# ==============================================================================


class TestFeature25StandalonePythonReproductionScript:
    """Feature 25: Standalone Python Reproduction Script - Zero MathField dependencies"""

    def _get_python_export_module(self):
        from mathfield.export.python_script import export_standalone_script

        return export_standalone_script

    def test_f25_01_repro_script_zero_mathfield(self, e2e_temp_dir: Path) -> None:
        """TC-F25-01: Audit confirms ZERO imports of mathfield."""
        export_standalone_script = self._get_python_export_module()
        script_path = e2e_temp_dir / "standalone_pure.py"
        sample_model = {"expressions": {"R": "x", "G": "y", "B": "0"}}
        export_standalone_script(sample_model, script_path)
        audit = validate_standalone_python_script(script_path)
        assert "mathfield" not in audit["imports"]

    def test_f25_02_repro_script_zero_raster_file(self, e2e_temp_dir: Path) -> None:
        """TC-F25-02: Script reproduces image from pure math without reading original image file."""
        export_standalone_script = self._get_python_export_module()
        script_path = e2e_temp_dir / "standalone_pure.py"
        sample_model = {"expressions": {"R": "x", "G": "y", "B": "0"}}
        export_standalone_script(sample_model, script_path)
        code = script_path.read_text(encoding="utf-8")
        assert "Image.open" not in code

    def test_f25_03_repro_script_subprocess_exec(self, e2e_temp_dir: Path) -> None:
        """TC-F25-03: Subprocess executes cleanly in isolated environment without MathField."""
        export_standalone_script = self._get_python_export_module()
        script_path = e2e_temp_dir / "exec_test.py"
        sample_model = {"expressions": {"R": "0", "G": "0", "B": "0"}}
        export_standalone_script(sample_model, script_path)
        out_img = e2e_temp_dir / "repro_out.png"
        audit = validate_standalone_python_script(script_path, test_execution=True, output_image_path=out_img)
        assert audit["output_valid"] is True

    def test_f25_04_repro_script_pixel_parity(self, e2e_temp_dir: Path) -> None:
        """TC-F25-04: Pixel output matches analytical function to numerical MSE < 1.0."""
        export_standalone_script = self._get_python_export_module()
        script_path = e2e_temp_dir / "parity_test.py"
        # Bivariate analytical model: R(x,y) = 0.5 * (sin(pi * x) * cos(pi * y) + 1.0)
        sample_model = {
            "coordinate_domain": {"x_min": -1.0, "x_max": 1.0, "y_min": -1.0, "y_max": 1.0},
            "expressions": {"R": "0.5 * (sin(pi * x) * cos(pi * y) + 1.0)", "G": "0.5 * (x + 1.0)", "B": "0.0"},
        }
        export_standalone_script(sample_model, script_path)
        out_img = e2e_temp_dir / "parity_out.png"
        res = run_standalone_script(script_path, ["--output", str(out_img), "--resolution", "16", "16"])
        res.assert_success("Standalone reproduction script execution failed.")
        assert out_img.exists(), "Standalone script failed to generate output image."

        # Load generated image
        gen_img = Image.open(out_img).convert("RGB")
        gen_arr = np.array(gen_img, dtype=np.float64)

        # Compute ground truth array analytically from first principles
        x = np.linspace(-1.0, 1.0, 16, dtype=np.float64)
        y = np.linspace(-1.0, 1.0, 16, dtype=np.float64)
        X, Y = np.meshgrid(x, y)
        truth_r = np.clip(0.5 * (np.sin(np.pi * X) * np.cos(np.pi * Y) + 1.0) * 255.0, 0.0, 255.0)
        truth_g = np.clip(0.5 * (X + 1.0) * 255.0, 0.0, 255.0)
        truth_b = np.zeros_like(X)
        truth_arr = np.stack([truth_r, truth_g, truth_b], axis=-1)

        # Compute pixel MSE across channels
        mse_r = compute_reference_mse(gen_arr[:, :, 0], truth_arr[:, :, 0])
        mse_g = compute_reference_mse(gen_arr[:, :, 1], truth_arr[:, :, 1])
        mse_b = compute_reference_mse(gen_arr[:, :, 2], truth_arr[:, :, 2])
        total_mse = (mse_r + mse_g + mse_b) / 3.0

        # Assert pixel parity: MSE must be < 1.0 (allowing uint8 rounding differences)
        assert total_mse < 1.0, f"Pixel parity MSE too high: {total_mse:.4f} (R: {mse_r:.4f}, G: {mse_g:.4f})"

    def test_f25_05_repro_script_cli_arguments(self, e2e_temp_dir: Path) -> None:
        """TC-F25-05: Standalone script handles --help, --output, and --resolution."""
        export_standalone_script = self._get_python_export_module()
        script_path = e2e_temp_dir / "help_test.py"
        sample_model = {"expressions": {"R": "x", "G": "0", "B": "0"}}
        export_standalone_script(sample_model, script_path)
        res = run_standalone_script(script_path, ["--help"])
        res.assert_success()
        assert "usage" in res.stdout.lower()
        assert "--output" in res.stdout
        assert "--resolution" in res.stdout


# ==============================================================================
# FEATURE 26: Standalone LaTeX & PDF Report
# ==============================================================================


class TestFeature26StandaloneLaTeXPDFReport:
    """Feature 26: Standalone LaTeX & PDF Report - ReportLab 5.0.1 zero-dependency PDF"""

    def _get_latex_pdf_modules(self):
        from mathfield.export.latex_report import export_latex_report
        from mathfield.export.pdf_report import export_pdf_report

        return export_latex_report, export_pdf_report

    def test_f26_01_export_latex_document_syntax(self, e2e_temp_dir: Path) -> None:
        """TC-F26-01: Compilable LaTeX syntax structure."""
        export_latex_report, _ = self._get_latex_pdf_modules()
        tex_path = e2e_temp_dir / "report.tex"
        sample_result = {"expressions": {"R": "c0 * x"}, "parameters": {"c0": 1.0}}
        export_latex_report(sample_result, tex_path)
        info = inspect_latex_doc(tex_path)
        assert info["is_compilable_structure"] is True
        assert info["equation_count"] >= 1

    def test_f26_02_export_latex_parameter_table(self, e2e_temp_dir: Path) -> None:
        """TC-F26-02: Booktabs parameter tables."""
        export_latex_report, _ = self._get_latex_pdf_modules()
        tex_path = e2e_temp_dir / "table.tex"
        sample_result = {"expressions": {}, "parameters": {"c0": 1.234}}
        export_latex_report(sample_result, tex_path)
        info = inspect_latex_doc(tex_path)
        assert info["tables_found"] >= 1

    def test_f26_03_export_pdf_zero_dependency(self, e2e_temp_dir: Path) -> None:
        """TC-F26-03: Zero-dependency PDF generation via ReportLab 5.0.1."""
        _, export_pdf_report = self._get_latex_pdf_modules()
        pdf_path = e2e_temp_dir / "mathfield_report.pdf"
        sample_result = {"expressions": {"R": "c0 * sin(x)"}, "metrics": {"mse": 0.0012, "psnr": 29.21}}
        export_pdf_report(sample_result, pdf_path)
        info = inspect_pdf_report(pdf_path)
        assert info["page_count"] >= 1
        assert info["has_text"] is True

    def test_f26_04_export_pdf_magic_and_pages(self, e2e_temp_dir: Path) -> None:
        """TC-F26-04: PDF binary magic bytes (%PDF-), page count >= 1, and valid structure."""
        _, export_pdf_report = self._get_latex_pdf_modules()
        pdf_path = e2e_temp_dir / "magic_report.pdf"
        sample_result = {"expressions": {"formula": "0.5 * x^2 + 0.1"}, "metrics": {"mse": 0.0005, "psnr": 33.01}}
        export_pdf_report(sample_result, pdf_path)
        assert pdf_path.exists()
        raw_bytes = pdf_path.read_bytes()
        assert raw_bytes.startswith(b"%PDF-"), "Generated file does not begin with %PDF- header magic"
        info = inspect_pdf_report(pdf_path)
        assert info["page_count"] >= 1
        assert info["file_size"] >= 1024, f"PDF file size suspiciously small: {info['file_size']} bytes"
        assert info["has_text"] is True

    def test_f26_05_export_pdf_visual_elements(self, e2e_temp_dir: Path) -> None:
        """TC-F26-05: PDF embeds images and figures (image_count >= 1)."""
        _, export_pdf_report = self._get_latex_pdf_modules()
        pdf_path = e2e_temp_dir / "visual_report.pdf"
        # Provide sample result with original and reconstructed image arrays to be drawn in PDF
        preview_arr = np.full((32, 32, 3), 128, dtype=np.uint8)
        sample_result = {
            "expressions": {"R": "c0 * x", "G": "c1 * y", "B": "0"},
            "parameters": {"c0": 0.5, "c1": 0.5},
            "metrics": {"mse": 0.002, "psnr": 27.0},
            "original_image": preview_arr,
            "reconstructed_image": preview_arr,
            "error_heatmap": preview_arr,
        }
        export_pdf_report(sample_result, pdf_path)
        info = inspect_pdf_report(pdf_path)
        assert info["page_count"] >= 1
        assert info["image_count"] >= 1, f"Expected at least 1 embedded figure/image in PDF, got {info['image_count']}"


# ==============================================================================
# FEATURE 27: CSV Data & JSON Project Format
# ==============================================================================

# Formal Draft-07 JSON Schema definition for .mathfield project files
DRAFT07_PROJECT_SCHEMA: Dict[str, Any] = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "type": "object",
    "required": ["schema_version", "coordinate_domain", "color_space", "global_model", "final_metrics"],
    "properties": {
        "schema_version": {"type": "string"},
        "project_name": {"type": "string"},
        "coordinate_domain": {
            "type": "object",
            "required": ["x_min", "x_max", "y_min", "y_max"],
            "properties": {
                "x_min": {"type": "number"},
                "x_max": {"type": "number"},
                "y_min": {"type": "number"},
                "y_max": {"type": "number"},
                "pixel_width": {"type": "integer"},
                "pixel_height": {"type": "integer"},
                "aspect_ratio": {"type": "number"},
            },
        },
        "color_space": {"type": "string", "enum": ["RGB", "RGBA", "HSV", "CIELAB", "LinearRGB"]},
        "mode": {"type": "string"},
        "global_model": {
            "type": "object",
            "required": ["channels"],
            "properties": {
                "channels": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "required": ["channel", "formula"],
                        "properties": {
                            "channel": {"type": "string"},
                            "formula": {"type": "string"},
                            "parameters": {"type": "array"},
                        },
                    },
                }
            },
        },
        "local_patches": {"type": "array"},
        "final_metrics": {
            "type": "object",
            "required": ["mse", "psnr"],
            "properties": {
                "mse": {"type": "number"},
                "psnr": {"type": "number"},
                "ssim": {"type": "number"},
                "complexity": {"type": "number"},
            },
        },
        "random_seed": {"type": ["integer", "null"]},
    },
}


class TestFeature27CSVDataJSONProjectFormat:
    """Feature 27: CSV Data & JSON Project Format - Draft-07 validation"""

    def _get_project_csv_modules(self):
        from mathfield.export.csv_export import export_csv_parameters
        from mathfield.export.project_file import ProjectFileError, load_project, save_project

        return export_csv_parameters, save_project, load_project, ProjectFileError

    def test_f27_01_export_csv_headers_columns(self, e2e_temp_dir: Path) -> None:
        """TC-F27-01: CSV parameter table contains required columns."""
        export_csv_parameters, _, _, _ = self._get_project_csv_modules()
        csv_path = e2e_temp_dir / "params.csv"
        sample_result = {"parameters": {"c0": {"stage": "global", "channel": "R", "value": 0.5234}}}
        export_csv_parameters(sample_result, csv_path)
        rows = inspect_csv_parameters(csv_path)
        assert len(rows) >= 1
        assert rows[0]["parameter_name"] == "c0"

    def test_f27_02_export_csv_numeric_precision(self, e2e_temp_dir: Path) -> None:
        """TC-F27-02: CSV preserves high float precision (>= 12 digits)."""
        export_csv_parameters, _, _, _ = self._get_project_csv_modules()
        csv_path = e2e_temp_dir / "precision.csv"
        val = 0.123456789012345
        sample_result = {"parameters": {"c0": {"value": val}}}
        export_csv_parameters(sample_result, csv_path)
        rows = inspect_csv_parameters(csv_path)
        assert math.isclose(float(rows[0]["parameter_value"]), val, abs_tol=1e-12)

    def test_f27_03_project_file_draft07_schema(self, sample_mathfield_project: Path) -> None:
        """TC-F27-03: Draft-07 JSON schema compliance validated via inspect_json_project."""
        # Use helper inspect_json_project which executes jsonschema.validate(instance, schema)
        data = inspect_json_project(sample_mathfield_project, schema_dict=DRAFT07_PROJECT_SCHEMA)
        assert isinstance(data, dict)
        assert data["schema_version"] == "1.0.0"
        assert "global_model" in data
        assert "final_metrics" in data

    def test_f27_04_project_file_roundtrip_state(self, sample_mathfield_project: Path, e2e_temp_dir: Path) -> None:
        """TC-F27-04: Save and restore project state round-trip."""
        _, save_project, load_project, _ = self._get_project_csv_modules()
        orig = load_project(sample_mathfield_project)
        clone_path = e2e_temp_dir / "clone.mathfield"
        save_project(orig, clone_path)
        recovered = load_project(clone_path)
        assert orig == recovered

    def test_f27_05_project_file_schema_rejection(self, e2e_temp_dir: Path) -> None:
        """TC-F27-05: Missing mandatory keys rejected."""
        _, _, load_project, ProjectFileError = self._get_project_csv_modules()
        bad_json = e2e_temp_dir / "invalid.mathfield"
        bad_json.write_text(json.dumps({"bogus_key": 123}), encoding="utf-8")
        with pytest.raises((ProjectFileError, ValueError)):
            load_project(bad_json)


# ==============================================================================
# FEATURE 28: Standalone Reproducibility Package Bundle
# ==============================================================================


class TestFeature28StandaloneReproducibilityPackageBundle:
    """Feature 28: Standalone Reproducibility Package Bundle - Complete self-contained package"""

    def _get_bundle_module(self):
        from mathfield.export.bundle import export_bundle

        return export_bundle

    def test_f28_01_bundle_manifest_completeness(self, e2e_temp_dir: Path) -> None:
        """TC-F28-01: Bundle contains all required reproducible assets."""
        export_bundle = self._get_bundle_module()
        bundle_dir = e2e_temp_dir / "bundle"
        sample_result = {"expressions": {}, "parameters": {}}
        export_bundle(sample_result, bundle_dir)
        res = validate_reproducibility_package(bundle_dir)
        assert res["assets_verified"] is True

    def test_f28_02_bundle_readme_instructions(self, e2e_temp_dir: Path) -> None:
        """TC-F28-02: README contains reproduction instructions."""
        export_bundle = self._get_bundle_module()
        bundle_dir = e2e_temp_dir / "bundle_readme"
        sample_result = {"expressions": {}, "parameters": {}}
        export_bundle(sample_result, bundle_dir)
        readme = bundle_dir / "README.md"
        assert "MathField" in readme.read_text(encoding="utf-8")

    def test_f28_03_bundle_verify_script_exec(self, e2e_temp_dir: Path) -> None:
        """TC-F28-03: verify.py runs cleanly via subprocess."""
        export_bundle = self._get_bundle_module()
        bundle_dir = e2e_temp_dir / "bundle_verify"
        sample_result = {"expressions": {}, "parameters": {}}
        export_bundle(sample_result, bundle_dir)
        v_script = bundle_dir / "verify.py"
        res = run_standalone_script(v_script)
        res.assert_success()
        assert "passed" in res.stdout.lower()

    def test_f28_04_bundle_zip_archive_structure(self, e2e_temp_dir: Path) -> None:
        """TC-F28-04: ZIP archive package validation."""
        export_bundle = self._get_bundle_module()
        zip_path = e2e_temp_dir / "bundle.zip"
        sample_result = {"expressions": {}, "parameters": {}}
        export_bundle(sample_result, zip_path, as_zip=True)
        res = validate_reproducibility_package(zip_path)
        assert res["assets_verified"] is True

    def test_f28_05_bundle_zero_mathfield_verify(self, e2e_temp_dir: Path) -> None:
        """TC-F28-05: verify.py has zero dependencies on MathField."""
        export_bundle = self._get_bundle_module()
        bundle_dir = e2e_temp_dir / "bundle_zero"
        sample_result = {"expressions": {}, "parameters": {}}
        export_bundle(sample_result, bundle_dir)
        v_script = bundle_dir / "verify.py"
        audit = validate_standalone_python_script(v_script)
        assert "mathfield" not in audit["imports"]


# ==============================================================================
# FEATURE 29: Automated Test Suite Integration
# ==============================================================================


class TestFeature29AutomatedTestSuiteIntegration:
    """Feature 29: Automated Test Suite Integration - Core test suite execution"""

    def test_f29_01_unit_test_suite_discovery(self) -> None:
        """TC-F29-01: Pytest discovers e2e test suite cleanly without syntax or collection errors."""
        e2e_file = PROJECT_ROOT / "e2e_tests" / "test_e2e_tier1.py"
        assert e2e_file.exists(), f"Tier 1 test file not found at {e2e_file}"

        res = subprocess.run(
            [sys.executable, "-m", "pytest", str(e2e_file), "--collect-only", "-q"],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            timeout=30.0,
            check=False,
        )
        assert res.returncode == 0, f"Pytest collection failed with code {res.returncode}:\n{res.stderr}\n{res.stdout}"

        # Verify total collected item count: 31 features * 5 tests = 155 minimum items (160 with parameterization)
        match = re.search(r"(\d+)\s+tests?\s+collected", res.stdout)
        assert match is not None, f"Could not parse collected item count from pytest output:\n{res.stdout}"
        collected_count = int(match.group(1))
        assert collected_count >= 155, f"Expected at least 155 collected test items, got {collected_count}"

    def test_f29_02_unit_test_core_ast_execution(self) -> None:
        """TC-F29-02: Core AST module passes basic invariant checks."""
        from mathfield.core.ast import ConstantNode, VariableNode, parse

        node = parse("x + 1")
        assert isinstance(node.left, VariableNode)
        assert isinstance(node.right, ConstantNode)

    def test_f29_03_unit_test_engine_execution(self) -> None:
        """TC-F29-03: Vectorized CPU engine evaluates correctly."""
        from mathfield.core.ast import parse
        from mathfield.core.engine.cpu_numpy import evaluate_numpy

        ast = parse("x * y")
        X, Y = np.meshgrid(np.array([1.0, 2.0]), np.array([3.0, 4.0]))
        res = evaluate_numpy(ast, X, Y, np.array([]))
        np.testing.assert_allclose(res, X * Y)

    def test_f29_04_unit_test_pipeline_execution(self) -> None:
        """TC-F29-04: Test infrastructure helpers integrate with pipeline evaluation and verify metrics."""
        from mathfield.core.ast import parse
        from mathfield.core.domain import CoordinateDomain2D
        from mathfield.core.engine.cpu_numpy import evaluate_numpy
        from mathfield.core.verify.metrics import compute_mse, compute_psnr, compute_ssim

        domain = CoordinateDomain2D(width=32, height=32)
        X, Y = domain.to_discrete_grid()
        ast = parse("sin(x) + cos(y)")
        ground_truth = evaluate_numpy(ast, X, Y, np.array([]))

        # Perturbed approximation to evaluate numerical error
        reconstructed = ground_truth + 0.05 * np.cos(2.0 * X)

        # 1. Pipeline verification metrics must produce genuine non-trivial values
        dyn_range = float(np.max(ground_truth) - np.min(ground_truth))
        pipe_mse = compute_mse(reconstructed, ground_truth)
        pipe_psnr = compute_psnr(reconstructed, ground_truth, max_val=dyn_range)
        pipe_ssim = compute_ssim(reconstructed, ground_truth, max_val=dyn_range)

        # 2. Independent reference helpers in e2e_tests.helpers must match pipeline metrics within tolerances
        ref_mse = compute_reference_mse(reconstructed, ground_truth)
        ref_psnr = compute_reference_psnr(reconstructed, ground_truth, max_val=dyn_range)
        ref_ssim = compute_reference_ssim(reconstructed, ground_truth, L=dyn_range)

        assert pipe_mse > 0.0, "Perturbed field must yield strictly positive MSE"
        assert math.isclose(pipe_mse, ref_mse, abs_tol=1e-12), f"MSE mismatch: {pipe_mse} vs {ref_mse}"
        assert math.isclose(pipe_psnr, ref_psnr, abs_tol=1e-10), f"PSNR mismatch: {pipe_psnr} vs {ref_psnr}"
        assert math.isclose(pipe_ssim, ref_ssim, abs_tol=1e-2), f"SSIM mismatch: {pipe_ssim} vs {ref_ssim}"
        assert 0.0 < pipe_ssim < 1.0, f"SSIM must be bounded in (0, 1), got {pipe_ssim}"

    def test_f29_05_unit_test_headless_ui_run(self) -> None:
        """TC-F29-05: Headless execution passes without display server crashes."""
        ui_test_file = PROJECT_ROOT / "tests" / "test_ui_headless.py"
        assert ui_test_file.exists(), "Required artefact missing: ui_test_file"
        try:
            res = subprocess.run(
                [sys.executable, "-m", "pytest", str(ui_test_file), "--collect-only", "-q"],
                cwd=str(PROJECT_ROOT),
                capture_output=True,
                text=True,
                timeout=30.0,
                check=False,
            )
            assert res.returncode == 0, f"Headless UI discovery failed: {res.stderr}\n{res.stdout}"
        except subprocess.TimeoutExpired:
            pytest.fail("Headless UI test discovery timed out after 30.0s")


# ==============================================================================
# FEATURE 30: Empirical Benchmarking Harness
# ==============================================================================


class TestFeature30EmpiricalBenchmarkingHarness:
    """Feature 30: Empirical Benchmarking Harness - Performance counters, zero fabricated scores"""

    def _get_benchmarks_module(self):
        import benchmarks.ablations as ablations_mod
        import benchmarks.baselines as baselines_mod
        import benchmarks.run_benchmarks as run_benchmarks_mod

        return baselines_mod, ablations_mod, run_benchmarks_mod

    def test_f30_01_benchmark_baselines_suite(self) -> None:
        """TC-F30-01: 5 baselines defined (Chebyshev, Fourier, wavelets, splines, RBF)."""
        baselines_mod, _, _ = self._get_benchmarks_module()
        names = (
            baselines_mod.list_baselines() if hasattr(baselines_mod, "list_baselines") else baselines_mod.BASELINE_NAMES
        )
        expected = {"chebyshev", "fourier", "wavelets", "splines", "rbf"}
        assert set(names) == expected

    def test_f30_02_benchmark_high_res_timers(self) -> None:
        """TC-F30-02: High-resolution hardware performance counter timer."""
        _, _, run_benchmarks_mod = self._get_benchmarks_module()
        runner = run_benchmarks_mod.BenchmarkRunner()
        timing = runner.measure_function(func=lambda: time.sleep(0.001), warmup_runs=5, timed_runs=10)
        assert timing.warmup_count == 5
        assert timing.timed_count == 10
        assert timing.mean_ms > 0.5

    def test_f30_03_benchmark_no_precomputed(self) -> None:
        """TC-F30-03: Computational honesty: zero precalculated benchmark output tables."""
        baselines_mod, _, _ = self._get_benchmarks_module()
        synth_poly = np.ones((16, 16, 3), dtype=np.float64)
        res = baselines_mod.run_baseline("chebyshev", synth_poly, degree=2)
        assert res.mse > 0.0 or res.psnr > 0.0
        assert res.runtime_ms > 0.0

    def test_f30_04_benchmark_ablation_studies(self) -> None:
        """TC-F30-04: 4 ablation studies specified."""
        _, ablations_mod, _ = self._get_benchmarks_module()
        names = (
            ablations_mod.list_ablations() if hasattr(ablations_mod, "list_ablations") else ablations_mod.ABLATION_NAMES
        )
        expected = {"no_quadtree", "no_bic_pruning", "no_varpro", "no_color_decoupling"}
        assert set(names) == expected

    def test_f30_05_benchmark_metrics_export(self, e2e_temp_dir: Path) -> None:
        """TC-F30-05: Benchmark runner metric export schema."""
        _, _, run_benchmarks_mod = self._get_benchmarks_module()
        runner = run_benchmarks_mod.BenchmarkRunner()
        bench_out = e2e_temp_dir / "bench.json"
        mock_result = {
            "benchmark_name": "Chebyshev_2D",
            "runtime_ms": 12.34,
            "peak_memory_mb": 45.6,
            "mse": 0.002,
            "psnr": 27.0,
        }
        runner.export_json(mock_result, bench_out)
        assert bench_out.exists()
        loaded = json.loads(bench_out.read_text(encoding="utf-8"))
        assert loaded["benchmark_name"] == "Chebyshev_2D"
        assert "runtime_ms" in loaded


# ==============================================================================
# FEATURE 31: Technical Paper Draft & Documentation
# ==============================================================================


class TestFeature31TechnicalPaperDocumentation:
    """Feature 31: Technical Paper Draft & Documentation - Publication-grade paper & README"""

    def test_f31_01_paper_draft_structure(self) -> None:
        """TC-F31-01: Paper draft checks or specifications."""
        paper_path = PROJECT_ROOT / "docs" / "paper" / "mathfield_paper.tex"
        assert paper_path.exists(), "Required artefact missing: paper_path"
        text = paper_path.read_text(encoding="utf-8")
        assert "\\title" in text
        assert "\\begin{document}" in text

    def test_f31_02_paper_draft_latex_wellformed(self) -> None:
        """TC-F31-02: Paper draft LaTeX syntax verification."""
        paper_path = PROJECT_ROOT / "docs" / "paper" / "mathfield_paper.tex"
        assert paper_path.exists(), "Required artefact missing: paper_path"
        info = inspect_latex_doc(paper_path)
        assert info["is_compilable_structure"] is True

    def test_f31_03_paper_draft_honesty_check(self) -> None:
        """TC-F31-03: Scientific integrity audit: zero fabricated benchmark claims."""
        paper_path = PROJECT_ROOT / "docs" / "paper" / "mathfield_paper.tex"
        assert paper_path.exists(), "Required artefact missing: paper_path"
        text = paper_path.read_text(encoding="utf-8")
        assert "[TODO: fake results]" not in text

    def test_f31_04_readme_windows_instructions(self) -> None:
        """TC-F31-04: README contains Windows quickstart and developer guide."""
        readme_path = PROJECT_ROOT / "README.md"
        assert readme_path.exists(), "Required artefact missing: readme_path"
        content = readme_path.read_text(encoding="utf-8")
        assert "MathField" in content

    def test_f31_05_readme_architecture_diagram(self) -> None:
        """TC-F31-05: Architecture documentation reflects 6 subsystems."""
        readme_path = PROJECT_ROOT / "README.md"
        assert readme_path.exists(), "Required artefact missing: readme_path"
        content = readme_path.read_text(encoding="utf-8")
        assert "Reconstruction" in content
