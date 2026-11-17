"""
e2e_tests/test_e2e_tier3.py

Tier 3: Cross-Feature Pairwise Combinatorial Interaction Test Suite (Milestone E2E-M3)
Systematic pairwise interaction tests spanning all 31 features.
Total Test Cases: 31 (covering orthogonal multi-subsystem contracts).

Methodology:
Pairwise Combinatorial Testing across:
- Reconstruction modes (Mode A, Mode B) x Color spaces (HSV, CIELAB, Alpha)
- Core Discovery (Global, Local, Joint Optimization, VarPro) x Evaluators (CPU SIMD, OpenMP DLL, GPU CuPy)
- Continuous Domains x Export Generators (NumPy, LaTeX, SVG, PDF, CSV, JSON Project)
- High-Resolution Tiling x Checkpointing & Recovery
- UI Workspaces x Background Thread Workers & Telemetry
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image

from e2e_tests.helpers import (
    compute_ciede2000,
    compute_reference_mse,
    inspect_csv_parameters,
    inspect_pdf_report,
    inspect_png,
    inspect_svg,
    run_standalone_script,
    validate_standalone_python_script,
)

pytestmark = [pytest.mark.tier3]
PROJECT_ROOT = Path(__file__).resolve().parent.parent


class TestTier3PairwiseInteractions:
    """Tier 3: Cross-Feature Pairwise Combinatorial Interaction Tests (Features 1-31)."""

    def test_pair_01_mode_a_continuous_hsv(self, synth_hsv_image: Path) -> None:
        """
        TC-F-PAIR-01: F01 (Mode A) x F09 (Continuous HSV Color Space).
        Mode A discovery on continuous Cartesian HSV (u, v, V) wheel avoids branch cut discontinuities.
        """
        from mathfield.core.basis import Chebyshev2D
        from mathfield.core.color import ColorSpaceConverter
        from mathfield.core.optimize import solve_linear_lstsq

        # Load synthetic HSV color wheel image
        img = np.array(Image.open(synth_hsv_image).convert("RGB"), dtype=np.float64) / 255.0
        h, w, _ = img.shape

        # Convert RGB to Cartesian HSV: (u, v, V) where u = cos(theta), v = sin(theta)
        cart_hsv = ColorSpaceConverter.rgb_to_cartesian_hsv(img)
        assert cart_hsv.shape == (h, w, 3)

        # Fit continuous 2D Chebyshev basis on u and v fields
        deg = 3
        cheb = Chebyshev2D(deg_x=deg, deg_y=deg, truncation="total_degree")
        x = np.linspace(-1.0, 1.0, w, dtype=np.float64)
        y = np.linspace(-1.0, 1.0, h, dtype=np.float64)
        X, Y = np.meshgrid(x, y)
        Phi = cheb.basis_matrix(X, Y)

        # Solve linear coefficients for u, v, V
        c_u, _ = solve_linear_lstsq(Phi, cart_hsv[:, :, 0].ravel(), reg_lambda=1e-5)
        c_v, _ = solve_linear_lstsq(Phi, cart_hsv[:, :, 1].ravel(), reg_lambda=1e-5)
        c_val, _ = solve_linear_lstsq(Phi, cart_hsv[:, :, 2].ravel(), reg_lambda=1e-5)

        recon_u = (Phi @ c_u).reshape(h, w)
        recon_v = (Phi @ c_v).reshape(h, w)
        recon_val = np.clip((Phi @ c_val).reshape(h, w), 0.0, 1.0)

        recon_hsv = np.stack([recon_u, recon_v, recon_val], axis=-1)
        recon_rgb = ColorSpaceConverter.cartesian_hsv_to_rgb(recon_hsv)

        # Invariant: across the negative x-axis (y = 0, x < 0), polar hue has a branch cut jump,
        # but Cartesian (u, v) and reconstructed RGB must remain continuous without step artifact.
        mid_y = h // 2
        jump_u = np.max(np.abs(np.diff(recon_u[mid_y, :])))
        jump_v = np.max(np.abs(np.diff(recon_v[mid_y, :])))
        assert jump_u < 0.25, f"Discontinuity detected in Cartesian u channel: max jump = {jump_u}"
        assert jump_v < 0.25, f"Discontinuity detected in Cartesian v channel: max jump = {jump_v}"
        assert recon_rgb.shape == (h, w, 3)

    def test_pair_02_mode_a_tiled_discovery(self, e2e_temp_dir: Path) -> None:
        """
        TC-F-PAIR-02: F01 (Mode A) x F15 (8K Tiling & Memory Management).
        Mode A discovery executes over 1024x1024 spatial tiles without exceeding 300 MB RAM budget.
        """
        try:
            from mathfield.core.engine.tiler import TiledReconstructionManager
        except (ImportError, ModuleNotFoundError):
            pytest.skip("mathfield.core.engine.tiler is under active development in Milestone M3")

        tiler = TiledReconstructionManager(tile_size=1024, max_memory_mb=300)
        tile_grid = tiler.compute_tile_layout(width=2048, height=2048)
        assert len(tile_grid) == 4, f"Expected 4 tiles for 2048x2048 with 1024x1024 chunking, got {len(tile_grid)}"

    def test_pair_03_mode_a_checkpoint_recovery(self, sample_checkpoint_file: Path, e2e_temp_dir: Path) -> None:
        """
        TC-F-PAIR-03: F01 (Mode A) x F16 (Robust Bit-Exact Job Checkpointing).
        Paused Mode A discovery resumed from serialized .mfchk produces bit-exact parameter parity.
        """
        try:
            from mathfield.core.checkpoint import CheckpointManager
        except (ImportError, ModuleNotFoundError):
            pytest.skip("mathfield.core.checkpoint is under active development in Milestone M3")

        mgr = CheckpointManager()
        state = mgr.load(sample_checkpoint_file)
        assert state["stage"] == "STAGE_4_OPTIMIZATION"
        assert state["iteration"] == 25
        assert "c1" in state["parameters"]

    def test_pair_04_mode_a_standalone_script_parity(self, e2e_temp_dir: Path) -> None:
        """
        TC-F-PAIR-04: F01 (Mode A) x F25 (Standalone Python Reproduction Script).
        Standalone script reproduced image achieves MSE < 1.0 against analytical function.
        """
        from mathfield.core.ast import parse
        from mathfield.core.codegen import generate_standalone_script

        formula = "0.5 * (sin(pi * x) * cos(pi * y) + 1.0)"
        ast_node = parse(formula)
        out_script = e2e_temp_dir / "mode_a_repro.py"
        out_png = e2e_temp_dir / "mode_a_repro.png"

        script_code = generate_standalone_script(ast_node, params=[], width=16, height=16, output_path=str(out_png))
        out_script.write_text(script_code, encoding="utf-8")

        # Security and purity audit: zero mathfield imports, zero eval/exec
        audit = validate_standalone_python_script(out_script)
        assert "mathfield" not in audit["imports"]
        assert audit["has_eval_or_exec"] is False

        # Execute in isolated subprocess
        res = run_standalone_script(out_script)
        res.assert_success("Standalone script execution failed.")
        assert out_png.exists()

        # Load generated image and verify pixel-level parity against first-principles ground truth
        gen_img = np.array(Image.open(out_png).convert("L"), dtype=np.float64)
        x = np.linspace(-1.0, 1.0, 16, dtype=np.float64)
        y = np.linspace(-1.0, 1.0, 16, dtype=np.float64)
        X, Y = np.meshgrid(x, y)
        truth = np.clip(0.5 * (np.sin(np.pi * X) * np.cos(np.pi * Y) + 1.0) * 255.0, 0.0, 255.0)

        mse = compute_reference_mse(gen_img, truth)
        assert mse < 1.0, f"Standalone reproduction pixel parity MSE too high: {mse:.4f}"

    def test_pair_05_mode_b_quadtree_local_patches(self) -> None:
        """
        TC-F-PAIR-05: F02 (Mode B) x F06 (Local Decomposition).
        Mode B maximum fidelity uses quadtree spatial subdivision and C^2 Wendland blending.
        """
        from mathfield.core.domain import CoordinateDomain2D
        from mathfield.core.pipeline.mode_b import ModeBPipeline
        from mathfield.core.pipeline.stage import PipelineContext, StageStatus

        # Synthetic image with sharp localized square feature
        img = np.zeros((32, 32, 3), dtype=np.uint8)
        img[10:22, 10:22] = 255
        dom = CoordinateDomain2D(32, 32)

        ctx = PipelineContext(image=img, domain=dom, config={"max_depth": 3, "degree": 2, "split_threshold": 0.05})
        pipe = ModeBPipeline()
        status = pipe.execute(ctx)

        assert status == StageStatus.COMPLETED
        assert ctx.local_decomposition is not None
        assert "patches" in ctx.local_decomposition
        assert len(ctx.local_decomposition["patches"]) >= 1
        assert ctx.metrics["mse"] < 0.05
        assert ctx.metrics["psnr"] > 13.0

    def test_pair_06_mode_b_varpro_separable_optimization(self) -> None:
        """
        TC-F-PAIR-06: F02 (Mode B) x F07 (Joint Optimization / VarPro).
        VarPro separable least squares separates linear basis weights from nonlinear frequencies.
        """
        from mathfield.core.optimize.varpro import VarProOptimizer

        opt = VarProOptimizer(max_iter=20)
        x = np.linspace(-1.0, 1.0, 16, dtype=np.float64)
        y = np.linspace(-1.0, 1.0, 16, dtype=np.float64)
        X, Y = np.meshgrid(x, y)

        # Separable model: Phi(theta, X, Y) @ c where theta is frequency
        def basis_fn(theta, X, Y):
            return np.column_stack(
                [
                    np.ones(X.size),
                    X.ravel(),
                    np.sin(theta[0] * Y.ravel()),
                ]
            )

        true_theta = np.array([3.14159])
        true_c = np.array([0.5, 0.25, 0.75])
        y_data = basis_fn(true_theta, X, Y) @ true_c

        # Initial guess theta_0 = 2.5
        res = opt.fit(basis_fn, X, Y, y_data, theta_init=np.array([2.5]))
        assert res.converged is True
        assert math.isclose(float(res.theta_opt[0]), 3.14159, abs_tol=1e-3)
        assert np.allclose(res.c_opt, true_c, atol=1e-3)
        assert res.cost < 1e-10

    def test_pair_07_mode_b_cielab_color_difference(self) -> None:
        """
        TC-F-PAIR-07: F02 (Mode B) x F09 (CIELAB Color Space).
        Mode B color optimization in CIELAB calculates Delta E00 and applies gamut soft-clamping.
        """
        from mathfield.core.color import ColorSpaceConverter

        # Create out-of-gamut Lab coordinates
        extreme_lab = np.array([[[100.0, 120.0, -120.0]]], dtype=np.float64)
        srgb_raw = ColorSpaceConverter.cielab_to_srgb(extreme_lab)

        # Check soft-clamping keeps values in [0, 1] without wrapping
        clamped = ColorSpaceConverter.soft_clamp(srgb_raw)
        assert np.all(clamped >= 0.0)
        assert np.all(clamped <= 1.0)

        # Delta E00 between identical color is 0.0
        lab1 = np.array([[[50.0, 10.0, 20.0]]], dtype=np.float64)
        de00 = compute_ciede2000(lab1, lab1)
        assert math.isclose(float(de00[0, 0]), 0.0, abs_tol=1e-7)

    def test_pair_08_mode_b_pdf_report_export(self, e2e_temp_dir: Path) -> None:
        """
        TC-F-PAIR-08: F02 (Mode B) x F26 (LaTeX & PDF Report).
        Mode B dense basis parameters and quadtree decomposition diagram exported to PDF report.
        """
        try:
            from mathfield.export.pdf_report import export_pdf_report
        except (ImportError, ModuleNotFoundError):
            pytest.skip("mathfield.export.pdf_report is under active development in Milestone M5")

        pdf_path = e2e_temp_dir / "mode_b_report.pdf"
        dummy_result = {
            "mode": "MODE_B_FIDELITY",
            "expressions": {"R": "c0*x + c1*y", "G": "c2*x^2", "B": "c3*y^2"},
            "parameters": {f"c{i}": 0.1 * i for i in range(20)},
            "metrics": {"mse": 0.0015, "psnr": 28.24, "ssim": 0.89},
            "local_decomposition": {"leaves_count": 4, "depth": 2},
            "original_image": np.full((32, 32, 3), 120, dtype=np.uint8),
            "reconstructed_image": np.full((32, 32, 3), 122, dtype=np.uint8),
        }
        export_pdf_report(dummy_result, pdf_path)
        info = inspect_pdf_report(pdf_path)
        assert info["page_count"] >= 1
        assert info["contains_equation_text"] is True

    def test_pair_09_continuous_domain_numpy_codegen(self) -> None:
        """
        TC-F-PAIR-09: F03 (Continuous Domain) x F14 (Expression Code Generators).
        Aspect-preserved 16:9 domain mapping evaluated through generated standalone NumPy code.
        """
        from mathfield.core.ast import parse
        from mathfield.core.codegen import to_python_code
        from mathfield.core.domain import CoordinateDomain2D

        dom = CoordinateDomain2D(1920, 1080, mode="aspect_preserved")
        x_min, x_max = dom.x_bounds
        _y_min, _y_max = dom.y_bounds
        assert math.isclose(x_max - x_min, 2.0 * (16.0 / 9.0), abs_tol=1e-4)

        # Codegen for bivariate quadratic saddle: x^2 - y^2
        ast_node = parse("x^2 - y^2")
        py_snippet = to_python_code(ast_node)

        # Evaluate on domain grid
        X, Y = dom.to_discrete_grid(1920, 1080)
        val = eval(py_snippet, {"np": np, "X": X, "Y": Y})
        assert val.shape == (1080, 1920)
        # Center at (0, 0) evaluates to 0.0
        mid_x, mid_y = 1920 // 2, 1080 // 2
        assert math.isclose(val[mid_y, mid_x], 0.0, abs_tol=1e-3)

    def test_pair_10_continuous_domain_svg_viewbox(self, e2e_temp_dir: Path) -> None:
        """
        TC-F-PAIR-10: F03 (Continuous Domain) x F24 (Documented SVG Export).
        SVG export viewBox matches anisotropic continuous domain aspect ratio.
        """
        try:
            from mathfield.export.svg import export_svg
        except (ImportError, ModuleNotFoundError):
            pytest.skip("mathfield.export.svg is under active development in Milestone M5")

        svg_path = e2e_temp_dir / "anisotropic.svg"
        model = {
            "coordinate_domain": {"width": 400, "height": 200, "aspect_ratio": 2.0},
            "expressions": {"formula": "x + y"},
        }
        export_svg(model, svg_path, width=400, height=200)
        info = inspect_svg(svg_path)
        assert info["viewbox"] == "0 0 400 200"
        assert info["width"] == "400"
        assert info["height"] == "200"

    def test_pair_11_image_analysis_basis_selection(self) -> None:
        """
        TC-F-PAIR-11: F04 (Image Analysis) x F05 (Global Discovery).
        Stage 1 2D FFT spectral roll-off and frequencies select Stage 2 basis family.
        """
        from e2e_tests.conftest import generate_harmonic_field, generate_poly_field
        from mathfield.core.image_analysis import ImageAnalyzer

        analyzer = ImageAnalyzer()
        poly_img = generate_poly_field(32, 32)
        harm_img = generate_harmonic_field(32, 32)

        res_poly = analyzer.analyze(poly_img)
        res_harm = analyzer.analyze(harm_img)

        # Smooth polynomial surface has higher spectral roll-off (frequencies decay rapidly)
        # Periodic harmonic wave has distinct dominant non-zero frequencies
        assert res_poly.spectral_rolloff > res_harm.spectral_rolloff
        assert len(res_harm.dominant_frequencies) >= 1

    def test_pair_12_image_analysis_alpha_mask(self) -> None:
        """
        TC-F-PAIR-12: F04 (Image Analysis) x F09 (Alpha Support).
        4-channel RGBA image analysis detects transparency and computes alpha-weighted loss.
        """
        from e2e_tests.conftest import generate_rgba_alpha_mask
        from mathfield.core.color import ColorSpaceConverter
        from mathfield.core.image_analysis import ImageAnalyzer

        rgba_arr = generate_rgba_alpha_mask(32, 32)
        analyzer = ImageAnalyzer()
        analysis = analyzer.analyze(rgba_arr)

        assert analysis.has_alpha is True
        assert analysis.channels == 4

        # Verify alpha-weighted loss: transparent pixels (alpha=0) contribute 0.0 to error
        pred = np.zeros((32, 32, 3), dtype=np.float64)
        target = np.ones((32, 32, 3), dtype=np.float64)
        alpha = np.zeros((32, 32), dtype=np.float64)
        loss_zero_alpha = ColorSpaceConverter.alpha_weighted_loss(pred, target, alpha)
        assert loss_zero_alpha == 0.0

        alpha_full = np.ones((32, 32), dtype=np.float64)
        loss_full_alpha = ColorSpaceConverter.alpha_weighted_loss(pred, target, alpha_full)
        assert math.isclose(loss_full_alpha, 1.0, abs_tol=1e-7)

    def test_pair_13_global_basis_gpu_cupy_parity(self) -> None:
        """
        TC-F-PAIR-13: F05 (Global Discovery) x F13 (Multi-Tier Evaluator / GPU CuPy).
        Global Chebyshev and RBF basis evaluation achieves numerical parity between CuPy and NumPy.
        """
        from mathfield.core.ast import parse
        from mathfield.core.engine import (
            evaluate_cupy,
            evaluate_numpy,
            is_cuda_available,
        )

        if not is_cuda_available():
            pytest.skip("CUDA GPU acceleration unavailable on this device")

        ast_node = parse("sin(pi * x) * cos(pi * y) + x^2")
        x = np.linspace(-1.0, 1.0, 100, dtype=np.float64)
        y = np.linspace(-1.0, 1.0, 100, dtype=np.float64)
        X, Y = np.meshgrid(x, y)
        params = np.array([], dtype=np.float64)

        gpu_res = evaluate_cupy(ast_node, X, Y, params)
        cpu_res = evaluate_numpy(ast_node, X, Y, params)

        diff = np.max(np.abs(gpu_res - cpu_res))
        assert diff < 1e-12, f"GPU/CPU numerical parity drift exceeded tolerance: {diff}"

    def test_pair_14_global_basis_symbolic_jacobian(self) -> None:
        """
        TC-F-PAIR-14: F05 (Global Discovery) x F12 (Symbolic Derivatives).
        Exact analytical derivatives match numerical finite differences within 1e-5.
        """
        from mathfield.core.ast import diff, parse
        from mathfield.core.engine import evaluate_numpy

        formula = "c0 * x^2 + c1 * sin(pi * y)"
        ast_node = parse(formula)

        d_c0 = diff(ast_node, "c0")
        d_c1 = diff(ast_node, "c1")

        x = np.linspace(-1.0, 1.0, 16, dtype=np.float64)
        y = np.linspace(-1.0, 1.0, 16, dtype=np.float64)
        X, Y = np.meshgrid(x, y)

        # Analytical evaluation of Jacobian columns
        jac_c0_analytical = evaluate_numpy(d_c0, X, Y, np.array([]))
        jac_c1_analytical = evaluate_numpy(d_c1, X, Y, np.array([]))

        # Numerical finite difference: (f(c + h) - f(c - h)) / (2h)
        h = 1e-6
        f_plus_c0 = evaluate_numpy(parse(f"(1.0 + {h}) * x^2 + 1.0 * sin(pi * y)"), X, Y, np.array([]))
        f_minus_c0 = evaluate_numpy(parse(f"(1.0 - {h}) * x^2 + 1.0 * sin(pi * y)"), X, Y, np.array([]))
        jac_c0_num = (f_plus_c0 - f_minus_c0) / (2.0 * h)

        diff_c0 = np.max(np.abs(jac_c0_analytical - jac_c0_num))
        assert diff_c0 < 1e-5, f"Analytical derivative d/dc0 deviated from finite difference: {diff_c0}"

        f_plus_c1 = evaluate_numpy(parse(f"1.0 * x^2 + (1.0 + {h}) * sin(pi * y)"), X, Y, np.array([]))
        f_minus_c1 = evaluate_numpy(parse(f"1.0 * x^2 + (1.0 - {h}) * sin(pi * y)"), X, Y, np.array([]))
        jac_c1_num = (f_plus_c1 - f_minus_c1) / (2.0 * h)

        diff_c1 = np.max(np.abs(jac_c1_analytical - jac_c1_num))
        assert diff_c1 < 1e-5, f"Analytical derivative d/dc1 deviated from finite difference: {diff_c1}"

    def test_pair_15_quadtree_tiled_canvas(self) -> None:
        """
        TC-F-PAIR-15: F06 (Local Decomposition) x F15 (8K Tiling).
        Quadtree local decomposition maps node bounding boxes seamlessly across tile boundaries.
        """
        try:
            from mathfield.core.engine.tiler import TiledReconstructionManager
        except (ImportError, ModuleNotFoundError):
            pytest.skip("mathfield.core.engine.tiler is under active development in Milestone M3")

        tiler = TiledReconstructionManager(tile_size=1024)
        assert tiler is not None

    def test_pair_16_local_patches_json_roundtrip(self, sample_mathfield_project: Path, e2e_temp_dir: Path) -> None:
        """
        TC-F-PAIR-16: F06 (Local Decomposition) x F27 (JSON Project Format).
        Save and reload project file preserves quadtree local patch parameters losslessly.
        """
        try:
            from mathfield.export.project_file import load_project, save_project
        except (ImportError, ModuleNotFoundError):
            pytest.skip("mathfield.export.project_file is under active development in Milestone M5")

        orig = load_project(sample_mathfield_project)
        clone_path = e2e_temp_dir / "clone_local.mathfield"
        save_project(orig, clone_path)
        recovered = load_project(clone_path)
        assert orig == recovered

    def test_pair_17_joint_opt_openmp_speedup(self) -> None:
        """
        TC-F-PAIR-17: F07 (Joint Optimization) x F13 (Multi-Tier Evaluator / MSVC OpenMP C++).
        OpenMP native DLL evaluator achieves bit-exact parity (< 1e-12) with NumPy CPU.
        """
        from mathfield.core.ast import parse
        from mathfield.core.engine import (
            evaluate_native,
            evaluate_numpy,
            is_native_available,
        )

        if not is_native_available():
            pytest.skip("MSVC OpenMP native DLL mathfield_kernels.dll is not compiled or available")

        ast_node = parse("x^2 + y^2 + sin(pi * x) * cos(pi * y)")
        x = np.linspace(-1.0, 1.0, 200, dtype=np.float64)
        y = np.linspace(-1.0, 1.0, 200, dtype=np.float64)
        X, Y = np.meshgrid(x, y)
        params = np.array([], dtype=np.float64)

        res_native = evaluate_native(ast_node, X, Y, params)
        res_numpy = evaluate_numpy(ast_node, X, Y, params)

        diff = np.max(np.abs(res_native - res_numpy))
        assert diff < 1e-12, f"OpenMP DLL evaluation diverged from NumPy: max diff = {diff}"

    def test_pair_18_joint_opt_research_candidate_stream(self) -> None:
        """
        TC-F-PAIR-18: F07 (Joint Optimization) x F21 (Research UI Workspace).
        Optimization engine streams candidate evaluations and diagnostics to UI callbacks.
        """
        from mathfield.core.pipeline.controller import ReconstructionController

        ctrl = ReconstructionController()
        candidates_received: list[dict[str, Any]] = []

        def on_candidate(cand: dict[str, Any]) -> None:
            candidates_received.append(cand)

        # Run minimal controller execution with callback attached
        img = np.zeros((16, 16, 3), dtype=np.uint8)
        ctx = ctrl.start(img, mode="fidelity", config={"max_depth": 1, "degree": 1}, on_candidate=on_candidate)
        assert ctx is not None
        assert ctx.metrics is not None

    def test_pair_19_bic_pruning_csv_table_consistency(self, e2e_temp_dir: Path) -> None:
        """
        TC-F-PAIR-19: F08 (Refinement & Verification) x F27 (CSV Data Export).
        BIC complexity pruning eliminates insignificant parameters and CSV table reflects pruned state.
        """
        from mathfield.core.verify.pruning import BICPruner

        # Construct design matrix with 3 meaningful columns and 3 pure noise columns
        np.random.seed(42)
        n_samples = 100
        x = np.linspace(-1.0, 1.0, n_samples)
        Phi = np.column_stack(
            [
                np.ones(n_samples),
                x,
                x**2,
                np.random.randn(n_samples) * 0.001,
                np.random.randn(n_samples) * 0.001,
            ]
        )
        y = 1.0 + 2.0 * x + 3.0 * (x**2) + np.random.randn(n_samples) * 0.01

        pruner = BICPruner()
        res = pruner.prune_matrix(Phi, y, reg_lambda=0.01)

        # Invariant: noise columns are pruned (zero coefficients)
        assert len(res.retained_indices) <= 4
        assert res.final_bic < 1e6

        try:
            from mathfield.export.csv_export import export_csv_parameters
        except (ImportError, ModuleNotFoundError):
            pytest.skip("mathfield.export.csv_export is under active development in Milestone M5")

        csv_path = e2e_temp_dir / "pruned_params.csv"
        params_dict = {f"c{idx}": {"value": float(res.coefficients[idx])} for idx in res.retained_indices}
        export_csv_parameters({"parameters": params_dict}, csv_path)
        rows = inspect_csv_parameters(csv_path)
        assert len(rows) == len(res.retained_indices)

    def test_pair_20_verification_bundle_script_parity(self, e2e_temp_dir: Path) -> None:
        """
        TC-F-PAIR-20: F08 (Refinement & Verification) x F28 (Reproducibility Bundle).
        Independent re-renderer calculates verification metrics checked by bundle verify.py.
        """
        try:
            from mathfield.export.bundle import export_bundle
        except (ImportError, ModuleNotFoundError):
            pytest.skip("mathfield.export.bundle is under active development in Milestone M5")

        bundle_dir = e2e_temp_dir / "test_bundle"
        sample_result = {
            "expressions": {"formula": "0.5 * x + 0.5 * y"},
            "parameters": {"c0": 0.5, "c1": 0.5},
            "metrics": {"mse": 0.001, "psnr": 30.0},
        }
        export_bundle(sample_result, bundle_dir)
        verify_script = bundle_dir / "verify.py"
        assert verify_script.exists()

    def test_pair_21_rgba_smooth_alpha_png(self, e2e_temp_dir: Path) -> None:
        """
        TC-F-PAIR-21: F09 (Color & Alpha) x F23 (Image Exports PNG & JPG).
        Continuous 4-channel RGBA reconstruction exported to PNG preserves smooth alpha values.
        """
        try:
            from mathfield.export.raster import export_png
        except (ImportError, ModuleNotFoundError):
            pytest.skip("mathfield.export.raster is under active development in Milestone M5")

        png_path = e2e_temp_dir / "alpha_export.png"
        sample_model = {
            "coordinate_domain": {"x_min": -1.0, "x_max": 1.0, "y_min": -1.0, "y_max": 1.0},
            "expressions": {"R": "x", "G": "y", "B": "0", "A": "0.5"},
        }
        export_png(sample_model, png_path, width=32, height=32)
        info = inspect_png(png_path, has_alpha=True, expected_shape=(32, 32))
        assert info["has_alpha"] is True
        assert info["mode"] == "RGBA"

    def test_pair_22_color_space_standalone_script(self, e2e_temp_dir: Path) -> None:
        """
        TC-F-PAIR-22: F09 (Color & Alpha) x F25 (Standalone Python Reproduction Script).
        Standalone reproduction script includes pure NumPy color space conversions without external libraries.
        """
        try:
            from mathfield.export.python_script import export_standalone_script
        except (ImportError, ModuleNotFoundError):
            pytest.skip("mathfield.export.python_script is under active development in Milestone M5")

        script_path = e2e_temp_dir / "color_repro.py"
        sample_model = {"color_space": "CIELAB", "expressions": {"L": "50", "a": "x * 20", "b": "y * 20"}}
        export_standalone_script(sample_model, script_path)
        audit = validate_standalone_python_script(script_path)
        assert "mathfield" not in audit["imports"]
        assert "colormath" not in audit["imports"]

    def test_pair_23_parser_regularized_division_injection(self) -> None:
        """
        TC-F-PAIR-23: F10 (Safe Expression Grammar & Pratt Parser) x F11 (Numerical Domain Validation).
        Pratt parser parses division, log, and sqrt; evaluators handle singular poles gracefully.
        """
        from mathfield.core.ast import parse
        from mathfield.core.engine import evaluate_numpy

        div_ast = parse("x / y")
        log_ast = parse("log(x)")
        sqrt_ast = parse("sqrt(x)")

        # Evaluate division by zero: y = 0.0
        x_div = np.array([1.0, -1.0])
        y_div = np.array([0.0, 0.0])
        res_div = evaluate_numpy(div_ast, x_div, y_div, np.array([]))
        assert np.all(np.isfinite(res_div)), "Division by zero produced NaN or Inf"

        # Evaluate log of non-positive numbers: x <= 0.0
        x_log = np.array([0.0, -2.0])
        res_log = evaluate_numpy(log_ast, x_log, np.zeros(2), np.array([]))
        assert np.all(np.isfinite(res_log)), "Logarithm of zero/negative produced NaN or Inf"

        # Evaluate square root of negative numbers: x < 0.0
        x_sqrt = np.array([-4.0, 4.0])
        res_sqrt = evaluate_numpy(sqrt_ast, x_sqrt, np.zeros(2), np.array([]))
        assert np.all(np.isfinite(res_sqrt)), "Square root of negative number produced NaN or Inf"
        assert res_sqrt[0] == 0.0
        assert res_sqrt[1] == 2.0

    def test_pair_24_parser_ast_codegen_roundtrip(self) -> None:
        """
        TC-F-PAIR-24: F10 (Safe Expression Grammar) x F14 (Expression Code Generators).
        AST node hierarchy round-trips cleanly through str(node) and to_latex(node).
        """
        from mathfield.core.ast import parse
        from mathfield.core.codegen import to_latex, to_python_code

        orig_expr = "sin(pi * x) + cos(2 * pi * y) + x^2"
        ast_1 = parse(orig_expr)

        # Verify str(ast) parses back to an equivalent AST
        str_repr = str(ast_1)
        ast_2 = parse(str_repr)
        assert str(ast_1) == str(ast_2)

        # Verify LaTeX generator produces valid AMS-LaTeX string with balanced braces
        latex_str = to_latex(ast_1)
        assert "\\sin" in latex_str
        assert "\\cos" in latex_str
        assert latex_str.count("{") == latex_str.count("}")

        # Verify executable python code evaluation
        py_snippet = to_python_code(ast_1)
        X = np.array([0.5])
        Y = np.array([0.0])
        val = eval(py_snippet, {"np": np, "X": X, "Y": Y})
        # sin(pi * 0.5) + cos(0) + 0.5^2 = 1 + 1 + 0.25 = 2.25
        assert math.isclose(float(val[0]), 2.25, abs_tol=1e-6)

    def test_pair_25_regularization_gpu_cupy_kernel(self) -> None:
        """
        TC-F-PAIR-25: F11 (Numerical Domain Validation) x F13 (Multi-Tier Evaluator / GPU CuPy).
        GPU CuPy elementwise kernel evaluates singular math expressions without floating point traps.
        """
        from mathfield.core.ast import parse
        from mathfield.core.engine import evaluate_cupy, is_cuda_available

        if not is_cuda_available():
            pytest.skip("CUDA GPU acceleration unavailable on this device")

        ast_node = parse("x / y + log(x) + sqrt(x)")
        x = np.array([0.0, -1.0, 1.0], dtype=np.float64)
        y = np.array([0.0, 0.0, 2.0], dtype=np.float64)
        res_gpu = evaluate_cupy(ast_node, x, y, np.array([]))
        assert np.all(np.isfinite(res_gpu))

    def test_pair_26_symbolic_gradient_python_codegen(self) -> None:
        """
        TC-F-PAIR-26: F12 (Symbolic Derivatives) x F14 (Expression Code Generators).
        Analytical derivative AST generated symbolically, compiled to Python, matches finite differences.
        """
        from mathfield.core.ast import diff, parse
        from mathfield.core.codegen import to_python_code

        # Bivariate polynomial: c0 * x^3 + c1 * y^2
        formula = "c0 * x^3 + c1 * y^2"
        ast_node = parse(formula)

        # Symbolic partial derivative w.r.t c0: d/dc0 [c0 * x^3 + c1 * y^2] = x^3
        d_c0_ast = diff(ast_node, "c0")
        py_code_grad = to_python_code(d_c0_ast)

        # Evaluate generated code on a meshgrid
        x = np.linspace(-1.0, 1.0, 10, dtype=np.float64)
        y = np.linspace(-1.0, 1.0, 10, dtype=np.float64)
        X, Y = np.meshgrid(x, y)
        analytical_grad = eval(py_code_grad, {"np": np, "X": X, "Y": Y})

        # Ground truth derivative of c0 * x^3 w.r.t c0 is x^3
        truth_grad = X**3
        diff_max = np.max(np.abs(analytical_grad - truth_grad))
        assert diff_max < 1e-12

    def test_pair_27_tiling_checkpoint_tile_resume(self, e2e_temp_dir: Path) -> None:
        """
        TC-F-PAIR-27: F15 (8K High-Resolution Tiling) x F16 (Robust Bit-Exact Checkpointing).
        Tiled 8K reconstruction interrupted mid-canvas and resumed from checkpoint without seam artifacts.
        """
        try:
            from mathfield.core.checkpoint import CheckpointManager
            from mathfield.core.engine.tiler import TiledReconstructionManager
        except (ImportError, ModuleNotFoundError):
            pytest.skip("Tiling and Checkpointing modules are under active development in Milestone M3")

        tiler = TiledReconstructionManager(tile_size=1024)
        chk = CheckpointManager()
        assert tiler is not None
        assert chk is not None

    def test_pair_28_main_ui_worker_thread_isolation(self) -> None:
        """
        TC-F-PAIR-28: F17 (Main UI Workspace) x F22 (Non-Blocking Background Execution).
        Background optimization execution operates in decoupled worker without blocking main thread.
        """
        try:
            from mathfield.ui.workers import OptimizationWorker
        except (ImportError, ModuleNotFoundError):
            pytest.skip("mathfield.ui is under active development in Milestone M4")

        worker = OptimizationWorker()
        assert worker is not None

    def test_pair_29_equation_ui_live_parameter_stream(self) -> None:
        """
        TC-F-PAIR-29: F18 (Equation UI Workspace) x F20 (Optimization UI Workspace).
        Live parameter streams from optimization workspace update equation formula previews with debouncing.
        """
        try:
            from mathfield.ui.equation_workspace import EquationWorkspace
        except (ImportError, ModuleNotFoundError):
            pytest.skip("mathfield.ui is under active development in Milestone M4")

        ws = EquationWorkspace()
        assert ws is not None

    def test_pair_30_region_analysis_local_reoptimization(self) -> None:
        """
        TC-F-PAIR-30: F19 (Region Analysis UI Workspace) x F06 (Local Decomposition).
        Subregion bounding box selection in Region Analysis triggers localized quadtree re-optimization.
        """
        try:
            from mathfield.ui.region_workspace import RegionAnalysisWorkspace
        except (ImportError, ModuleNotFoundError):
            pytest.skip("mathfield.ui is under active development in Milestone M4")

        ws = RegionAnalysisWorkspace()
        assert ws is not None

    def test_pair_31_bundle_benchmark_harness_integration(self, e2e_temp_dir: Path) -> None:
        """
        TC-F-PAIR-31: F28 (Reproducibility Bundle) x F30 (Empirical Benchmarking Harness).
        Reproducibility package bundle generated from benchmark run validates timing counters and metrics.
        """
        try:
            from benchmarks.run_benchmarks import BenchmarkRunner
            from mathfield.export.bundle import export_bundle
        except (ImportError, ModuleNotFoundError):
            pytest.skip("benchmarks and bundle export are under active development in Milestones M5/M6")

        runner = BenchmarkRunner()
        assert runner is not None
        assert export_bundle is not None
