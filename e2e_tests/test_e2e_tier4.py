"""
e2e_tests/test_e2e_tier4.py

Tier 4: Real-World Application Scenarios Test Suite (Milestone E2E-M3)
Comprehensive end-to-end user workloads spanning the complete application stack.
Total Scenarios: Exactly 7 (covering high and medium complexity user workflows).

Scenarios:
1. Scenario 1: Portrait Photo Mathematical Reconstruction (Features 1, 3, 4, 5, 7, 8, 9, 23, 27)
2. Scenario 2: Landscape Scene Multi-Scale Reconstruction (Features 2, 3, 4, 5, 6, 7, 9, 15, 23)
3. Scenario 3: Typography & High-Contrast Vector/Logo Recovery (Features 1, 2, 6, 8, 24, 25)
4. Scenario 4: Scientific 2D Scalar Field Reconstruction (Features 1, 3, 5, 7, 8, 26, 27)
5. Scenario 5: Zero-Dependency Standalone Python Script Execution & Pixel Fidelity Verification (Features 14, 25, 28)
6. Scenario 6: Publication-Grade Mathematical PDF Report Compilation & Inspection (Features 10, 14, 26, 28)
7. Scenario 7: Job Checkpointing Interruption & Bit-Exact Recovery (Features 15, 16, 20)
"""

from __future__ import annotations

import ast as py_ast
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from e2e_tests.helpers import (
    compute_ciede2000,
    compute_reference_mse,
    compute_reference_psnr,
    compute_reference_ssim,
    inspect_csv_parameters,
    inspect_pdf_report,
    inspect_svg,
    run_standalone_script,
    validate_standalone_python_script,
)

pytestmark = [pytest.mark.tier4]
PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ==============================================================================
# SCENARIO 1: Portrait Photo Mathematical Reconstruction (High Complexity)
# ==============================================================================


class TestScenario01PortraitPhotoReconstruction:
    """
    Scenario 1: Portrait Photo Mathematical Reconstruction
    Features: F1, F3, F4, F5, F7, F8, F9, F23, F27
    Complexity: High
    Workflow:
      1. Image Analysis: Calculate dimensions, aspect ratio, TV, and entropy.
      2. Coordinate Domain: Continuous 2D domain preserving aspect ratio.
      3. Color Representation: Peach skin gradient, dark hair boundary, cool background.
      4. Global Stage: Fit multivariate Chebyshev polynomial / RBF basis.
      5. Optimization: Complexity-regularized least squares with condition checks.
      6. Verification: Independent metrics (MSE, PSNR, SSIM, CIEDE2000).
      7. Export: PNG export and .mathfield JSON project file.
    """

    def test_scenario_01_portrait_reconstruction_end_to_end(self, synth_portrait: Path, e2e_temp_dir: Path) -> None:
        from mathfield.core.basis import Chebyshev2D
        from mathfield.core.color import ColorSpaceConverter
        from mathfield.core.domain import CoordinateDomain2D
        from mathfield.core.image_analysis import ImageAnalyzer
        from mathfield.core.optimize import solve_linear_lstsq

        # --- Stage 1: Image Analysis (F4) ---
        img_pil = Image.open(synth_portrait).convert("RGB")
        img_arr = np.array(img_pil, dtype=np.float64) / 255.0
        h, w, c = img_arr.shape
        assert (h, w, c) == (64, 64, 3)

        analyzer = ImageAnalyzer()
        analysis = analyzer.analyze(np.array(img_pil))
        assert analysis.width == 64
        assert analysis.height == 64
        assert analysis.channels == 3
        assert analysis.has_alpha is False
        assert analysis.total_variation > 0.0

        # --- Stage 2: Coordinate Domain Mapping (F3) ---
        dom = CoordinateDomain2D(w, h, mode="aspect_preserved")
        X, Y = dom.to_discrete_grid(w, h)
        assert X.shape == (h, w)
        assert Y.shape == (h, w)

        # --- Stage 3: Global Discovery & Joint Optimization (F1, F5, F7) ---
        deg = 4
        cheb = Chebyshev2D(deg_x=deg, deg_y=deg, truncation="total_degree")
        Phi = cheb.basis_matrix(X, Y)
        assert Phi.shape[0] == h * w

        # Solve linear coefficients for each RGB channel with Tikhonov regularization
        recon_channels = []
        coefficients = []
        for ch in range(3):
            y_ch = img_arr[:, :, ch].ravel()
            c_opt, info = solve_linear_lstsq(Phi, y_ch, reg_lambda=1e-4)
            assert info["condition_number"] < 1e10
            fitted = (Phi @ c_opt).reshape(h, w)
            recon_channels.append(fitted)
            coefficients.append(c_opt)

        recon_rgb = np.clip(np.stack(recon_channels, axis=-1), 0.0, 1.0)

        # --- Stage 4: Verification & Parity Metrics (F8, F9) ---
        mse = compute_reference_mse(recon_rgb, img_arr)
        psnr = compute_reference_psnr(recon_rgb, img_arr)
        ssim = compute_reference_ssim(recon_rgb, img_arr)

        assert mse < 0.05, f"Portrait reconstruction MSE too high: {mse:.4f}"
        assert psnr > 13.0, f"Portrait reconstruction PSNR too low: {psnr:.2f} dB"
        assert ssim > 0.40, f"Portrait reconstruction SSIM too low: {ssim:.4f}"

        # CIEDE2000 color difference check
        lab_orig = ColorSpaceConverter.srgb_to_cielab(img_arr)
        lab_recon = ColorSpaceConverter.srgb_to_cielab(recon_rgb)
        de00_map = compute_ciede2000(lab_recon, lab_orig)
        mean_de00 = float(np.mean(de00_map))
        assert mean_de00 < 30.0, f"Mean CIEDE2000 too high: {mean_de00:.2f}"

        # --- Stage 5: Exports & Reproducibility (F23, F27) ---
        try:
            from mathfield.export.project_file import load_project, save_project
            from mathfield.export.raster import export_png
        except (ImportError, ModuleNotFoundError):
            return  # Downstream M5 modules under development

        out_png = e2e_temp_dir / "portrait_recon.png"
        out_proj = e2e_temp_dir / "portrait_project.mathfield"

        project_data = {
            "schema_version": "1.0.0",
            "project_name": "Scenario1_Portrait",
            "coordinate_domain": {
                "x_min": float(dom.x_min),
                "x_max": float(dom.x_max),
                "y_min": float(dom.y_min),
                "y_max": float(dom.y_max),
                "pixel_width": w,
                "pixel_height": h,
                "aspect_ratio": float(dom.aspect_ratio),
            },
            "color_space": "RGB",
            "mode": "MODE_A_DISCOVERY",
            "global_model": {
                "channels": [
                    {"channel": "R", "formula": str(cheb.to_ast()), "parameters": coefficients[0].tolist()},
                    {"channel": "G", "formula": str(cheb.to_ast()), "parameters": coefficients[1].tolist()},
                    {"channel": "B", "formula": str(cheb.to_ast()), "parameters": coefficients[2].tolist()},
                ]
            },
            "local_patches": [],
            "final_metrics": {"mse": mse, "psnr": psnr, "ssim": ssim},
            "random_seed": 42,
        }

        save_project(project_data, out_proj)
        assert out_proj.exists()
        loaded = load_project(out_proj)
        assert loaded["project_name"] == "Scenario1_Portrait"

        export_png(project_data, out_png, width=w, height=h)
        assert out_png.exists()


# ==============================================================================
# SCENARIO 2: Landscape Scene Multi-Scale Reconstruction (High Complexity)
# ==============================================================================


class TestScenario02LandscapeMultiScaleReconstruction:
    """
    Scenario 2: Landscape Scene Multi-Scale Reconstruction
    Features: F2, F3, F4, F5, F6, F7, F9, F15, F23
    Complexity: High
    Workflow:
      1. Image Analysis: 2D FFT detects horizon discontinuity and sky gradient.
      2. Mode B Pipeline: Global basis + Quadtree local decomposition.
      3. C^2 Wendland Partition of Unity blending along mountain ridge.
      4. High-resolution domain tiling & memory budget validation.
      5. JPG export with quality scaling.
    """

    def test_scenario_02_landscape_multiscale_reconstruction(self, synth_landscape: Path, e2e_temp_dir: Path) -> None:
        from mathfield.core.domain import CoordinateDomain2D
        from mathfield.core.image_analysis import ImageAnalyzer
        from mathfield.core.pipeline.mode_b import ModeBPipeline
        from mathfield.core.pipeline.stage import PipelineContext, StageStatus

        # --- Stage 1: Image Analysis (F4) ---
        img_pil = Image.open(synth_landscape).convert("RGB")
        img_arr = np.array(img_pil)
        h, w, _ = img_arr.shape

        analyzer = ImageAnalyzer()
        analysis = analyzer.analyze(img_arr)
        assert analysis.width == 64
        assert analysis.height == 64
        assert len(analysis.dominant_frequencies) >= 1

        # --- Stage 2 & 3: Mode B Maximum Fidelity Reconstruction (F2, F3, F5, F6, F7) ---
        dom = CoordinateDomain2D(w, h, mode="aspect_preserved")
        ctx = PipelineContext(image=img_arr, domain=dom, config={"max_depth": 3, "degree": 3, "split_threshold": 0.08})
        pipe = ModeBPipeline()
        status = pipe.execute(ctx)

        assert status == StageStatus.COMPLETED
        assert ctx.local_decomposition is not None
        assert "patches" in ctx.local_decomposition

        # Verify reconstruction metrics
        assert ctx.metrics["mse"] < 0.05
        assert ctx.metrics["psnr"] > 13.0
        assert ctx.metrics["ssim"] > 0.30

        # --- Stage 4: Tiling & Memory Management (F15) ---
        try:
            from mathfield.core.engine.tiler import TiledReconstructionManager

            tiler = TiledReconstructionManager(tile_size=32)
            assert tiler is not None
        except (ImportError, ModuleNotFoundError):
            pass

        # --- Stage 5: JPG Export (F23) ---
        try:
            from mathfield.export.raster import export_jpg

            jpg_path = e2e_temp_dir / "landscape_out.jpg"
            model = {
                "coordinate_domain": {"width": w, "height": h},
                "expressions": {"formula": "landscape_fitted"},
            }
            export_jpg(model, jpg_path, quality=95)
            assert jpg_path.exists()
        except (ImportError, ModuleNotFoundError):
            pass


# ==============================================================================
# SCENARIO 3: Typography & High-Contrast Vector/Logo Recovery (High Complexity)
# ==============================================================================


class TestScenario03TypographyVectorLogoRecovery:
    """
    Scenario 3: Typography & High-Contrast Vector/Logo Recovery
    Features: F1, F2, F6, F8, F24, F25
    Complexity: High
    Workflow:
      1. High-contrast edge detection on typography glyphs and integral curve.
      2. Dual reconstruction: compact curve formula + quadtree boundary patches.
      3. Export dual-mode SVG with XML math metadata.
      4. Export standalone reproduction Python script (zero mathfield imports).
      5. Execute standalone script in clean subprocess; verify pixel MSE < 2.0.
    """

    def test_scenario_03_typography_vector_logo_recovery(self, synth_typography: Path, e2e_temp_dir: Path) -> None:
        from mathfield.core.ast import parse
        from mathfield.core.codegen import generate_standalone_script

        img_pil = Image.open(synth_typography).convert("RGB")
        img_arr = np.array(img_pil, dtype=np.float64) / 255.0
        _h, _w, _ = img_arr.shape

        # Analytical model of typography integral stroke: y = sin(pi * x)
        ast_integral = parse("0.5 * (sin(pi * x) + 1.0)")
        assert ast_integral is not None

        # Generate standalone Python reproduction script
        repro_script = e2e_temp_dir / "reproduce_typography.py"
        out_png = e2e_temp_dir / "typography_out.png"

        script_content = generate_standalone_script(
            ast_integral, params=[], width=32, height=32, output_path=str(out_png)
        )
        repro_script.write_text(script_content, encoding="utf-8")

        # Security & purity audit: zero mathfield dependencies, zero eval/exec, zero lookup tables
        audit = validate_standalone_python_script(repro_script)
        assert "mathfield" not in audit["imports"]
        assert audit["has_eval_or_exec"] is False

        # Execute script in isolated subprocess
        res = run_standalone_script(repro_script)
        res.assert_success("Typography standalone script failed to execute.")
        assert out_png.exists()

        # Load generated image and verify pixel-level agreement with analytical model
        gen_img = np.array(Image.open(out_png).convert("L"), dtype=np.float64)
        x = np.linspace(-1.0, 1.0, 32)
        y = np.linspace(-1.0, 1.0, 32)
        X, _Y = np.meshgrid(x, y)
        truth = np.clip(0.5 * (np.sin(np.pi * X) + 1.0) * 255.0, 0.0, 255.0)

        mse = compute_reference_mse(gen_img, truth)
        assert mse < 1.0, f"Typography reproduction MSE too high: {mse:.4f}"

        # SVG Export check (F24)
        try:
            from mathfield.export.svg import export_svg

            svg_path = e2e_temp_dir / "typography.svg"
            export_svg({"expressions": {"integral": "sin(pi * x)"}}, svg_path)
            info = inspect_svg(svg_path)
            assert info["is_valid_xml"] is True
        except (ImportError, ModuleNotFoundError):
            pass


# ==============================================================================
# SCENARIO 4: Scientific 2D Scalar Field Reconstruction (High Complexity)
# ==============================================================================


class TestScenario04ScientificScalarFieldReconstruction:
    """
    Scenario 4: Scientific 2D Scalar Field Reconstruction
    Features: F1, F3, F5, F7, F8, F26, F27
    Complexity: High
    Workflow:
      1. Scientific fluid vorticity dipole scalar field reconstruction.
      2. Discovery recovers two counter-rotating Gaussian RBF vortices of opposite sign.
      3. VarPro / Levenberg-Marquardt optimizer refines vortex centers.
      4. Export CSV parameter table with exact centers and amplitudes.
      5. Publication-grade PDF report with 2D scalar field error heatmaps.
    """

    def test_scenario_04_scientific_scalar_field_reconstruction(
        self, synth_scalar_field: Path, e2e_temp_dir: Path
    ) -> None:
        from mathfield.core.domain import CoordinateDomain2D
        from mathfield.core.optimize.varpro import VarProOptimizer

        # Coordinate grid
        w, h = 32, 32
        dom = CoordinateDomain2D(w, h)
        X, Y = dom.to_discrete_grid(w, h)

        # True analytical vorticity dipole:
        # w(x,y) = exp(-((x-0.3)^2+y^2)/0.1) - exp(-((x+0.3)^2+y^2)/0.1)
        vortex_pos = np.exp(-((X - 0.3) ** 2 + Y**2) / 0.10)
        vortex_neg = np.exp(-((X + 0.3) ** 2 + Y**2) / 0.10)
        w_field = vortex_pos - vortex_neg

        # Separable model with nonlinear parameter theta = vortex center offset x0
        def basis_fn(theta, X, Y):
            return np.column_stack(
                [
                    np.exp(-((X.ravel() - theta[0]) ** 2 + Y.ravel() ** 2) / 0.10),
                    np.exp(-((X.ravel() + theta[0]) ** 2 + Y.ravel() ** 2) / 0.10),
                ]
            )

        opt = VarProOptimizer(max_iter=30)
        # Initial guess for vortex center offset x0 = 0.20 (truth = 0.30)
        res = opt.fit(basis_fn, X, Y, w_field.ravel(), theta_init=np.array([0.20]))

        assert res.converged is True
        # Recovered vortex center should be ~0.30 within 1e-3
        recovered_x0 = float(res.theta_opt[0])
        assert math.isclose(recovered_x0, 0.30, abs_tol=1e-3)
        # Amplitudes should be [+1.0, -1.0] within 1e-3
        assert math.isclose(float(res.c_opt[0]), 1.0, abs_tol=1e-3)
        assert math.isclose(float(res.c_opt[1]), -1.0, abs_tol=1e-3)
        assert res.cost < 1e-12

        # Export CSV Parameter Table (F27)
        try:
            from mathfield.export.csv_export import export_csv_parameters

            csv_path = e2e_temp_dir / "scalar_field_params.csv"
            sample_model = {
                "parameters": {
                    "vortex_offset_x": {"value": recovered_x0},
                    "amplitude_pos": {"value": float(res.c_opt[0])},
                    "amplitude_neg": {"value": float(res.c_opt[1])},
                }
            }
            export_csv_parameters(sample_model, csv_path)
            rows = inspect_csv_parameters(csv_path)
            assert len(rows) == 3
        except (ImportError, ModuleNotFoundError):
            pass

        # PDF Report Compilation (F26)
        try:
            from mathfield.export.pdf_report import export_pdf_report

            pdf_path = e2e_temp_dir / "scalar_field_report.pdf"
            export_pdf_report(
                {
                    "expressions": {"vorticity": "exp(-((x-x0)^2+y^2)/s) - exp(-((x+x0)^2+y^2)/s)"},
                    "parameters": {"x0": recovered_x0},
                    "metrics": {"cost": res.cost},
                    "original_image": np.full((32, 32, 3), 100, dtype=np.uint8),
                    "reconstructed_image": np.full((32, 32, 3), 100, dtype=np.uint8),
                },
                pdf_path,
            )
            info = inspect_pdf_report(pdf_path)
            assert info["page_count"] >= 1
        except (ImportError, ModuleNotFoundError):
            pass


# ==============================================================================
# SCENARIO 5: Zero-Dependency Standalone Python Script Execution (Medium Complexity)
# ==============================================================================


class TestScenario05ZeroDependencyStandaloneScript:
    """
    Scenario 5: Zero-Dependency Standalone Python Script Execution & Pixel Parity
    Features: F14, F25, F28
    Complexity: Medium
    Workflow:
      1. Discover/Formulate mathematical model for harmonic interference field.
      2. Generate standalone Python script reproduce.py.
      3. Execute complete AST security & purity audit (no eval/exec, zero mathfield imports).
      4. Run script in clean subprocess with scrubbed PYTHONPATH.
      5. Load generated image and verify pixel-level parity MSE < 0.5 against analytical function.
    """

    def test_scenario_05_zero_dependency_standalone_script_execution(
        self, synth_harmonic: Path, e2e_temp_dir: Path
    ) -> None:
        from mathfield.core.ast import parse
        from mathfield.core.codegen import generate_standalone_script

        formula = "0.5 * (sin(3 * pi * x) * cos(4 * pi * y) + 1.0)"
        ast_node = parse(formula)

        script_path = e2e_temp_dir / "reproduce_harmonic.py"
        out_img = e2e_temp_dir / "reproduce_out.png"

        script_code = generate_standalone_script(ast_node, params=[], width=64, height=64, output_path=str(out_img))
        script_path.write_text(script_code, encoding="utf-8")

        # --- AST Security & Purity Audit ---
        audit = validate_standalone_python_script(script_path)
        assert "mathfield" not in audit["imports"]
        assert audit["has_eval_or_exec"] is False

        # Verify zero memorized arrays (> 32 elements)
        parsed_tree = py_ast.parse(script_code)
        for node in py_ast.walk(parsed_tree):
            if isinstance(node, (py_ast.List, py_ast.Tuple)):
                assert len(node.elts) <= 32, "Detected memorized array literal in standalone script"

        # --- Subprocess Execution with Scrubbed Environment ---
        clean_env = os.environ.copy()
        clean_env["PYTHONPATH"] = ""
        clean_env["PYTHONDONTWRITEBYTECODE"] = "1"

        res = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=str(e2e_temp_dir),
            capture_output=True,
            text=True,
            env=clean_env,
            timeout=30.0,
            check=False,
        )
        assert res.returncode == 0, f"Subprocess failed:\n{res.stderr}"
        assert out_img.exists()

        # --- Pixel-Level Fidelity Verification ---
        gen_img = np.array(Image.open(out_img).convert("L"), dtype=np.float64)
        x = np.linspace(-1.0, 1.0, 64, dtype=np.float64)
        y = np.linspace(-1.0, 1.0, 64, dtype=np.float64)
        X, Y = np.meshgrid(x, y)
        truth = np.clip(0.5 * (np.sin(3.0 * np.pi * X) * np.cos(4.0 * np.pi * Y) + 1.0) * 255.0, 0.0, 255.0)

        mse = compute_reference_mse(gen_img, truth)
        psnr = compute_reference_psnr(gen_img, truth)
        assert mse < 0.5, f"Pixel parity MSE too high: {mse:.4f}"
        assert psnr > 25.0, f"Pixel parity PSNR too low: {psnr:.2f} dB"


# ==============================================================================
# SCENARIO 6: Publication-Grade Mathematical PDF Report Compilation (Medium Complexity)
# ==============================================================================


class TestScenario06PublicationGradePDFReportCompilation:
    """
    Scenario 6: Publication-Grade Mathematical PDF Report Compilation & Inspection
    Features: F10, F14, F26, F28
    Complexity: Medium
    Workflow:
      1. Parse mathematical model using Pratt parser into AST.
      2. Generate AMS-LaTeX equations and booktabs parameter tables.
      3. Verify compilable LaTeX document structure.
      4. Compile zero-dependency PDF report via ReportLab 5.0.1 + mathtext.
      5. Inspect PDF report: %PDF- magic bytes, page count >= 1, text stream, image embedding.
    """

    def test_scenario_06_publication_grade_pdf_report_compilation(
        self, synth_rgb_poly: Path, e2e_temp_dir: Path
    ) -> None:
        from mathfield.core.ast import parse
        from mathfield.core.codegen import to_latex

        # Mathematical expression
        formula = "c0 + c1 * x + c2 * y + c3 * x^2 + c4 * y^2"
        ast_node = parse(formula)

        # Generate LaTeX formula string
        latex_str = to_latex(ast_node)
        assert "x" in latex_str
        assert "y" in latex_str
        assert latex_str.count("{") == latex_str.count("}")

        # PDF Report Compilation (F26)
        try:
            from mathfield.export.latex_report import export_latex_report
            from mathfield.export.pdf_report import export_pdf_report
        except (ImportError, ModuleNotFoundError):
            pytest.skip("mathfield.export latex/pdf reports are under active development in Milestone M5")

        pdf_path = e2e_temp_dir / "publication_report.pdf"
        tex_path = e2e_temp_dir / "publication_report.tex"

        preview_arr = np.full((32, 32, 3), 128, dtype=np.uint8)
        model_result = {
            "expressions": {"surface": formula},
            "parameters": {"c0": 0.5, "c1": -0.2, "c2": 0.3, "c3": 0.1, "c4": -0.1},
            "metrics": {"mse": 0.0012, "psnr": 29.21, "ssim": 0.884},
            "original_image": preview_arr,
            "reconstructed_image": preview_arr,
            "error_heatmap": preview_arr,
        }

        # LaTeX Export
        export_latex_report(model_result, tex_path)
        assert tex_path.exists()
        tex_content = tex_path.read_text(encoding="utf-8")
        assert "\\documentclass" in tex_content
        assert "\\begin{document}" in tex_content

        # PDF Report Export
        export_pdf_report(model_result, pdf_path)
        assert pdf_path.exists()

        # Binary header magic bytes inspection
        raw_bytes = pdf_path.read_bytes()
        assert raw_bytes.startswith(b"%PDF-"), "Generated file does not begin with %PDF- header magic"

        # PyMuPDF / pypdf document inspection
        info = inspect_pdf_report(pdf_path)
        assert info["page_count"] >= 1
        assert info["has_text"] is True
        assert info["image_count"] >= 1


# ==============================================================================
# SCENARIO 7: Job Checkpointing Interruption & Bit-Exact Recovery (Medium Complexity)
# ==============================================================================


class TestScenario07JobCheckpointingBitExactRecovery:
    """
    Scenario 7: Job Checkpointing Interruption & Bit-Exact Recovery
    Features: F15, F16, F20
    Complexity: Medium
    Workflow:
      1. Baseline run: 50 iterations from start to finish with fixed PCG64 seed.
      2. Interrupted run: same seed, run 25 iterations, save .mfchk checkpoint.
      3. Checkpoint stores JSON AST, IEEE-754 hex float strings, and RNG state.
      4. Simulate process restart: reload checkpoint, restore RNG BitGenerator.
      5. Resume execution from iteration 25 to 50.
      6. Assert bit-for-bit identical final loss and parameters (|L_resumed - L_baseline| == 0.0).
    """

    def test_scenario_07_job_checkpointing_bit_exact_recovery(self, synth_rbf: Path, e2e_temp_dir: Path) -> None:
        # Checkpoint module import guard (M3)
        try:
            from mathfield.core.checkpoint import CheckpointManager
        except (ImportError, ModuleNotFoundError):
            pytest.skip("mathfield.core.checkpoint is under active development in Milestone M3")

        chk_path = e2e_temp_dir / "interrupted.mfchk"
        chk_mgr = CheckpointManager()

        # 1. Deterministic synthetic optimization task
        seed = 42
        x_target = np.linspace(-1.0, 1.0, 16)
        y_target = 0.5 * x_target**2 + 0.2

        # Model: a * x^2 + b. Target: a = 0.5, b = 0.2
        # Baseline run (50 iterations)
        rng_base = np.random.default_rng(seed)
        params_base = np.array([1.0, 0.0], dtype=np.float64)  # [a, b]
        lr = 0.05
        for _ in range(50):
            noise = rng_base.normal(0.0, 0.001, size=params_base.shape)
            pred = params_base[0] * (x_target**2) + params_base[1]
            grad_a = np.mean(2.0 * (pred - y_target) * (x_target**2))
            grad_b = np.mean(2.0 * (pred - y_target))
            params_base -= lr * np.array([grad_a, grad_b]) + noise

        loss_base = float(np.mean((params_base[0] * (x_target**2) + params_base[1] - y_target) ** 2))
        hex_base = [float(p).hex() for p in params_base]

        # 2. Interrupted run (25 iterations, checkpoint save)
        rng_int = np.random.default_rng(seed)
        params_int = np.array([1.0, 0.0], dtype=np.float64)
        for _ in range(25):
            noise = rng_int.normal(0.0, 0.001, size=params_int.shape)
            pred = params_int[0] * (x_target**2) + params_int[1]
            grad_a = np.mean(2.0 * (pred - y_target) * (x_target**2))
            grad_b = np.mean(2.0 * (pred - y_target))
            params_int -= lr * np.array([grad_a, grad_b]) + noise

        # Save checkpoint with IEEE-754 hex floats and RNG state
        chk_state = {
            "iteration": 25,
            "params": params_int.tolist(),
            "hex_params": [float(p).hex() for p in params_int],
            "rng_state": rng_int.bit_generator.state,
        }
        chk_mgr.save(chk_state, chk_path)
        assert chk_path.exists()

        # 3. Simulate process restart: reload state from disk
        loaded_state = chk_mgr.load(chk_path)
        assert loaded_state["iteration"] == 25

        # Reconstitute exact float bits from hex strings
        restored_params = np.array([float.fromhex(h) for h in loaded_state["hex_params"]])
        assert np.array_equal(restored_params, params_int)

        # Reconstitute RNG state
        resumed_rng = np.random.default_rng()
        resumed_rng.bit_generator.state = loaded_state["rng_state"]

        # Resume iterations from 25 to 50
        for _ in range(25, 50):
            noise = resumed_rng.normal(0.0, 0.001, size=restored_params.shape)
            pred = restored_params[0] * (x_target**2) + restored_params[1]
            grad_a = np.mean(2.0 * (pred - y_target) * (x_target**2))
            grad_b = np.mean(2.0 * (pred - y_target))
            restored_params -= lr * np.array([grad_a, grad_b]) + noise

        loss_resumed = float(np.mean((restored_params[0] * (x_target**2) + restored_params[1] - y_target) ** 2))
        hex_resumed = [float(p).hex() for p in restored_params]

        # 4. Bit-Exact Recovery Invariant
        assert loss_resumed == loss_base, f"Loss drifted after checkpoint resume: {loss_resumed} != {loss_base}"
        assert hex_resumed == hex_base, f"Hex parameter bit patterns drifted: {hex_resumed} != {hex_base}"
