"""Regressions from the independent post-hardening code reviews."""

from __future__ import annotations

import json
import math
import threading
import time
from pathlib import Path

import numpy as np
import pytest

from cli import _format_mode_b_channel, main as cli_main
from mathfield.core.ast.parser import parse
from mathfield.core.basis.fourier import Fourier2D
from mathfield.core.basis.polynomial import Chebyshev2D
from mathfield.core.domain import CoordinateDomain2D
from mathfield.core.engine.cpu_numpy import evaluate_numpy
from mathfield.core.optimize.solvers import LevenbergMarquardtOptimizer
from mathfield.core.verify.metrics import compute_mae, compute_mse
from mathfield.export.bundle import export_bundle
from mathfield.export.python_script import export_standalone_script
from mathfield.export.raster import natural_key
from mathfield.ui.workers import OptimizationWorker
from mathfield.ui.workspaces.equation_workspace import EquationWorkspace
from mathfield.ui.workspaces.main_workspace import MainWorkspace
from mathfield.ui.workspaces.optimization_workspace import OptimizationWorkspace
from mathfield.ui.workspaces.region_workspace import RegionWorkspace


def test_levenberg_marquardt_converges_with_correct_damping() -> None:
    optimizer = LevenbergMarquardtOptimizer(max_iter=80, initial_lambda=1e-3)
    result = optimizer.minimize(
        residual_fn=lambda x: np.array([x[0] - 3.0, 2.0 * x[1] + 1.0]),
        jacobian_fn=lambda x: np.array([[1.0, 0.0], [0.0, 2.0]]),
        x0=np.array([0.0, 0.0]),
    )
    assert result["converged"] is True
    np.testing.assert_allclose(result["x"], [3.0, -0.5], atol=1e-5)


def test_scalar_hyperbolic_and_clamp_propagate_nan() -> None:
    for expression in ("sinh(c0)", "cosh(c0)", "sign(c0)", "clamp(c0, -1, 1)"):
        tree = parse(expression)
        value = tree.eval_scalar(0.0, 0.0, [float("nan")])
        assert math.isnan(value), expression
    # clamp must follow np.clip ordering when lo > hi.
    assert parse("clamp(3, 5, 1)").eval_scalar(0.0, 0.0, []) == 1.0


def test_weighted_metrics_are_channel_consistent() -> None:
    zeros = np.zeros((2, 2, 3))
    ones = np.ones((2, 2, 3))
    weights = np.ones((2, 2))
    assert compute_mse(zeros, ones, weights) == pytest.approx(1.0)
    assert compute_mae(zeros, ones, weights) == pytest.approx(1.0)
    assert compute_mse(zeros, ones, np.ones(3)) == pytest.approx(1.0)


def test_cli_machine_expression_uses_chebyshev_basis() -> None:
    domain = CoordinateDomain2D(32, 24)
    X, Y = domain.to_discrete_grid()
    basis = Chebyshev2D(deg_x=3, deg_y=3, truncation="total_degree")
    fourier = Fourier2D(freq_x=1, freq_y=1, truncation="rectangular")
    size = basis.num_terms + 2 * len(fourier.freq_pairs)
    coeffs = np.zeros(size)
    coeffs[: basis.num_terms] = np.linspace(-2.0, 2.0, basis.num_terms)
    model = {
        "chebyshev_basis": basis,
        "fourier_basis": fourier,
        "coefficients": np.column_stack([coeffs, coeffs, coeffs]),
    }
    expr = _format_mode_b_channel(model, 0)
    actual = evaluate_numpy(parse(expr), X, Y, np.array([]))
    expected = (basis.basis_matrix(X, Y) @ coeffs[: basis.num_terms]).reshape(X.shape)
    np.testing.assert_allclose(actual, expected, atol=1e-10)
    assert "·" not in expr and "π" not in expr


def test_natural_key_handles_mixed_stage_names() -> None:
    names = ["p_000_t_01", "g_term_0001", "c10", "c2", "c1"]
    assert sorted(names, key=natural_key) == ["c1", "c2", "c10", "g_term_0001", "p_000_t_01"]


def test_standalone_script_reserves_parameter_indices(tmp_path: Path) -> None:
    expr = " + ".join(f"c{i} * x" for i in range(41))
    out = tmp_path / "reproduce.py"
    export_standalone_script({"expressions": {"R": expr, "G": "0", "B": "0"}}, out)
    source = out.read_text(encoding="utf-8")
    namespace: dict = {"__name__": "reproduce_test"}
    exec(compile(source, str(out), "exec"), namespace)
    image = namespace["reconstruct"](8, 8)
    assert np.asarray(image).shape == (8, 8, 3)


def test_worker_concurrent_start_job_is_rejected() -> None:
    release = threading.Event()
    worker = OptimizationWorker(lambda: release.wait(5.0))
    gate = threading.Barrier(8)
    results: list[bool] = []

    def attempt() -> None:
        gate.wait()
        results.append(worker.start_job())

    threads = [threading.Thread(target=attempt) for _ in range(8)]
    for thread in threads:
        thread.start()
    deadline = time.monotonic() + 3.0
    while len(results) < 8 and time.monotonic() < deadline:
        time.sleep(0.01)
    release.set()
    for thread in threads:
        thread.join(timeout=2.0)
    assert sum(results) == 1
    worker.wait(2000)


def test_terminal_workspace_states_are_immutable() -> None:
    ws = OptimizationWorkspace()
    ws.cancel()
    ws.pause()
    assert ws.get_status() == "CANCELLED"
    diverged = OptimizationWorkspace()
    diverged.update_metrics(1, {"loss": float("nan")})
    diverged.pause()
    assert diverged.get_status() == "DIVERGED"
    negative = OptimizationWorkspace()
    negative.update_metrics(1, {"loss": -1e20})
    assert negative.has_diverged is True


def test_ui_arrays_are_copied_and_normalised() -> None:
    ws = MainWorkspace()
    rgba = np.zeros((4, 4, 4), dtype=np.uint8)
    assert ws.load_image(rgba) is True
    assert ws._original_image.shape == (4, 4, 3)
    rgba[:] = 255
    assert np.all(ws._original_image == 0)
    preview = np.zeros((4, 4, 3), dtype=np.uint8)
    ws.update_preview(preview)
    preview[:] = 255
    assert np.all(ws.get_preview_buffer() == 0)

    equation = EquationWorkspace()
    with pytest.raises(ValueError):
        equation.set_parameter_value("c0", float("nan"))


def test_region_heatmap_accepts_real_residual() -> None:
    ws = RegionWorkspace()
    residual = np.array([[0.1, -0.2], [0.3, 0.0]])
    mse_map = ws.generate_heatmap("mse", error_map=residual, synthetic=False)
    np.testing.assert_allclose(mse_map, residual**2)
    with pytest.raises(ValueError):
        ws.generate_heatmap("mse", synthetic=False)


def test_poster_and_latex_accept_pi_adjacent_terms(tmp_path: Path) -> None:
    from mathfield.export.artwork_poster import export_artwork_poster
    from mathfield.export.latex_report import export_latex_report

    model = {
        "coordinate_domain": {
            "x_min": -2.0,
            "x_max": 2.0,
            "y_min": -1.0,
            "y_max": 1.0,
            "pixel_width": 320,
            "pixel_height": 200,
        },
        "global_model": {"degree": 3},
        "global_equations": {"R": "+0.5·cos(2πx) +0.2·T_2(y)", "G": "0", "B": "0"},
        "rendered_image": np.zeros((32, 32, 3)),
    }
    export_artwork_poster(model, tmp_path / "poster.png", width=640)
    export_latex_report(model, tmp_path / "report.tex")
    tex = (tmp_path / "report.tex").read_text(encoding="utf-8")
    assert r"\pix" not in tex
    assert r"\pi " in tex


def test_demo_scripts_emit_machine_expressions() -> None:
    root = Path(__file__).resolve().parents[1]
    for name in ("run_lotus_reconstruction.py", "run_web_reconstruction.py"):
        source = (root / name).read_text(encoding="utf-8")
        assert 'machine_expressions = {' in source
        assert '"R": machine_expressions["R"]' in source


def test_bundle_does_not_change_global_reportlab_state(tmp_path: Path) -> None:
    from reportlab import rl_config

    original = rl_config.invariant
    model = {"expressions": {"R": "0", "G": "0", "B": "0"}, "parameters": {}}
    export_bundle(model, tmp_path / "bundle")
    assert rl_config.invariant == original


def test_cli_remains_reproducible_after_machine_expression_changes(tmp_path: Path) -> None:
    import subprocess
    import sys

    from PIL import Image

    from examples.generate_sample import build_sample

    image = tmp_path / "input.png"
    Image.fromarray((build_sample(64, 48) * 255.0).astype(np.uint8)).save(image)
    out = tmp_path / "out"
    assert cli_main(
        ["-i", str(image), "-o", str(out), "-m", "fidelity", "--export-bundle", "-q"]
    ) == 0
    project = json.loads((out / "mathfield_out.mathfield").read_text(encoding="utf-8"))
    assert project["final_metrics"]["psnr"] > 20.0

    bundle = out / "bundle"
    script = bundle / "reproduce.py"
    assert "R = (0.0 * X)" not in script.read_text(encoding="utf-8")
    rendered = bundle / "standalone.png"
    result = subprocess.run(
        [sys.executable, str(script), "-o", str(rendered), "-r", "64", "48"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    pixels = np.asarray(Image.open(rendered).convert("RGB"), dtype=np.float64)
    assert pixels.std() > 1.0, "Standalone reproduction rendered a uniform image"
