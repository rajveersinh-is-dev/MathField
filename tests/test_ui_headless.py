"""
tests/test_ui_headless.py

Headless UI Workspace Verification Tests.
Verifies instantiation, event processing, rendering, and interaction across all 5 UI workspaces
without requiring a GUI display server or window manager.
"""

from __future__ import annotations

import numpy as np

from mathfield.ui.main_window import MainWindow
from mathfield.ui.workers import OptimizationWorker, WorkerSignals
from mathfield.ui.workspaces.equation_workspace import EquationWorkspace
from mathfield.ui.workspaces.main_workspace import MainWorkspace
from mathfield.ui.workspaces.optimization_workspace import OptimizationWorkspace
from mathfield.ui.workspaces.region_workspace import RegionWorkspace
from mathfield.ui.workspaces.research_workspace import ResearchWorkspace


def test_main_workspace_headless():
    ws = MainWorkspace()
    ws.resize(800, 600)
    assert ws.width() == 800
    assert ws.height() == 600

    # Image loading and dimensions
    arr = np.zeros((128, 128, 3), dtype=np.uint8)
    ok = ws.load_image(arr)
    assert ok is True
    assert ws.image_dimensions == (128, 128)

    # Zoom clamping
    ws.set_zoom(0.001)
    assert ws.current_zoom >= 0.1
    ws.set_zoom(100.0)
    assert ws.current_zoom <= 50.0

    # Curtain split
    ws.set_curtain_split(0.75)
    assert ws.curtain_split_position == 0.75

    # Preview buffer
    preview = np.ones((64, 64, 3), dtype=np.uint8) * 128
    ws.update_preview(preview)
    assert ws.get_preview_buffer().shape == (64, 64, 3)
    assert isinstance(ws.get_status_text(), str)


def test_equation_workspace_headless():
    ws = EquationWorkspace()
    tree = ws.get_equation_tree()
    assert isinstance(tree, dict)

    # Mathtext rendering
    card = ws.render_latex_card("c_0 \\cdot x + c_1 \\cdot y")
    assert card is not None

    # AST inspection
    info = ws.inspect_ast_node("root")
    assert "type" in info

    # Parameter setting and retrieval
    ws.set_parameter_value("c0", 3.14)
    assert ws.get_parameter_value("c0") == 3.14

    # Event debouncing
    ws.debounce_parameter_events([1.0, 2.0, 3.0])
    assert ws.last_dispatched_value == 3.0

    # Validation
    valid_res = ws.validate_formula_input("sin(x) + cos(y)")
    assert valid_res["is_valid"] is True
    invalid_res = ws.validate_formula_input("sin(x + ")
    assert invalid_res["is_valid"] is False


def test_region_workspace_headless():
    ws = RegionWorkspace()
    ws.set_roi_selection(-0.25, -0.25, 0.25, 0.25)
    assert ws.get_roi_selection() == (-0.25, -0.25, 0.25, 0.25)
    assert ws.is_roi_valid() is True

    # Re-optimization trigger
    res = ws.trigger_local_reoptimization()
    assert res["status"] == "SUCCESS"

    # C2 boundary blending
    blended = ws.blend_subdomain_patch(None, np.zeros((32, 32)))
    assert blended.shape == (32, 32)

    # 5 Heatmap types
    for m in ["mse", "ssim", "ciede2000", "abs_diff", "signed_diff"]:
        hm = ws.generate_heatmap(m)
        assert hm.shape == (64, 64)

    # Colormap normalization
    raw = np.linspace(0.0, 10.0, 20)
    norm = ws.normalize_heatmap(raw)
    assert norm.dtype == np.uint8
    assert norm[0] == 0
    assert norm[-1] == 255


def test_optimization_workspace_headless():
    ws = OptimizationWorkspace()
    for it in range(5):
        ws.update_metrics(it, {"loss": 1.0 / (it + 1)})
    curves = ws.get_metric_curves()
    assert len(curves["loss"]) == 5

    telem = ws.get_hardware_telemetry()
    assert "cpu" in telem and "ram" in telem

    ws.pause()
    assert ws.is_paused is True
    ws.resume()
    assert ws.is_paused is False
    ws.cancel()
    assert ws.is_cancelled is True


def test_research_workspace_headless(tmp_path):
    ws = ResearchWorkspace()
    ws.log_candidate(1, "sin(x)", "ACCEPTED", "NONE", -0.1)
    ws.log_candidate(2, "x^7", "REJECTED", "BIC_INCREASE", 0.05)
    log = ws.get_candidate_log()
    assert len(log) == 2

    reasons = ws.get_rejection_reasons()
    assert "BIC_INCREASE" in reasons
    assert "ILL_CONDITIONED" in reasons

    ws.update_parameter_trajectories(1, {"c0": 1.0})
    ws.update_parameter_trajectories(2, {"c0": 0.5})
    assert len(ws.get_trajectories()["c0"]) == 2

    s = np.array([10.0, 2.0, 0.01])
    kappa = ws.update_svd_spectrum(s)
    assert kappa == 1000.0

    csv_path = tmp_path / "research_log.csv"
    ws.export_research_log(csv_path)
    assert csv_path.exists()


def test_workers_and_main_window():
    win = MainWindow()
    assert win.main_workspace is not None
    assert win.equation_workspace is not None

    worker = win.create_worker()
    assert isinstance(worker, OptimizationWorker)
    assert isinstance(worker.signals, WorkerSignals)

    executed = []
    worker.set_task(lambda: executed.append(1))
    worker.start()
    worker.wait(1000)
    assert len(executed) == 1
