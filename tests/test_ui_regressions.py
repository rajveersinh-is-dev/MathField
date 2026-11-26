"""Behavioural regression tests for UI worker and workspace state handling."""

from __future__ import annotations

import json
import threading

import numpy as np
import pytest

from mathfield.ui.workers import OptimizationWorker
from mathfield.ui.workspaces.equation_workspace import EquationWorkspace
from mathfield.ui.workspaces.main_workspace import MainWorkspace
from mathfield.ui.workspaces.optimization_workspace import OptimizationWorkspace
from mathfield.ui.workspaces.region_workspace import RegionWorkspace
from mathfield.ui.workspaces.research_workspace import ResearchWorkspace


def test_worker_reuse_and_error_signal_delivery() -> None:
    worker = OptimizationWorker()
    seen: list[str] = []
    worker.signals.error.connect(lambda message: seen.append(message))

    assert worker.start_job(lambda: 1 / 0) is True
    assert worker.wait(2000)
    assert seen and "division by zero" in seen[0]

    results: list[object] = []
    worker.signals.finished.connect(lambda value: results.append(value))
    assert worker.start_job(lambda: 42) is True
    assert worker.wait(2000)
    assert results == [42]
    assert worker.is_busy is False


def test_worker_cancellation_notifies_once_and_can_be_reused() -> None:
    task_started = threading.Event()
    release_task = threading.Event()

    def cancellable_task() -> str:
        task_started.set()
        assert release_task.wait(timeout=2.0)
        return "discarded"

    worker = OptimizationWorker(cancellable_task)
    events: list[tuple[str, object]] = []
    worker.signals.progress.connect(lambda value: events.append(("progress", value)))
    worker.signals.iteration_completed.connect(
        lambda iteration, info: events.append(("iteration_completed", (iteration, info)))
    )
    worker.signals.finished.connect(lambda result: events.append(("finished", result)))
    worker.signals.error.connect(lambda message: events.append(("error", message)))

    assert worker.start_job() is True
    assert task_started.wait(timeout=2.0)
    worker.request_cancel()
    events.clear()
    release_task.set()
    assert worker.wait(2000) is True

    assert events == [("error", "Operation cancelled")]
    assert worker.is_busy is False

    events.clear()
    assert worker.start_job(lambda: 42) is True
    assert worker.wait(2000) is True

    assert events == [
        ("progress", 0.0),
        ("iteration_completed", (1, {"status": "complete"})),
        ("progress", 1.0),
        ("finished", 42),
    ]
    assert worker.is_busy is False


def test_optimization_workspace_terminal_states_and_downsampling() -> None:
    ws = OptimizationWorkspace()
    ws.cancel()
    ws.resume()
    assert ws.get_status() == "CANCELLED"

    ws2 = OptimizationWorkspace()
    ws2.update_metrics(0, {"loss": float("nan")})
    ws2.resume()
    assert ws2.get_status() == "DIVERGED"

    ws3 = OptimizationWorkspace()
    for i in range(2500):
        ws3.update_metrics(i, {"loss": float(i)})
    assert len(ws3.get_metric_curves()["loss"]) <= 1000


def test_research_log_honours_requested_format(tmp_path) -> None:
    ws = ResearchWorkspace()
    ws.log_candidate(iteration=1, expression="x", status="ACCEPTED", rationale="NONE", delta_loss=-0.1)
    json_path = tmp_path / "log.json"
    ws.export_research_log(json_path, format="json")
    assert json.loads(json_path.read_text(encoding="utf-8"))[0]["expression"] == "x"
    with pytest.raises(ValueError):
        ws.export_research_log(tmp_path / "bad.xml", format="xml")


def test_region_reoptimization_rejects_invalid_roi() -> None:
    ws = RegionWorkspace()
    ws.set_roi_selection(0.0, 0.0, 0.0, 0.0)
    result = ws.trigger_local_reoptimization()
    assert result["status"] == "INVALID_ROI"
    assert result["iterations"] == 0
    assert result["delta_mse"] == 0.0


def test_main_workspace_copies_arrays_and_validates_shape(tmp_path) -> None:
    ws = MainWorkspace()
    source = np.zeros((8, 8, 3), dtype=np.float64)
    assert ws.load_image(source) is True
    source[:] = 1.0
    assert np.all(ws._original_image == 0.0)
    assert ws.load_image(np.zeros((8, 8), dtype=np.float64)) is False


def test_equation_card_renders_and_falls_back() -> None:
    ws = EquationWorkspace()
    assert ws.render_latex_card(r"$x^2 + \frac{1}{y}$").startswith(b"\x89PNG")
    assert ws.render_latex_card("unmatched $ delimiter here").startswith(b"\x89PNG")
