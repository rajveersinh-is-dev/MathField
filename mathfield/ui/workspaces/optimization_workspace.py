"""
mathfield/ui/workspaces/optimization_workspace.py

Feature 20: Optimization UI Workspace.
Real-time convergence curves, hardware resource telemetry (psutil fallback),
divergence detection, pause/resume/cancel controls, and cooperative token signaling.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List


class OptimizationWorkspace:
    """
    Optimization Workspace: Metric plotting, hardware monitoring,
    job control state machine, and divergence protection.
    """

    def __init__(self) -> None:
        self.is_paused: bool = False
        self.is_cancelled: bool = False
        self.has_diverged: bool = False
        self._status: str = "IDLE"
        self._curves: Dict[str, List[float]] = {"loss": [], "psnr": [], "mse": []}
        self._iterations: List[int] = []

    def pause(self) -> None:
        """Pause active optimization loop."""
        self.is_paused = True
        self._status = "PAUSED"

    def resume(self) -> None:
        """Resume paused optimization loop."""
        self.is_paused = False
        self._status = "RUNNING"

    def cancel(self) -> None:
        """Issue cooperative cancellation token."""
        self.is_cancelled = True
        self._status = "CANCELLED"

    def get_status(self) -> str:
        """Retrieve current workspace execution status."""
        return self._status

    def update_metrics(self, iteration: int, metrics: Dict[str, Any]) -> None:
        """
        Record iteration metric values and detect numerical divergence.
        """
        self._iterations.append(iteration)
        for k, v in metrics.items():
            if k not in self._curves:
                self._curves[k] = []
            val = float(v)
            self._curves[k].append(val)
            if math.isnan(val) or math.isinf(val) or val > 1e12:
                self.has_diverged = True
                self._status = "DIVERGED"

    def get_metric_curves(self) -> Dict[str, List[float]]:
        """
        Retrieve metric history curves.
        Downsamples to at most 1000 points if history is large for responsive UI.
        """
        res: Dict[str, List[float]] = {}
        for k, vals in self._curves.items():
            if len(vals) > 1000:
                stride = len(vals) // 1000
                res[k] = vals[::stride]
            else:
                res[k] = list(vals)
        return res

    def get_hardware_telemetry(self) -> Dict[str, Any]:
        """
        Query system resource telemetry via psutil with safe fallback.
        """
        try:
            import psutil

            cpu_pct = float(psutil.cpu_percent(interval=None))
            ram_pct = float(psutil.virtual_memory().percent)
            return {
                "cpu": cpu_pct,
                "ram": ram_pct,
                "gpu": "N/A",
                "fallback": False,
            }
        except Exception:
            return {
                "cpu": 25.0,
                "ram": 50.0,
                "gpu": "N/A",
                "fallback": True,
            }
