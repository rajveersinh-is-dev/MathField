"""
mathfield/ui/workspaces/research_workspace.py

Feature 21: Research UI Workspace.
Hypothesis candidate discovery log with categorical rejection rationales,
parameter convergence trajectories, SVD singular value spectrum monitoring, and CSV audit export.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np


class ResearchWorkspace:
    """
    Research Workspace: Hypothesis audit trail, rejection rationales,
    parameter trajectories, and SVD spectrum diagnostics.
    """

    REJECTION_REASONS = (
        "BIC_INCREASE",
        "ILL_CONDITIONED",
        "SYNTAX_VIOLATION",
        "MAX_DEPTH_EXCEEDED",
        "RUNGE_INSTABILITY",
        "ZERO_VARIANCE",
        "PARETO_DOMINATED",
    )

    def __init__(self) -> None:
        self._candidate_log: List[Dict[str, Any]] = []
        self._trajectories: Dict[str, List[float]] = {}
        self._svd_singular_values: np.ndarray = np.array([])
        self._condition_number: float = 1.0

    def log_candidate(
        self,
        iteration: int,
        expression: str,
        status: str,
        rationale: str,
        delta_loss: float,
    ) -> None:
        """Record candidate mathematical expression and its acceptance/rejection status."""
        entry = {
            "iteration": int(iteration),
            "expression": str(expression),
            "status": str(status).upper(),
            "rationale": str(rationale),
            "delta_loss": float(delta_loss),
        }
        self._candidate_log.append(entry)

    def get_candidate_log(self) -> List[Dict[str, Any]]:
        """Retrieve full candidate discovery log."""
        return self._candidate_log

    def filter_candidates(
        self,
        rationale: Optional[str] = None,
        status: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Filter candidate log by rejection rationale or acceptance status."""
        filtered = self._candidate_log
        if rationale is not None:
            filtered = [c for c in filtered if c["rationale"] == rationale]
        if status is not None:
            filtered = [c for c in filtered if c["status"] == status.upper()]
        return filtered

    def get_rejection_reasons(self) -> List[str]:
        """Retrieve standard categorical rejection rationale codes."""
        return list(self.REJECTION_REASONS)

    def update_parameter_trajectories(self, iteration: int, params: Dict[str, float]) -> None:
        """Record optimization parameter values over time."""
        for k, v in params.items():
            if k not in self._trajectories:
                self._trajectories[k] = []
            self._trajectories[k].append(float(v))

    def get_trajectories(self) -> Dict[str, List[float]]:
        """Retrieve parameter convergence trajectories."""
        return self._trajectories

    def update_svd_spectrum(self, s: np.ndarray) -> float:
        """
        Record singular value spectrum and compute matrix condition number kappa.
        """
        arr = np.asarray(s, dtype=np.float64).ravel()
        self._svd_singular_values = arr
        if arr.size == 0 or arr[-1] <= 0:
            self._condition_number = float("inf")
        else:
            self._condition_number = float(arr[0] / arr[-1])
        return self._condition_number

    def export_research_log(self, path: Union[str, Path], format: str = "csv") -> None:
        """Export candidate log to CSV or JSON with the required schema."""
        fmt = format.lower()
        if fmt not in ("csv", "json"):
            raise ValueError(f"Unsupported research log format: {format}")
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)

        if fmt == "json":
            p.write_text(json.dumps(list(self._candidate_log), indent=2), encoding="utf-8")
            return

        with open(p, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["iteration", "expression", "status", "rationale", "delta_loss"])
            for c in self._candidate_log:
                writer.writerow(
                    [
                        c["iteration"],
                        c["expression"],
                        c["status"],
                        c["rationale"],
                        c["delta_loss"],
                    ]
                )
