"""
mathfield/ui/main_window.py

Top-level MathField Desktop Application Main Window.
Coordinates Main, Equation, Region Analysis, Optimization, and Research workspaces.
"""

from __future__ import annotations

from typing import Optional

from mathfield.ui.workers import OptimizationWorker
from mathfield.ui.workspaces.equation_workspace import EquationWorkspace
from mathfield.ui.workspaces.main_workspace import MainWorkspace
from mathfield.ui.workspaces.optimization_workspace import OptimizationWorkspace
from mathfield.ui.workspaces.region_workspace import RegionWorkspace
from mathfield.ui.workspaces.research_workspace import ResearchWorkspace


class MainWindow:
    """
    Main Application Window integrating the 5 core research workspaces.
    """

    def __init__(self) -> None:
        self.main_workspace = MainWorkspace()
        self.equation_workspace = EquationWorkspace()
        self.region_workspace = RegionWorkspace()
        self.optimization_workspace = OptimizationWorkspace()
        self.research_workspace = ResearchWorkspace()
        self.worker: Optional[OptimizationWorker] = None

    def create_worker(self) -> OptimizationWorker:
        """Create and configure background optimization worker thread."""
        self.worker = OptimizationWorker()
        return self.worker
