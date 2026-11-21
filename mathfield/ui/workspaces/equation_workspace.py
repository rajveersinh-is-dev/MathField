"""
mathfield/ui/workspaces/equation_workspace.py

Feature 18: Equation UI Workspace.
Hierarchical equation tree view, LaTeX formula card rendering via matplotlib.mathtext,
AST node inspector, real-time parameter sliders with event debouncing, and formula validation.
"""

from __future__ import annotations

import io
import math
from typing import Any, Dict, List

import numpy as np

from mathfield.core.ast.parser import PrattParser


class EquationWorkspace:
    """
    Equation Workspace: Interactive formula hierarchy, AST inspection,
    parameter tuning, and LaTeX mathtext card visualization.
    """

    def __init__(self) -> None:
        self._parameters: Dict[str, float] = {"c0": 1.0, "c1": 0.5}
        self.last_dispatched_value: float = 0.0
        self._equation_tree: Dict[str, Any] = {
            "root": {
                "id": "root",
                "label": "f(x, y)",
                "type": "Add",
                "children": ["left_term", "right_term"],
            },
            "left_term": {
                "id": "left_term",
                "label": "c0 * sin(x)",
                "type": "Mul",
                "children": [],
            },
            "right_term": {
                "id": "right_term",
                "label": "c1 * cos(y)",
                "type": "Mul",
                "children": [],
            },
        }

    def get_equation_tree(self) -> Dict[str, Any]:
        """Retrieve equation hierarchy tree structure."""
        return self._equation_tree

    def render_latex_card(self, latex_str: str) -> Any:
        """
        Render LaTeX formula card via matplotlib.mathtext without pdflatex.
        Handles long formulas gracefully by wrapping or clipping.
        """
        from matplotlib.font_manager import FontProperties
        from matplotlib.mathtext import math_to_image
        from PIL import Image, ImageDraw

        display_str = latex_str
        if len(display_str) > 120:
            display_str = display_str[:115] + " \\dots"
        if not display_str.startswith("$"):
            display_str = f"${display_str}$"

        buf = io.BytesIO()
        try:
            # Direct mathtext rendering never changes the global matplotlib
            # backend, which matters when the workspace is embedded in a GUI.
            math_to_image(display_str, buf, prop=FontProperties(size=12), dpi=100, format="png", color="black")
            return buf.getvalue()
        except Exception:
            # Guaranteed fallback: draw sanitized plain text with Pillow.
            clean = latex_str.replace("$", "")[:100]
            image = Image.new("RGB", (max(160, 8 * len(clean) + 20), 28), "white")
            ImageDraw.Draw(image).text((10, 8), clean, fill="black")
            fallback = io.BytesIO()
            image.save(fallback, format="PNG")
            return fallback.getvalue()

    def inspect_ast_node(self, node_id: str) -> Dict[str, Any]:
        """Inspect AST node properties, type, parameters, and child references."""
        if node_id in self._equation_tree:
            return self._equation_tree[node_id]
        return {
            "id": node_id,
            "type": "VariableNode",
            "name": node_id,
            "children": [],
            "properties": {"tunable": True},
        }

    def set_parameter_value(self, key: str, value: float) -> None:
        """Set bounded parameter slider value, clamping to finite range."""
        numeric = float(value)
        if not math.isfinite(numeric):
            raise ValueError("Parameter value must be finite")
        self._parameters[key] = float(np.clip(numeric, -1e6, 1e6))

    def get_parameter_value(self, key: str) -> float:
        """Retrieve current value for parameter slider."""
        return self._parameters.get(key, 0.0)

    def debounce_parameter_events(self, values: List[float]) -> None:
        """Consolidate rapid slider adjustment events; dispatch latest value."""
        if values:
            self.last_dispatched_value = float(values[-1])

    def validate_formula_input(self, formula_str: str) -> Dict[str, Any]:
        """Validate user-entered mathematical formula string without crashing."""
        try:
            parser = PrattParser()
            ast = parser.parse(formula_str)
            return {"is_valid": True, "ast": ast}
        except Exception as exc:
            return {"is_valid": False, "error": str(exc)}
