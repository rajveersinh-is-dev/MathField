"""
mathfield/core/verify/renderer.py

Stage 5: Independent Reference Re-Renderer and Parity Verifier.
Provides decoupled re-rendering of mathematical expressions and composite patches,
independent verification against target rasters, and discrepancy detection.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple, Union

import numpy as np

from mathfield.core.ast.nodes import ASTNode
from mathfield.core.color import ColorSpace, ColorSpaceConverter
from mathfield.core.domain import CoordinateDomain2D
from mathfield.core.engine.cpu_numpy import evaluate_numpy
from mathfield.core.verify.metrics import compute_metrics


class ReferenceRenderer:
    """
    Decoupled reference re-renderer and numerical parity verifier.
    """

    def __init__(self) -> None:
        pass

    def render_ast(
        self,
        ast: ASTNode,
        domain: CoordinateDomain2D,
        width: Optional[int] = None,
        height: Optional[int] = None,
        params: Optional[np.ndarray] = None,
        color_space: Union[str, ColorSpace] = ColorSpace.SRGB,
        soft_clip: bool = False,
        hard_clamp: bool = True,
    ) -> np.ndarray:
        """
        Evaluate AST on continuous grid defined by domain, applying color and clipping policy.
        """
        X, Y = domain.to_discrete_grid(width=width, height=height)
        p = np.array([], dtype=np.float64) if params is None else np.asarray(params, dtype=np.float64)

        raw = evaluate_numpy(ast, X, Y, p)

        if soft_clip:
            raw = ColorSpaceConverter.soft_clamp(raw)
        if hard_clamp:
            raw = ColorSpaceConverter.hard_clamp(raw)

        cs = ColorSpace.from_str(color_space)
        if cs != ColorSpace.SRGB:
            raw = ColorSpaceConverter.convert(raw, src_space=cs, dst_space=ColorSpace.SRGB)

        return raw

    def verify(
        self,
        truth: np.ndarray,
        candidate: np.ndarray,
        tolerance: float = 1e-4,
    ) -> Tuple[bool, Dict[str, float]]:
        """
        Verify that candidate array matches ground truth within tolerance.
        Returns (is_verified, metrics_dict).
        """
        t = np.asarray(truth, dtype=np.float64)
        c = np.asarray(candidate, dtype=np.float64)

        if t.shape != c.shape:
            # Shape mismatch implies verification failure
            metrics = {
                "mse": float("inf"),
                "mae": float("inf"),
                "linf": float("inf"),
                "psnr": 0.0,
                "ssim": 0.0,
            }
            return False, metrics

        metrics = compute_metrics(c, t)
        verified = bool(metrics["mse"] <= tolerance)
        return verified, metrics
