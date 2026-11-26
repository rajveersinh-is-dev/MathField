"""
mathfield/core/pipeline/mode_a.py

Mode A: Continuous Mathematical Discovery Pipeline.
Discovers compact, interpretable symbolic formulas minimizing:
J(M) = E(M) + lambda * C(M)
Zero lookup tables or direct raster memorization.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional

import numpy as np

from mathfield.core.basis.polynomial import Chebyshev2D
from mathfield.core.image_analysis import ImageAnalyzer
from mathfield.core.optimize.evolutionary import EvolutionaryOptimizer
from mathfield.core.pipeline.stage import PipelineContext, StageStatus
from mathfield.core.verify.metrics import compute_metrics
from mathfield.core.verify.pruning import BICPruner


class ModeAPipeline:
    """
    Orchestrates Mode A (Discovery) reconstruction stages.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config = config or {}

    def execute(
        self,
        context: PipelineContext,
        progress_callback: Optional[Callable[[str, int, int], None]] = None,
    ) -> StageStatus:
        """Execute full Mode A discovery pipeline."""
        if context.is_cancelled():
            return StageStatus.CANCELLED

        # --- Stage 1: Image Analysis ---
        if progress_callback:
            progress_callback("Stage 1: Image Analysis", 1, 5)
        context.check_pause()
        if context.is_cancelled():
            return StageStatus.CANCELLED

        analyzer = ImageAnalyzer()
        context.analysis = analyzer.analyze(context.image)

        # Normalize target image to float64 in [0, 1]
        img_arr = np.asarray(context.image, dtype=np.float64)
        if img_arr.max() > 1.0:
            img_arr = img_arr / 255.0

        h, w = img_arr.shape[:2]
        c = img_arr.shape[2] if img_arr.ndim == 3 else 1

        X, Y = context.domain.to_discrete_grid(width=w, height=h)

        # --- Stage 2 & 4: Global Discovery & Optimization ---
        if progress_callback:
            progress_callback("Stage 2 & 4: Symbolic Discovery", 2, 5)
        context.check_pause()
        if context.is_cancelled():
            return StageStatus.CANCELLED

        lambda_reg = float(self.config.get("lambda_reg", 0.01))
        max_iter = int(self.config.get("max_iter", 15))
        seed = int(self.config.get("seed", 42))

        # Candidate 1: Low-order Chebyshev polynomial expansion
        deg = 3 if lambda_reg < 0.1 else 2
        cheb = Chebyshev2D(deg_x=deg, deg_y=deg, truncation="total_degree")
        Phi = cheb.basis_matrix(X, Y)

        # Candidate 2: Deterministic Evolutionary STGP search on luminance
        target_lum = (
            0.299 * img_arr[:, :, 0] + 0.587 * img_arr[:, :, 1] + 0.114 * img_arr[:, :, 2]
            if c >= 3
            else img_arr[:, :, 0]
            if img_arr.ndim == 3
            else img_arr
        )

        # Solve linear coefficients for polynomial basis per channel
        pruner = BICPruner()
        channel_asts = []
        channel_coefficients = []
        fitted_channels = []

        for ch in range(c):
            y_ch = img_arr[:, :, ch] if img_arr.ndim == 3 else img_arr
            # BIC pruning backward elimination
            prune_res = pruner.prune_matrix(Phi, y_ch, reg_lambda=lambda_reg)
            cheb_ch = Chebyshev2D(deg_x=deg, deg_y=deg, truncation="total_degree", coefficients=prune_res.coefficients)
            ast_ch = cheb_ch.to_ast(prune_threshold=0.0)
            channel_asts.append(ast_ch)
            channel_coefficients.append(prune_res.coefficients.copy())
            fitted_channels.append(ast_ch.compile_numpy()(X, Y, np.empty(0, dtype=np.float64)).reshape(h, w))

        fitted_arr = np.stack(fitted_channels, axis=-1) if c > 1 else fitted_channels[0]
        fitted_arr = np.clip(fitted_arr, 0.0, 1.0)

        # STGP search exploration
        evo = EvolutionaryOptimizer(
            random_seed=seed,
            population_size=20,
            generations=max(max_iter, 5),
            lambda_reg=lambda_reg,
        )
        best_ast, evo_diag = evo.evolve(X, Y, target_lum)

        # Store model in context
        context.global_model = {
            "type": "mode_a_discovery",
            "channel_asts": channel_asts,
            "channel_coefficients": channel_coefficients,
            "symbolic_ast": best_ast,
            "formula": str(best_ast),
            "complexity": best_ast.complexity(),
            "evo_diagnostics": evo_diag,
        }

        # --- Stage 5: Verification & Metrics ---
        if progress_callback:
            progress_callback("Stage 5: Verification & Parity", 5, 5)
        context.check_pause()
        if context.is_cancelled():
            return StageStatus.CANCELLED

        metrics = compute_metrics(fitted_arr, img_arr)
        context.metrics = metrics
        context.rendered_image = fitted_arr

        return StageStatus.COMPLETED
