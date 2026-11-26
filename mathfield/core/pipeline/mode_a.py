"""
mathfield/core/pipeline/mode_a.py

Mode A: Continuous Mathematical Discovery Pipeline.
Discovers compact, interpretable symbolic formulas minimizing:
J(M) = E(M) + lambda * C(M)
Zero lookup tables or direct raster memorization.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

import numpy as np

from mathfield.core.ast.nodes import ASTNode, BinaryOpNode, ConstantNode, FunctionCallNode, UnaryOpNode
from mathfield.core.basis.polynomial import Chebyshev2D
from mathfield.core.image_analysis import ImageAnalyzer
from mathfield.core.optimize.evolutionary import EvolutionaryOptimizer
from mathfield.core.pipeline.stage import PipelineContext, StageStatus
from mathfield.core.verify.metrics import compute_metrics
from mathfield.core.verify.pruning import BICPruner


def _ast_constants(node: ASTNode) -> List[float]:
    if isinstance(node, ConstantNode):
        return [float(node.value)]
    if isinstance(node, BinaryOpNode):
        return _ast_constants(node.left) + _ast_constants(node.right)
    if isinstance(node, UnaryOpNode):
        return _ast_constants(node.operand)
    if isinstance(node, FunctionCallNode):
        return [value for arg in node.args for value in _ast_constants(arg)]
    return []


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
        tolerance = float(self.config.get("tolerance", 1e-4))
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
        alpha_weights = img_arr[:, :, 3].ravel() if img_arr.ndim == 3 and c >= 4 else None

        for ch in range(c):
            y_ch = img_arr[:, :, ch] if img_arr.ndim == 3 else img_arr
            channel_weights = alpha_weights if ch < 3 else None
            # BIC pruning backward elimination
            prune_res = pruner.prune_matrix(Phi, y_ch, reg_lambda=lambda_reg, weights=channel_weights)
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
            tolerance=tolerance,
        )
        best_ast, evo_diag = evo.evolve(X, Y, target_lum, sample_weights=alpha_weights)

        chebyshev_metrics = compute_metrics(fitted_arr, img_arr)
        chebyshev_complexity = sum(ast.complexity() for ast in channel_asts)
        candidate_channel = np.clip(
            best_ast.compile_numpy()(X, Y, np.empty(0, dtype=np.float64)),
            0.0,
            1.0,
        )
        candidate_arr = np.repeat(candidate_channel[:, :, np.newaxis], c, axis=-1) if c > 1 else candidate_channel
        candidate_metrics = compute_metrics(candidate_arr, img_arr)
        candidate_complexity = c * best_ast.complexity()
        chebyshev_objective = chebyshev_metrics["mse"] + lambda_reg * chebyshev_complexity
        candidate_objective = candidate_metrics["mse"] + lambda_reg * candidate_complexity
        use_candidate = candidate_metrics["mse"] < chebyshev_metrics["mse"] and candidate_objective < chebyshev_objective
        if use_candidate:
            channel_asts = [best_ast] * c
            channel_coefficients = [_ast_constants(best_ast) for _ in range(c)]
            fitted_arr = candidate_arr
            selected_metrics = candidate_metrics
            delivered_model = "evolutionary_luminance"
            delivered_ast = best_ast
        else:
            selected_metrics = chebyshev_metrics
            delivered_model = "per_channel_chebyshev"
            delivered_ast = channel_asts[0]
        delivered_complexity = sum(ast.complexity() for ast in channel_asts)
        evo_diag.update(
            {
                "delivered": use_candidate,
                "candidate_mse": candidate_metrics["mse"],
                "candidate_complexity": float(candidate_complexity),
                "candidate_objective": candidate_objective,
                "chebyshev_objective": chebyshev_objective,
            }
        )

        # Store model in context
        context.global_model = {
            "type": "mode_a_discovery",
            "channel_asts": channel_asts,
            "channel_coefficients": channel_coefficients,
            "symbolic_ast": best_ast,
            "luminance_candidate": str(best_ast),
            "formula": str(delivered_ast),
            "delivered_model": delivered_model,
            "complexity": delivered_complexity,
            "evo_diagnostics": evo_diag,
        }

        # --- Stage 5: Verification & Metrics ---
        if progress_callback:
            progress_callback("Stage 5: Verification & Parity", 5, 5)
        context.check_pause()
        if context.is_cancelled():
            return StageStatus.CANCELLED

        metrics = dict(selected_metrics)
        metrics["converged"] = float(evo_diag["converged"])
        metrics["generations_run"] = float(evo_diag["generations_run"])
        context.metrics = metrics
        context.rendered_image = fitted_arr

        return StageStatus.COMPLETED
