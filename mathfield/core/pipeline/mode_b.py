"""
mathfield/core/pipeline/mode_b.py

Mode B: Maximum Fidelity Reconstruction Pipeline.
Combines dense global expansions with localized adaptive quadtree subdivision,
Wendland C2 partition of unity blending, SVD condition monitoring, and condition reporting.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

import numpy as np

from mathfield.core.basis.fourier import Fourier2D
from mathfield.core.basis.polynomial import Chebyshev2D
from mathfield.core.image_analysis import ImageAnalyzer
from mathfield.core.local.local_patch import LocalPatch
from mathfield.core.local.partition_of_unity import WendlandPartitionOfUnity
from mathfield.core.local.quadtree import QuadtreeDecomposition
from mathfield.core.optimize.conditioning import ConditionMonitor
from mathfield.core.pipeline.stage import PipelineContext, PipelineStage, StageStatus
from mathfield.core.verify.metrics import compute_metrics
from mathfield.core.verify.renderer import ReferenceRenderer


class ModeBPipeline:
    """
    Orchestrates Mode B (Maximum Fidelity) reconstruction stages.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config = config or {}

    def execute(
        self,
        context: PipelineContext,
        progress_callback: Optional[Callable[[str, int, int], None]] = None,
    ) -> StageStatus:
        """Execute full Mode B high-fidelity pipeline."""
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

        img_arr = np.asarray(context.image, dtype=np.float64)
        if img_arr.max() > 1.0:
            img_arr = img_arr / 255.0

        h, w = img_arr.shape[:2]
        c = img_arr.shape[2] if img_arr.ndim == 3 else 1

        X, Y = context.domain.to_discrete_grid(width=w, height=h)

        # --- Stage 2: Dense Global Expansion ---
        if progress_callback:
            progress_callback("Stage 2: Dense Global Expansion", 2, 5)
        context.check_pause()
        if context.is_cancelled():
            return StageStatus.CANCELLED

        global_deg = int(self.config.get("global_degree", 6))
        global_basis = Chebyshev2D(deg_x=global_deg, deg_y=global_deg, truncation="total_degree")
        Phi_cheb = global_basis.basis_matrix(X, Y)

        # Dense 2D Fourier harmonic expansion for high-frequency patterns
        fourier_freq = int(self.config.get("fourier_freq", 12))
        fourier_basis = Fourier2D(freq_x=fourier_freq, freq_y=fourier_freq, truncation="rectangular")
        Phi_four = fourier_basis.basis_matrix(X, Y)

        # Combine polynomial Chebyshev with harmonic Fourier terms (dropping redundant DC term)
        Phi_global = np.hstack([Phi_cheb, Phi_four[:, 1:]])
        n_terms = Phi_global.shape[1]

        # Monitor conditioning of global basis
        monitor = ConditionMonitor(
            max_condition_number=float(self.config.get("max_condition_number", 1e10)),
            tikhonov_alpha=float(self.config.get("tikhonov_alpha", 1e-4)),
        )
        cond_rep = monitor.analyze_matrix(Phi_global)
        context.condition_report = cond_rep

        if cond_rep["is_ill_conditioned"]:
            Phi_solve = monitor.regularize(Phi_global)
        else:
            Phi_solve = Phi_global

        global_channels = []
        for ch in range(c):
            y_ch = img_arr[:, :, ch] if img_arr.ndim == 3 else img_arr
            b_ch = y_ch.ravel()
            if cond_rep["is_ill_conditioned"]:
                b_ch = np.concatenate([b_ch, np.zeros(n_terms)])
            c_ch, _ = np.linalg.lstsq(Phi_solve, b_ch, rcond=1e-10)[:2]
            fit_ch = Phi_global @ c_ch
            global_channels.append(fit_ch.reshape(h, w))

        global_recon = np.stack(global_channels, axis=-1) if c > 1 else global_channels[0]
        context.global_model = {
            "type": "mode_b_global_hybrid",
            "degree": global_deg,
            "basis_terms_count": n_terms,
            "condition_number": cond_rep["condition_number"],
        }

        # --- Stage 3: Quadtree Local Decomposition ---
        if progress_callback:
            progress_callback("Stage 3: Quadtree Local Decomposition", 3, 5)
        context.check_pause()
        if context.is_cancelled():
            return StageStatus.CANCELLED

        residual = img_arr - global_recon
        res_mag = np.sqrt(np.mean(residual ** 2, axis=-1)) if residual.ndim == 3 else residual
        root_mse = float(np.mean(res_mag ** 2))
        default_thresh = max(1e-4, min(0.05, 0.25 * root_mse))
        split_thresh = float(self.config.get("split_threshold", default_thresh))
        max_depth = int(self.config.get("max_depth", 3))

        quadtree = QuadtreeDecomposition(
            max_depth=max_depth,
            split_threshold=split_thresh,
            min_patch_size=4,
            sobolev_mu=0.1,
        )
        root = quadtree.decompose(residual, domain=context.domain)
        leaves = quadtree.get_leaf_nodes(root)

        # --- Stage 4: Local Patch Fitting & Partition of Unity Blending ---
        if progress_callback:
            progress_callback("Stage 4: Local Patch Blending", 4, 5)
        context.check_pause()
        if context.is_cancelled():
            return StageStatus.CANCELLED

        pou = WendlandPartitionOfUnity(overlap_ratio=float(self.config.get("overlap_ratio", 0.33)))
        weights = pou.compute_weights(leaves, X, Y)  # (K, H, W)

        local_correction = np.zeros_like(img_arr)
        patches: List[LocalPatch] = []

        for k, leaf in enumerate(leaves):
            patch = LocalPatch(bounds=leaf.bounds)
            # Weighted least squares on residual
            leaf_weight = weights[k]
            if c > 1:
                patch_ch_fits = []
                for ch in range(c):
                    r_ch = residual[:, :, ch]
                    patch.fit(r_ch, X, Y, weights=leaf_weight, reg_lambda=1e-4)
                    fit_k = patch.evaluate(X, Y)
                    patch_ch_fits.append(fit_k)
                patch_fit = np.stack(patch_ch_fits, axis=-1)
            else:
                patch.fit(residual, X, Y, weights=leaf_weight, reg_lambda=1e-4)
                patch_fit = patch.evaluate(X, Y)

            local_correction += np.expand_dims(leaf_weight, axis=-1) * patch_fit if c > 1 else leaf_weight * patch_fit
            patches.append(patch)

        context.local_decomposition = {
            "root": root,
            "leaves_count": len(leaves),
            "patches": patches,
        }

        # Composite surface
        final_recon = np.clip(global_recon + local_correction, 0.0, 1.0)
        context.rendered_image = final_recon

        # --- Stage 5: Verification & Condition Reporting ---
        if progress_callback:
            progress_callback("Stage 5: Verification & Reporting", 5, 5)
        context.check_pause()
        if context.is_cancelled():
            return StageStatus.CANCELLED

        metrics = compute_metrics(final_recon, img_arr)
        context.metrics = metrics

        return StageStatus.COMPLETED
