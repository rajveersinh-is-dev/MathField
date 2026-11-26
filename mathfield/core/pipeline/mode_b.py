"""
mathfield/core/pipeline/mode_b.py

Mode B: Maximum Fidelity Reconstruction Pipeline.
Combines dense global expansions with localized adaptive quadtree subdivision,
Wendland C2 partition of unity blending, SVD condition monitoring, and condition reporting.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

import numpy as np

from mathfield.core.basis.fourier import DCT2Basis, Fourier2D
from mathfield.core.basis.polynomial import Chebyshev2D
from mathfield.core.image_analysis import ImageAnalyzer
from mathfield.core.local.local_patch import LocalPatch
from mathfield.core.local.partition_of_unity import WendlandPartitionOfUnity
from mathfield.core.local.quadtree import QuadtreeDecomposition
from mathfield.core.optimize.conditioning import ConditionMonitor
from mathfield.core.pipeline.stage import PipelineContext, StageStatus
from mathfield.core.verify.metrics import compute_metrics


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

        # Accepted aliases keep CLI flags and ablation studies meaningful.
        cfg = self.config
        reg_lambda = float(cfg.get("reg_lambda", cfg.get("lambda_reg", 1e-5)))
        use_conditioning = bool(cfg.get("enable_varpro", True))
        decouple_color = bool(cfg.get("decouple_color", True))
        enable_quadtree = bool(cfg.get("enable_quadtree", True))

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
            max_condition_number=float(cfg.get("max_condition_number", 1e10)),
            tikhonov_alpha=float(cfg.get("tikhonov_alpha", reg_lambda)),
        )
        cond_rep = monitor.analyze_matrix(Phi_global)
        context.condition_report = cond_rep

        regularize = use_conditioning and bool(cond_rep["is_ill_conditioned"])
        Phi_solve = monitor.regularize(Phi_global) if regularize else Phi_global

        def solve_channel(b_ch: np.ndarray) -> np.ndarray:
            rhs = np.concatenate([b_ch, np.zeros(n_terms)]) if regularize else b_ch
            return np.linalg.lstsq(Phi_solve, rhs, rcond=1e-10)[0]

        shared_coeffs: Optional[np.ndarray] = None
        if not decouple_color and img_arr.ndim == 3:
            luminance = (
                0.299 * img_arr[:, :, 0] + 0.587 * img_arr[:, :, 1] + 0.114 * img_arr[:, :, 2]
                if c >= 3
                else img_arr[:, :, 0]
            )
            shared_coeffs = solve_channel(luminance.ravel())

        if shared_coeffs is not None:
            coefficient_matrix = np.repeat(shared_coeffs[:, np.newaxis], c, axis=1)
        else:
            targets = img_arr.reshape(-1, c) if img_arr.ndim == 3 else img_arr.reshape(-1, 1)
            rhs = np.vstack([targets, np.zeros((n_terms, c))]) if regularize else targets
            coefficient_matrix = np.linalg.lstsq(Phi_solve, rhs, rcond=1e-10)[0]

        global_values = Phi_global @ coefficient_matrix
        global_recon = global_values.reshape(h, w, c) if c > 1 else global_values[:, 0].reshape(h, w)
        context.global_model = {
            "type": "mode_b_global_hybrid",
            "degree": global_deg,
            "fourier_freq": fourier_freq,
            "basis_terms_count": n_terms,
            "condition_number": cond_rep["condition_number"],
            "coefficients": coefficient_matrix if c > 1 else coefficient_matrix[:, 0],
            "chebyshev_basis": global_basis,
            "fourier_basis": fourier_basis,
        }

        # --- Stage 3: Quadtree Local Decomposition ---
        if progress_callback:
            progress_callback("Stage 3: Quadtree Local Decomposition", 3, 5)
        context.check_pause()
        if context.is_cancelled():
            return StageStatus.CANCELLED

        residual = img_arr - global_recon
        res_mag = np.sqrt(np.mean(residual**2, axis=-1)) if residual.ndim == 3 else residual
        root_mse = float(np.mean(res_mag**2))
        default_thresh = max(1e-4, min(0.05, 0.25 * root_mse))
        split_thresh = float(cfg.get("split_threshold", default_thresh))
        max_depth = int(cfg.get("max_depth", 3)) if enable_quadtree else 0

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

        overlap_ratio = float(cfg.get("overlap_ratio", 0.20))
        pou = WendlandPartitionOfUnity(overlap_ratio=overlap_ratio, separable=True)
        weights = pou.compute_weights(leaves, X, Y)  # (K, H, W)

        local_correction = np.zeros_like(img_arr)
        patches: List[LocalPatch] = []

        local_basis_type = str(cfg.get("local_basis_type", "dct2")).lower()
        local_freq = int(cfg.get("local_freq", 8))

        for k, leaf in enumerate(leaves):
            leaf_weight = weights[k]
            active = leaf_weight > 1e-12
            if not np.any(active):
                patch = LocalPatch(bounds=leaf.bounds)
                patches.append(patch)
                continue

            rows, cols = np.where(active)
            r0, r1 = int(rows.min()), int(rows.max()) + 1
            c0, c1 = int(cols.min()), int(cols.max()) + 1

            sub_X = X[r0:r1, c0:c1]
            sub_Y = Y[r0:r1, c0:c1]
            sub_W = leaf_weight[r0:r1, c0:c1]
            sub_res = residual[r0:r1, c0:c1]

            if local_basis_type == "dct2":
                fx = max(1, min(local_freq, c1 - c0 - 1))
                fy = max(1, min(local_freq, r1 - r0 - 1))
                patch_basis = DCT2Basis(freq_x=fx, freq_y=fy, x_bounds=(-1.0, 1.0), y_bounds=(-1.0, 1.0))
            else:
                deg = int(self.config.get("degree", 3))
                patch_basis = Chebyshev2D(deg_x=deg, deg_y=deg, truncation="total_degree")

            patch = LocalPatch(bounds=leaf.bounds, basis=patch_basis)
            patch.fit(sub_res, sub_X, sub_Y, weights=sub_W, reg_lambda=reg_lambda)
            sub_fit = patch.evaluate(sub_X, sub_Y)

            if c > 1:
                local_correction[r0:r1, c0:c1] += np.expand_dims(sub_W, axis=-1) * sub_fit
            else:
                local_correction[r0:r1, c0:c1] += sub_W * sub_fit
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
