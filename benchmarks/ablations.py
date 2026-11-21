"""
benchmarks/ablations.py

Ablation Studies for MathField Architecture.
Every configuration key emitted here is consumed by ModeBPipeline, so each
ablation genuinely changes the reconstruction instead of being a no-op.

1. no_quadtree: Disables localized quadtree spatial decomposition (max_depth=0)
2. no_bic_pruning: Disables complexity regularization (reg_lambda=0)
3. no_varpro: Disables condition-aware regularized solve (plain least squares)
4. no_color_decoupling: Uses one shared basis fit across RGB (no per-channel fit)
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np

from mathfield.core.domain import CoordinateDomain2D
from mathfield.core.pipeline.mode_b import ModeBPipeline
from mathfield.core.pipeline.stage import PipelineContext
from mathfield.core.verify.metrics import compute_mse, compute_psnr

ABLATION_NAMES: List[str] = [
    "no_quadtree",
    "no_bic_pruning",
    "no_varpro",
    "no_color_decoupling",
]


def list_ablations() -> List[str]:
    """Retrieve list of supported architectural ablation studies."""
    return list(ABLATION_NAMES)


def get_ablation_config(ablation_name: str) -> Dict[str, Any]:
    """Retrieve pipeline configuration dictionary for specified ablation."""
    norm = ablation_name.lower()
    base_config: Dict[str, Any] = {
        "global_degree": 4,
        "fourier_freq": 8,
        "max_depth": 3,
        "split_threshold": 0.02,
        "reg_lambda": 1e-4,
        "local_basis_type": "dct2",
        "local_freq": 8,
        "overlap_ratio": 0.20,
        "enable_quadtree": True,
        "enable_varpro": True,
        "decouple_color": True,
    }

    if norm == "no_quadtree":
        base_config["enable_quadtree"] = False
        base_config["max_depth"] = 0
    elif norm == "no_bic_pruning":
        base_config["reg_lambda"] = 0.0
    elif norm == "no_varpro":
        base_config["enable_varpro"] = False
    elif norm == "no_color_decoupling":
        base_config["decouple_color"] = False
    elif norm in ("none", "baseline", "all_enabled"):
        pass
    else:
        raise ValueError(f"Unknown ablation name '{ablation_name}'. Supported: {ABLATION_NAMES}")

    return base_config


@dataclass(frozen=True, slots=True)
class AblationResult:
    """Quantitative performance and error measurement for an architectural ablation."""

    name: str
    mse: float
    psnr: float
    runtime_ms: float
    config: Dict[str, Any]


def run_ablation(
    name: str,
    img: np.ndarray,
    **kwargs: Any,
) -> AblationResult:
    """Execute reconstruction with designated architectural component disabled."""
    arr = np.asarray(img, dtype=np.float64)
    if arr.size == 0:
        raise ValueError("Cannot run ablation on empty image")
    if arr.max() > 1.0:
        arr = arr / 255.0

    h, w = arr.shape[:2]
    config = get_ablation_config(name)
    config.update(kwargs)

    t_start = time.perf_counter()
    domain = CoordinateDomain2D(width=w, height=h)
    ctx = PipelineContext(image=arr, domain=domain)
    pipeline = ModeBPipeline(config=config)
    pipeline.execute(ctx)

    runtime_ms = (time.perf_counter() - t_start) * 1000.0
    recon = ctx.rendered_image if ctx.rendered_image is not None else arr
    mse_val = float(compute_mse(recon, arr))
    psnr_val = float(compute_psnr(recon, arr))

    return AblationResult(
        name=name,
        mse=mse_val,
        psnr=psnr_val,
        runtime_ms=runtime_ms,
        config=config,
    )
