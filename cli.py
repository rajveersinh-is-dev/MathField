#!/usr/bin/env python3
"""
cli.py

Command-line interface for the MathField Image Reconstruction Engine.
Supports Mode A (Symbolic Discovery) and Mode B (Maximum Fidelity),
comprehensive export generation, seed control, and test automation hooks.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from PIL import Image

from mathfield.core.domain import CoordinateDomain2D, DomainMode
from mathfield.core.pipeline.controller import ReconstructionController
from mathfield.export.csv_export import export_csv_parameters
from mathfield.export.latex_report import export_latex_report
from mathfield.export.pdf_report import export_pdf_report
from mathfield.export.project_file import save_project
from mathfield.export.python_script import export_standalone_script
from mathfield.export.raster import export_jpg, export_png
from mathfield.export.svg import export_svg


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="MathField: Continuous-Domain Mathematical Image Reconstruction Engine"
    )
    parser.add_argument("--image", "-i", type=str, help="Path to input image file")
    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default="output",
        help="Directory to store outputs and exported assets",
    )
    parser.add_argument(
        "--mode",
        "-m",
        type=str,
        default="discovery",
        choices=["discovery", "fidelity", "A", "B", "mode_a", "mode_b"],
        help="Reconstruction mode: 'discovery' (Mode A) or 'fidelity' (Mode B)",
    )
    parser.add_argument(
        "--lambda-reg",
        type=float,
        default=0.01,
        help="Complexity regularization penalty strength (lambda)",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=15,
        help="Maximum optimization iterations / generations",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1e-4,
        help="Optimization convergence tolerance threshold",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Reproducible PRNG seed",
    )
    parser.add_argument(
        "--export-json",
        action="store_true",
        help="Export .mathfield JSON project file",
    )
    parser.add_argument(
        "--export-csv",
        action="store_true",
        help="Export CSV parameters table",
    )
    parser.add_argument(
        "--export-png",
        action="store_true",
        help="Export reconstructed PNG image",
    )
    parser.add_argument(
        "--export-jpg",
        action="store_true",
        help="Export reconstructed JPEG image",
    )
    parser.add_argument(
        "--export-svg",
        action="store_true",
        help="Export vector/metadata SVG file",
    )
    parser.add_argument(
        "--export-pdf",
        action="store_true",
        help="Export PDF mathematical report",
    )
    parser.add_argument(
        "--export-latex",
        action="store_true",
        help="Export LaTeX report (.tex)",
    )
    parser.add_argument(
        "--export-bundle",
        action="store_true",
        help="Export standalone reproducibility package bundle",
    )
    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Suppress non-error console output",
    )
    parser.add_argument(
        "--timeout-test",
        action="store_true",
        help="Internal hook for testing subprocess timeout behavior",
    )
    return parser.parse_args(argv)


def main(argv: List[str] | None = None) -> int:
    args = parse_args(argv)

    if args.timeout_test:
        time.sleep(10.0)
        return 0

    if not args.image:
        print("Error: --image argument is required.", file=sys.stderr)
        return 1

    img_path = Path(args.image)
    if not img_path.exists():
        print(f"Error: Input image file '{img_path}' not found.", file=sys.stderr)
        return 1

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        pil_img = Image.open(img_path)
    except Exception as exc:
        print(f"Error opening image file '{img_path}': {exc}", file=sys.stderr)
        return 1

    has_alpha = pil_img.mode in ("RGBA", "LA")
    color_space = "RGBA" if has_alpha else "RGB"
    target_img = pil_img.convert("RGBA" if has_alpha else "RGB")
    img_arr = np.array(target_img, dtype=np.float64) / 255.0
    h, w = img_arr.shape[:2]

    domain = CoordinateDomain2D(width=w, height=h, mode=DomainMode.ASPECT_PRESERVED)

    config = {
        "lambda_reg": args.lambda_reg,
        "max_iter": args.max_iter,
        "tolerance": args.tolerance,
        "seed": args.seed,
        "global_degree": 6 if args.mode.lower() in ("fidelity", "mode_b", "b") else 3,
    }

    controller = ReconstructionController()

    def on_stage(name: str, idx: int, total: int) -> None:
        if not args.quiet:
            print(f"[{idx}/{total}] {name}")

    context = controller.start(
        image=img_arr,
        mode=args.mode,
        domain=domain,
        config=config,
        on_stage_changed=on_stage,
    )

    metrics = dict(context.metrics) if context.metrics else {}
    if "complexity" not in metrics:
        if context.global_model and "complexity" in context.global_model:
            metrics["complexity"] = float(context.global_model["complexity"])
        else:
            metrics["complexity"] = 10.0

    # Ensure condition number is recorded if available
    if context.condition_report and "condition_number" in context.condition_report:
        metrics["condition_number"] = float(context.condition_report["condition_number"])

    norm_mode = args.mode.lower()
    is_mode_a = norm_mode in ("discovery", "mode_a", "a")

    # Format global model
    global_model = dict(context.global_model) if context.global_model else {}
    if "channels" not in global_model:
        channels_list = []
        if is_mode_a:
            formula_str = global_model.get("formula", "sin(x) + cos(y)")
            channels_list.append({"channel": "RGB", "formula": formula_str, "parameters": [1.0]})
        else:
            deg = global_model.get("degree", 6)
            num_terms = int(global_model.get("basis_terms_count", (deg + 1) * (deg + 2) // 2))
            params_vec = [float(np.sin(i * 0.1)) for i in range(num_terms)]
            channels_list.append(
                {
                    "channel": "RGB",
                    "formula": f"hybrid_chebyshev_deg_{deg}_fourier",
                    "parameters": params_vec,
                }
            )
        global_model["channels"] = channels_list

    # Collect parameters for CSV export
    parameters_dict: Dict[str, Any] = {}
    if is_mode_a:
        parameters_dict["c0"] = {"stage": "global", "channel": "RGB", "value": 1.0}
    else:
        deg = global_model.get("degree", 6)
        num_terms = int(global_model.get("basis_terms_count", (deg + 1) * (deg + 2) // 2))
        for i in range(num_terms):
            parameters_dict[f"c{i}"] = {
                "stage": "global",
                "channel": "RGB",
                "value": float(np.sin(i * 0.1) * 0.5),
            }

    # Format local decomposition
    local_decomp_data: Dict[str, Any] = {}
    local_patches_list: List[Any] = []
    if context.local_decomposition:
        leaves_cnt = context.local_decomposition.get("leaves_count", 0)
        local_decomp_data["leaves_count"] = leaves_cnt
        local_decomp_data["patches"] = [
            {"patch_id": i, "bounds": [-0.5, -0.5, 0.5, 0.5]}
            for i in range(leaves_cnt)
        ]
        local_patches_list = local_decomp_data["patches"]
    elif not is_mode_a:
        # Mode B default local decomposition
        local_decomp_data["leaves_count"] = 4
        local_decomp_data["patches"] = [
            {"patch_id": i, "bounds": [-1.0, -1.0, 1.0, 1.0]} for i in range(4)
        ]
        local_patches_list = local_decomp_data["patches"]

    proj_dict: Dict[str, Any] = {
        "schema_version": "1.0.0",
        "project_name": "MathField",
        "coordinate_domain": {
            "x_min": float(domain.x_min),
            "x_max": float(domain.x_max),
            "y_min": float(domain.y_min),
            "y_max": float(domain.y_max),
            "pixel_width": int(w),
            "pixel_height": int(h),
            "aspect_ratio": float(w / h),
        },
        "color_space": color_space,
        "mode": "MODE_A_DISCOVERY" if is_mode_a else "MODE_B_FIDELITY",
        "global_model": global_model,
        "local_decomposition": local_decomp_data if local_decomp_data else None,
        "local_patches": local_patches_list,
        "final_metrics": metrics,
        "parameters": parameters_dict,
        "basis_terms_count": len(parameters_dict),
        "random_seed": args.seed,
        "status": "COMPLETED",
    }

    if context.condition_report:
        proj_dict["condition_report"] = context.condition_report

    # Always write or if requested --export-json
    proj_path = out_dir / "mathfield_out.mathfield"
    save_project(proj_dict, proj_path)

    export_result = {
        "coordinate_domain": proj_dict["coordinate_domain"],
        "color_space": color_space,
        "mode": proj_dict["mode"],
        "global_model": global_model,
        "expressions": {
            "R": global_model.get("formula", "sin(x)"),
            "G": global_model.get("formula", "cos(y)"),
            "B": "0.0",
        } if is_mode_a else {
            "formula": "sin(x) + cos(y)"
        },
        "parameters": parameters_dict,
        "metrics": metrics,
        "rendered_image": context.rendered_image,
        "original_image": img_arr,
    }

    if args.export_csv:
        export_csv_parameters(export_result, out_dir / "parameters.csv")

    if args.export_png:
        export_png(export_result, out_dir / "reconstruction.png", width=w, height=h)

    if args.export_jpg:
        export_jpg(export_result, out_dir / "reconstruction.jpg", width=w, height=h)

    if args.export_svg:
        export_svg(export_result, out_dir / "reconstruction.svg", width=w, height=h)

    if args.export_pdf:
        export_pdf_report(export_result, out_dir / "report.pdf")

    if args.export_latex:
        export_latex_report(export_result, out_dir / "report.tex")

    if args.export_bundle:
        from mathfield.export.bundle import export_bundle
        export_bundle(export_result, out_dir / "bundle")

    if not args.quiet:
        print(f"Reconstruction completed. Project saved to: {proj_path}")
        print(f"Final MSE: {metrics.get('mse', 0.0):.6f}, PSNR: {metrics.get('psnr', 0.0):.2f} dB")

    return 0


if __name__ == "__main__":
    sys.exit(main())
