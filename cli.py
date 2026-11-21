#!/usr/bin/env python3
"""
cli.py

Command-line interface for the MathField Image Reconstruction Engine.
Supports Mode A (Symbolic Discovery) and Mode B (Maximum Fidelity),
comprehensive export generation, seed control, and test automation hooks.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from PIL import Image

from mathfield.core.ast.nodes import ASTNode, BinaryOpNode, ConstantNode, FunctionCallNode, UnaryOpNode
from mathfield.core.domain import CoordinateDomain2D, DomainMode
from mathfield.core.pipeline.controller import ReconstructionController
from mathfield.export.csv_export import export_csv_parameters
from mathfield.export.latex_report import export_latex_report
from mathfield.export.pdf_report import export_pdf_report
from mathfield.export.project_file import save_project
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


def _ast_constants(node: ASTNode) -> List[float]:
    """Collect fitted numeric constants from a discovered symbolic AST."""
    if isinstance(node, ConstantNode):
        return [float(node.value)]
    if isinstance(node, BinaryOpNode):
        return _ast_constants(node.left) + _ast_constants(node.right)
    if isinstance(node, UnaryOpNode):
        return _ast_constants(node.operand)
    if isinstance(node, FunctionCallNode):
        return [v for arg in node.args for v in _ast_constants(arg)]
    return []


def _format_mode_b_channel(
    global_model: Dict[str, Any],
    channel: int,
    max_terms: int | None = None,
    human: bool = False,
) -> str:
    """Render a real analytical equation from the fitted global coefficients.

    With human=False the output uses only MathField-parseable syntax so the
    standalone reproduction script can compile it. With human=True it uses
    conventional mathematical notation for reports and posters.
    """
    if max_terms is None:
        # Machine expressions must retain every non-negligible coefficient:
        # truncating a large cancellation-heavy expansion corrupts the model.
        max_terms = 8 if human else 100000
    cheb = global_model.get("chebyshev_basis")
    fourier = global_model.get("fourier_basis")
    raw_coeffs = global_model.get("coefficients")
    if cheb is None or fourier is None or raw_coeffs is None:
        return global_model.get("formula", "0")
    coeffs = np.asarray(raw_coeffs)
    c_vec = coeffs[:, channel] if coeffs.ndim == 2 else coeffs

    terms: List[Tuple[float, str]] = []
    for k, (i, j) in enumerate(cheb.indices):
        value = float(c_vec[k])
        if abs(value) <= 1e-4:
            continue
        if human:
            name = "1" if i == 0 and j == 0 else f"T_{i}(x)" if j == 0 else f"T_{j}(y)" if i == 0 else f"T_{i}(x)T_{j}(y)"
        else:
            name = "1" if i == 0 and j == 0 else f"x^{i}" if j == 0 else f"y^{j}" if i == 0 else f"x^{i}*y^{j}"
        terms.append((abs(value), f"{value:+.6g}*{name}" if not human else f"{value:+.4f}·{name}"))

    base = cheb.num_terms
    for m, (kx, ky) in enumerate(fourier.freq_pairs):
        cos_v = float(c_vec[base + 2 * m])
        sin_v = float(c_vec[base + 2 * m + 1])
        if human:
            arg = f"({int(kx)}πx" if kx else "("
            if ky:
                arg += f" + {int(ky)}πy)" if kx else f"{int(ky)}πy)"
            else:
                arg += ")"
        else:
            arg = f"({int(kx)}*pi*x" if kx else "("
            if ky:
                arg += f" + {int(ky)}*pi*y)" if kx else f"{int(ky)}*pi*y)"
            else:
                arg += ")"
        for value, func in ((cos_v, "cos"), (sin_v, "sin")):
            if abs(value) > 1e-4:
                text = f"{value:+.6g}*{func}{arg}" if not human else f"{value:+.4f}·{func}{arg}"
                terms.append((abs(value), text))

    terms.sort(key=lambda item: item[0], reverse=True)
    leading = " ".join(text for _, text in terms[:max_terms])
    return f"{leading} + ... [{len(terms)} active terms]" if human else (leading or "0.0")


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
        with Image.open(img_path) as pil_img:
            has_alpha = pil_img.mode in ("RGBA", "LA")
            color_space = "RGBA" if has_alpha else "RGB"
            target_img = pil_img.convert("RGBA" if has_alpha else "RGB")
            img_arr = np.array(target_img, dtype=np.float64) / 255.0
    except Exception as exc:
        print(f"Error opening image file '{img_path}': {exc}", file=sys.stderr)
        return 1

    h, w = img_arr.shape[:2]
    if h == 0 or w == 0:
        print("Error: Input image has zero width or height.", file=sys.stderr)
        return 1

    domain = CoordinateDomain2D(width=w, height=h, mode=DomainMode.ASPECT_PRESERVED)

    config = {
        "lambda_reg": args.lambda_reg,
        "reg_lambda": args.lambda_reg,
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
        metrics["complexity"] = float(context.global_model.get("complexity", 0.0)) if context.global_model else 0.0

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
            if "coefficients" in global_model and global_model["coefficients"] is not None:
                actual = np.asarray(global_model["coefficients"])
                flat_coeffs = actual[:, 0] if actual.ndim == 2 else actual.ravel()
                params_vec = [float(x) for x in flat_coeffs]
            else:
                num_terms = int(global_model.get("basis_terms_count", (deg + 1) * (deg + 2) // 2))
                params_vec = [0.0 for _ in range(num_terms)]
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
        channel_asts = global_model.get("channel_asts") or []
        constants = [
            value
            for ast in channel_asts
            for value in _ast_constants(ast)
            if abs(value) > 1e-12
        ]
        if not constants:
            best_ast = global_model.get("symbolic_ast")
            constants = _ast_constants(best_ast) if best_ast is not None else []
        for i, value in enumerate(constants):
            parameters_dict[f"c{i}"] = {"stage": "global", "channel": "RGB", "value": float(value)}
    else:
        if "coefficients" in global_model and global_model["coefficients"] is not None:
            actual = np.asarray(global_model["coefficients"])
            flat_coeffs = actual[:, 0] if actual.ndim == 2 else actual.ravel()
            for i, val in enumerate(flat_coeffs):
                parameters_dict[f"c{i}"] = {
                    "stage": "global",
                    "channel": "RGB",
                    "value": float(val),
                }
        else:
            deg = global_model.get("degree", 6)
            num_terms = int(global_model.get("basis_terms_count", (deg + 1) * (deg + 2) // 2))
            for i in range(num_terms):
                parameters_dict[f"c{i}"] = {
                    "stage": "global",
                    "channel": "RGB",
                    "value": 0.0,
                }

    # Format local decomposition
    local_decomp_data: Dict[str, Any] = {}
    local_patches_list: List[Any] = []
    if context.local_decomposition:
        actual_patches = context.local_decomposition.get("patches") or []
        local_decomp_data["leaves_count"] = int(
            context.local_decomposition.get("leaves_count", len(actual_patches))
        )
        local_decomp_data["patches"] = [
            {"patch_id": i, "bounds": [float(v) for v in patch.bounds]}
            for i, patch in enumerate(actual_patches)
            if hasattr(patch, "bounds")
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

    if is_mode_a:
        mode_expr = {
            "R": str(global_model.get("symbolic_ast", global_model.get("formula", "0"))),
            "G": str(global_model.get("symbolic_ast", global_model.get("formula", "0"))),
            "B": str(global_model.get("symbolic_ast", global_model.get("formula", "0"))),
            "formula": str(global_model.get("formula", "0")),
        }
    else:
        mode_expr = {
            "R": _format_mode_b_channel(global_model, 0),
            "G": _format_mode_b_channel(global_model, 1),
            "B": _format_mode_b_channel(global_model, 2),
            "formula": f"chebyshev2d_deg_{global_model.get('degree', 6)} + fourier2d_harmonics + local_patches",
        }

    global_equations = {
        channel: _format_mode_b_channel(global_model, index, human=True)
        for index, channel in enumerate("RGB")
    } if not is_mode_a else {
        channel: mode_expr[channel] for channel in "RGB"
    }

    export_result = {
        "coordinate_domain": proj_dict["coordinate_domain"],
        "color_space": color_space,
        "mode": proj_dict["mode"],
        "global_model": global_model,
        "global_equations": global_equations,
        "expressions": mode_expr,
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
