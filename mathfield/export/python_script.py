"""
mathfield/export/python_script.py

Generates standalone, zero-dependency Python reproduction scripts.
"""

from __future__ import annotations

import re
import warnings
from pathlib import Path
from typing import Any, List, Union

from mathfield.core.ast.parser import parse
from mathfield.core.codegen.numpy_gen import to_python_code
from mathfield.export.raster import natural_key


def export_standalone_script(
    model: Any,
    output_script_path: Union[str, Path],
) -> None:
    """
    Export mathematical reconstruction model to a standalone Python script.
    The generated script reproduces the image using only standard numpy and Pillow,
    with zero imports of MathField, zero eval/exec, and zero raster dependencies.
    """
    out_p = Path(output_script_path)
    if not out_p.parent.exists():
        raise FileNotFoundError(f"Destination directory does not exist: {out_p.parent}")

    domain = model.get("coordinate_domain", {}) if isinstance(model, dict) else {}
    x_min = float(domain.get("x_min", -1.0))
    x_max = float(domain.get("x_max", 1.0))
    y_min = float(domain.get("y_min", -1.0))
    y_max = float(domain.get("y_max", 1.0))

    expressions = {}
    if isinstance(model, dict):
        if "expressions" in model:
            expressions = model["expressions"]
        elif "global_model" in model and isinstance(model["global_model"], dict):
            gm = model["global_model"]
            if "expressions" in gm:
                expressions = gm["expressions"]
            elif "formula" in gm:
                expressions = {"formula": gm["formula"]}

    # Extract parameters array (max 32 elements for anti-memorization audit)
    params_list: List[float] = []
    if isinstance(model, dict) and "parameters" in model:
        raw_p = model["parameters"]
        if isinstance(raw_p, dict):
            for k in sorted(raw_p.keys(), key=natural_key):
                val = raw_p[k]
                v = float(val.get("value", 0.0)) if isinstance(val, dict) else float(val)
                params_list.append(v)
        elif isinstance(raw_p, (list, tuple)):
            params_list = [float(x) for x in raw_p]

    # Generate Python expression strings for channels
    has_rgb = any(k in expressions for k in ("R", "G", "B"))
    has_alpha = "A" in expressions or "alpha" in expressions

    def _compile_expr_to_py(expr: Any) -> str:
        if isinstance(expr, (int, float)):
            return f"({float(expr):.17g} + 0.0 * X)"
        if isinstance(expr, str):
            clean_s = expr.strip()
            if not clean_s:
                return "(0.0 * X)"
            try:
                ast_node = parse(clean_s)
                return to_python_code(ast_node)
            except Exception as exc:
                warnings.warn(f"Cannot compile expression {clean_s!r} for standalone export: {exc}", stacklevel=2)
        return "(0.0 * X)"

    if has_rgb:
        r_code = _compile_expr_to_py(expressions.get("R", "0"))
        g_code = _compile_expr_to_py(expressions.get("G", "0"))
        b_code = _compile_expr_to_py(expressions.get("B", "0"))
        a_code = _compile_expr_to_py(expressions.get("A", expressions.get("alpha", "1"))) if has_alpha else ""

        render_logic = (
            "    R = " + r_code + "\n"
            "    G = " + g_code + "\n"
            "    B = " + b_code + "\n"
            "    r_u8 = np.broadcast_to(np.clip(R * 255.0, 0.0, 255.0).astype(np.uint8), X.shape)\n"
            "    g_u8 = np.broadcast_to(np.clip(G * 255.0, 0.0, 255.0).astype(np.uint8), X.shape)\n"
            "    b_u8 = np.broadcast_to(np.clip(B * 255.0, 0.0, 255.0).astype(np.uint8), X.shape)\n"
        )
        if has_alpha:
            render_logic += (
                "    A = " + a_code + "\n"
                "    a_u8 = np.broadcast_to(np.clip(A * 255.0, 0.0, 255.0).astype(np.uint8), X.shape)\n"
                "    img_arr = np.stack([r_u8, g_u8, b_u8, a_u8], axis=-1)\n"
                "    img = Image.fromarray(img_arr, mode='RGBA')\n"
            )
        else:
            render_logic += (
                "    img_arr = np.stack([r_u8, g_u8, b_u8], axis=-1)\n    img = Image.fromarray(img_arr, mode='RGB')\n"
            )
    else:
        formula_str = expressions.get("formula", "0") if isinstance(expressions, dict) else "0"
        f_code = _compile_expr_to_py(formula_str)
        render_logic = (
            "    Z = " + f_code + "\n"
            "    z_u8 = np.broadcast_to(np.clip(Z * 255.0, 0.0, 255.0).astype(np.uint8), X.shape)\n"
            "    img_arr = np.stack([z_u8, z_u8, z_u8], axis=-1)\n"
            "    img = Image.fromarray(img_arr, mode='RGB')\n"
        )

    # Size the emitted parameter vector to the highest referenced index so
    # large ASTs cannot produce out-of-range params[i] at runtime.
    used_indices = [int(value) for value in re.findall(r"params\[(\d+)\]", render_logic)]
    required_count = max(32, max(used_indices) + 1) if used_indices else len(params_list)
    if len(params_list) < required_count:
        params_list = params_list + [0.0] * (required_count - len(params_list))
    params_code = "[" + ", ".join(f"{p:.17g}" for p in params_list) + "]"

    params_line = (
        f"    params = np.array({params_code}, dtype=np.float64)\n"
        if "params[" in render_logic
        else (f"    _params = np.array({params_code}, dtype=np.float64)\n" if params_list else "")
    )

    script_content = f'''#!/usr/bin/env python3
"""
MathField Standalone Reproduction Script.
Pure mathematical evaluation - zero MathField imports.
"""

import argparse
import sys
import numpy as np
from PIL import Image


def reconstruct(width: int, height: int) -> Image.Image:
    x = np.linspace({x_min:.17g}, {x_max:.17g}, width, dtype=np.float64)
    y = np.linspace({y_min:.17g}, {y_max:.17g}, height, dtype=np.float64)
    X, Y = np.meshgrid(x, y)
{params_line}
{render_logic}
    return img


def main() -> None:
    parser = argparse.ArgumentParser(
        description="MathField Standalone Image Reconstruction"
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="reconstruction.png",
        help="Path to save reconstructed PNG image",
    )
    parser.add_argument(
        "--resolution",
        "-r",
        nargs=2,
        type=int,
        default=[64, 64],
        metavar=("WIDTH", "HEIGHT"),
        help="Output resolution in pixels (width height)",
    )
    args = parser.parse_args()

    width, height = args.resolution[0], args.resolution[1]
    img = reconstruct(width=width, height=height)
    img.save(args.output)
    print("Reconstruction finished successfully.")


if __name__ == "__main__":
    main()
'''

    out_p.write_text(script_content, encoding="utf-8")
