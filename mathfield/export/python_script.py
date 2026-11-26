"""
mathfield/export/python_script.py

Generates standalone, zero-dependency Python reproduction scripts.
"""

from __future__ import annotations

import re
import warnings
from pathlib import Path
from typing import Any, Dict, List, Union

import numpy as np

from mathfield.core.ast.parser import parse
from mathfield.core.codegen.numpy_gen import to_python_code
from mathfield.core.model_codec import mode_b_payload_from_result
from mathfield.export.raster import natural_key

_MODE_B_HELPERS = '''def _chebyshev_indices(deg_x, deg_y, truncation):
    indices = []
    max_degree = max(deg_x, deg_y)
    for i in range(deg_x + 1):
        for j in range(deg_y + 1):
            if truncation == "total_degree" and i + j > max_degree:
                continue
            if truncation == "hyperbolic_cross" and (i + 1) * (j + 1) > max_degree + 1:
                continue
            indices.append((i, j))
    return indices


def _chebyshev_matrix(X, Y, deg_x, deg_y, truncation):
    x_flat = X.ravel()
    y_flat = Y.ravel()
    tx = np.zeros((x_flat.size, deg_x + 1), dtype=np.float64)
    ty = np.zeros((y_flat.size, deg_y + 1), dtype=np.float64)
    tx[:, 0] = 1.0
    ty[:, 0] = 1.0
    if deg_x >= 1:
        tx[:, 1] = x_flat
        for degree in range(2, deg_x + 1):
            tx[:, degree] = 2.0 * x_flat * tx[:, degree - 1] - tx[:, degree - 2]
    if deg_y >= 1:
        ty[:, 1] = y_flat
        for degree in range(2, deg_y + 1):
            ty[:, degree] = 2.0 * y_flat * ty[:, degree - 1] - ty[:, degree - 2]
    indices = _chebyshev_indices(deg_x, deg_y, truncation)
    matrix = np.empty((x_flat.size, len(indices)), dtype=np.float64)
    for term, (i, j) in enumerate(indices):
        matrix[:, term] = tx[:, i] * ty[:, j]
    return matrix


def _fourier_matrix(X, Y, frequencies):
    x_flat = X.ravel()
    y_flat = Y.ravel()
    matrix = np.empty((x_flat.size, 1 + 2 * len(frequencies)), dtype=np.float64)
    matrix[:, 0] = 1.0
    column = 1
    for kx, ky in frequencies:
        argument = kx * np.pi * x_flat + ky * np.pi * y_flat
        matrix[:, column] = np.cos(argument)
        matrix[:, column + 1] = np.sin(argument)
        column += 2
    return matrix


def _dct_matrix(X, Y, freq_x, freq_y, x_bounds, y_bounds):
    x_norm = (X.ravel() - x_bounds[0]) / (x_bounds[1] - x_bounds[0])
    y_norm = (Y.ravel() - y_bounds[0]) / (y_bounds[1] - y_bounds[0])
    cos_x = np.ones((freq_x + 1, X.size), dtype=np.float64)
    cos_y = np.ones((freq_y + 1, X.size), dtype=np.float64)
    for p in range(1, freq_x + 1):
        cos_x[p] = np.cos(p * np.pi * x_norm)
    for q in range(1, freq_y + 1):
        cos_y[q] = np.cos(q * np.pi * y_norm)
    matrix = np.empty((X.size, (freq_x + 1) * (freq_y + 1)), dtype=np.float64)
    sqrt_two = np.sqrt(2.0)
    term = 0
    for p in range(freq_x + 1):
        alpha_p = 1.0 if p == 0 else sqrt_two
        for q in range(freq_y + 1):
            alpha_q = 1.0 if q == 0 else sqrt_two
            matrix[:, term] = (alpha_p * alpha_q) * (cos_x[p] * cos_y[q])
            term += 1
    return matrix


def _partition_chunks(X, Y, bounds, overlap_ratio, separable):
    patch_count = len(bounds)
    if patch_count == 0:
        return
    if patch_count == 1:
        indices = np.arange(X.size)
        yield 0, indices, np.ones(X.size, dtype=np.float64)
        return
    x_flat = X.ravel()
    y_flat = Y.ravel()
    centers = np.column_stack(
        (0.5 * (bounds[:, 0] + bounds[:, 2]), 0.5 * (bounds[:, 1] + bounds[:, 3]))
    )
    half_widths = np.column_stack(
        (0.5 * (bounds[:, 2] - bounds[:, 0]), 0.5 * (bounds[:, 3] - bounds[:, 1]))
    )

    def build(factor):
        sum_weights = np.zeros(X.size, dtype=np.float64)
        sparse = []
        for k in range(patch_count):
            radius_x = factor * max(half_widths[k, 0], 1e-6)
            radius_y = factor * max(half_widths[k, 1], 1e-6)
            dx = (x_flat - centers[k, 0]) / radius_x
            dy = (y_flat - centers[k, 1]) / radius_y
            if separable:
                adx = np.minimum(np.abs(dx), 1.0)
                ady = np.minimum(np.abs(dy), 1.0)
                inside = (np.abs(dx) <= 1.0 + 1e-12) & (np.abs(dy) <= 1.0 + 1e-12)
                wx = ((1.0 - adx) ** 4.0) * (4.0 * adx + 1.0)
                wy = ((1.0 - ady) ** 4.0) * (4.0 * ady + 1.0)
                values = np.where(inside, wx * wy, 0.0)
            else:
                radius = np.sqrt(dx * dx + dy * dy)
                scaled = np.minimum(radius, 1.0)
                values = np.where(radius <= 1.0 + 1e-12, ((1.0 - scaled) ** 4.0) * (4.0 * scaled + 1.0), 0.0)
            indices = np.flatnonzero(values > 0.0)
            selected = values[indices]
            sparse.append((indices, selected))
            sum_weights[indices] += selected
        return sum_weights, sparse

    factor = 1.0 + max(overlap_ratio, 1e-6)
    sum_weights, sparse = build(factor)
    for _ in range(12):
        if np.all(sum_weights > 1e-12):
            break
        factor *= 1.25
        sum_weights, sparse = build(factor)
    zero_mask = sum_weights <= 1e-12
    if np.any(zero_mask):
        zero_indices = np.flatnonzero(zero_mask)
        zx = x_flat[zero_indices]
        zy = y_flat[zero_indices]
        distance_squared = (centers[:, 0:1] - zx[None, :]) ** 2 + (centers[:, 1:2] - zy[None, :]) ** 2
        scale = max(float(np.max(distance_squared)), 1e-12)
        fallback = np.exp(-distance_squared / scale)
        for k in range(patch_count):
            selected = fallback[k] > 0.0
            indices, values = sparse[k]
            sparse[k] = (
                np.concatenate((indices, zero_indices[selected])),
                np.concatenate((values, fallback[k, selected])),
            )
            sum_weights[zero_indices[selected]] += fallback[k, selected]
    for k, (indices, values) in enumerate(sparse):
        yield k, indices, values / np.maximum(sum_weights[indices], 1e-300)
'''


def _coefficient_assignment_lines(variable: str, coefficients: Any, indent: str = "    ") -> List[str]:
    array = np.asarray(coefficients, dtype=np.float64)
    matrix = array.reshape(-1, 1) if array.ndim == 1 else array
    lines = [f"{indent}{variable} = np.empty({matrix.shape!r}, dtype=np.float64)"]
    for term in range(matrix.shape[0]):
        for channel in range(matrix.shape[1]):
            lines.append(f"{indent}{variable}[{term}, {channel}] = {float(matrix[term, channel]):.17g}")
    return lines


def _mode_b_script_source(payload: Dict[str, Any], domain: Dict[str, Any]) -> str:
    """Emit scalar model assignments plus pure NumPy global/local evaluation."""
    basis = payload["global_basis"]
    chebyshev = basis["chebyshev"]
    fourier = basis["fourier"]
    frequencies = np.asarray(fourier["frequencies"], dtype=np.float64).reshape(-1, 2)
    coefficients = np.asarray(payload["coefficients"], dtype=np.float64)
    coefficient_matrix = coefficients.reshape(-1, 1) if coefficients.ndim == 1 else coefficients
    patches = payload["local_patches"]
    partition = payload["partition_of_unity"]

    lines = [
        f"    cheb_deg_x = {int(chebyshev['deg_x'])}",
        f"    cheb_deg_y = {int(chebyshev['deg_y'])}",
        f"    cheb_truncation = {chebyshev['truncation']!r}",
        f"    include_fourier_dc = {bool(basis['include_fourier_dc'])!r}",
        f"    channel_count = {coefficient_matrix.shape[1]}",
        f"    overlap_ratio = {float(partition['overlap_ratio']):.17g}",
        f"    separable = {bool(partition['separable'])!r}",
        f"    fourier_frequencies = np.empty({frequencies.shape!r}, dtype=np.float64)",
    ]
    for term in range(frequencies.shape[0]):
        for axis in range(2):
            lines.append(f"    fourier_frequencies[{term}, {axis}] = {float(frequencies[term, axis]):.17g}")
    lines.extend(_coefficient_assignment_lines("global_coefficients", coefficient_matrix))

    if patches:
        lines.append(f"    patch_bounds = np.empty({(len(patches), 4)!r}, dtype=np.float64)")
        for patch_index, patch in enumerate(patches):
            for axis, value in enumerate(patch["bounds"]):
                lines.append(f"    patch_bounds[{patch_index}, {axis}] = {float(value):.17g}")
    else:
        lines.append("    patch_bounds = np.empty((0, 4), dtype=np.float64)")

    lines.extend(["    patch_basis_specs = []", "    patch_fourier_frequencies = []", "    patch_coefficients = []"])
    for patch in patches:
        config = patch["basis"]
        spec = {key: value for key, value in config.items() if key != "frequencies"}
        lines.append(f"    patch_basis_specs.append({spec!r})")
        if config["type"] == "fourier2d":
            patch_frequencies = np.asarray(config["frequencies"], dtype=np.float64).reshape(-1, 2)
            lines.append(f"    patch_fourier_frequencies.append(np.empty({patch_frequencies.shape!r}, dtype=np.float64))")
            for term in range(patch_frequencies.shape[0]):
                for axis in range(2):
                    lines.append(
                        "    patch_fourier_frequencies[-1]"
                        f"[{term}, {axis}] = {float(patch_frequencies[term, axis]):.17g}"
                    )
        else:
            lines.append("    patch_fourier_frequencies.append(None)")
        patch_coefficients = np.asarray(patch["coefficients"], dtype=np.float64)
        patch_matrix = patch_coefficients.reshape(-1, 1) if patch_coefficients.ndim == 1 else patch_coefficients
        lines.append(f"    patch_coefficients.append(np.empty({patch_matrix.shape!r}, dtype=np.float64))")
        for term in range(patch_matrix.shape[0]):
            for channel in range(patch_matrix.shape[1]):
                lines.append(
                    f"    patch_coefficients[-1][{term}, {channel}] = {float(patch_matrix[term, channel]):.17g}"
                )

    render_logic = '''
    composite = np.empty((height, width, channel_count), dtype=np.float64)
    sample_block = max(global_coefficients.shape[0], 4096)
    row_block = max(1, sample_block // max(width, 1))
    for row_start in range(0, height, row_block):
        row_stop = min(row_start + row_block, height)
        sub_x = X[row_start:row_stop]
        sub_y = Y[row_start:row_stop]
        phi_chebyshev = _chebyshev_matrix(sub_x, sub_y, cheb_deg_x, cheb_deg_y, cheb_truncation)
        phi_fourier = _fourier_matrix(sub_x, sub_y, fourier_frequencies)
        if include_fourier_dc:
            phi_global = np.hstack((phi_chebyshev, phi_fourier))
        else:
            phi_global = np.hstack((phi_chebyshev, phi_fourier[:, 1:]))
        block_values = phi_global @ global_coefficients
        composite[row_start:row_stop] = block_values.reshape(row_stop - row_start, width, channel_count)
    local_correction = np.zeros_like(composite)
    for patch_index, flat_indices, flat_weights in _partition_chunks(
        X, Y, patch_bounds, overlap_ratio, separable
    ):
        active = flat_weights > 1e-12
        if not np.any(active):
            continue
        flat_indices = flat_indices[active]
        flat_weights = flat_weights[active]
        rows, cols = np.divmod(flat_indices, width)
        row_start, row_stop = int(rows.min()), int(rows.max()) + 1
        col_start, col_stop = int(cols.min()), int(cols.max()) + 1
        bounds = patch_bounds[patch_index]
        center_x = 0.5 * (bounds[0] + bounds[2])
        center_y = 0.5 * (bounds[1] + bounds[3])
        half_x = max(0.5 * (bounds[2] - bounds[0]), 1e-12)
        half_y = max(0.5 * (bounds[3] - bounds[1]), 1e-12)
        sub_x = X[row_start:row_stop, col_start:col_stop]
        sub_y = Y[row_start:row_stop, col_start:col_stop]
        local_x = (sub_x - center_x) / half_x
        local_y = (sub_y - center_y) / half_y
        spec = patch_basis_specs[patch_index]
        if spec["type"] == "chebyshev2d":
            phi_patch = _chebyshev_matrix(local_x, local_y, spec["deg_x"], spec["deg_y"], spec["truncation"])
        elif spec["type"] == "dct2":
            phi_patch = _dct_matrix(
                local_x,
                local_y,
                spec["freq_x"],
                spec["freq_y"],
                spec["x_bounds"],
                spec["y_bounds"],
            )
        else:
            phi_patch = _fourier_matrix(local_x, local_y, patch_fourier_frequencies[patch_index])
        patch_values = phi_patch @ patch_coefficients[patch_index]
        patch_image = patch_values.reshape(row_stop - row_start, col_stop - col_start, channel_count)
        sub_weights = np.zeros((row_stop - row_start, col_stop - col_start), dtype=np.float64)
        sub_weights[rows - row_start, cols - col_start] = flat_weights
        local_correction[row_start:row_stop, col_start:col_stop] += sub_weights[:, :, None] * patch_image
    composite = np.clip(composite + local_correction, 0.0, 1.0)
    if channel_count == 1:
        pixels = np.clip(composite[:, :, 0] * 255.0, 0.0, 255.0).astype(np.uint8)
        img = Image.fromarray(pixels, mode="L")
    else:
        pixels = np.clip(composite * 255.0, 0.0, 255.0).astype(np.uint8)
        mode = {2: "LA", 3: "RGB", 4: "RGBA"}.get(channel_count)
        if mode is None:
            raise ValueError("Unsupported Mode B channel count")
        img = Image.fromarray(pixels, mode=mode)
    return img
'''
    default_width = int(domain.get("pixel_width") or 64)
    default_height = int(domain.get("pixel_height") or 64)
    return f'''#!/usr/bin/env python3
"""
MathField Standalone Reproduction Script.
Pure mathematical evaluation - zero MathField imports.
"""

import argparse
import numpy as np
from PIL import Image


{_MODE_B_HELPERS}


def reconstruct(width: int, height: int) -> Image.Image:
    x = np.linspace({x_min_value(domain):.17g}, {x_max_value(domain):.17g}, width, dtype=np.float64)
    y = np.linspace({y_min_value(domain):.17g}, {y_max_value(domain):.17g}, height, dtype=np.float64)
    X, Y = np.meshgrid(x, y)
{chr(10).join(lines)}
{render_logic}


def main() -> None:
    parser = argparse.ArgumentParser(description="MathField Standalone Image Reconstruction")
    parser.add_argument("--output", "-o", type=str, default="reconstruction.png")
    parser.add_argument(
        "--resolution",
        "-r",
        nargs=2,
        type=int,
        default=[{default_width}, {default_height}],
        metavar=("WIDTH", "HEIGHT"),
    )
    args = parser.parse_args()
    width, height = args.resolution[0], args.resolution[1]
    reconstruct(width=width, height=height).save(args.output)
    print("Reconstruction finished successfully.")


if __name__ == "__main__":
    main()
'''


def x_min_value(domain: Dict[str, Any]) -> float:
    return float(domain.get("x_min", -1.0))


def x_max_value(domain: Dict[str, Any]) -> float:
    return float(domain.get("x_max", 1.0))


def y_min_value(domain: Dict[str, Any]) -> float:
    return float(domain.get("y_min", -1.0))


def y_max_value(domain: Dict[str, Any]) -> float:
    return float(domain.get("y_max", 1.0))


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

    mode_b_payload = mode_b_payload_from_result(model)
    if mode_b_payload is not None:
        out_p.write_text(_mode_b_script_source(mode_b_payload, domain), encoding="utf-8")
        return

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
    if "params[" in render_logic:
        parameter_name = "params"
    elif params_list:
        parameter_name = "_params"
    else:
        parameter_name = ""
    parameter_lines: List[str] = []
    if parameter_name:
        parameter_lines.append(f"    {parameter_name} = np.empty({required_count}, dtype=np.float64)")
        for index in range(required_count):
            value = params_list[index] if index < len(params_list) else 0.0
            parameter_lines.append(f"    {parameter_name}[{index}] = {value:.17g}")
    params_line = "\n".join(parameter_lines) + ("\n" if parameter_lines else "")

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
