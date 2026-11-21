"""
mathfield/export/raster.py

Continuous-resolution raster image exporter (PNG with alpha channel, quality-scaled JPEG).
"""

from __future__ import annotations

import re
import warnings
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
from PIL import Image
from PIL.PngImagePlugin import PngInfo

from mathfield.core.ast.parser import parse
from mathfield.core.engine.cpu_numpy import evaluate_numpy


def to_uint8_array(arr: np.ndarray) -> np.ndarray:
    """Convert float image data in [0,1] or [-1,1] to uint8 bytes."""
    data = np.asarray(arr)
    if data.dtype in (np.float32, np.float64):
        if data.min() < 0.0 and data.max() <= 1.0:
            data = (data + 1.0) * 127.5
        elif data.max() <= 1.0:
            data = data * 255.0
    return np.clip(data, 0, 255).astype(np.uint8)


def natural_key(name: str) -> tuple:
    """Sort alternating text/number runs (c1, c2, ..., c10; p_000_t_01)."""
    tokens = re.findall(r"\d+|\D+", name)
    return tuple((0, int(token)) if token.isdigit() else (1, token) for token in tokens)


def _model_to_image(
    model: Any,
    width: Optional[int] = None,
    height: Optional[int] = None,
) -> Image.Image:
    """Helper to evaluate or extract an Image from model or expression dict."""
    if width is not None and width <= 0:
        raise ValueError(f"Image width must be strictly positive, got {width}")
    if height is not None and height <= 0:
        raise ValueError(f"Image height must be strictly positive, got {height}")

    w = width or 64
    h = height or 64

    # If model is already an image or array
    if isinstance(model, Image.Image):
        if width and height and model.size != (width, height):
            return model.resize((width, height), Image.Resampling.BILINEAR)
        return model

    if isinstance(model, np.ndarray):
        arr = model
        arr = to_uint8_array(arr)
        img = Image.fromarray(arr)
        if width and height and img.size != (width, height):
            return img.resize((width, height), Image.Resampling.BILINEAR)
        return img

    if not isinstance(model, dict):
        raise TypeError(f"Expected dict, ndarray, or PIL Image, got {type(model)}")

    if "rendered_image" in model and isinstance(model["rendered_image"], np.ndarray):
        arr = model["rendered_image"]
        arr = to_uint8_array(arr)
        img = Image.fromarray(arr)
        if width and height and img.size != (width, height):
            return img.resize((width, height), Image.Resampling.BILINEAR)
        return img

    expressions = model.get("expressions", {})
    if not expressions and "global_model" in model:
        gm = model["global_model"]
        if isinstance(gm, dict) and "expressions" in gm:
            expressions = gm["expressions"]

    # Evaluate expressions
    domain = model.get("coordinate_domain", {})
    x_min = float(domain.get("x_min", -1.0))
    x_max = float(domain.get("x_max", 1.0))
    y_min = float(domain.get("y_min", -1.0))
    y_max = float(domain.get("y_max", 1.0))

    xs = np.linspace(x_min, x_max, w, dtype=np.float64)
    ys = np.linspace(y_min, y_max, h, dtype=np.float64)
    X, Y = np.meshgrid(xs, ys)

    params = model.get("parameters", {})
    if isinstance(params, dict):
        p_list = []
        for k in sorted(params.keys(), key=natural_key):
            val = params[k]
            if isinstance(val, dict):
                p_list.append(float(val.get("value", 0.0)))
            else:
                p_list.append(float(val))
        p_arr = np.array(p_list, dtype=np.float64)
    elif isinstance(params, (list, tuple, np.ndarray)):
        p_arr = np.array(params, dtype=np.float64)
    else:
        p_arr = np.array([], dtype=np.float64)

    has_alpha = "A" in expressions or "alpha" in expressions

    def _eval_expr(expr_val: Any, allow_label: bool = False) -> np.ndarray:
        if isinstance(expr_val, (int, float)):
            return np.full((h, w), float(expr_val), dtype=np.float64)
        if isinstance(expr_val, str):
            try:
                ast_node = parse(expr_val)
                val = evaluate_numpy(ast_node, X, Y, p_arr)
                return val
            except Exception as exc:
                if allow_label:
                    # Some projects store a descriptive model label in the
                    # `formula` metadata field; use a documented neutral field
                    # instead of silently inventing coefficients.
                    warnings.warn(f"Formula {expr_val!r} is metadata, not an expression: {exc}", stacklevel=2)
                    return np.full((h, w), 0.5, dtype=np.float64)
                raise ValueError(f"Cannot evaluate channel expression {expr_val!r}: {exc}") from exc
        return np.zeros((h, w), dtype=np.float64)

    if has_alpha:
        r_expr = expressions.get("R", "0")
        g_expr = expressions.get("G", "0")
        b_expr = expressions.get("B", "0")
        a_expr = expressions.get("A", expressions.get("alpha", "1"))

        R = _eval_expr(r_expr)
        G = _eval_expr(g_expr)
        B = _eval_expr(b_expr)
        A = _eval_expr(a_expr)

        # Normalize to [0, 255]
        # In domain [-1, 1], linear coords 'x' or 'y' map to [-1, 1].
        # If in [-1, 1], map to [0, 1] if max <= 1.0 and min >= -1.0 or clip
        def to_uint8(ch_data: np.ndarray) -> np.ndarray:
            if ch_data.min() < 0.0 or ch_data.max() <= 1.0:
                # If in [0, 1]
                if ch_data.min() >= 0.0 and ch_data.max() <= 1.0:
                    scaled = ch_data * 255.0
                else:
                    # In [-1, 1]
                    scaled = (ch_data + 1.0) * 127.5
            else:
                scaled = ch_data
            return np.clip(scaled, 0.0, 255.0).astype(np.uint8)

        rgba = np.stack([to_uint8(R), to_uint8(G), to_uint8(B), to_uint8(A)], axis=-1)
        return Image.fromarray(rgba, mode="RGBA")
    else:
        if "formula" in expressions:
            f_val = _eval_expr(expressions["formula"], allow_label=True)
            scaled = np.clip((f_val + 1.0) * 127.5 if f_val.min() < 0.0 else f_val * 255.0, 0, 255).astype(np.uint8)
            rgb = np.stack([scaled, scaled, scaled], axis=-1)
            return Image.fromarray(rgb, mode="RGB")

        r_expr = expressions.get("R", "0")
        g_expr = expressions.get("G", "0")
        b_expr = expressions.get("B", "0")

        R = _eval_expr(r_expr)
        G = _eval_expr(g_expr)
        B = _eval_expr(b_expr)

        def to_uint8_rgb(ch_data: np.ndarray) -> np.ndarray:
            if ch_data.min() >= 0.0 and ch_data.max() <= 1.0:
                scaled = ch_data * 255.0
            elif ch_data.min() < 0.0 and ch_data.max() <= 1.0:
                scaled = (ch_data + 1.0) * 127.5
            else:
                scaled = ch_data
            return np.clip(scaled, 0.0, 255.0).astype(np.uint8)

        rgb = np.stack([to_uint8_rgb(R), to_uint8_rgb(G), to_uint8_rgb(B)], axis=-1)
        return Image.fromarray(rgb, mode="RGB")


def export_png(
    model: Any,
    output_path: Union[str, Path],
    width: Optional[int] = None,
    height: Optional[int] = None,
    metadata: Optional[Dict[str, str]] = None,
) -> None:
    """
    Export reconstructed image or mathematical model to continuous-resolution PNG.
    Supports RGBA alpha channel transparency and metadata embedding.
    """
    out_p = Path(output_path)
    if not out_p.parent.exists():
        raise FileNotFoundError(f"Destination directory does not exist: {out_p.parent}")

    img = _model_to_image(model, width=width, height=height)

    pnginfo = PngInfo()
    if metadata:
        for k, v in metadata.items():
            pnginfo.add_text(str(k), str(v))

    img.save(out_p, format="PNG", pnginfo=pnginfo)


def export_jpg(
    model: Any,
    output_path: Union[str, Path],
    quality: int = 75,
    width: Optional[int] = None,
    height: Optional[int] = None,
    metadata: Optional[Dict[str, str]] = None,
) -> None:
    """
    Export reconstructed image or mathematical model to quality-scaled JPEG.
    """
    out_p = Path(output_path)
    if not out_p.parent.exists():
        raise FileNotFoundError(f"Destination directory does not exist: {out_p.parent}")

    img = _model_to_image(model, width=width, height=height)
    if img.mode != "RGB":
        img = img.convert("RGB")

    save_kwargs: Dict[str, Any] = {"format": "JPEG", "quality": int(quality)}
    img.save(out_p, **save_kwargs)
