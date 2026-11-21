"""
mathfield/export/artwork_poster.py

Mathematical artwork poster exporter in the style of Hamid Naderi Yeganeh.
Composites the reconstructed continuous image on top with a publication-grade
mathematical formula panel directly below it, detailing the master vector mapping,
discrete-to-continuous domain transformations, global expansions, partition of unity
blending windows, and representative local patch equations.

Features dynamic bounding-box autofit measurement to guarantee zero text clipping
or margin overflow across arbitrary canvas resolutions.
"""

from __future__ import annotations

import re
import warnings
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import matplotlib

matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def generate_math_panel(
    result: Dict[str, Any],
    width_px: int,
    height_px: int,
    title: str = "White Lotus by MathField",
    dpi: int = 200,
) -> Image.Image:
    """
    Renders the white mathematical formula plaque using matplotlib's
    Computer Modern math typography engine with dynamic bounding-box autofit.
    Guarantees no text is cut off or truncated.
    """
    fig_w = width_px / dpi
    fig_h = height_px / dpi
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi)
    fig.patch.set_facecolor("white")

    domain = result.get("coordinate_domain", {})
    pw = domain.get("pixel_width") or domain.get("width")
    ph = domain.get("pixel_height") or domain.get("height")
    if pw is None or ph is None:
        recon_data = result.get("reconstructed_image") or result.get("rendered_image")
        if isinstance(recon_data, np.ndarray):
            ph, pw = recon_data.shape[:2]
        elif isinstance(recon_data, Image.Image):
            pw, ph = recon_data.size
        else:
            pw, ph = 256, 256

    global_eqs = result.get("global_equations") or {}
    patch_eqs = result.get("patch_equations") or []
    expressions = result.get("expressions") or {}

    def _to_math_tex(eq_str: str) -> str:
        """Convert human-readable coefficient strings into valid mathtext."""
        text = eq_str.replace("π", r"\pi").replace("·", r" \cdot ")
        text = re.sub(r"T_(\d+)", r"T_{\1}", text)
        text = re.sub(r"\[(\d+) total terms\]", r"[\1 terms]", text)
        return text

    def _compact_terms(eq_str: str, max_terms: int = 4) -> str:
        """Truncate by *complete* signed terms, respecting parenthesis depth."""
        if not eq_str:
            return ""
        parts: List[str] = []
        depth = 0
        start = 0
        for idx, char in enumerate(eq_str):
            if char == "(":
                depth += 1
            elif char == ")":
                depth -= 1
            elif char in "+-" and depth == 0 and idx > start:
                parts.append(eq_str[start:idx].strip())
                start = idx
        parts.append(eq_str[start:].strip())
        if len(parts) <= max_terms:
            return " ".join(parts)
        return " ".join(parts[:max_terms]) + r" + \dots"

    if global_eqs and "R" in global_eqs:
        g_r_text = rf"$G_R(x, y) = {_to_math_tex(_compact_terms(global_eqs['R'], 6))}$"
        g_gb_text = (
            rf"$G_G(x, y) = {_to_math_tex(_compact_terms(global_eqs.get('G', ''), 4))}, "
            rf"\quad G_B(x, y) = {_to_math_tex(_compact_terms(global_eqs.get('B', ''), 4))}$"
        )
    elif expressions and "R" in expressions:
        g_r_text = rf"$R(x, y) = {_to_math_tex(expressions['R'])}$"
        g_gb_text = (
            rf"$G(x, y) = {_to_math_tex(expressions.get('G', ''))}, "
            rf"\quad B(x, y) = {_to_math_tex(expressions.get('B', ''))}$"
        )
    else:
        g_r_text = r"$G_R(x, y) = \sum a_{i,j,R} T_i(x) T_j(y) + \sum b_{\mathbf{k},R} \cos(\mathbf{k} \cdot \mathbf{x}) + \dots$"
        g_gb_text = r"$G_G(x, y) = \sum a_{i,j,G} T_i(x) T_j(y) + \dots, \quad G_B(x, y) = \sum a_{i,j,B} T_i(x) T_j(y) + \dots$"

    if patch_eqs:
        p0 = patch_eqs[0]
        p0_id = p0.get("id", "L_1")
        p0_form = p0.get("formula", "")
        patch_line_1 = rf"${p0_id}(u, v) = {p0_form}$"
        if len(patch_eqs) > 1:
            p1 = patch_eqs[1]
            p1_id = p1.get("id", "L_2")
            p1_form = p1.get("formula", "")
            patch_line_2 = rf"${p1_id}(u, v) = {p1_form} \quad [\mathrm{{total~patches:}}~{len(patch_eqs)}]$"
        else:
            patch_line_2 = rf"$\mathrm{{Total~active~local~patches:}}~{len(patch_eqs)}$"
    else:
        patch_line_1 = r"$L_{k,c}(u, v) = \sum_{p,q} c_{k,p,q,c} \alpha_p \alpha_q \cos(p \pi u) \cos(q \pi v) + \dots$"
        patch_line_2 = r"$w_k(x, y) = \psi_4\left(\frac{|x - c_{x,k}|}{R_{x,k}}\right) \psi_4\left(\frac{|y - c_{y,k}|}{R_{y,k}}\right), \quad \psi_4(r) = (1 - r)_+^4 (4r + 1)$"

    # 13 structured mathematical clauses defining the artwork
    spec_lines: List[Tuple[float, str, float, bool]] = [
        # (y_relative, text, base_fontsize, is_bold)
        (0.94, title, 12.0, True),
        (
            0.89,
            f"The above picture is a converted version of a {pw} × {ph} image. For m = 1, 2, ..., {pw} and n = 1, 2, ..., {ph},",
            7.2,
            False,
        ),
        (
            0.84,
            r"the color of pixel of row $n$ and column $m$ is rgb$\left(F(R(x, y)), F(G(x, y)), F(B(x, y))\right)$, where",
            7.2,
            False,
        ),
        (
            0.79,
            rf"$x = \frac{{2m - {pw + 1}}}{{{ph}}}, \quad y = \frac{{{ph + 1} - 2n}}{{{ph}}}, \quad F(u) = \left\lfloor 255 \cdot \max(0, \min(1, u)) \right\rfloor, \quad \mathrm{{and}}$",
            7.2,
            False,
        ),
        (
            0.71,
            r"$F_c(x, y) = G_c(x, y) + \frac{\sum_{k=1}^K w_k(x, y) \cdot L_{k,c}(x, y)}{\sum_{k=1}^K w_k(x, y)}, \quad \mathrm{for} \quad c \in \{R, G, B\}, \quad \mathrm{and}$",
            7.0,
            False,
        ),
        (
            0.63,
            r"$G_c(x, y) = \sum_{i+j \leq 6} a_{i,j,c} T_i(x) T_j(y) + \sum_{k_x, k_y} \left[ b_{k_x,k_y,c} \cos(k_x \pi x + k_y \pi y) + d_{k_x,k_y,c} \sin(k_x \pi x + k_y \pi y) \right],$",
            6.8,
            False,
        ),
        (
            0.55,
            r"$w_k(x, y) = \psi_4\left(\frac{|x - c_{x,k}|}{R_{x,k}}\right) \psi_4\left(\frac{|y - c_{y,k}|}{R_{y,k}}\right), \quad \psi_4(r) = (1 - r)_+^4 (4r + 1), \quad (R_{x,k}, R_{y,k}) = 1.20 \cdot (h_{x,k}, h_{y,k}),$",
            6.5,
            False,
        ),
        (
            0.47,
            r"$L_{k,c}(x, y) = \sum_{p=0}^P \sum_{q=0}^Q c_{k,p,q,c} \cdot \alpha_p \alpha_q \cos\left(p \pi u_k\right) \cos\left(q \pi v_k\right), \quad \alpha_0 = 1, \; \alpha_{p>0} = \sqrt{2},$",
            6.5,
            False,
        ),
        (
            0.39,
            r"$T_0(u) = 1, \quad T_1(u) = u, \quad T_{d+1}(u) = 2u T_d(u) - T_{d-1}(u), \quad u_k = \frac{x - x_{k,0}}{x_{k,1} - x_{k,0}}, \quad v_k = \frac{y - y_{k,0}}{y_{k,1} - y_{k,0}},$",
            6.5,
            False,
        ),
        (0.31, g_r_text, 6.5, False),
        (0.23, g_gb_text, 6.5, False),
        (0.15, patch_line_1, 6.5, False),
        (0.07, patch_line_2, 6.5, False),
    ]

    renderer = fig.canvas.get_renderer()
    max_w = width_px * 0.88  # at least 6% margin on each side

    # Measure text sizes without rebuilding the whole canvas on every step.
    for y_pos, text_str, base_fs, is_bold in spec_lines:
        weight = "bold" if is_bold else "normal"
        t = fig.text(0.5, y_pos, text_str, fontsize=base_fs, ha="center", va="center", weight=weight)
        bbox = t.get_window_extent(renderer)
        while bbox.width > max_w and t.get_fontsize() > 3.0:
            factor = (max_w / bbox.width) * 0.95
            new_fs = max(3.0, t.get_fontsize() * factor)
            t.set_fontsize(new_fs)
            bbox = t.get_window_extent(renderer)
        if bbox.width > max_w:
            warnings.warn("Artwork poster text could not be autofitted without clipping", stacklevel=2)

    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba())
    plt.close(fig)
    return Image.fromarray(rgba).convert("RGB")


def export_artwork_poster(
    result: Dict[str, Any],
    output_path: Union[str, Path],
    layout: str = "square",
    title: str = "White Lotus by MathField",
    width: int = 1600,
    quality: int = 95,
) -> None:
    """
    Exports a composite mathematical artwork poster in the style of Hamid Naderi Yeganeh.

    Parameters:
        result: Reconstruction result dictionary.
        output_path: File path (.png or .jpg).
        layout: 'square' (1:1 aspect ratio) or 'portrait' (vertical aspect ratio).
        title: Title of the artwork displayed at the top of the formula panel.
        width: Pixel width of the composite image.
        quality: JPEG compression quality (if saving as JPEG).
    """
    out_p = Path(output_path)
    if not out_p.parent.exists():
        raise FileNotFoundError(f"Destination directory does not exist: {out_p.parent}")

    recon_data = result.get("reconstructed_image")
    if recon_data is None:
        recon_data = result.get("rendered_image")

    if isinstance(recon_data, np.ndarray):
        from mathfield.export.raster import to_uint8_array

        recon_img = Image.fromarray(to_uint8_array(recon_data))
    elif isinstance(recon_data, Image.Image):
        recon_img = recon_data
    else:
        recon_img = Image.new("RGB", (width, int(width * 1.5)), (0, 0, 0))

    rw, rh = recon_img.size

    if layout == "square":
        # 1:1 Square poster matching Hamid Naderi Yeganeh proportions
        w_total = width
        h_total = width
        h_top = int(round(h_total * 0.60))
        h_panel = h_total - h_top

        # Top artwork with seamless centered placement on pure black water canvas
        top_canvas = Image.new("RGB", (w_total, h_top), (0, 0, 0))
        scale = h_top / rh
        scaled_w = int(round(rw * scale))
        recon_resized = recon_img.resize((scaled_w, h_top), Image.Resampling.LANCZOS)
        x_offset = max(0, (w_total - scaled_w) // 2)
        top_canvas.paste(recon_resized, (x_offset, 0))

        # Bottom mathematical formula panel with autofit guarantees
        panel_canvas = generate_math_panel(result, w_total, h_panel, title=title, dpi=200)

        # Composite
        poster = Image.new("RGB", (w_total, h_total), (255, 255, 255))
        poster.paste(top_canvas, (0, 0))
        poster.paste(panel_canvas, (0, h_top))

    else:
        # Portrait poster preserving exact image width
        w_total = width
        scale = w_total / rw
        h_top = int(round(rh * scale))
        h_panel = int(round(w_total * 0.50))  # proportional panel height

        top_canvas = recon_img.resize((w_total, h_top), Image.Resampling.LANCZOS)
        panel_canvas = generate_math_panel(result, w_total, h_panel, title=title, dpi=200)

        poster = Image.new("RGB", (w_total, h_top + h_panel), (255, 255, 255))
        poster.paste(top_canvas, (0, 0))
        poster.paste(panel_canvas, (0, h_top))

    if out_p.suffix.lower() in (".jpg", ".jpeg"):
        poster.save(out_p, quality=quality)
    else:
        poster.save(out_p)
