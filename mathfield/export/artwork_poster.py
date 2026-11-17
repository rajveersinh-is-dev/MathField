"""
mathfield/export/artwork_poster.py

Mathematical artwork poster exporter in the style of Hamid Naderi Yeganeh.
Composites the reconstructed continuous image on top with a publication-grade
mathematical formula panel directly below it, detailing the master vector mapping,
discrete-to-continuous domain transformations, global expansions, partition of unity
blending windows, and representative local patch equations.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Union

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
    dpi: int = 250,
) -> Image.Image:
    """
    Renders the white mathematical formula plaque using matplotlib's
    Computer Modern math typography engine.
    """
    fig_w = width_px / dpi
    fig_h = height_px / dpi
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi)
    fig.patch.set_facecolor("white")

    domain = result.get("coordinate_domain", {})
    pw = domain.get("pixel_width", 236)
    ph = domain.get("pixel_height", 354)

    # 1. Title
    fig.text(0.5, 0.95, title, fontsize=12.5, weight="bold", ha="center", va="center", fontfamily="sans-serif")

    # 2. Explanatory intro text
    t1 = f"The above picture is a converted version of a {pw} × {ph} image. For m = 1, 2, 3, ..., {pw} and n = 1, 2, 3, ..., {ph}, the color of the pixel of the row n and the column m is"
    fig.text(0.5, 0.895, t1, fontsize=7.8, ha="center", va="center", fontfamily="sans-serif")

    t2 = rf"rgb$\left(F(R(x, y)), F(G(x, y)), F(B(x, y))\right)$, where $x = \frac{{2m - {pw + 1}}}{{{ph}}}$, $y = \frac{{{ph + 1} - 2n}}{{{ph}}}$, $F(u) = \left\lfloor 255 \cdot \max(0, \min(1, u)) \right\rfloor$, and"
    fig.text(0.5, 0.84, t2, fontsize=7.8, ha="center", va="center")

    # 3. Master Vector Field Equation
    eq1 = r"$F_c(x, y) = G_c(x, y) + \frac{\sum_{k=1}^K w_k(x, y) \cdot L_{k,c}(x, y)}{\sum_{k=1}^K w_k(x, y)}, \quad \mathrm{for} \quad c \in \{R, G, B\}, \quad \mathrm{and}$"
    fig.text(0.5, 0.75, eq1, fontsize=7.8, ha="center", va="center")

    # 4. Global Color Field
    eq2 = r"$G_c(x, y) = \sum_{i+j \leq 6} a_{i,j,c} T_i(x) T_j(y) + \sum_{k_x, k_y} \left[ b_{k_x,k_y,c} \cos(k_x \pi x + k_y \pi y) + d_{k_x,k_y,c} \sin(k_x \pi x + k_y \pi y) \right],$"
    fig.text(0.5, 0.66, eq2, fontsize=7.5, ha="center", va="center")

    # 5. Partition of Unity Window
    eq3 = r"$w_k(x, y) = \psi_4\left(\frac{|x - c_{x,k}|}{R_{x,k}}\right) \cdot \psi_4\left(\frac{|y - c_{y,k}|}{R_{y,k}}\right), \quad \mathrm{where} \quad \psi_4(r) = (1 - r)_+^4 (4r + 1), \quad (R_{x,k}, R_{y,k}) = 1.20 \cdot (h_{x,k}, h_{y,k}),$"
    fig.text(0.5, 0.57, eq3, fontsize=7.2, ha="center", va="center")

    # 6. Local DCT expansions
    eq4 = r"$L_{k,c}(x, y) = \sum_{p=0}^P \sum_{q=0}^Q c_{k,p,q,c} \cdot \alpha_p \alpha_q \cos\left(p \pi \frac{x - x_{k,0}}{x_{k,1} - x_{k,0}}\right) \cos\left(q \pi \frac{y - y_{k,0}}{y_{k,1} - y_{k,0}}\right), \quad \alpha_0 = 1, \; \alpha_p = \sqrt{2},$"
    fig.text(0.5, 0.48, eq4, fontsize=7.2, ha="center", va="center")

    # 7. Coordinate normalization & Chebyshev definitions
    eq5 = r"$T_0(u) = 1, \quad T_1(u) = u, \quad T_{d+1}(u) = 2u T_d(u) - T_{d-1}(u), \quad u_k = \frac{x - x_{k,0}}{x_{k,1} - x_{k,0}}, \quad v_k = \frac{y - y_{k,0}}{y_{k,1} - y_{k,0}},$"
    fig.text(0.5, 0.39, eq5, fontsize=7.2, ha="center", va="center")

    # 8. Discovered Global Channels
    eq6 = r"$G_R(x, y) = 22.782 T_1(x)T_2(y) - 16.474 T_2(x)T_2(y) - 16.363 T_1(x) - 15.347 T_3(x) + 10.216 \sin(\pi x - \pi y) + 7.303 \cos(\pi x - \pi y) - 6.999 \sin(\pi y) + \dots$"
    fig.text(0.5, 0.30, eq6, fontsize=6.8, ha="center", va="center")

    eq7 = r"$G_G(x, y) = 22.951 T_1(x)T_2(y) - 16.796 T_1(x) - 16.760 T_2(x)T_2(y) + \dots, \quad G_B(x, y) = 22.935 T_1(x)T_2(y) - 16.221 T_2(x)T_2(y) + \dots$"
    fig.text(0.5, 0.21, eq7, fontsize=6.8, ha="center", va="center")

    # 9. Discovered Local Patches
    eq8 = r"$L_{107}(x, y) = 0.284 \cos(7\pi u)\cos(3\pi v) + 0.239 \cos(8\pi u)\cos(3\pi v) + 0.185 \cos(7\pi u)\cos(2\pi v) - 0.181 \cos(\pi v) + \dots$"
    fig.text(0.5, 0.12, eq8, fontsize=6.8, ha="center", va="center")

    eq9 = r"$L_{58}(x, y) = 0.216 \cos(7\pi u)\cos(\pi v) - 0.205 - 0.200 \cos(7\pi u) + \dots, \quad L_{237}(x, y) = -0.299 \cos(\pi v) + 0.082 \cos(3\pi v) - 0.043 \cos(5\pi v) + \dots$"
    fig.text(0.5, 0.04, eq9, fontsize=6.8, ha="center", va="center")

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
        arr = recon_data
        if arr.max() <= 1.0 and arr.dtype in (np.float32, np.float64):
            arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
        recon_img = Image.fromarray(arr)
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

        # Bottom mathematical formula panel
        panel_canvas = generate_math_panel(result, w_total, h_panel, title=title, dpi=250)

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
