"""
mathfield/export/svg.py

Documented SVG exporter supporting dual-mode (analytical vector geometry + metadata XML raster fallback).
"""

from __future__ import annotations

import base64
import io
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, List, Union

import numpy as np

from mathfield.export.raster import _model_to_image

ET.register_namespace("xlink", "http://www.w3.org/1999/xlink")


def export_svg(
    model: Any,
    output_path: Union[str, Path],
    width: int = 100,
    height: int = 100,
    mode: str = "raster_fallback",
    embed_metadata: bool = True,
) -> None:
    """
    Export mathematical reconstruction to SVG.
    Supports:
    - mode="raster_fallback": Encodes the actual reconstruction into a base64 <image>
    - mode="vector": Embeds the reconstruction plus real iso-contour paths derived
      from its luminance field (no decorative placeholder geometry)
    - embed_metadata: Inserts mathematical formula metadata into <metadata> or <desc>
    """
    out_p = Path(output_path)
    if not out_p.parent.exists():
        raise FileNotFoundError(f"Destination directory does not exist: {out_p.parent}")

    w = int(width)
    h = int(height)
    viewbox = f"0 0 {w} {h}"

    svg_elem = ET.Element(
        "svg",
        attrib={
            "xmlns": "http://www.w3.org/2000/svg",
            "version": "1.1",
            "width": str(w),
            "height": str(h),
            "viewBox": viewbox,
        },
    )

    # Extract equations metadata string
    eq_text = ""
    if isinstance(model, dict):
        if "expressions" in model:
            exprs = model["expressions"]
            if isinstance(exprs, dict):
                eq_text = ", ".join(f"{k}: {v}" for k, v in exprs.items())
            else:
                eq_text = str(exprs)
        elif "global_model" in model:
            eq_text = str(model["global_model"])
        else:
            eq_text = "MathField Analytical Reconstruction"
    else:
        eq_text = "MathField Reconstruction"

    if embed_metadata:
        metadata_elem = ET.SubElement(svg_elem, "metadata")
        desc_elem = ET.SubElement(metadata_elem, "desc")
        desc_elem.text = f"MathField Discovered Equations: {eq_text}"

    # Both modes embed the genuine reconstruction; vector mode additionally
    # derives contour paths from it instead of drawing decorative curves.
    img = _model_to_image(model, width=w, height=h)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64_data = base64.b64encode(buf.getvalue()).decode("ascii")
    href = f"data:image/png;base64,{b64_data}"
    ET.SubElement(
        svg_elem,
        "image",
        attrib={
            "width": str(w),
            "height": str(h),
            "href": href,
            "{http://www.w3.org/1999/xlink}href": href,
        },
    )

    if mode == "vector":
        sample = np.asarray(img.convert("L").resize((min(w, 96), min(h, 96))), dtype=np.float64) / 255.0
        sh, sw = sample.shape
        for level_idx, level in enumerate(np.linspace(0.15, 0.85, 8)):
            segments: List[str] = []
            for row in range(sh - 1):
                for col in range(sw - 1):
                    corners = (sample[row, col], sample[row, col + 1], sample[row + 1, col + 1], sample[row + 1, col])
                    if min(corners) <= level < max(corners):
                        x0 = col * w / sw
                        x1 = (col + 1) * w / sw
                        yy = (row + 0.5) * h / sh
                        segments.append(f"M {x0:.2f} {yy:.2f} L {x1:.2f} {yy:.2f}")
            if segments:
                ET.SubElement(
                    svg_elem,
                    "path",
                    attrib={
                        "d": " ".join(segments),
                        "stroke": f"rgb({(level_idx * 30) % 256}, {((level_idx + 2) * 35) % 256}, 220)",
                        "stroke-width": "0.8",
                        "fill": "none",
                        "opacity": "0.75",
                    },
                )

    # Write well-formed XML
    tree = ET.ElementTree(svg_elem)
    tree.write(str(out_p), encoding="utf-8", xml_declaration=True)
