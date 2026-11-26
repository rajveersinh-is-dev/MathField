"""
mathfield/export/pdf_report.py

Publication-grade PDF report exporter using ReportLab.
Embeds the delivered mode-specific model, quantitative metrics, visual diagnostics,
and a local patch catalog only when authoritative serialized patches are present.
"""

from __future__ import annotations

import io
from html import escape
from pathlib import Path
from typing import Any, Dict, List, Union

import numpy as np
from PIL import Image
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.platypus import (
    HRFlowable,
    Image as RLImage,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from mathfield.export.raster import to_uint8_array


def _report_mode(result: Dict[str, Any]) -> str:
    raw_mode = result.get("mode")
    if raw_mode is None or not str(raw_mode).strip():
        return "MODE_B_FIDELITY" if result.get("mode_b_model") is not None else "MODE_A_DISCOVERY"
    return str(raw_mode)


def _is_mode_a(mode: str) -> bool:
    normalized = mode.strip().lower().replace("-", "_")
    return normalized in ("a", "discovery", "mode_a", "mode_a_discovery") or normalized.startswith("mode_a_")


def _partition_is_separable(result: Dict[str, Any]) -> bool:
    """Read persisted partition geometry, preferring the authoritative Mode B payload."""
    for container in (result.get("mode_b_model"), result.get("local_decomposition"), result):
        if not isinstance(container, dict):
            continue
        partition = container.get("partition_of_unity")
        if isinstance(partition, dict) and "separable" in partition:
            return bool(partition["separable"])
    return True


def _local_patches(result: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Return only authoritative serialized/local patch records, never display metadata."""
    mode_b_model = result.get("mode_b_model")
    if isinstance(mode_b_model, dict) and "local_patches" in mode_b_model:
        raw_patches = mode_b_model.get("local_patches")
    else:
        raw_patches = result.get("local_patches", [])
    if not isinstance(raw_patches, (list, tuple)):
        return []
    return [patch for patch in raw_patches if isinstance(patch, dict)]


def _patch_id(patch: Dict[str, Any]) -> str:
    if "id" in patch:
        return str(patch["id"])
    if "patch_id" in patch:
        raw_id = patch["patch_id"]
        try:
            return f"P_{int(raw_id):03d}"
        except (TypeError, ValueError):
            return str(raw_id)
    return "N/A"


def _numeric_pair(value: Any) -> tuple | None:
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        return None
    try:
        return float(value[0]), float(value[1])
    except (TypeError, ValueError):
        return None


def _patch_bounds(patch: Dict[str, Any]) -> tuple | None:
    bounds = patch.get("bounds")
    if not isinstance(bounds, (list, tuple)) or len(bounds) < 4:
        return None
    try:
        return tuple(float(value) for value in bounds[:4])
    except (TypeError, ValueError):
        return None


def _patch_basis_label(patch: Dict[str, Any]) -> str:
    basis = patch.get("basis")
    if isinstance(basis, dict) and basis.get("type"):
        basis_type = str(basis["type"])
        labels = {"dct2": "DCT2Basis", "chebyshev2d": "Chebyshev2D", "fourier2d": "Fourier2D"}
        label = labels.get(basis_type.lower(), basis_type)
        details = [f"{key}={basis[key]}" for key in ("deg_x", "deg_y", "freq_x", "freq_y", "truncation") if key in basis]
        return f"{label} ({', '.join(details)})" if details else label
    formula = patch.get("formula")
    return str(formula) if formula is not None and str(formula).strip() else "N/A"


def _format_number(value: Any, format_spec: str) -> str:
    return "N/A" if value is None else format(value, format_spec)


def export_pdf_report(
    result: Dict[str, Any],
    output_path: Union[str, Path],
) -> None:
    """
    Generates a publication-grade PDF mathematical report for the delivered mode.
    Embeds analytical equations, metric tables, visual diagnostics, and only real
    serialized local patch records when present.
    """
    out_p = Path(output_path)
    if not out_p.parent.exists():
        raise FileNotFoundError(f"Destination directory does not exist: {out_p.parent}")

    doc = SimpleDocTemplate(
        str(out_p),
        pagesize=letter,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36,
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "ReportTitle",
        parent=styles["Title"],
        fontName="Helvetica-Bold",
        fontSize=18,
        leading=22,
        textColor=colors.HexColor("#0f172a"),
        spaceAfter=4,
    )
    subtitle_style = ParagraphStyle(
        "ReportSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=10,
        leading=14,
        textColor=colors.HexColor("#475569"),
        spaceAfter=10,
    )
    heading_style = ParagraphStyle(
        "ReportHeading",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=12,
        leading=15,
        textColor=colors.HexColor("#1e293b"),
        spaceBefore=12,
        spaceAfter=6,
    )
    body_style = ParagraphStyle(
        "ReportBody",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8.5,
        leading=12,
        textColor=colors.HexColor("#334155"),
    )
    caption_style = ParagraphStyle(
        "ReportCaption",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=8,
        leading=10,
        alignment=1,  # Center
        textColor=colors.HexColor("#1e293b"),
    )
    eq_style = ParagraphStyle(
        "ReportEq",
        parent=styles["Code"],
        fontName="Courier",
        fontSize=8,
        leading=11,
        textColor=colors.HexColor("#0284c7"),
        backColor=colors.HexColor("#f8fafc"),
    )
    table_header_style = ParagraphStyle(
        "TableHeader",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=8,
        leading=10,
        textColor=colors.HexColor("#0f172a"),
    )
    table_cell_style = ParagraphStyle(
        "TableCell",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=7.5,
        leading=9.5,
        textColor=colors.HexColor("#334155"),
    )
    table_code_style = ParagraphStyle(
        "TableCode",
        parent=styles["Code"],
        fontName="Courier",
        fontSize=7,
        leading=8.5,
        textColor=colors.HexColor("#0369a1"),
    )

    story: List[Any] = []

    # Title & Metadata Header
    story.append(Paragraph("MathField Mathematical Reconstruction Report", title_style))
    story.append(
        Paragraph(
            "<b>Rigorous Continuous-Domain Image Representation & Analytical Equation Discovery Engine</b>",
            subtitle_style,
        )
    )
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#cbd5e1"), spaceAfter=8))

    # Executive Metadata Summary Table
    domain = result.get("coordinate_domain", {})
    x_min = float(domain.get("x_min", -1.0))
    x_max = float(domain.get("x_max", 1.0))
    y_min = float(domain.get("y_min", -1.0))
    y_max = float(domain.get("y_max", 1.0))
    pw = domain.get("pixel_width", "N/A")
    ph = domain.get("pixel_height", "N/A")
    color_space = result.get("color_space", "RGB")
    mode_str = _report_mode(result)
    is_mode_a = _is_mode_a(mode_str)
    local_patches = [] if is_mode_a else _local_patches(result)
    partition_is_separable = _partition_is_separable(result) if not is_mode_a else True
    channel_names = ("R", "G", "B", "A") if str(color_space).upper() == "RGBA" else ("R", "G", "B")
    channel_set = ", ".join(channel_names)
    metrics = result.get("metrics", {})
    condition_report = result.get("condition_report") or {}
    raw_condition = (
        condition_report.get("condition_number")
        or (result.get("global_model") or {}).get("condition_number")
        or metrics.get("condition_number")
    )
    try:
        condition_value = float(raw_condition) if raw_condition is not None else None
    except (TypeError, ValueError):
        condition_value = None

    psnr_display = _format_number(metrics.get("psnr"), ".2f")
    ssim_value = metrics.get("ssim")
    ssim_display = _format_number(ssim_value, ".4f")
    ssim_percent = "N/A" if ssim_value is None else f"{ssim_value * 100:.2f}%"

    meta_data = [
        [
            Paragraph("<b>Target Resolution:</b>", table_header_style),
            Paragraph(f"{pw} × {ph} pixels", table_cell_style),
            Paragraph("<b>Continuous Domain Ω:</b>", table_header_style),
            Paragraph(f"[{x_min:.3f}, {x_max:.3f}] × [{y_min:.3f}, {y_max:.3f}]", table_cell_style),
        ],
        [
            Paragraph("<b>Reconstruction Mode:</b>", table_header_style),
            Paragraph(str(mode_str), table_cell_style),
            Paragraph("<b>Color Space:</b>", table_header_style),
            Paragraph(str(color_space), table_cell_style),
        ],
        [
            Paragraph("<b>Peak SNR (PSNR):</b>", table_header_style),
            Paragraph(f"<b>{psnr_display} dB</b>" if psnr_display != "N/A" else "<b>N/A</b>", table_cell_style),
            Paragraph("<b>Structural Similarity (SSIM):</b>", table_header_style),
            Paragraph(f"<b>{ssim_percent}</b> ({ssim_display})", table_cell_style),
        ],
    ]
    meta_table = Table(meta_data, colWidths=[120, 150, 120, 150])
    meta_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f8fafc")),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    story.append(meta_table)
    story.append(Spacer(1, 10))

    # --- Section 1: Master Continuous Vector Field Formulation ---
    story.append(Paragraph("1. Master Continuous Vector Field Formulation", heading_style))
    if is_mode_a:
        master_intro = (
            "Mode A discovery delivers an independent symbolic expression for each image channel. "
            "This report uses only those delivered per-channel equations:"
        )
        master_formula = f"F_c(x, y) = f_c(x, y),   c ∈ {{{channel_set}}}"
        master_detail = (
            "Each <b>f<sub>c</sub>(x, y)</b> is the delivered symbolic model for channel c; "
            "its numerical parameters and measured reconstruction metrics are reported below."
        )
    elif local_patches:
        master_intro = (
            "The image is represented as a deterministic continuous 2D vector field "
            "<b>F</b>(x, y) defined over the continuous domain "
            f"Ω = [{x_min:.4f}, {x_max:.4f}] × [{y_min:.4f}, {y_max:.4f}]. "
            "The spatial function is reconstructed using a hierarchical continuous hybrid expansion:"
        )
        master_formula = (
            "F_c(x, y) = G_c(x, y) + [ ∑_{k=1}^K w_k(x, y) · L_{k,c}(x, y) ] / [ ∑_{k=1}^K w_k(x, y) ], "
            f"  c ∈ {{{channel_set}}}"
        )
        master_detail = (
            "where <b>G<sub>c</sub>(x, y)</b> is the global continuous background field, "
            "<b>L<sub>k,c</sub>(x, y)</b> is a serialized localized patch expansion, "
            "and <b>w<sub>k</sub>(x, y)</b> is the smooth C<sup>2</sup> Wendland partition of unity window."
        )
    else:
        master_intro = (
            "The image is represented as a deterministic continuous 2D vector field "
            "<b>F</b>(x, y) defined over the continuous domain "
            f"Ω = [{x_min:.4f}, {x_max:.4f}] × [{y_min:.4f}, {y_max:.4f}]. "
            "The serialized Mode B model contains a global field and no local patches:"
        )
        master_formula = f"F_c(x, y) = G_c(x, y),   c ∈ {{{channel_set}}}"
        master_detail = "The global field <b>G<sub>c</sub>(x, y)</b> supplies the complete delivered reconstruction."
    story.append(Paragraph(master_intro, body_style))
    story.append(Spacer(1, 4))
    story.append(Paragraph(master_formula, eq_style))
    story.append(Spacer(1, 4))
    story.append(Paragraph(master_detail, body_style))
    story.append(Spacer(1, 8))

    # --- Section 2: Delivered channel equations ---
    global_heading = "Discovered Per-Channel Symbolic Equations" if is_mode_a else "Discovered Global Colour-Field Equations"
    story.append(Paragraph(f"2. {global_heading}", heading_style))
    if is_mode_a:
        global_intro = "Each equation below is a symbolic channel model delivered by Mode A discovery:"
    else:
        global_intro = (
            "The global continuous colour field G<sub>c</sub>(x, y) captures low-frequency gradients, background illumination, "
            "and broad structural harmonics across the entire canvas using orthogonal bivariate Chebyshev polynomials "
            "T<sub>i</sub>(x)T<sub>j</sub>(y) and harmonic 2D Fourier components cos/sin(k<sub>x</sub>πx + k<sub>y</sub>πy):"
        )
    story.append(Paragraph(global_intro, body_style))
    story.append(Spacer(1, 4))

    global_eqs = result.get("global_equations", {})
    expressions = result.get("expressions", {})
    if is_mode_a:
        global_channels = {ch: eq for ch, eq in global_eqs.items() if str(ch).lower() != "formula"}
        expression_channels = {ch: eq for ch, eq in expressions.items() if str(ch).lower() != "formula"}
        equations = global_channels or expression_channels or global_eqs or expressions
    else:
        equations = global_eqs or expressions
    global_model = result.get("global_model") or {}
    if not equations and isinstance(global_model, dict):
        channels = global_model.get("channels") or []
        if isinstance(channels, list):
            equations = {
                str(record.get("channel", "c")): record.get("formula", "")
                for record in channels
                if isinstance(record, dict) and record.get("formula")
            }

    displayed_equations = 0
    for ch, eq_str in equations.items():
        if not str(eq_str).strip():
            continue
        if not is_mode_a and str(ch).lower() not in ("r", "g", "b", "alpha", "formula"):
            continue
        clean_ch = escape(str(ch))
        clean_eq = escape(str(eq_str))
        lhs = f"<b>{clean_ch}(x, y)</b>" if is_mode_a else f"<b>G<sub>{clean_ch}</sub>(x, y)</b>"
        story.append(Paragraph(f"{lhs} = {clean_eq}", eq_style))
        story.append(Spacer(1, 3))
        displayed_equations += 1
    if not displayed_equations:
        if is_mode_a:
            story.append(Paragraph("No per-channel symbolic equations were supplied in the export result.", body_style))
        else:
            story.append(
                Paragraph(
                    "G_c(x, y) = ∑_{i+j ≤ d} a_{i,j,c} T_i(x) T_j(y) + ∑_{k} [ b_{k,c} cos(k_x π x + k_y π y) + d_{k,c} sin(k_x π x + k_y π y) ]",
                    eq_style,
                )
            )
    story.append(Spacer(1, 8))

    if not is_mode_a:
        # --- Section 3: Smooth C2 Wendland Partition of Unity Blending ---
        story.append(Paragraph("3. Smooth C² Partition of Unity Blending Windows", heading_style))
        if local_patches:
            window_geometry = "separable" if partition_is_separable else "radial"
            story.append(
                Paragraph(
                    "To blend serialized local patches without step discontinuities or boundary voids, "
                    f"each localized patch k is modulated by a {window_geometry} bivariate C² Wendland weighting window:",
                    body_style,
                )
            )
            story.append(Spacer(1, 4))
            if partition_is_separable:
                wendland_formula = (
                    "w_k(x, y) = ψ₄( |x - c_{x,k}| / R_{x,k} ) · ψ₄( |y - c_{y,k}| / R_{y,k} ), "
                    "  where ψ₄(r) = (1 - r)₊⁴ (4r + 1)"
                )
                window_detail = (
                    "Here, (c_{x,k}, c_{y,k}) is the patch center and (R_{x,k}, R_{y,k}) is its expanded support radius. "
                )
            else:
                wendland_formula = (
                    "w_k(x, y) = ψ₄( sqrt(dx^2 + dy^2) ), "
                    "  where ψ₄(r) = (1 - r)₊⁴ (4r + 1)"
                )
                window_detail = (
                    "Here, dx = (x - c_{x,k}) / R_{x,k} and dy = (y - c_{y,k}) / R_{y,k} are normalized displacements. "
                    "The patch center is (c_{x,k}, c_{y,k}) and its expanded support radius is (R_{x,k}, R_{y,k}). "
                )
            story.append(Paragraph(wendland_formula, eq_style))
            story.append(Spacer(1, 4))
            story.append(
                Paragraph(
                    f"{window_detail}"
                    "The Wendland weights and their normal and tangential derivatives vanish at patch boundaries, "
                    "and Shepard normalization combines the weighted local fields.",
                    body_style,
                )
            )
        else:
            story.append(
                Paragraph(
                    "No local patches are present in the serialized Mode B model, so no partition-of-unity blending windows are active.",
                    body_style,
                )
            )
        story.append(Spacer(1, 8))

        # --- Section 4: Catalog of Serialized Local Patch Equations ---
        story.append(Paragraph("4. Catalog of Serialized Local Patch Equations", heading_style))
        if local_patches:
            all_dct = all(
                isinstance(patch.get("basis"), dict) and str(patch["basis"].get("type", "")).lower() == "dct2"
                for patch in local_patches
            )
            if all_dct:
                story.append(
                    Paragraph(
                        "Localized details are represented by the serialized 2D Type-II Discrete Cosine bases (DCT2Basis) cataloged below:",
                        body_style,
                    )
                )
                local_formula = (
                    "L_{k,c}(x, y) = ∑_{p=0}^{P_k} ∑_{q=0}^{Q_k} c_{k,p,q,c} · α_p α_q · "
                    "cos( p π [x - x_{k,min}] / [x_{k,max} - x_{k,min}] ) · "
                    "cos( q π [y - y_{k,min}] / [y_{k,max} - y_{k,min}] )"
                )
            else:
                story.append(
                    Paragraph(
                        "Each localized field is evaluated from the basis configuration and coefficients serialized with its patch:",
                        body_style,
                    )
                )
                local_formula = "L_{k,c}(x, y) = Φ_{k,c}(x, y)"
            story.append(Paragraph(local_formula, eq_style))
            story.append(Spacer(1, 6))

            display_count = min(len(local_patches), 35)
            story.append(
                Paragraph(
                    f"Below is a structured catalog generated only from Mode B local-patch data "
                    f"(displaying the first {display_count} of {len(local_patches)} serialized patches). "
                    "No placeholder patch rows are generated:",
                    body_style,
                )
            )
            story.append(Spacer(1, 4))

            patch_table_data = [
                [
                    Paragraph("<b>Patch ID</b>", table_header_style),
                    Paragraph("<b>Spatial Region [x0, x1] × [y0, y1]</b>", table_header_style),
                    Paragraph("<b>Center & Radii</b>", table_header_style),
                    Paragraph("<b>Equation / Serialized Basis</b>", table_header_style),
                ]
            ]

            for patch in local_patches[:display_count]:
                bounds = _patch_bounds(patch)
                center = _numeric_pair(patch.get("center"))
                radius = _numeric_pair(patch.get("radius"))
                if bounds is None:
                    bounds_str = "N/A"
                else:
                    x0, y0, x1, y1 = bounds
                    bounds_str = f"[{x0:.3f}, {x1:.3f}] × [{y0:.3f}, {y1:.3f}]"
                    if center is None:
                        center = ((x0 + x1) / 2.0, (y0 + y1) / 2.0)
                    if radius is None:
                        radius = ((x1 - x0) / 2.0, (y1 - y0) / 2.0)
                center_str = "N/A" if center is None else f"({center[0]:.2f}, {center[1]:.2f})"
                radius_str = "N/A" if radius is None else f"±({radius[0]:.2f}, {radius[1]:.2f})"
                cr_str = f"{center_str}<br/>{radius_str}"
                patch_table_data.append(
                    [
                        Paragraph(f"<b>{escape(_patch_id(patch))}</b>", table_cell_style),
                        Paragraph(bounds_str, table_cell_style),
                        Paragraph(cr_str, table_cell_style),
                        Paragraph(escape(_patch_basis_label(patch)), table_code_style),
                    ]
                )

            patch_table = Table(patch_table_data, colWidths=[50, 140, 80, 270], repeatRows=1)
            patch_table.setStyle(
                TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f1f5f9")),
                        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
                        ("TOPPADDING", (0, 0), (-1, -1), 3),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                        ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ]
                )
            )
            story.append(patch_table)
        else:
            story.append(
                Paragraph(
                    "No local patches are present in the serialized Mode B model; the local-patch catalog is empty.",
                    body_style,
                )
            )
        story.append(Spacer(1, 10))

    # --- Quantitative Verification Metrics Table ---
    metrics_section_no = 3 if is_mode_a else 5
    story.append(Paragraph(f"{metrics_section_no}. Quantitative Reconstruction Verification", heading_style))
    metric_rows = [
        [
            Paragraph("<b>Verification Metric</b>", table_header_style),
            Paragraph("<b>Measured Value</b>", table_header_style),
            Paragraph("<b>Physical Significance & Interpretation</b>", table_header_style),
        ]
    ]

    metric_defs = [
        (
            "PSNR",
            f"{_format_number(metrics.get('psnr'), '.2f')} dB" if metrics.get("psnr") is not None else "N/A",
            "Peak Signal-to-Noise Ratio (fidelity against discrete ground truth)",
        ),
        (
            "SSIM",
            f"{ssim_display} ({ssim_percent})",
            "Structural Similarity Index (geometric edge and gradient alignment)",
        ),
        ("MSE", _format_number(metrics.get("mse"), ".8f"), "Normalized Mean Squared Error (variance across all channels)"),
        ("MAE", _format_number(metrics.get("mae"), ".8f"), "Mean Absolute Error (average pixel deviation across canvas)"),
        ("L_INF", _format_number(metrics.get("linf"), ".6f"), "Maximum Absolute Error (worst-case deviation)"),
        (
            "PARAMETER_COUNT",
            f"{len(result.get('parameters') or {}):,}",
            "Total delivered continuous mathematical coefficients",
        ),
    ]
    if not is_mode_a:
        metric_defs.append(
            (
                "LOCAL_PATCHES",
                f"{len(local_patches):,}",
                "Serialized local patches with localized basis expansions",
            )
        )
    metric_defs.append(
        (
            "CONDITIONING",
            "MONITORED" if condition_report.get("is_ill_conditioned") else ("STABLE" if condition_report else "N/A"),
            (
                f"SVD condition number κ = {condition_value:.3g}"
                if condition_value is not None
                else "Condition number unavailable"
            ),
        )
    )

    for m_name, m_val, m_desc in metric_defs:
        metric_rows.append(
            [
                Paragraph(f"<b>{m_name}</b>", table_cell_style),
                Paragraph(f"<b>{m_val}</b>", table_cell_style),
                Paragraph(m_desc, table_cell_style),
            ]
        )

    metric_table = Table(metric_rows, colWidths=[120, 110, 310])
    metric_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f1f5f9")),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
                ("TOPPADDING", (0, 0), (-1, -1), 3.5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3.5),
            ]
        )
    )
    story.append(metric_table)
    story.append(Spacer(1, 10))

    # --- Section 6: Visual Verification Diagnostics ---
    orig_img = result.get("original_image")
    recon_img = result.get("reconstructed_image")
    if recon_img is None:
        recon_img = result.get("rendered_image")
    err_img = result.get("error_heatmap")

    if err_img is None and isinstance(orig_img, np.ndarray) and isinstance(recon_img, np.ndarray):
        diff = np.abs(recon_img.astype(float) - orig_img.astype(float))
        diff_mag = np.mean(diff, axis=-1) if diff.ndim == 3 else diff
        err_img = np.clip(diff_mag * 5.0 * 255.0, 0, 255).astype(np.uint8)

    img_triplet = [
        ("Original Target", orig_img),
        ("Continuous Reconstruction", recon_img),
        ("Residual Error (5× Amplified)", err_img),
    ]
    valid_triplet = [(name, img) for name, img in img_triplet if img is not None]

    if valid_triplet:
        story.append(Paragraph(f"{metrics_section_no + 1}. Visual Verification Diagnostics", heading_style))
        fig_cells = []
        label_cells = []
        for name, img_data in valid_triplet:
            buf = io.BytesIO()
            if isinstance(img_data, np.ndarray):
                arr = to_uint8_array(img_data)
                if arr.ndim == 2:
                    pil_img = Image.fromarray(arr, mode="L")
                else:
                    pil_img = Image.fromarray(arr)
            elif isinstance(img_data, Image.Image):
                pil_img = img_data
            else:
                continue

            pil_img.save(buf, format="PNG")
            buf.seek(0)
            rl_img = RLImage(buf, width=160, height=160)
            label_cells.append(Paragraph(f"<b>{name}</b>", caption_style))
            fig_cells.append(rl_img)

        if fig_cells:
            img_table = Table([label_cells, fig_cells], colWidths=[175] * len(fig_cells))
            img_table.setStyle(
                TableStyle(
                    [
                        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                        ("BOTTOMPADDING", (0, 0), (-1, 0), 4),
                        ("TOPPADDING", (0, 1), (-1, 1), 2),
                    ]
                )
            )
            story.append(img_table)
            story.append(Spacer(1, 10))

    # --- Model Parameter Vectors ---
    parameters = result.get("parameters", {})
    if parameters or is_mode_a:
        story.append(Paragraph(f"{metrics_section_no + 2}. Model Parameter Vectors", heading_style))
        if parameters:
            if is_mode_a:
                param_intro = (
                    f"The delivered per-channel symbolic model incorporates {len(parameters):,} numerical coefficients. "
                    "The table reports the parameter records supplied with the export:"
                )
            elif local_patches:
                param_intro = (
                    f"The continuous model incorporates {len(parameters):,} discovered mathematical coefficients. "
                    "Below is a structured table of the primary global field parameters and serialized local patch coefficients:"
                )
            else:
                param_intro = (
                    f"The global Mode B model incorporates {len(parameters):,} discovered mathematical coefficients. "
                    "No local patch parameters are present:"
                )
            story.append(Paragraph(param_intro, body_style))
            story.append(Spacer(1, 4))

            param_rows = [
                [
                    Paragraph("<b>Coefficient ID</b>", table_header_style),
                    Paragraph("<b>Expansion Stage</b>", table_header_style),
                    Paragraph("<b>Channel</b>", table_header_style),
                    Paragraph("<b>Discovered Value</b>", table_header_style),
                ]
            ]
            default_stage = "symbolic" if is_mode_a else "global"

            if isinstance(parameters, dict):
                for k in list(parameters.keys())[:20]:
                    val = parameters[k]
                    if isinstance(val, dict):
                        raw_value = val.get("value")
                        v_num = float(raw_value) if raw_value is not None else None
                        v_stage = str(val.get("stage", default_stage))
                        v_ch = str(val.get("channel", color_space))
                    else:
                        v_num = float(val)
                        v_stage = default_stage
                        v_ch = str(color_space)
                    value_text = "N/A" if v_num is None else f"{v_num:+.8f}"
                    param_rows.append(
                        [
                            Paragraph(escape(str(k)), table_code_style),
                            Paragraph(escape(v_stage), table_cell_style),
                            Paragraph(escape(v_ch), table_cell_style),
                            Paragraph(value_text, table_code_style),
                        ]
                    )
            elif isinstance(parameters, (list, tuple)):
                for index, val in enumerate(parameters[:20]):
                    param_rows.append(
                        [
                            Paragraph(f"c_{{{index}}}", table_code_style),
                            Paragraph(default_stage, table_cell_style),
                            Paragraph(str(color_space), table_cell_style),
                            Paragraph(f"{float(val):+.8f}", table_code_style),
                        ]
                    )

            p_table = Table(param_rows, colWidths=[140, 120, 80, 200])
            p_table.setStyle(
                TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f1f5f9")),
                        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
                        ("TOPPADDING", (0, 0), (-1, -1), 2.5),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
                    ]
                )
            )
            story.append(p_table)
        else:
            story.append(Paragraph("No model parameters were supplied in the export result.", body_style))
        story.append(Spacer(1, 10))

    # --- Standalone Reproducibility Instructions ---
    story.append(Paragraph(f"{metrics_section_no + 3}. Standalone Mathematical Reproducibility", heading_style))
    story.append(
        Paragraph(
            "All exported mathematical representations are 100% self-contained and reproducible without external APIs, "
            "neural networks, or proprietary runtime libraries. The accompanying standalone Python script "
            "<code>reproduce.py</code> evaluates these exact continuous basis functions on arbitrary coordinate grids.",
            body_style,
        )
    )
    story.append(Spacer(1, 4))
    domain = result.get("coordinate_domain", {})
    out_w = int(domain.get("pixel_width") or 64)
    out_h = int(domain.get("pixel_height") or 64)
    repro_code = (
        "# Zero-dependency mathematical evaluation command:\n"
        f"python reproduce.py --output reconstruction.png --resolution {out_w} {out_h}\n"
        "# Continuous 2x super-resolution direct evaluation:\n"
        f"python reproduce.py --output superres.png --resolution {out_w * 2} {out_h * 2}"
    )
    story.append(Paragraph(repro_code, eq_style))

    doc.build(story)
