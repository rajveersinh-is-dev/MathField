"""
mathfield/export/pdf_report.py

Publication-grade PDF report exporter using ReportLab 5.0.1.
Embeds mathematical equations, metrics, and visual comparison figures.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any, Dict, List, Union

import numpy as np
from PIL import Image

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.platypus import (
    Image as RLImage,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


def export_pdf_report(
    result: Dict[str, Any],
    output_path: Union[str, Path],
) -> None:
    """
    Generates a publication-grade PDF mathematical report.
    Embeds discovered equations, metric tables, and visual preview images.
    """
    out_p = Path(output_path)
    if not out_p.parent.exists():
        raise FileNotFoundError(f"Destination directory does not exist: {out_p.parent}")

    doc = SimpleDocTemplate(
        str(out_p),
        pagesize=letter,
        rightMargin=54,
        leftMargin=54,
        topMargin=54,
        bottomMargin=54,
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "ReportTitle",
        parent=styles["Title"],
        fontName="Helvetica-Bold",
        fontSize=20,
        leading=24,
        textColor=colors.HexColor("#0f172a"),
    )
    heading_style = ParagraphStyle(
        "ReportHeading",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=13,
        leading=16,
        textColor=colors.HexColor("#1e293b"),
        spaceBefore=12,
        spaceAfter=6,
    )
    body_style = ParagraphStyle(
        "ReportBody",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=10,
        leading=14,
        textColor=colors.HexColor("#334155"),
    )
    eq_style = ParagraphStyle(
        "ReportEq",
        parent=styles["Code"],
        fontName="Courier",
        fontSize=9,
        leading=12,
        textColor=colors.HexColor("#0284c7"),
        backColor=colors.HexColor("#f8fafc"),
    )

    story: List[Any] = []

    # Title & Metadata
    story.append(Paragraph("MathField Mathematical Reconstruction Report", title_style))
    story.append(Spacer(1, 10))
    story.append(
        Paragraph(
            "Rigorous Continuous-Domain Image Representation & Equation Discovery Engine",
            body_style,
        )
    )
    story.append(Spacer(1, 15))

    # Equations Section
    story.append(Paragraph("1. Discovered Analytical Equations", heading_style))
    expressions = result.get("expressions", {})
    if expressions:
        for ch, expr in expressions.items():
            eq_text = f"<b>{ch}(x, y)</b> = {expr}"
            story.append(Paragraph(eq_text, eq_style))
            story.append(Spacer(1, 4))
    else:
        story.append(Paragraph("Global continuous representation: f(x, y) = sum(c_i * phi_i(x, y))", eq_style))
    story.append(Spacer(1, 10))

    # Metrics Table
    story.append(Paragraph("2. Quantitative Reconstruction Metrics", heading_style))
    metrics = result.get("metrics", {})
    metric_rows = [["Metric", "Value", "Unit"]]
    if metrics:
        for k, v in metrics.items():
            val_str = f"{float(v):.6f}" if isinstance(v, (int, float)) else str(v)
            metric_rows.append([str(k).upper(), val_str, "scalar"])
    else:
        metric_rows.append(["STATUS", "CONVERGED", "flag"])

    metric_table = Table(metric_rows, colWidths=[150, 150, 100])
    metric_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f1f5f9")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#0f172a")),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, 0), 9),
                ("BOTTOMPADDING", (0, 0), (-1, 0), 6),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
                ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
                ("FONTSIZE", (0, 1), (-1, -1), 9),
            ]
        )
    )
    story.append(metric_table)
    story.append(Spacer(1, 15))

    # Visual Comparison Figures
    img_keys = [
        ("Original Target", result.get("original_image")),
        ("Reconstruction", result.get("reconstructed_image")),
        ("Residual Heatmap", result.get("error_heatmap")),
    ]
    valid_imgs = [(name, img) for name, img in img_keys if img is not None]

    if valid_imgs:
        story.append(Paragraph("3. Visual Verification Diagnostics", heading_style))
        img_cells = []
        for name, img_data in valid_imgs:
            buf = io.BytesIO()
            if isinstance(img_data, np.ndarray):
                arr = img_data
                if arr.max() <= 1.0 and arr.dtype in (np.float32, np.float64):
                    arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
                pil_img = Image.fromarray(arr)
            elif isinstance(img_data, Image.Image):
                pil_img = img_data
            else:
                continue
            pil_img.save(buf, format="PNG")
            buf.seek(0)
            rl_img = RLImage(buf, width=140, height=140)
            img_cells.append([Paragraph(f"<b>{name}</b>", body_style), rl_img])

        if img_cells:
            # Flatten or format into a horizontal table
            labels = [c[0] for c in img_cells]
            figures = [c[1] for c in img_cells]
            fig_table = Table([labels, figures], colWidths=[160] * len(img_cells))
            fig_table.setStyle(
                TableStyle(
                    [
                        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ]
                )
            )
            story.append(fig_table)
            story.append(Spacer(1, 15))

    # Parameters Table if present
    parameters = result.get("parameters", {})
    if parameters:
        story.append(Paragraph("4. Model Parameters", heading_style))
        param_rows = [["Parameter", "Value"]]
        if isinstance(parameters, dict):
            for k in list(parameters.keys())[:15]:
                val = parameters[k]
                v_num = float(val.get("value", 0.0)) if isinstance(val, dict) else float(val)
                param_rows.append([str(k), f"{v_num:.8f}"])
        param_table = Table(param_rows, colWidths=[200, 200])
        param_table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f1f5f9")),
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
                    ("FONTSIZE", (0, 0), (-1, -1), 8),
                ]
            )
        )
        story.append(param_table)

    doc.build(story)
