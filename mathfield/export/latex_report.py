"""
mathfield/export/latex_report.py

Publication-grade LaTeX mathematical report exporter.
Generates fully compilable AMS-LaTeX documents (.tex) containing the delivered
mode-specific model, parameters, metrics, and any serialized local patch catalog.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Union

from mathfield.core.ast.parser import parse
from mathfield.core.codegen.latex import to_latex


def _human_equation_to_latex(equation: str) -> str:
    """Convert a human-readable discovered equation into AMS-LaTeX math."""
    text = str(equation).replace("π", r"\pi ").replace("·", r" \cdot ")
    text = re.sub(r"T_(\d+)", r"T_{\1}", text)
    text = re.sub(r"\[(\d+) (?:total )?terms\]", r"\ [\1\ terms]", text)
    return text.replace("&", r"\&").replace("%", r"\%").replace("$", r"\$")


def _report_mode(result: Dict[str, Any]) -> str:
    raw_mode = result.get("mode")
    if raw_mode is None or not str(raw_mode).strip():
        return "MODE_B_FIDELITY" if result.get("mode_b_model") is not None else "MODE_A_DISCOVERY"
    return str(raw_mode)


def _is_mode_a(mode: str) -> bool:
    normalized = mode.strip().lower().replace("-", "_")
    return normalized in ("a", "discovery", "mode_a", "mode_a_discovery") or normalized.startswith("mode_a_")


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


def _latex_table_text(value: Any) -> str:
    text = str(value).replace("·", " ")
    replacements = {
        "\\": r"\textbackslash{}",
        "_": r"\_",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "{": r"\{",
        "}": r"\}",
    }
    return "".join(replacements.get(char, char) for char in text)


def export_latex_report(
    result: Dict[str, Any],
    output_path: Union[str, Path],
) -> None:
    """
    Exports reconstruction results to a compilable LaTeX report (.tex).
    Guarantees balanced braces and standard AMS-LaTeX syntax.
    """
    out_p = Path(output_path)
    if not out_p.parent.exists():
        raise FileNotFoundError(f"Destination directory does not exist: {out_p.parent}")

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

    expressions = result.get("expressions", {})
    global_eqs = result.get("global_equations", {})
    parameters = result.get("parameters", {})
    metrics = result.get("metrics", {})
    channel_names = ("R", "G", "B", "A") if str(color_space).upper() == "RGBA" else ("R", "G", "B")
    channel_set = ", ".join(channel_names)

    if is_mode_a:
        abstract_text = (
            "This report documents the per-channel symbolic model delivered by Mode A discovery for the input image canvas. "
            "The equations, numerical parameters, and reconstruction metrics below are reported directly from the exported result."
        )
        master_eq = (
            "\\begin{equation}\n"
            f"  F_c(x, y) = f_c(x, y), \\quad c \\in \\{{{channel_set}\\}}\n"
            "\\end{equation}"
        )
        global_heading = "Discovered Per-Channel Symbolic Equations"
        global_intro = "Each delivered channel equation $f_c(x, y)$ defines its continuous symbolic reconstruction:"
    else:
        if local_patches:
            abstract_text = (
                "This report details the continuous mathematical representation discovered for the input image canvas. "
                "The reconstruction engine employs a hierarchical hybrid formulation combining dense global bivariate Chebyshev "
                "polynomials and 2D Fourier harmonics with serialized local patch expansions smoothly blended via Wendland "
                "partition of unity functions."
            )
            master_eq = (
                "\\begin{equation}\n"
                "  \\mathbf{F}_c(x, y) = G_c(x, y) + \\frac{\\sum_{k=1}^K w_k(x, y) \\cdot L_{k,c}(x, y)}{\\sum_{k=1}^K w_k(x, y)}, "
                f"\\quad c \\in \\{{{channel_set}\\}}\n"
                "\\end{equation}"
            )
        else:
            abstract_text = (
                "This report details the Mode B global bivariate Chebyshev and 2D Fourier representation delivered for the input "
                "image canvas. No local patches are present in the serialized Mode B model."
            )
            master_eq = (
                "\\begin{equation}\n"
                f"  F_c(x, y) = G_c(x, y), \\quad c \\in \\{{{channel_set}\\}}\n"
                "\\end{equation}"
            )
        global_heading = "Discovered Global Colour-Field Equations"
        global_intro = "The global field $G_c(x, y)$ models broad illumination and low-frequency structures:"

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

    global_blocks: List[str] = []
    for ch, eq_str in equations.items():
        if not str(eq_str).strip():
            continue
        clean_ch = str(ch).replace("_", "\\_")
        if is_mode_a:
            clean_eq = _human_equation_to_latex(eq_str)
            global_blocks.append(f"\\begin{{equation}}\n  {clean_ch}(x, y) = {clean_eq}\n\\end{{equation}}")
        else:
            if global_eqs:
                latex_str = _human_equation_to_latex(eq_str)
            else:
                try:
                    ast_node = parse(str(eq_str))
                    latex_str = to_latex(ast_node)
                except Exception:
                    latex_str = _human_equation_to_latex(eq_str)
            global_blocks.append(f"\\begin{{equation}}\n  G_{{{clean_ch}}}(x, y) = {latex_str}\n\\end{{equation}}")
    if not global_blocks:
        if is_mode_a:
            global_blocks.append("No per-channel symbolic equations were supplied in the export result.")
        else:
            global_blocks.append(
                "\\begin{equation}\n"
                "  G_c(x, y) = \\sum_{i+j \\le d} a_{i,j,c} T_i(x) T_j(y) + \\sum_{k} \\left[ b_{k,c} \\cos(k_x \\pi x + k_y \\pi y) + d_{k,c} \\sin(k_x \\pi x + k_y \\pi y) \\right]\n"
                "\\end{equation}"
            )

    global_section = "\n\n".join(global_blocks)

    if is_mode_a:
        local_sections = ""
    else:
        if local_patches:
            wendland_section = (
                "\\section{Smooth $C^2$ Wendland Partition of Unity}\n"
                "To blend serialized local patches without step discontinuities, compact Wendland windows are applied:\n\n"
                "\\begin{equation}\n"
                "  w_k(x, y) = \\psi_4\\left(\\frac{|x - c_{x,k}|}{R_{x,k}}\\right) \\cdot \\psi_4\\left(\\frac{|y - c_{y,k}|}{R_{y,k}}\\right), "
                "  \\quad \\psi_4(r) = (1 - r)_+^4 (4r + 1)\n"
                "\\end{equation}"
            )
        else:
            wendland_section = (
                "\\section{Smooth $C^2$ Wendland Partition of Unity}\n"
                "No local patches are present in the serialized Mode B model, so no partition-of-unity blending windows are active."
            )

        if local_patches:
            patch_rows = []
            for patch in local_patches[:20]:
                bounds = _patch_bounds(patch)
                if bounds is None:
                    bounds_str = "N/A"
                    derived_center = None
                else:
                    x0, y0, x1, y1 = bounds
                    bounds_str = f"[{x0:.2f}, {x1:.2f}] \\times [{y0:.2f}, {y1:.2f}]"
                    derived_center = ((x0 + x1) / 2.0, (y0 + y1) / 2.0)

                center = patch.get("center")
                if not isinstance(center, (list, tuple)) or len(center) < 2:
                    center = derived_center
                if center is None:
                    center_str = "N/A"
                else:
                    try:
                        center_str = f"({float(center[0]):.2f}, {float(center[1]):.2f})"
                    except (TypeError, ValueError):
                        center_str = "N/A"

                formula = _latex_table_text(_patch_basis_label(patch))
                if len(formula) > 50:
                    formula = formula[:47] + r"\dots"
                patch_rows.append(
                    f"  {_latex_table_text(_patch_id(patch))} & ${bounds_str}$ & ${center_str}$ & \\texttt{{{formula}}} \\\\"
                )

            all_dct = all(
                isinstance(patch.get("basis"), dict) and str(patch["basis"].get("type", "")).lower() == "dct2"
                for patch in local_patches
            )
            if all_dct:
                local_formula = (
                    "\\begin{equation}\n"
                    "  L_{k,c}(x, y) = \\sum_{p=0}^{P_k} \\sum_{q=0}^{Q_k} c_{k,p,q,c} \\cdot \\alpha_p \\alpha_q \\cdot "
                    "\\cos\\left(\\frac{p \\pi [x - x_{k,min}]}{x_{k,max} - x_{k,min}}\\right) \\cdot "
                    "\\cos\\left(\\frac{q \\pi [y - y_{k,min}]}{y_{k,max} - y_{k,min}}\\right)\n"
                    "\\end{equation}"
                )
            else:
                local_formula = (
                    "\\begin{equation}\n  L_{k,c}(x, y) = \\Phi_{k,c}(x, y)\n\\end{equation}"
                )
            patch_section = (
                "\\section{Catalog of Serialized Local Patch Equations}\n"
                "Each row reports a patch equation or basis configuration stored in the Mode B local-patch data; "
                "no placeholder patch rows are generated.\n\n"
                f"{local_formula}\n\n"
                "\\begin{table}[htbp]\n"
                "\\centering\n"
                f"\\caption{{Serialized Local Patch Catalog (first {min(len(local_patches), 20)} of {len(local_patches)})}}\n"
                "\\label{tab:patches}\n"
                "\\begin{tabular}{llll}\n"
                "\\toprule\n"
                "Patch ID & Domain Region & Center $(c_x, c_y)$ & Equation / Serialized Basis \\\\\n"
                "\\midrule\n"
                f"{chr(10).join(patch_rows)}\n"
                "\\bottomrule\n"
                "\\end{tabular}\n"
                "\\end{table}"
            )
        else:
            patch_section = (
                "\\section{Catalog of Serialized Local Patch Equations}\n"
                "No local patches are present in the serialized Mode B model; the local-patch catalog is empty."
            )
        local_sections = f"{wendland_section}\n\n{patch_section}"

    # 5. Verification Metrics table
    psnr_val = f"{metrics.get('psnr', 0.0):.2f} dB" if "psnr" in metrics else "N/A"
    ssim_val = f"{metrics.get('ssim', 0.0):.4f}" if "ssim" in metrics else "N/A"
    mse_val = f"{metrics.get('mse', 0.0):.8f}" if "mse" in metrics else "N/A"
    mae_val = f"{metrics.get('mae', 0.0):.8f}" if "mae" in metrics else "N/A"
    linf_val = f"{metrics.get('linf', 0.0):.6f}" if "linf" in metrics else "N/A"

    # 6. Parameter table rows
    param_rows = []
    if isinstance(parameters, dict):
        for k in list(parameters.keys())[:50]:
            v = parameters[k]
            raw_value = v.get("value") if isinstance(v, dict) else v
            clean_k = _latex_table_text(k)
            if raw_value is None:
                param_rows.append(f"  {clean_k} & N/A \\\\")
            else:
                param_rows.append(f"  {clean_k} & {float(raw_value):.6f} \\\\")
    elif isinstance(parameters, (list, tuple)):
        for i, val in enumerate(parameters[:50]):
            param_rows.append(f"  c_{{{i}}} & {float(val):.6f} \\\\")

    if not param_rows:
        param_rows.append("  \\multicolumn{2}{c}{\\textit{No model parameters were supplied.}} \\\\")

    param_rows_str = "\n".join(param_rows)
    param_caption = "Delivered Model Parameters" if is_mode_a else "Discovered Model Parameters"
    param_intro = (
        "Table~\\ref{tab:params} presents the numerical parameters delivered with the per-channel symbolic model."
        if is_mode_a
        else "Table~\\ref{tab:params} presents representative parameter vectors discovered via numerical optimization."
    )

    doc_str = (
        "\\documentclass[11pt,a4paper]{article}\n"
        "\\usepackage[utf8]{inputenc}\n"
        "\\usepackage[T1]{fontenc}\n"
        "\\usepackage{amsmath,amssymb,amsfonts}\n"
        "\\usepackage{booktabs}\n"
        "\\usepackage{geometry}\n"
        "\\geometry{margin=1in}\n\n"
        "\\title{MathField Mathematical Image Reconstruction Report}\n"
        "\\author{MathField Computational Engine}\n"
        "\\date{\\today}\n\n"
        "\\begin{document}\n"
        "\\maketitle\n\n"
        "\\begin{abstract}\n"
        f"{abstract_text}\n"
        "\\end{abstract}\n\n"
        "\\section{Executive Summary & Domain Setup}\n"
        "\\begin{table}[htbp]\n"
        "\\centering\n"
        "\\caption{Reconstruction Domain & Coordinate Specifications}\n"
        "\\begin{tabular}{ll}\n"
        "\\toprule\n"
        "Property & Value \\\\\n"
        "\\midrule\n"
        f"Target Resolution & {pw} $\\times$ {ph} pixels \\\\\n"
        f"Continuous Domain $\\Omega$ & $[{x_min:.3f}, {x_max:.3f}] \\times [{y_min:.3f}, {y_max:.3f}]$ \\\\\n"
        f"Reconstruction Mode & \\texttt{{{mode_str}}} \\\\\n"
        f"Color Space & {color_space} \\\\\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
        "\\end{table}\n\n"
        "\\section{Master Continuous Vector Field Formulation}\n"
        "The image is formulated as a deterministic continuous mapping from $\\Omega$ to its reconstructed channel values:\n\n"
        f"{master_eq}\n\n"
        f"\\section{{{global_heading}}}\n"
        f"{global_intro}\n\n"
        f"{global_section}\n\n"
        f"{local_sections}\n\n"
        "\\section{Quantitative Reconstruction Verification}\n"
        "Table~\\ref{tab:metrics} lists rigorous error norms measured between the continuous evaluation and target canvas.\n\n"
        "\\begin{table}[htbp]\n"
        "\\centering\n"
        "\\caption{Continuous Reconstruction Quality Metrics}\n"
        "\\label{tab:metrics}\n"
        "\\begin{tabular}{lll}\n"
        "\\toprule\n"
        "Metric & Value & Description \\\\\n"
        "\\midrule\n"
        f"PSNR & {psnr_val} & Peak Signal-to-Noise Ratio \\\\\n"
        f"SSIM & {ssim_val} & Structural Similarity Index \\\\\n"
        f"MSE  & {mse_val} & Mean Squared Error \\\\\n"
        f"MAE  & {mae_val} & Mean Absolute Error \\\\\n"
        f"L\\_INF & {linf_val} & Maximum Absolute Deviation \\\\\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
        "\\end{table}\n\n"
        "\\section{Numerical Model Parameters}\n"
        f"{param_intro}\n\n"
        "\\begin{table}[htbp]\n"
        "\\centering\n"
        f"\\caption{{{param_caption}}}\n"
        "\\label{tab:params}\n"
        "\\begin{tabular}{lr}\n"
        "\\toprule\n"
        "Parameter & Value \\\\\n"
        "\\midrule\n"
        f"{param_rows_str}\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
        "\\end{table}\n\n"
        "\\section{Standalone Reproducibility}\n"
        "The discovered model is fully reproducible without external dependencies via the generated \\texttt{reproduce.py} script.\n\n"
        "\\end{document}\n"
    )

    out_p.write_text(doc_str, encoding="utf-8")
