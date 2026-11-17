"""
run_lotus_reconstruction.py

Execute maximum-quality Mode B reconstruction on the user-uploaded water lily / lotus image.
Generates full suite of deliverables:
- High-fidelity continuous raster reconstruction (PNG, JPG)
- Continuous mathematical super-resolved rendering (2x resolution from equations)
- Standalone zero-dependency Python reproduction script
- Publication-grade PDF and LaTeX mathematical reports
- High-precision parameter CSV table
- Versioned JSON project file (.mathfield)
- Standalone reproducibility package bundle (.zip)
"""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from mathfield.core.domain import CoordinateDomain2D, DomainMode
from mathfield.core.pipeline.mode_b import ModeBPipeline
from mathfield.core.pipeline.stage import PipelineContext
from mathfield.export.artwork_poster import export_artwork_poster
from mathfield.export.bundle import export_bundle
from mathfield.export.csv_export import export_csv_parameters
from mathfield.export.latex_report import export_latex_report
from mathfield.export.pdf_report import export_pdf_report
from mathfield.export.project_file import save_project
from mathfield.export.python_script import export_standalone_script
from mathfield.export.raster import export_jpg, export_png
from mathfield.export.svg import export_svg

INPUT_IMAGE = Path(
    r"C:/Users/davea/.gemini/antigravity/brain/8953e3b7-d601-4777-8b63-fe5f1fe32be4/.user_uploaded/media_1794844888796_2afd2c38.jpg"
)
OUTPUT_DIR = Path(r"c:/Users/davea/OneDrive/Desktop/MathField/output/lotus_maximum_quality")
ARTIFACT_DIR = Path(r"C:/Users/davea/.gemini/antigravity/brain/8953e3b7-d601-4777-8b63-fe5f1fe32be4")


def render_continuous_super_resolution(
    ctx: PipelineContext,
    domain: CoordinateDomain2D,
    out_w: int,
    out_h: int,
    overlap_ratio: float = 0.20,
) -> np.ndarray:
    """
    Renders continuous mathematical super-resolution directly from the
    discovered basis functions (Chebyshev, Fourier, and localized DCT patches)
    without raster interpolation.
    """
    X_s, Y_s = domain.to_discrete_grid(width=out_w, height=out_h)

    # 1. Global expansion evaluation
    cheb_b = ctx.global_model["chebyshev_basis"]
    four_b = ctx.global_model["fourier_basis"]
    g_coeffs = ctx.global_model["coefficients"]
    c = g_coeffs.shape[1] if g_coeffs.ndim == 2 else 1

    Phi_cheb = cheb_b.basis_matrix(X_s, Y_s)
    Phi_four = four_b.basis_matrix(X_s, Y_s)
    Phi_global = np.hstack([Phi_cheb, Phi_four[:, 1:]])
    global_s = (
        (Phi_global @ g_coeffs).reshape(out_h, out_w, c) if c > 1 else (Phi_global @ g_coeffs).reshape(out_h, out_w)
    )

    # 2. Local patches evaluation with separable Wendland C2 partition of unity
    patches = ctx.local_decomposition.get("patches", [])
    local_corr_s = (
        np.zeros((out_h, out_w, c), dtype=np.float64) if c > 1 else np.zeros((out_h, out_w), dtype=np.float64)
    )
    sum_W_s = np.zeros((out_h, out_w), dtype=np.float64)

    xs = np.linspace(domain.x_min, domain.x_max, out_w)
    ys = np.linspace(domain.y_min, domain.y_max, out_h)

    for patch in patches:
        lx0, ly0, lx1, ly1 = patch.bounds
        cx, cy = patch.center
        hx, hy = patch.half_widths
        Rx = (1.0 + overlap_ratio) * hx
        Ry = (1.0 + overlap_ratio) * hy

        px_mask = (xs >= cx - Rx) & (xs <= cx + Rx)
        py_mask = (ys >= cy - Ry) & (ys <= cy + Ry)
        cols = np.where(px_mask)[0]
        rows = np.where(py_mask)[0]
        if len(cols) == 0 or len(rows) == 0:
            continue

        r0, r1 = int(rows.min()), int(rows.max()) + 1
        c0, c1 = int(cols.min()), int(cols.max()) + 1

        sub_X = X_s[r0:r1, c0:c1]
        sub_Y = Y_s[r0:r1, c0:c1]

        dx = (sub_X - cx) / Rx
        dy = (sub_Y - cy) / Ry
        adx = np.abs(dx)
        ady = np.abs(dy)
        wx = np.where(adx < 1.0, ((1.0 - adx) ** 4.0) * (4.0 * adx + 1.0), 0.0)
        wy = np.where(ady < 1.0, ((1.0 - ady) ** 4.0) * (4.0 * ady + 1.0), 0.0)
        w_k = wx * wy

        fit_s = patch.evaluate(sub_X, sub_Y)
        if c > 1:
            local_corr_s[r0:r1, c0:c1] += fit_s * np.expand_dims(w_k, axis=-1)
        else:
            local_corr_s[r0:r1, c0:c1] += fit_s * w_k
        sum_W_s[r0:r1, c0:c1] += w_k

    sum_W_safe = np.maximum(sum_W_s, 1e-12)
    if c > 1:
        local_corr_norm = local_corr_s / np.expand_dims(sum_W_safe, axis=-1)
    else:
        local_corr_norm = local_corr_s / sum_W_safe

    super_recon = np.clip(global_s + local_corr_norm, 0.0, 1.0)
    return super_recon


def main() -> None:
    print(f"Loading input image from: {INPUT_IMAGE}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    img_pil = Image.open(INPUT_IMAGE).convert("RGB")
    arr = np.array(img_pil, dtype=np.float64) / 255.0
    h, w, c = arr.shape
    print(f"Source Image Dimensions: {w} x {h} (Total Pixels: {w * h:,})")

    # Continuous coordinate domain [-1, 1]^2 preserving aspect ratio
    domain = CoordinateDomain2D(width=w, height=h, mode=DomainMode.ASPECT_PRESERVED)

    # Maximum quality configuration: Degree 6 Chebyshev + Freq 8 Fourier + Depth 5 Quadtree with Local DCT2
    config = {
        "global_degree": 6,
        "fourier_freq": 8,
        "max_depth": 5,
        "split_threshold": 0.0005,
        "local_basis_type": "dct2",
        "local_freq": 10,
        "overlap_ratio": 0.20,
        "reg_lambda": 1e-5,
    }

    ctx = PipelineContext(image=arr, domain=domain, config=config)
    pipeline = ModeBPipeline(config=config)

    print("\n--- Starting Maximum Quality Mode B Reconstruction ---")
    t0 = time.perf_counter()

    def on_progress(stage_name: str, current: int, total: int) -> None:
        print(f"[{current}/{total}] {stage_name}")

    status = pipeline.execute(ctx, progress_callback=on_progress)
    t1 = time.perf_counter()
    total_time = t1 - t0

    metrics = ctx.metrics
    print(f"\nReconstruction completed in {total_time:.2f} seconds.")
    print(f"Status: {status}")
    print(f"Final Peak Signal-to-Noise Ratio (PSNR): {metrics['psnr']:.2f} dB")
    print(f"Structural Similarity Index (SSIM): {metrics['ssim']:.4f} ({metrics['ssim'] * 100:.1f}%)")
    print(f"Mean Squared Error (MSE): {metrics['mse']:.8f}")
    print(f"Mean Absolute Error (MAE): {metrics['mae']:.8f}")

    recon_arr = ctx.rendered_image

    # Compute absolute error heatmap (5x amplified for visualization)
    diff = np.abs(recon_arr - arr)
    diff_mag = np.mean(diff, axis=-1)
    diff_norm = np.clip(diff_mag * 5.0 * 255.0, 0, 255).astype(np.uint8)

    # 1. Build model parameters dictionary from genuine fitted coefficients
    parameters_dict = {}
    g_coeffs = ctx.global_model["coefficients"]
    n_global = g_coeffs.shape[0] if isinstance(g_coeffs, np.ndarray) else 100
    for i in range(n_global):
        val = float(np.mean(g_coeffs[i])) if isinstance(g_coeffs, np.ndarray) else 0.0
        parameters_dict[f"g_term_{i:04d}"] = {
            "stage": "global_hybrid",
            "channel": "RGB",
            "value": val,
        }

    patches = ctx.local_decomposition.get("patches", [])
    for p_idx, patch in enumerate(patches):
        if patch.coefficients is not None:
            c_flat = np.mean(patch.coefficients, axis=-1) if patch.coefficients.ndim == 2 else patch.coefficients
            for t_idx, val in enumerate(c_flat):
                parameters_dict[f"p_{p_idx:03d}_t_{t_idx:02d}"] = {
                    "stage": "local_patch",
                    "channel": "RGB",
                    "value": float(val),
                }

    print(f"Total Discovered Parameter Count: {len(parameters_dict):,} coefficients")

    # 2. Extract explicit analytical global equations for each colour channel
    cheb_b = ctx.global_model["chebyshev_basis"]
    four_b = ctx.global_model["fourier_basis"]

    def format_global_channel(c_idx: int, max_terms: int = 8) -> str:
        terms = []
        # Chebyshev polynomial terms
        for k, (i, j) in enumerate(cheb_b.indices):
            coeff = g_coeffs[k, c_idx]
            if abs(coeff) > 1e-4:
                if i == 0 and j == 0:
                    name = "1"
                elif i > 0 and j == 0:
                    name = f"T_{i}(x)"
                elif i == 0 and j > 0:
                    name = f"T_{j}(y)"
                else:
                    name = f"T_{i}(x)T_{j}(y)"
                terms.append((abs(coeff), f"{coeff:+.4f}·{name}"))
        # Fourier harmonic terms
        base_idx = cheb_b.num_terms
        for m, (kx, ky) in enumerate(four_b.freq_pairs):
            cos_coeff = g_coeffs[base_idx + 2 * m, c_idx]
            sin_coeff = g_coeffs[base_idx + 2 * m + 1, c_idx]
            arg = f"({int(kx)}πx" if kx != 0 else "("
            if ky != 0:
                arg += f" + {int(ky)}πy)" if kx != 0 else f"{int(ky)}πy)"
            else:
                arg += ")"
            if abs(cos_coeff) > 1e-4:
                terms.append((abs(cos_coeff), f"{cos_coeff:+.4f}·cos{arg}"))
            if abs(sin_coeff) > 1e-4:
                terms.append((abs(sin_coeff), f"{sin_coeff:+.4f}·sin{arg}"))

        terms.sort(key=lambda t: t[0], reverse=True)
        leading = [t[1] for t in terms[:max_terms]]
        return " ".join(leading) + f" + ... [{len(terms)} total terms]"

    global_equations = {
        "R": format_global_channel(0),
        "G": format_global_channel(1),
        "B": format_global_channel(2),
    }

    # 3. Extract explicit analytical formulas for all discovered local patches
    def format_patch_formula(patch: Any, max_terms: int = 5) -> str:
        if patch.coefficients is None or patch.basis is None:
            return "0"
        c = patch.coefficients
        if c.ndim == 2:
            c_mag = np.linalg.norm(c, axis=-1)
            c_mean = np.mean(c, axis=-1)
        else:
            c_mag = np.abs(c)
            c_mean = c
        indices = getattr(patch.basis, "indices", [])
        terms = []
        for k, (p, q) in enumerate(indices):
            coeff = c_mean[k]
            mag = c_mag[k]
            if mag > 1e-4:
                if p == 0 and q == 0:
                    name = "1"
                elif p > 0 and q == 0:
                    name = f"cos({p}πu)"
                elif p == 0 and q > 0:
                    name = f"cos({q}πv)"
                else:
                    name = f"cos({p}πu)·cos({q}πv)"
                terms.append((mag, f"{coeff:+.4f}·{name}"))
        terms.sort(key=lambda t: t[0], reverse=True)
        if not terms:
            return "0.0000"
        leading = [t[1] for t in terms[:max_terms]]
        return " ".join(leading) + (f" + ... [{len(terms)} terms]" if len(terms) > max_terms else "")

    patch_equations = []
    for p_idx, patch in enumerate(patches):
        lx0, ly0, lx1, ly1 = patch.bounds
        cx, cy = patch.center
        hx, hy = patch.half_widths
        Rx = (1.0 + config["overlap_ratio"]) * hx
        Ry = (1.0 + config["overlap_ratio"]) * hy
        c = patch.coefficients
        energy = float(np.sum(c**2)) if c is not None and c.size > 0 else 0.0
        formula = format_patch_formula(patch, max_terms=5)
        patch_equations.append(
            {
                "id": f"P_{p_idx + 1:03d}",
                "bounds": (float(lx0), float(ly0), float(lx1), float(ly1)),
                "center": (float(cx), float(cy)),
                "radius": (float(Rx), float(Ry)),
                "energy": energy,
                "formula": formula,
                "terms_count": patch.basis.num_terms if patch.basis else 0,
            }
        )

    # Sort patches by harmonic energy descending so high-detail flower patches are cataloged first
    patch_equations_sorted = sorted(patch_equations, key=lambda p: p["energy"], reverse=True)

    # Save full catalog of all 298 patch equations to JSON for research completeness
    with open(OUTPUT_DIR / "patch_equations.json", "w", encoding="utf-8") as f:
        json.dump(patch_equations_sorted, f, indent=2)
    print(f"✔ Exported patch_equations.json ({len(patch_equations_sorted)} analytical patch equations)")

    global_model = {
        "type": "mode_b_global_hybrid_chebyshev_fourier",
        "degree": config["global_degree"],
        "fourier_freq": config["fourier_freq"],
        "basis_terms_count": n_global,
        "local_patches_count": len(patches),
        "channels": [
            {
                "channel": "RGB",
                "formula": f"chebyshev2d_deg_{config['global_degree']} + fourier2d_freq_{config['fourier_freq']} + quadtree_dct2_local_patches",
                "parameters": [parameters_dict[k]["value"] for k in list(parameters_dict.keys())[:30]],
            }
        ],
    }

    export_result = {
        "coordinate_domain": {
            "x_min": float(domain.x_min),
            "x_max": float(domain.x_max),
            "y_min": float(domain.y_min),
            "y_max": float(domain.y_max),
            "pixel_width": w,
            "pixel_height": h,
            "aspect_ratio": float(w / h),
        },
        "color_space": "RGB",
        "mode": "MODE_B_FIDELITY",
        "global_model": global_model,
        "global_equations": global_equations,
        "patch_equations": patch_equations_sorted,
        "expressions": {
            "formula": f"chebyshev2d_deg_{config['global_degree']} + fourier2d_freq_{config['fourier_freq']} + quadtree_dct2_local_patches",
            "R": global_equations["R"],
            "G": global_equations["G"],
            "B": global_equations["B"],
        },
        "parameters": parameters_dict,
        "metrics": metrics,
        "rendered_image": recon_arr,
        "reconstructed_image": recon_arr,
        "original_image": arr,
        "error_heatmap": diff_norm,
    }

    print("\n--- Exporting Deliverables ---")

    # 1. Native PNG and JPG
    export_png(export_result, OUTPUT_DIR / "reconstruction.png", width=w, height=h)
    export_jpg(export_result, OUTPUT_DIR / "reconstruction.jpg", quality=95, width=w, height=h)
    print("✔ Exported reconstruction.png and reconstruction.jpg (Native resolution)")

    # 2. Super-resolved continuous-domain rendering (2x resolution from continuous equations)
    super_w, super_h = w * 2, h * 2
    print(f"Rendering continuous 2x super-resolution ({super_w}x{super_h}) from equations...")
    super_recon = render_continuous_super_resolution(
        ctx, domain, super_w, super_h, overlap_ratio=config["overlap_ratio"]
    )
    Image.fromarray((super_recon * 255.0).astype(np.uint8)).save(OUTPUT_DIR / "reconstruction_2x_superres.png")
    print(f"✔ Exported reconstruction_2x_superres.png ({super_w}x{super_h} via continuous mathematical evaluation)")

    # 3. Vector SVG
    export_svg(export_result, OUTPUT_DIR / "reconstruction.svg", width=w, height=h)
    print("✔ Exported reconstruction.svg (Vector & mathematical metadata)")

    # 4. Parameters CSV table
    export_csv_parameters(export_result, OUTPUT_DIR / "parameters.csv")
    print(f"✔ Exported parameters.csv ({len(parameters_dict):,} basis terms)")

    # 5. Standalone reproduction script
    export_standalone_script(export_result, OUTPUT_DIR / "reproduce.py")
    print("✔ Exported reproduce.py (Zero-dependency standalone reproduction script)")

    # 6. JSON project file
    proj_dict = {
        "schema_version": "1.0.0",
        "project_name": "MathField_Lotus_Waterlily_Reconstruction",
        "coordinate_domain": export_result["coordinate_domain"],
        "color_space": "RGB",
        "mode": "MODE_B_FIDELITY",
        "global_model": global_model,
        "local_decomposition": {
            "leaves_count": ctx.local_decomposition.get("leaves_count", 0) if ctx.local_decomposition else 0
        },
        "final_metrics": metrics,
        "parameters": parameters_dict,
        "random_seed": 42,
        "status": "COMPLETED",
    }
    save_project(proj_dict, OUTPUT_DIR / "lotus_model.mathfield")
    print("✔ Exported lotus_model.mathfield (Draft-07 validated project file)")

    # 7. PDF and LaTeX reports
    export_pdf_report(export_result, OUTPUT_DIR / "reconstruction_report.pdf")
    export_latex_report(export_result, OUTPUT_DIR / "reconstruction_report.tex")
    print("✔ Exported reconstruction_report.pdf and reconstruction_report.tex")

    # 8. Reproducibility package bundle
    export_bundle(export_result, OUTPUT_DIR / "reproducibility_bundle.zip", as_zip=True)
    print("✔ Exported reproducibility_bundle.zip (Complete standalone archive)")

    # 9. Mathematical Artwork Posters (Hamid Naderi Yeganeh Style)
    export_artwork_poster(export_result, OUTPUT_DIR / "lotus_mathematical_artwork.png", layout="square", width=1600)
    export_artwork_poster(
        export_result, OUTPUT_DIR / "lotus_mathematical_artwork.jpg", layout="square", width=1600, quality=95
    )
    export_artwork_poster(
        export_result, OUTPUT_DIR / "lotus_mathematical_artwork_portrait.png", layout="portrait", width=1200
    )
    export_artwork_poster(
        export_result, OUTPUT_DIR / "lotus_mathematical_artwork_portrait.jpg", layout="portrait", width=1200, quality=95
    )
    print("✔ Exported lotus_mathematical_artwork (Square & Portrait posters in Hamid Naderi Yeganeh style)")

    # Copy visual comparison images and artwork posters into artifact directory
    artifact_orig = ARTIFACT_DIR / "lotus_original.jpg"
    artifact_recon = ARTIFACT_DIR / "lotus_reconstructed.png"
    artifact_diff = ARTIFACT_DIR / "lotus_error_map.png"

    shutil.copy2(INPUT_IMAGE, artifact_orig)
    Image.fromarray((np.clip(recon_arr, 0.0, 1.0) * 255.0).astype(np.uint8)).save(artifact_recon)

    # Compute absolute error heatmap
    diff = np.abs(recon_arr - arr)
    diff_mag = np.mean(diff, axis=-1)
    diff_norm = np.clip(diff_mag * 5.0 * 255.0, 0, 255).astype(np.uint8)  # 5x amplified for visualization
    Image.fromarray(diff_norm, mode="L").save(artifact_diff)

    shutil.copy2(OUTPUT_DIR / "lotus_mathematical_artwork.png", ARTIFACT_DIR / "lotus_mathematical_artwork.png")
    shutil.copy2(OUTPUT_DIR / "lotus_mathematical_artwork.jpg", ARTIFACT_DIR / "lotus_mathematical_artwork.jpg")
    shutil.copy2(
        OUTPUT_DIR / "lotus_mathematical_artwork_portrait.png", ARTIFACT_DIR / "lotus_mathematical_artwork_portrait.png"
    )

    print("\n✔ Visual assets and mathematical artwork posters copied to artifact directory for presentation.")
    print("All tasks completed successfully!")


if __name__ == "__main__":
    main()
