#!/usr/bin/env python3
"""Render the MathField demonstration video from a live reconstruction.

The reconstructed images, metrics and equation terms come from a real Mode B
fit of the deterministic analytic sample image. The faint contour backdrop on
the title card is decorative; no stock imagery, canned metrics or pre-rendered
frames are used.

Usage:
    python docs/demo/make_demo.py

Outputs ``docs/demo/demo.mp4`` and ``docs/demo/preview.png``. Requires ffmpeg
(the binary bundled with ``imageio-ffmpeg`` is used when available).
"""

from __future__ import annotations

import math
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parents[2]
DEMO_DIR = Path(__file__).resolve().parent
SAMPLE_PATH = REPO_ROOT / "examples" / "sample_spectrum.png"
VIDEO_PATH = DEMO_DIR / "demo.mp4"
PREVIEW_PATH = DEMO_DIR / "preview.png"

WIDTH, HEIGHT, FPS = 1280, 720, 24
SEGMENTS = (2.0, 2.5, 2.0, 2.5, 1.5)
TOTAL_FRAMES = int(sum(SEGMENTS) * FPS)

# Professional dark research palette.
BG = (10, 17, 30)
PANEL = (16, 26, 44)
BORDER = (38, 56, 84)
TEXT = (232, 240, 248)
MUTED = (148, 166, 190)
ACCENT = (45, 212, 191)
BLUE = (96, 165, 250)
GOLD = (250, 204, 21)
RED = (248, 113, 113)

CONFIG = {
    "global_degree": 4,
    "fourier_freq": 6,
    "max_depth": 3,
    "local_basis_type": "dct2",
    "local_freq": 6,
    "overlap_ratio": 0.20,
    "reg_lambda": 1e-5,
}


def ffmpeg_binary() -> str:
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as exc:
        binary = shutil.which("ffmpeg")
        if binary is None:
            raise RuntimeError("ffmpeg is required to render the demo video") from exc
        return binary


_FONT_CACHE: dict[tuple[int, bool], ImageFont.FreeTypeFont | ImageFont.ImageFont] = {}


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    key = (size, bold)
    if key in _FONT_CACHE:
        return _FONT_CACHE[key]
    candidates = (
        [r"C:\Windows\Fonts\segoeuib.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"]
        if bold
        else [r"C:\Windows\Fonts\segoeui.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"]
    )
    for path in candidates:
        if Path(path).exists():
            _FONT_CACHE[key] = ImageFont.truetype(path, size)
            return _FONT_CACHE[key]
    _FONT_CACHE[key] = ImageFont.load_default(size=size)
    return _FONT_CACHE[key]


def mono(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for path in (r"C:\Windows\Fonts\consola.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"):
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default(size=size)


def fit(image: Image.Image, box: tuple[int, int, int, int]) -> Image.Image:
    left, top, right, bottom = box
    scale = min((right - left) / image.width, (bottom - top) / image.height)
    size = (max(1, int(image.width * scale)), max(1, int(image.height * scale)))
    return image.resize(size, Image.Resampling.LANCZOS)


def ellipsize(draw: ImageDraw.ImageDraw, text: str, fnt: ImageFont.FreeTypeFont | ImageFont.ImageFont, max_w: int) -> str:
    if draw.textlength(text, font=fnt) <= max_w:
        return text
    trimmed = text
    while trimmed and draw.textlength(trimmed + "…", font=fnt) > max_w:
        trimmed = trimmed[:-1]
    return trimmed + "…"


def base_canvas() -> tuple[Image.Image, ImageDraw.ImageDraw]:
    canvas = Image.new("RGB", (WIDTH, HEIGHT), BG)
    draw = ImageDraw.Draw(canvas)
    return canvas, draw


def draw_header(draw: ImageDraw.ImageDraw, kicker: str) -> None:
    draw.rounded_rectangle((44, 26, 78, 60), radius=8, fill=ACCENT)
    draw.text((52, 31), "M", font=font(22, bold=True), fill=BG)
    draw.text((92, 24), "MathField", font=font(30, bold=True), fill=TEXT)
    draw.text((94, 59), "CONTINUOUS MATHEMATICAL IMAGE RECONSTRUCTION", font=font(12), fill=MUTED)
    draw.text((WIDTH - 44, 37), kicker, font=font(13, bold=True), fill=ACCENT, anchor="ra")
    draw.rectangle((44, 82, WIDTH - 44, 84), fill=ACCENT)


def draw_title(draw: ImageDraw.ImageDraw, title: str, subtitle: str | None = None) -> None:
    draw.text((44, 100), title, font=font(31, bold=True), fill=TEXT)
    if subtitle:
        draw.text((46, 141), subtitle, font=font(15), fill=MUTED)


def draw_progress(draw: ImageDraw.ImageDraw, progress: float) -> None:
    draw.rectangle((44, HEIGHT - 10, WIDTH - 44, HEIGHT - 8), fill=BORDER)
    draw.rectangle((44, HEIGHT - 10, 44 + int((WIDTH - 88) * progress), HEIGHT - 8), fill=ACCENT)


def draw_badge(
    draw: ImageDraw.ImageDraw,
    x: int,
    y: int,
    label: str,
    value: str,
    accent: tuple[int, int, int] = ACCENT,
    width: int = 232,
) -> None:
    draw.rounded_rectangle((x, y, x + width, y + 56), radius=10, fill=PANEL, outline=BORDER, width=1)
    draw.rectangle((x + 14, y + 14, x + 18, y + 42), fill=accent)
    draw.text((x + 28, y + 10), label.upper(), font=font(11), fill=MUTED)
    draw.text((x + 28, y + 26), value, font=font(18, bold=True), fill=TEXT)


def draw_panel(
    canvas: Image.Image,
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    image: Image.Image,
    label: str,
    accent: tuple[int, int, int] = ACCENT,
    sublabel: str | None = None,
) -> None:
    x0, y0, x1, y1 = box
    draw.rounded_rectangle(box, radius=14, fill=PANEL, outline=BORDER, width=2)
    draw.rounded_rectangle((x0 + 16, y0 + 16, x0 + 21, y0 + 36), radius=3, fill=accent)
    draw.text((x0 + 32, y0 + 14), label, font=font(14, bold=True), fill=TEXT)

    inner = (x0 + 14, y0 + 48, x1 - 14, y1 - 38)
    shown = fit(image, inner)
    px = x0 + (x1 - x0 - shown.width) // 2
    py = y0 + 48 + (inner[3] - inner[1] - shown.height) // 2
    canvas.paste(shown, (px, py))
    draw.rectangle((px, py, px + shown.width - 1, py + shown.height - 1), outline=BORDER, width=1)
    if sublabel:
        draw.text((x0 + 16, y1 - 30), sublabel, font=font(12), fill=MUTED)


def draw_equation_panel(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    equations: tuple[str, str, str],
) -> None:
    x0, y0, x1, y1 = box
    draw.rounded_rectangle(box, radius=14, fill=PANEL, outline=BORDER, width=2)
    draw.rounded_rectangle((x0 + 16, y0 + 16, x0 + 21, y0 + 36), radius=3, fill=BLUE)
    draw.text((x0 + 32, y0 + 14), "DISCOVERED ANALYTICAL TERMS", font=font(14, bold=True), fill=TEXT)
    for index, (channel, color) in enumerate(zip("RGB", (RED, (74, 222, 128), BLUE), strict=True)):
        yy = y0 + 62 + index * 92
        draw.rounded_rectangle((x0 + 18, yy, x0 + 50, yy + 30), radius=6, fill=color)
        draw.text((x0 + 34, yy + 5), channel, font=font(14, bold=True), fill=BG, anchor="ma")
        draw.text((x0 + 62, yy - 2), f"{channel}(x, y) =", font=mono(13), fill=TEXT)
        draw.text(
            (x0 + 62, yy + 22),
            ellipsize(draw, equations[index], mono(13), x1 - x0 - 90),
            font=mono(13),
            fill=ACCENT,
        )
    draw.text(
        (x0 + 18, y1 - 30),
        "Global terms shown; adaptive local patches live in project.mathfield.",
        font=font(12),
        fill=MUTED,
    )


def make_title_frame(index: int) -> Image.Image:
    canvas, draw = base_canvas()
    # Faint analytic contour backdrop, generated from a simple harmonic field.
    for band in range(9):
        phase = band * 0.32 + index * 0.02
        points = [
            (x, int(664 + 26 * math.sin(x / 92 + phase) + 14 * math.cos(x / 47 - phase)))
            for x in range(0, WIDTH, 8)
        ]
        draw.line(points, fill=(20, 38, 58), width=2)
    draw_header(draw, "RESEARCH DEMO  ·  MODE B")
    draw.text((44, 250), "From pixels to closed-form equations", font=font(46, bold=True), fill=TEXT)
    draw.text((46, 320), "MathField reconstructs images as continuous mathematical colour fields —", font=font(20), fill=MUTED)
    draw.text((46, 352), "deterministic, interpretable, and independent of any neural network.", font=font(20), fill=MUTED)
    cards = [
        ("01", "IMAGE ANALYSIS", "Geometry, entropy, spectrum", ACCENT),
        ("02", "GLOBAL EXPANSION", "Chebyshev + Fourier basis", BLUE),
        ("03", "LOCAL REFINEMENT", "Sobolev quadtree + C² blend", GOLD),
    ]
    for slot, (number, heading, detail, color) in enumerate(cards):
        x0 = 44 + slot * 400
        draw.rounded_rectangle((x0, 430, x0 + 376, 570), radius=12, fill=PANEL, outline=BORDER, width=2)
        draw.rounded_rectangle((x0 + 18, 448, x0 + 23, 478), radius=3, fill=color)
        draw.text((x0 + 36, 444), number, font=font(28, bold=True), fill=color)
        draw.text((x0 + 18, 492), heading, font=font(15, bold=True), fill=TEXT)
        draw.text((x0 + 18, 522), detail, font=font(13), fill=MUTED)
    draw_progress(draw, index / TOTAL_FRAMES)
    return canvas


def make_comparison_frame(
    index: int,
    total: int,
    target: Image.Image,
    recon: Image.Image,
    metrics: dict[str, float],
    reveal: float,
) -> Image.Image:
    canvas, draw = base_canvas()
    draw_header(draw, "TARGET  ↔  RECONSTRUCTION")
    draw_title(draw, "A discrete image, represented by continuous equations")
    right = Image.blend(target, recon, reveal)
    draw_panel(
        canvas,
        draw,
        (44, 176, 616, 584),
        target,
        "TARGET RASTER  ·  320 × 200",
        BLUE,
        "Sampled once on a discrete pixel grid.",
    )
    draw_panel(
        canvas,
        draw,
        (664, 176, 1236, 584),
        right,
        "CONTINUOUS RECONSTRUCTION",
        ACCENT,
        f"Analytical fit  ·  reveal {reveal * 100:5.1f}%",
    )
    draw_badge(draw, 44, 604, "PSNR", f"{metrics.get('psnr', 0):.2f} dB", ACCENT)
    draw_badge(draw, 292, 604, "SSIM", f"{metrics.get('ssim', 0):.4f}", BLUE)
    draw_badge(draw, 540, 604, "MSE", f"{metrics.get('mse', 0):.6f}", GOLD)
    draw.text((800, 620), "Chebyshev + Fourier + adaptive local patches", font=font(16), fill=MUTED)
    draw.text((800, 646), "No learned weights. No lookup tables.", font=font(14), fill=ACCENT)
    draw_progress(draw, index / TOTAL_FRAMES)
    return canvas


def make_curtain_frame(
    index: int,
    total: int,
    target: Image.Image,
    recon: Image.Image,
    split: float,
) -> Image.Image:
    canvas, draw = base_canvas()
    draw_header(draw, "CURTAIN COMPARISON")
    draw_title(draw, "Two representations, one coordinate domain")
    box = (214, 170, 1066, 604)
    draw.rounded_rectangle(box, radius=14, fill=PANEL, outline=BORDER, width=2)
    shown_target = fit(target, (box[0] + 10, box[1] + 10, box[2] - 10, box[3] - 10))
    shown_recon = fit(recon, (box[0] + 10, box[1] + 10, box[2] - 10, box[3] - 10))
    x0 = box[0] + (box[2] - box[0] - shown_target.width) // 2
    y0 = box[1] + (box[3] - box[1] - shown_target.height) // 2
    canvas.paste(shown_recon, (x0, y0))
    cut = int(shown_target.width * split)
    canvas.paste(shown_target.crop((0, 0, max(cut, 1), shown_target.height)), (x0, y0))
    draw.line((x0 + cut, y0, x0 + cut, y0 + shown_target.height), fill=ACCENT, width=3)
    draw.rounded_rectangle((214, 148, 219, 166), radius=3, fill=BLUE)
    draw.text((230, 146), "TARGET RASTER", font=font(14, bold=True), fill=TEXT)
    draw.rounded_rectangle((868, 148, 873, 166), radius=3, fill=ACCENT)
    draw.text((884, 146), "CONTINUOUS FIELD", font=font(14, bold=True), fill=TEXT)
    draw.text((44, 628), "A continuous field evaluates at any resolution — the curtain is only a visual aid.", font=font(17), fill=MUTED)
    draw_progress(draw, index / TOTAL_FRAMES)
    return canvas


def error_image(target: Image.Image, recon: Image.Image) -> Image.Image:
    from scipy.ndimage import gaussian_filter

    diff = np.abs(np.asarray(recon, dtype=np.float64) - np.asarray(target, dtype=np.float64)).mean(axis=-1)
    # Mild smoothing removes uint8 quantisation speckle while preserving the
    # localized structure that drives the adaptive quadtree.
    smooth = gaussian_filter(diff, sigma=1.2)
    scale = max(float(np.percentile(smooth, 99.5)), 1e-6)
    t = np.clip(smooth / scale, 0.0, 1.0)
    import matplotlib

    rgb = (matplotlib.colormaps["inferno"](t)[:, :, :3] * 255.0).astype(np.uint8)
    return Image.fromarray(rgb, "RGB")


def make_verify_frame(
    index: int,
    total: int,
    heat: Image.Image,
    metrics: dict[str, float],
    equations: tuple[str, str, str],
    show_equations: bool,
) -> Image.Image:
    canvas, draw = base_canvas()
    draw_header(draw, "VERIFICATION & DISCOVERY")
    draw_title(draw, "Error diagnostics and the equations behind the image")
    draw_panel(
        canvas,
        draw,
        (44, 176, 616, 584),
        heat,
        "ABSOLUTE ERROR  ·  P99.5 NORMALIZED",
        GOLD,
        "Cool = accurate, warm = localized residual.",
    )
    if show_equations:
        draw_equation_panel(draw, (664, 176, 1236, 584), equations)
    else:
        draw_panel(
            canvas,
            draw,
            (664, 176, 1236, 584),
            heat,
            "LOCALIZED RESIDUAL STRUCTURE",
            BLUE,
            "Residuals guide the adaptive quadtree.",
        )
    draw_badge(draw, 44, 604, "PSNR", f"{metrics.get('psnr', 0):.2f} dB", ACCENT)
    draw_badge(draw, 292, 604, "SSIM", f"{metrics.get('ssim', 0):.4f}", BLUE)
    draw_badge(draw, 540, 604, "MSE", f"{metrics.get('mse', 0):.6f}", GOLD)
    draw.text((800, 620), "Fully deterministic and reproducible from exported coefficients.", font=font(16), fill=MUTED)
    draw_progress(draw, index / TOTAL_FRAMES)
    return canvas


def make_closing_frame(index: int, total: int, metrics: dict[str, float]) -> Image.Image:
    canvas, draw = base_canvas()
    draw_header(draw, "REPRODUCIBILITY")
    draw.text((44, 220), "Verifiable by construction", font=font(42, bold=True), fill=TEXT)
    items = [
        (ACCENT, "No machine learning, pretrained weights, or cloud inference."),
        (BLUE, "Standalone reproduction scripts evaluate the exported equations."),
        (GOLD, f"Reconstruction verified at {metrics.get('psnr', 0):.2f} dB PSNR / {metrics.get('ssim', 0):.4f} SSIM."),
    ]
    for row, (color, text) in enumerate(items):
        yy = 320 + row * 64
        draw.rounded_rectangle((48, yy, 54, yy + 32), radius=3, fill=color)
        draw.text((76, yy + 3), text, font=font(20), fill=TEXT)
    draw.rectangle((44, 560, WIDTH - 44, 562), fill=BORDER)
    draw.text((44, 588), "mathfield  ·  research-grade mathematical image reconstruction", font=font(15), fill=MUTED)
    draw.text((WIDTH - 44, 588), "github.com/rajveersinh-is-dev/MathField", font=font(15), fill=ACCENT, anchor="ra")
    draw_progress(draw, index / TOTAL_FRAMES)
    return canvas


def build_frames(
    target: Image.Image,
    recon: Image.Image,
    metrics: dict[str, float],
    equations: tuple[str, str, str],
) -> list[Image.Image]:
    frames: list[Image.Image] = []
    target_big = fit(target, (44, 176, 616, 584))
    recon_big = fit(recon, (664, 176, 1236, 584))
    heat = error_image(target, recon)

    for idx in range(int(SEGMENTS[0] * FPS)):
        frames.append(make_title_frame(idx))
    for idx in range(int(SEGMENTS[1] * FPS)):
        reveal = (idx + 1) / int(SEGMENTS[1] * FPS)
        frames.append(make_comparison_frame(len(frames), TOTAL_FRAMES, target_big, recon_big, metrics, reveal))
    for idx in range(int(SEGMENTS[2] * FPS)):
        count = max(int(SEGMENTS[2] * FPS) - 1, 1)
        split = 0.5 + 0.5 * math.sin(math.pi * idx / count)
        frames.append(make_curtain_frame(len(frames), TOTAL_FRAMES, target, recon, split))
    verify_count = int(SEGMENTS[3] * FPS)
    for idx in range(verify_count):
        frames.append(
            make_verify_frame(
                len(frames),
                TOTAL_FRAMES,
                heat,
                metrics,
                equations,
                show_equations=idx >= int(0.12 * verify_count),
            )
        )
    for _idx in range(int(SEGMENTS[4] * FPS)):
        frames.append(make_closing_frame(len(frames), TOTAL_FRAMES, metrics))
    return frames


def main() -> None:
    sys.path.insert(0, str(REPO_ROOT))
    from mathfield.core.domain import CoordinateDomain2D, DomainMode
    from mathfield.core.pipeline.mode_b import ModeBPipeline
    from mathfield.core.pipeline.stage import PipelineContext

    if not SAMPLE_PATH.is_file():
        raise FileNotFoundError("Run `python examples/generate_sample.py` first")
    target = Image.open(SAMPLE_PATH).convert("RGB")
    arr = np.asarray(target, dtype=np.float64) / 255.0
    domain = CoordinateDomain2D(width=target.width, height=target.height, mode=DomainMode.ASPECT_PRESERVED)

    started = time.perf_counter()
    ctx = PipelineContext(image=arr, domain=domain, config=CONFIG)
    ModeBPipeline(config=CONFIG).execute(ctx)
    print(f"Reconstruction fitted in {time.perf_counter() - started:.1f}s")

    recon_arr = np.clip(ctx.rendered_image, 0.0, 1.0)
    recon = Image.fromarray((recon_arr * 255.0 + 0.5).astype(np.uint8))
    metrics = dict(ctx.metrics)
    global_model = ctx.global_model
    cheb, four = global_model["chebyshev_basis"], global_model["fourier_basis"]
    coeffs = np.asarray(global_model["coefficients"])

    def channel_equation(channel: int) -> str:
        vec = coeffs[:, channel] if coeffs.ndim == 2 else coeffs
        terms: list[tuple[float, str]] = []
        for k, (i, j) in enumerate(cheb.indices):
            value = float(vec[k])
            if abs(value) > 1e-3:
                name = "1" if i == j == 0 else f"T{i}(x)" if j == 0 else f"T{j}(y)" if i == 0 else f"T{i}(x)T{j}(y)"
                terms.append((abs(value), f"{value:+.3f} {name}"))
        base = cheb.num_terms
        for m, (kx, ky) in enumerate(four.freq_pairs):
            for value, func in ((float(vec[base + 2 * m]), "cos"), (float(vec[base + 2 * m + 1]), "sin")):
                if abs(value) > 1e-3:
                    parts = [f"{int(k)}x" for k in (kx,) if k] + [f"{int(k)}y" for k in (ky,) if k]
                    terms.append((abs(value), f"{value:+.3f} {func}(π({' + '.join(parts) or '0'}))"))
        terms.sort(key=lambda item: item[0], reverse=True)
        return " ".join(text for _, text in terms[:5]) + " + …"

    equations = tuple(channel_equation(i) for i in range(3))  # type: ignore[assignment]
    frames = build_frames(target, recon, metrics, equations)

    # Still preview: the fully reconstructed comparison frame (README thumbnail).
    preview = make_comparison_frame(
        len(frames) - 1,
        TOTAL_FRAMES,
        fit(target, (44, 176, 616, 584)),
        fit(recon, (664, 176, 1236, 584)),
        metrics,
        1.0,
    )
    preview.save(PREVIEW_PATH)
    print(f"Wrote {PREVIEW_PATH}")

    command = [
        ffmpeg_binary(),
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{WIDTH}x{HEIGHT}",
        "-r",
        str(FPS),
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "20",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(VIDEO_PATH),
    ]
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    assert process.stdin is not None
    for frame in frames:
        process.stdin.write(frame.tobytes())
    process.stdin.close()
    stderr = process.stderr.read().decode("utf-8", errors="replace") if process.stderr else ""
    if process.wait() != 0:
        raise RuntimeError(f"ffmpeg failed: {stderr[-2000:]}")
    print(f"Wrote {VIDEO_PATH} ({len(frames)} frames, {len(frames) / FPS:.1f}s)")


if __name__ == "__main__":
    main()
