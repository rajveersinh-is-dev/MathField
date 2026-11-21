#!/usr/bin/env python3
"""Render the MathField demonstration video from a live reconstruction.

Everything shown in the video is produced by MathField itself: the target is a
deterministic analytic sample image, the reconstruction is fitted by Mode B,
and the captions quote the equations actually discovered by the engine.

Usage:
    python docs/demo/make_demo.py

Requires ffmpeg (the bundled binary from ``imageio-ffmpeg`` is used when
available). Outputs ``docs/demo/demo.mp4`` and ``docs/demo/preview.png``.
"""

from __future__ import annotations

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

WIDTH, HEIGHT, FPS = 960, 540, 24
SEGMENTS = (2.0, 2.5, 2.0, 2.0, 2.0)  # title, target, reveal, curtain, error+equations
TOTAL_FRAMES = int(sum(SEGMENTS) * FPS)

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


def load_font(size: int, monospace: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = (
        [r"C:\Windows\Fonts\consola.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"]
        if monospace
        else [
            r"C:\Windows\Fonts\segoeui.ttf",
            r"C:\Windows\Fonts\arial.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        ]
    )
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default(size=size)


def fit(image: Image.Image, box: tuple[int, int, int, int]) -> Image.Image:
    left, top, right, bottom = box
    max_w, max_h = right - left, bottom - top
    scale = min(max_w / image.width, max_h / image.height)
    size = (max(1, int(image.width * scale)), max(1, int(image.height * scale)))
    return image.resize(size, Image.Resampling.LANCZOS)


def make_frame(
    *,
    title: str,
    caption: str,
    image: Image.Image | None,
    overlay: str = "",
    progress: float = 0.0,
    equations: tuple[str, str, str] | None = None,
    metrics: dict[str, float] | None = None,
    split: float | None = None,
    second_image: Image.Image | None = None,
) -> Image.Image:
    canvas = Image.new("RGB", (WIDTH, HEIGHT), (12, 16, 24))
    draw = ImageDraw.Draw(canvas)
    title_font = load_font(26)
    small_font = load_font(15)
    mono_font = load_font(13, monospace=True)

    draw.rectangle((0, 0, WIDTH, 4), fill=(0, 209, 178))
    draw.text((30, 20), "MathField", font=load_font(30), fill=(0, 209, 178))
    draw.text((215, 28), "/ continuous mathematical image reconstruction", font=small_font, fill=(160, 176, 194))
    draw.text((30, 66), title, font=title_font, fill=(238, 244, 250))

    if image is not None:
        box = (30, 112, WIDTH - 30, 400)
        shown = fit(image, box)
        x0 = box[0] + (box[2] - box[0] - shown.width) // 2
        y0 = box[1] + (box[3] - box[1] - shown.height) // 2
        canvas.paste(shown, (x0, y0))
        draw.rectangle((x0, y0, x0 + shown.width - 1, y0 + shown.height - 1), outline=(72, 92, 112), width=1)

        if split is not None and second_image is not None:
            other = fit(second_image, box)
            cut = int(shown.width * split)
            crop = other.crop((0, 0, max(cut, 1), other.height))
            canvas.paste(crop, (x0, y0))
            draw.line((x0 + cut, y0, x0 + cut, y0 + shown.height), fill=(0, 209, 178), width=2)
            draw.text((x0 + cut - 34, y0 + 8), "target", font=small_font, fill=(255, 255, 255))
            draw.text((x0 + cut + 10, y0 + 8), "equations", font=small_font, fill=(0, 209, 178))

    draw.text((30, 412), caption, font=small_font, fill=(198, 212, 226))
    if overlay:
        draw.text((30, 438), overlay, font=mono_font, fill=(0, 209, 178))

    if equations:
        for idx, equation in enumerate(equations):
            draw.text((30, 436 + idx * 22), f"{'RGB'[idx]}(x,y) = {equation[:96]}", font=mono_font, fill=(170, 226, 214))
    if metrics:
        text = (
            f"PSNR {metrics.get('psnr', 0):.2f} dB   "
            f"SSIM {metrics.get('ssim', 0):.4f}   "
            f"MSE {metrics.get('mse', 0):.6f}"
        )
        draw.text((30, 508), text, font=small_font, fill=(255, 255, 255))

    draw.rectangle((0, HEIGHT - 3, int(WIDTH * progress), HEIGHT - 1), fill=(0, 209, 178))
    return canvas


def build_frames(target: Image.Image, recon: Image.Image, metrics: dict[str, float], equations: tuple[str, str, str]):
    frames: list[Image.Image] = []
    target_scaled = fit(target, (30, 112, WIDTH - 30, 400))
    recon_scaled = fit(recon, (30, 112, WIDTH - 30, 400))
    diff = np.abs(np.asarray(recon, dtype=np.float64) - np.asarray(target, dtype=np.float64))
    heat = np.clip(diff.mean(axis=-1) * 8.0 * 255.0, 0, 255).astype(np.uint8)
    heat_img = Image.fromarray(heat, mode="L").convert("RGB")

    def add(count: int, **kwargs) -> None:
        for _idx in range(count):
            frames.append(make_frame(progress=len(frames) / TOTAL_FRAMES, **kwargs))

    add(
        int(SEGMENTS[0] * FPS),
        title="From pixels to closed-form equations",
        caption="A deterministic, first-principles reconstruction pipeline - no neural networks, no pretrained weights.",
        image=target_scaled,
        overlay="Stage 1: analyse  >  Stage 2: global expansion  >  Stage 3: quadtree  >  Stage 4: blend  >  Stage 5: verify",
    )
    add(
        int(SEGMENTS[1] * FPS),
        title="Target: a continuous analytic test field",
        caption="320 x 200 RGB image generated from smooth harmonics, a sharp disc and localized highlights.",
        image=target_scaled,
        overlay="Sampled once on a discrete grid; MathField never sees the generating formula.",
    )
    for idx in range(int(SEGMENTS[2] * FPS)):
        alpha = (idx + 1) / int(SEGMENTS[2] * FPS)
        reveal = Image.blend(target_scaled, recon_scaled, alpha)
        frames.append(
            make_frame(
                title="Mode B: fitted continuous field",
                caption="Chebyshev + Fourier expansion with Sobolev-guided local patches and Wendland C2 blending.",
                image=reveal,
                overlay=f"Reconstruction from fitted coefficients  -  reveal {alpha * 100:5.1f}%",
                progress=len(frames) / TOTAL_FRAMES,
            )
        )
    for idx in range(int(SEGMENTS[3] * FPS)):
        split = 0.5 + 0.5 * np.sin(np.pi * idx / max(int(SEGMENTS[3] * FPS) - 1, 1))
        frames.append(
            make_frame(
                title="Raster target  |  continuous reconstruction",
                caption="Curtain comparison: identical coordinate grids, two representations of the same field.",
                image=recon_scaled,
                second_image=target_scaled,
                split=float(split),
                progress=len(frames) / TOTAL_FRAMES,
            )
        )
    for idx in range(int(SEGMENTS[4] * FPS)):
        show_eq = idx >= int(0.45 * SEGMENTS[4] * FPS)
        frames.append(
            make_frame(
                title="Verification and discovered equations",
                caption="Absolute error (8x amplified) and the leading analytical terms fitted to this image.",
                image=heat_img,
                equations=equations if show_eq else None,
                overlay=None if show_eq else "Five independent fidelity metrics computed against the target raster.",
                metrics=None if show_eq else metrics,
                progress=len(frames) / TOTAL_FRAMES,
            )
        )
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

    def channel_equation(ch: int) -> str:
        c_vec = coeffs[:, ch] if coeffs.ndim == 2 else coeffs
        terms: list[tuple[float, str]] = []
        for k, (i, j) in enumerate(cheb.indices):
            value = float(c_vec[k])
            if abs(value) > 1e-3:
                name = "1" if i == j == 0 else f"T{i}(x)" if j == 0 else f"T{j}(y)" if i == 0 else f"T{i}(x)T{j}(y)"
                terms.append((abs(value), f"{value:+.3f} {name}"))
        base = cheb.num_terms
        for m, (kx, ky) in enumerate(four.freq_pairs):
            for value, func in ((float(c_vec[base + 2 * m]), "cos"), (float(c_vec[base + 2 * m + 1]), "sin")):
                if abs(value) > 1e-3:
                    terms.append((abs(value), f"{value:+.3f} {func}({int(kx)}x,{int(ky)}y)"))
        terms.sort(key=lambda item: item[0], reverse=True)
        return " ".join(text for _, text in terms[:5]) + " + ..."

    equations = tuple(channel_equation(index) for index in range(3))  # type: ignore[assignment]

    frames = build_frames(target, recon, metrics, equations)
    make_frame(
        title="MathField - reconstructed by continuous equations",
        caption=f"PSNR {metrics['psnr']:.2f} dB  |  SSIM {metrics['ssim']:.4f}  |  MSE {metrics['mse']:.6f}",
        image=fit(recon, (30, 112, WIDTH - 30, 400)),
        second_image=fit(target, (30, 112, WIDTH - 30, 400)),
        split=0.5,
        progress=1.0,
    ).save(PREVIEW_PATH)
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
