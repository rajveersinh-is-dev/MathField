#!/usr/bin/env python3
"""Generate the deterministic sample image used by the MathField quickstart.

The image is built entirely from closed-form continuous functions, so it is a
fair target for MathField: there is no bitmap trickery, precomputed lookup
table or learned prior involved. It combines smooth low-frequency bands with
sharp edges and localized detail, which exercises both the global basis and
the adaptive local patches.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

OUTPUT = Path(__file__).with_name("sample_spectrum.png")
WIDTH, HEIGHT = 320, 200


def build_sample(width: int = WIDTH, height: int = HEIGHT) -> np.ndarray:
    """Return a float RGB image in [0, 1] built from analytic fields."""
    x = np.linspace(-1.0, 1.0, width, dtype=np.float64)
    y = np.linspace(-1.0, 1.0, height, dtype=np.float64)
    X, Y = np.meshgrid(x, y)

    r = np.sqrt(X**2 + Y**2)
    angle = np.arctan2(Y, X)

    red = 0.50 + 0.30 * np.sin(4.0 * X + 1.2) * np.cos(3.0 * Y)
    green = 0.48 + 0.28 * np.sin(3.0 * (X + Y)) * np.exp(-0.9 * r)
    blue = 0.45 + 0.35 * np.cos(11.0 * r - 2.0 * angle) * np.exp(-0.6 * r)

    # A crisp disc and two compact Gaussian highlights give the quadtree real
    # discontinuities to localize rather than a purely smooth target.
    disc = (X - 0.42) ** 2 + (Y + 0.18) ** 2 < 0.11**2
    for channel, value in ((red, 0.95), (green, 0.86), (blue, 0.55)):
        channel[disc] = value

    highlights = np.exp(-28.0 * ((X + 0.45) ** 2 + (Y - 0.35) ** 2))
    red += 0.35 * highlights
    green += 0.40 * highlights

    return np.clip(np.stack([red, green, blue], axis=-1), 0.0, 1.0)


def main() -> None:
    arr = build_sample()
    Image.fromarray((arr * 255.0 + 0.5).astype(np.uint8)).save(OUTPUT)
    print(f"Wrote {OUTPUT} ({WIDTH}x{HEIGHT})")


if __name__ == "__main__":
    main()
