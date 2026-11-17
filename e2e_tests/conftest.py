"""
Pytest configuration, global hooks, custom markers, and first-principles
synthetic test data generators for MathField E2E testing track.

All synthetic data is dynamically generated from mathematical first principles
using pure NumPy and Pillow without precalculated constants or lookup tables.
"""

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pytest
from PIL import Image

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def pytest_configure(config: pytest.Config) -> None:
    """Register custom markers and configure execution environment."""
    config.addinivalue_line("markers", "tier1: Tier 1 Feature Coverage tests (Features 1-31)")
    config.addinivalue_line("markers", "tier2: Tier 2 Boundary & Corner Case tests")
    config.addinivalue_line("markers", "tier3: Tier 3 Cross-Feature Interaction tests")
    config.addinivalue_line("markers", "tier4: Tier 4 Real-World Application Scenario tests")
    config.addinivalue_line("markers", "gpu: Tests requiring NVIDIA CUDA acceleration")
    config.addinivalue_line("markers", "slow: Long-running benchmark or high-res tests")

    # Set deterministic threading and unbuffered execution
    os.environ["PYTHONUNBUFFERED"] = "1"
    os.environ["OMP_NUM_THREADS"] = "4"

    # Enforce strict offline execution invariant across all tests
    try:
        from dataclasses import replace

        import pytest_socket

        # 1. Process-wide immediate disable
        pytest_socket.disable_socket()
        # 2. Update pytest-socket plugin configuration stash so plugin setup enforces it per test
        if hasattr(pytest_socket, "_STASH_KEY") and pytest_socket._STASH_KEY in config.stash:
            cfg = config.stash[pytest_socket._STASH_KEY]
            config.stash[pytest_socket._STASH_KEY] = replace(cfg, socket_disabled=True)
    except Exception:
        pass


@pytest.fixture(autouse=True)
def enforce_offline_sockets():
    """Guarantees network sockets are disabled before each test, surviving teardown resets."""
    try:
        import pytest_socket

        pytest_socket.disable_socket()
    except Exception:
        pass
    yield


# ============================================================================
# Directory & Isolation Fixtures
# ============================================================================


@pytest.fixture
def e2e_temp_dir(tmp_path: Path) -> Path:
    """Provides an isolated temporary directory for test outputs."""
    scratch = tmp_path / "scratch"
    scratch.mkdir(parents=True, exist_ok=True)
    return scratch


@pytest.fixture
def tmp_e2e_dir(e2e_temp_dir: Path) -> Path:
    """Alias for e2e_temp_dir for compatibility across specs."""
    return e2e_temp_dir


# ============================================================================
# First-Principles Synthetic Image Generators
# ============================================================================


def _make_grid(width: int, height: int) -> Tuple[np.ndarray, np.ndarray]:
    """Generates 2D coordinate matrices X, Y in [-1.0, 1.0]."""
    x = np.linspace(-1.0, 1.0, width, dtype=np.float64)
    y = np.linspace(-1.0, 1.0, height, dtype=np.float64)
    return np.meshgrid(x, y)


def generate_poly_field(width: int, height: int) -> np.ndarray:
    """
    Synthetic Bivariate Polynomial Surface:
    f(x, y) = 0.5 * x^2 - 0.3 * y^2 + 0.2 * x * y + 0.1
    Returns uint8 array of shape (height, width, 3).
    """
    X, Y = _make_grid(width, height)
    f = 0.5 * (X**2) - 0.3 * (Y**2) + 0.2 * (X * Y) + 0.1
    # Normalize f to [0, 1]
    f_min, f_max = float(np.min(f)), float(np.max(f))
    denom = f_max - f_min if f_max > f_min else 1.0
    norm_f = (f - f_min) / denom

    r = norm_f
    g = 0.5 * (1.0 - norm_f) + 0.5 * np.sin(np.pi * X) ** 2
    b = norm_f**2
    rgb = np.stack([r, g, b], axis=-1)
    return np.clip(rgb * 255.0, 0, 255).astype(np.uint8)


def generate_harmonic_field(width: int, height: int) -> np.ndarray:
    """
    Synthetic Harmonic Wave Interference:
    f(x, y) = sin(3*pi*x)*cos(4*pi*y) + 0.5*cos(6*pi*x)
    Returns uint8 array of shape (height, width, 3).
    """
    X, Y = _make_grid(width, height)
    f1 = np.sin(3.0 * np.pi * X) * np.cos(4.0 * np.pi * Y)
    f2 = 0.5 * np.cos(6.0 * np.pi * X) * np.sin(2.0 * np.pi * Y)
    f3 = 0.25 * np.cos(np.pi * (X + Y))
    f = f1 + f2 + f3

    f_min, f_max = float(np.min(f)), float(np.max(f))
    denom = f_max - f_min if f_max > f_min else 1.0
    norm_f = (f - f_min) / denom

    r = norm_f
    g = np.abs(np.sin(2.0 * np.pi * norm_f))
    b = 1.0 - norm_f
    rgb = np.stack([r, g, b], axis=-1)
    return np.clip(rgb * 255.0, 0, 255).astype(np.uint8)


def generate_rbf_field(width: int, height: int) -> np.ndarray:
    """
    Synthetic Multi-Modal Gaussian RBF Mixture:
    Sum of 4 Gaussian peaks.
    Returns uint8 array of shape (height, width, 3).
    """
    X, Y = _make_grid(width, height)
    peaks = [
        (-0.5, -0.5, 0.35, 1.0),
        (0.5, -0.5, 0.25, 0.8),
        (0.0, 0.5, 0.40, 0.9),
        (0.4, 0.4, 0.20, 0.7),
    ]
    f = np.zeros_like(X, dtype=np.float64)
    for x_c, y_c, sigma, amp in peaks:
        dist_sq = (X - x_c) ** 2 + (Y - y_c) ** 2
        f += amp * np.exp(-dist_sq / (2.0 * (sigma**2)))

    f_max = float(np.max(f)) if float(np.max(f)) > 0 else 1.0
    norm_f = f / f_max
    r = norm_f
    g = norm_f**1.5
    b = np.sin(np.pi * norm_f)
    rgb = np.stack([r, g, b], axis=-1)
    return np.clip(rgb * 255.0, 0, 255).astype(np.uint8)


def generate_checker_field(width: int, height: int) -> np.ndarray:
    """
    Sharp Geometric Discs & Checkerboard:
    Combines high-frequency checkerboard with smooth circular discs.
    Returns uint8 array of shape (height, width, 3).
    """
    X, Y = _make_grid(width, height)
    checker = np.sign(np.sin(6.0 * np.pi * X) * np.sin(6.0 * np.pi * Y))
    checker_norm = (checker + 1.0) * 0.5

    # Center disc
    disc = np.where(X**2 + Y**2 < 0.25, 1.0, 0.0)

    r = 0.7 * checker_norm + 0.3 * disc
    g = 0.3 * checker_norm + 0.7 * disc
    b = disc
    rgb = np.stack([r, g, b], axis=-1)
    return np.clip(rgb * 255.0, 0, 255).astype(np.uint8)


def generate_portrait_field(width: int, height: int) -> np.ndarray:
    """
    Synthetic Portrait:
    Analytical head ellipse, neck, hair boundary, and skin tone gradient.
    Returns uint8 array of shape (height, width, 3).
    """
    X, Y = _make_grid(width, height)
    # Head ellipse centered at (0.0, -0.1)
    head_mask = ((X / 0.45) ** 2 + ((Y + 0.1) / 0.60) ** 2) <= 1.0
    hair_mask = head_mask & (Y < -0.3)
    skin_mask = head_mask & (~hair_mask)
    neck_mask = (np.abs(X) < 0.22) & (Y >= 0.45) & (Y <= 0.85)

    r = np.zeros_like(X)
    g = np.zeros_like(X)
    b = np.zeros_like(X)

    # Background: soft cool gradient
    r += 0.2 * (1.0 - Y)
    g += 0.3 * (1.0 - Y)
    b += 0.5 * (1.0 - Y)

    # Skin tones: warm peach gradient
    skin_val_r = 0.90 + 0.08 * X
    skin_val_g = 0.72 + 0.05 * Y
    skin_val_b = 0.60 + 0.04 * (X + Y)

    r = np.where(skin_mask | neck_mask, skin_val_r, r)
    g = np.where(skin_mask | neck_mask, skin_val_g, g)
    b = np.where(skin_mask | neck_mask, skin_val_b, b)

    # Hair: dark brown/black
    r = np.where(hair_mask, 0.15, r)
    g = np.where(hair_mask, 0.10, g)
    b = np.where(hair_mask, 0.08, b)

    rgb = np.stack([r, g, b], axis=-1)
    return np.clip(rgb * 255.0, 0, 255).astype(np.uint8)


def generate_landscape_field(width: int, height: int) -> np.ndarray:
    """
    Synthetic Landscape:
    Sky Rayleigh scattering gradient, mountain ridge horizon, and textured ground.
    Returns uint8 array of shape (height, width, 3).
    """
    X, Y = _make_grid(width, height)
    # Mountain ridge horizon function
    horizon = 0.15 * np.sin(2.0 * np.pi * X) + 0.08 * np.cos(5.0 * np.pi * X)
    sky_mask = horizon > Y
    ground_mask = ~sky_mask

    r = np.zeros_like(X)
    g = np.zeros_like(X)
    b = np.zeros_like(X)

    # Sky: Rayleigh gradient from deep blue at top to sunset orange near horizon
    sky_t = (Y + 1.0) / 1.15
    sky_t = np.clip(sky_t, 0.0, 1.0)
    r = np.where(sky_mask, 0.3 + 0.6 * sky_t, r)
    g = np.where(sky_mask, 0.5 + 0.3 * sky_t, g)
    b = np.where(sky_mask, 0.9 - 0.2 * sky_t, b)

    # Ground: earth tones and terrain slope
    ground_t = (Y - horizon) / (1.0 - horizon + 1e-6)
    ground_t = np.clip(ground_t, 0.0, 1.0)
    r = np.where(ground_mask, 0.2 + 0.2 * ground_t, r)
    g = np.where(ground_mask, 0.4 + 0.1 * ground_t, g)
    b = np.where(ground_mask, 0.1 + 0.1 * ground_t, b)

    rgb = np.stack([r, g, b], axis=-1)
    return np.clip(rgb * 255.0, 0, 255).astype(np.uint8)


def generate_typography_field(width: int, height: int) -> np.ndarray:
    """
    Synthetic Typography & High-Contrast Geometric Glyphs:
    Sharp binary edges resembling letters/symbols.
    Returns uint8 array of shape (height, width, 3).
    """
    X, Y = _make_grid(width, height)
    # Background: clean light gray
    bg = 0.95
    glyph = np.zeros_like(X, dtype=bool)

    # Vertical bar for letter 'M' left
    glyph |= (np.abs(X + 0.5) < 0.05) & (np.abs(Y) < 0.6)
    # Vertical bar for letter 'M' right
    glyph |= (np.abs(X - 0.5) < 0.05) & (np.abs(Y) < 0.6)
    # Diagonal strokes
    diag1 = np.abs((Y - X) + 0.0) < 0.06
    diag2 = np.abs((Y + X) + 0.0) < 0.06
    glyph |= (diag1 | diag2) & (Y > -0.1) & (Y < 0.6)

    # Integral curve symbol: y = sin(pi * x)
    integral_stroke = np.abs(Y - np.sin(np.pi * X)) < 0.04
    glyph |= integral_stroke & (np.abs(X) < 0.8)

    r = np.where(glyph, 0.05, bg)
    g = np.where(glyph, 0.15, bg)
    b = np.where(glyph, 0.40, bg)

    rgb = np.stack([r, g, b], axis=-1)
    return np.clip(rgb * 255.0, 0, 255).astype(np.uint8)


def generate_scalar_field(width: int, height: int) -> np.ndarray:
    """
    Scientific 2D Scalar Field:
    Fluid vorticity dipole w(x,y) = exp(-((x-0.3)^2+y^2)/0.1) - exp(-((x+0.3)^2+y^2)/0.1).
    Mapped to pseudo-viridis colormap.
    Returns uint8 array of shape (height, width, 3).
    """
    X, Y = _make_grid(width, height)
    vortex_pos = np.exp(-((X - 0.3) ** 2 + Y**2) / 0.10)
    vortex_neg = np.exp(-((X + 0.3) ** 2 + Y**2) / 0.10)
    w = vortex_pos - vortex_neg

    # Normalize to [0, 1]
    norm_w = (w + 1.0) * 0.5
    # Viridis-like analytical colormap approximation
    r = np.clip(np.sin(np.pi * norm_w - 0.2), 0.0, 1.0)
    g = np.clip(norm_w**0.8, 0.0, 1.0)
    b = np.clip(np.cos(np.pi * norm_w * 0.5), 0.0, 1.0)

    rgb = np.stack([r, g, b], axis=-1)
    return np.clip(rgb * 255.0, 0, 255).astype(np.uint8)


def generate_rgba_alpha_mask(width: int, height: int) -> np.ndarray:
    """
    Synthetic 4-Channel RGBA Image:
    Smooth trigonometric color gradient with circular alpha mask.
    Returns uint8 array of shape (height, width, 4).
    """
    X, Y = _make_grid(width, height)
    r = 0.5 * (1.0 + np.sin(np.pi * X))
    g = 0.5 * (1.0 + np.cos(np.pi * Y))
    b = 0.5 * (1.0 + np.sin(np.pi * (X + Y)))

    # Circular mask: opaque in center, transparent outside radius 0.707
    radius_sq = X**2 + Y**2
    alpha = np.where(radius_sq <= 0.50, 1.0, 0.0)
    # Smooth edge rolloff
    edge = (radius_sq > 0.40) & (radius_sq <= 0.50)
    alpha[edge] = (0.50 - radius_sq[edge]) / 0.10

    rgba = np.stack([r, g, b, alpha], axis=-1)
    return np.clip(rgba * 255.0, 0, 255).astype(np.uint8)


def generate_continuous_hsv(width: int, height: int) -> np.ndarray:
    """
    Continuous Cartesian HSV Color Wheel:
    Hue = (atan2(y, x) + pi) / (2 * pi), Saturation = radius, Value = 1.0.
    Dynamically converted to RGB via analytical formula.
    Returns uint8 array of shape (height, width, 3).
    """
    X, Y = _make_grid(width, height)
    hue = (np.arctan2(Y, X) + np.pi) / (2.0 * np.pi)  # in [0, 1]
    radius = np.sqrt(X**2 + Y**2)
    sat = np.clip(radius, 0.0, 1.0)
    val = np.ones_like(X)

    # Pure NumPy HSV to RGB conversion
    h_i = (hue * 6.0).astype(int) % 6
    f = (hue * 6.0) - (hue * 6.0).astype(int)
    p = val * (1.0 - sat)
    q = val * (1.0 - f * sat)
    t = val * (1.0 - (1.0 - f) * sat)

    r = np.zeros_like(X)
    g = np.zeros_like(X)
    b = np.zeros_like(X)

    idx0 = h_i == 0
    r[idx0], g[idx0], b[idx0] = val[idx0], t[idx0], p[idx0]
    idx1 = h_i == 1
    r[idx1], g[idx1], b[idx1] = q[idx1], val[idx1], p[idx1]
    idx2 = h_i == 2
    r[idx2], g[idx2], b[idx2] = p[idx2], val[idx2], t[idx2]
    idx3 = h_i == 3
    r[idx3], g[idx3], b[idx3] = p[idx3], q[idx3], val[idx3]
    idx4 = h_i == 4
    r[idx4], g[idx4], b[idx4] = t[idx4], p[idx4], val[idx4]
    idx5 = h_i == 5
    r[idx5], g[idx5], b[idx5] = val[idx5], p[idx5], q[idx5]

    rgb = np.stack([r, g, b], axis=-1)
    return np.clip(rgb * 255.0, 0, 255).astype(np.uint8)


def generate_cielab_field(width: int, height: int) -> np.ndarray:
    """
    Rich CIELAB Gamut Image:
    L* in [20, 90], a* in [-60, 60], b* in [-60, 60].
    Dynamically converted to RGB via analytical standard formulas.
    Returns uint8 array of shape (height, width, 3).
    """
    X, Y = _make_grid(width, height)
    # L* varies vertically
    L = 20.0 + 70.0 * ((Y + 1.0) * 0.5)
    # a* and b* vary horizontally and diagonally
    a = 60.0 * X
    b = 60.0 * np.sin(np.pi * Y)

    # Standard analytical Lab -> XYZ -> sRGB conversion
    fy = (L + 16.0) / 116.0
    fx = a / 500.0 + fy
    fz = fy - b / 200.0

    delta = 6.0 / 29.0

    def f_inv(t: np.ndarray) -> np.ndarray:
        return np.where(t > delta, t**3, 3.0 * (delta**2) * (t - 4.0 / 29.0))

    # Standard illuminant D65 reference white
    X_n, Y_n, Z_n = 0.95047, 1.00000, 1.08883
    X_val = X_n * f_inv(fx)
    Y_val = Y_n * f_inv(fy)
    Z_val = Z_n * f_inv(fz)

    # XYZ to linear sRGB matrix
    r_lin = 3.2406 * X_val - 1.5372 * Y_val - 0.4986 * Z_val
    g_lin = -0.9689 * X_val + 1.8758 * Y_val + 0.0415 * Z_val
    b_lin = 0.0557 * X_val - 0.2040 * Y_val + 1.0570 * Z_val

    # Gamma companding
    def srgb_compand(c: np.ndarray) -> np.ndarray:
        c_clipped = np.clip(c, 0.0, 1.0)
        return np.where(c_clipped <= 0.0031308, 12.92 * c_clipped, 1.055 * (c_clipped ** (1.0 / 2.4)) - 0.055)

    r = srgb_compand(r_lin)
    g = srgb_compand(g_lin)
    b = srgb_compand(b_lin)
    rgb = np.stack([r, g, b], axis=-1)
    return np.clip(rgb * 255.0, 0, 255).astype(np.uint8)


def create_test_image(
    path: Path, kind: str = "poly", width: int = 64, height: int = 64, color_space: str = "RGB"
) -> Path:
    """Creates and saves a synthetic test image to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if color_space == "RGBA" or kind == "rgba":
        arr = generate_rgba_alpha_mask(width, height)
        img = Image.fromarray(arr, mode="RGBA")
    elif kind == "poly":
        arr = generate_poly_field(width, height)
        img = Image.fromarray(arr, mode="RGB")
    elif kind == "harmonic":
        arr = generate_harmonic_field(width, height)
        img = Image.fromarray(arr, mode="RGB")
    elif kind == "rbf":
        arr = generate_rbf_field(width, height)
        img = Image.fromarray(arr, mode="RGB")
    elif kind == "checker":
        arr = generate_checker_field(width, height)
        img = Image.fromarray(arr, mode="RGB")
    elif kind == "portrait":
        arr = generate_portrait_field(width, height)
        img = Image.fromarray(arr, mode="RGB")
    elif kind == "landscape":
        arr = generate_landscape_field(width, height)
        img = Image.fromarray(arr, mode="RGB")
    elif kind == "typography":
        arr = generate_typography_field(width, height)
        img = Image.fromarray(arr, mode="RGB")
    elif kind == "scalar_field":
        arr = generate_scalar_field(width, height)
        img = Image.fromarray(arr, mode="RGB")
    elif kind == "hsv" or color_space == "HSV":
        arr = generate_continuous_hsv(width, height)
        img = Image.fromarray(arr, mode="RGB")
    elif kind == "cielab" or color_space == "CIELAB":
        arr = generate_cielab_field(width, height)
        img = Image.fromarray(arr, mode="RGB")
    elif kind == "black":
        arr = np.zeros((height, width, 3), dtype=np.uint8)
        img = Image.fromarray(arr, mode="RGB")
    elif kind == "white":
        arr = np.full((height, width, 3), 255, dtype=np.uint8)
        img = Image.fromarray(arr, mode="RGB")
    else:
        arr = generate_poly_field(width, height)
        img = Image.fromarray(arr, mode="RGB")

    img.save(str(path))
    return path


# ============================================================================
# Test Image Fixtures
# ============================================================================


@pytest.fixture
def synth_rgb_poly(e2e_temp_dir: Path) -> Path:
    """Generates a 64x64 synthetic RGB polynomial test image."""
    path = e2e_temp_dir / "synth_poly.png"
    return create_test_image(path, kind="poly", width=64, height=64, color_space="RGB")


@pytest.fixture
def synth_harmonic(e2e_temp_dir: Path) -> Path:
    """Generates a 64x64 synthetic harmonic wave interference test image."""
    path = e2e_temp_dir / "synth_harmonic.png"
    return create_test_image(path, kind="harmonic", width=64, height=64, color_space="RGB")


@pytest.fixture
def synth_rbf(e2e_temp_dir: Path) -> Path:
    """Generates a 64x64 synthetic multi-modal RBF test image."""
    path = e2e_temp_dir / "synth_rbf.png"
    return create_test_image(path, kind="rbf", width=64, height=64, color_space="RGB")


@pytest.fixture
def synth_checker(e2e_temp_dir: Path) -> Path:
    """Generates a 64x64 synthetic checkerboard test image."""
    path = e2e_temp_dir / "synth_checker.png"
    return create_test_image(path, kind="checker", width=64, height=64, color_space="RGB")


@pytest.fixture
def synth_portrait(e2e_temp_dir: Path) -> Path:
    """Generates a 64x64 synthetic portrait test image."""
    path = e2e_temp_dir / "synth_portrait.png"
    return create_test_image(path, kind="portrait", width=64, height=64, color_space="RGB")


@pytest.fixture
def synth_landscape(e2e_temp_dir: Path) -> Path:
    """Generates a 64x64 synthetic landscape test image."""
    path = e2e_temp_dir / "synth_landscape.png"
    return create_test_image(path, kind="landscape", width=64, height=64, color_space="RGB")


@pytest.fixture
def synth_typography(e2e_temp_dir: Path) -> Path:
    """Generates a 64x64 synthetic typography test image."""
    path = e2e_temp_dir / "synth_typography.png"
    return create_test_image(path, kind="typography", width=64, height=64, color_space="RGB")


@pytest.fixture
def synth_scalar_field(e2e_temp_dir: Path) -> Path:
    """Generates a 64x64 synthetic scalar field test image."""
    path = e2e_temp_dir / "synth_scalar_field.png"
    return create_test_image(path, kind="scalar_field", width=64, height=64, color_space="RGB")


@pytest.fixture
def synth_rgba_image(e2e_temp_dir: Path) -> Path:
    """Generates a 64x64 synthetic RGBA image with alpha circular mask."""
    path = e2e_temp_dir / "synth_rgba.png"
    return create_test_image(path, kind="rgba", width=64, height=64, color_space="RGBA")


@pytest.fixture
def synth_hsv_image(e2e_temp_dir: Path) -> Path:
    """Generates a 64x64 synthetic continuous HSV color wheel test image."""
    path = e2e_temp_dir / "synth_hsv.png"
    return create_test_image(path, kind="hsv", width=64, height=64, color_space="HSV")


@pytest.fixture
def synth_cielab_image(e2e_temp_dir: Path) -> Path:
    """Generates a 64x64 synthetic CIELAB gamut test image."""
    path = e2e_temp_dir / "synth_cielab.png"
    return create_test_image(path, kind="cielab", width=64, height=64, color_space="CIELAB")


@pytest.fixture
def synth_tiny_image(e2e_temp_dir: Path) -> Path:
    """Generates a 2x2 boundary case image."""
    path = e2e_temp_dir / "synth_tiny.png"
    return create_test_image(path, kind="checker", width=2, height=2, color_space="RGB")


# Cross-specification fixture aliases
@pytest.fixture
def synthetic_image_rgb(synth_rgb_poly: Path) -> str:
    """Returns str path to synthetic RGB image."""
    return str(synth_rgb_poly)


@pytest.fixture
def synthetic_image_rgba(synth_rgba_image: Path) -> str:
    """Returns str path to synthetic RGBA image."""
    return str(synth_rgba_image)


@pytest.fixture
def synthetic_image_hsv(synth_hsv_image: Path) -> str:
    """Returns str path to synthetic HSV image."""
    return str(synth_hsv_image)


@pytest.fixture
def synthetic_image_cielab(synth_cielab_image: Path) -> str:
    """Returns str path to synthetic CIELAB image."""
    return str(synth_cielab_image)


@pytest.fixture
def synthetic_image_extreme(e2e_temp_dir: Path) -> Dict[str, str]:
    """Generates degenerate boundary case images."""
    extreme_dir = e2e_temp_dir / "extreme"
    extreme_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        "all_black": str(create_test_image(extreme_dir / "all_black.png", kind="black", width=32, height=32)),
        "all_white": str(create_test_image(extreme_dir / "all_white.png", kind="white", width=32, height=32)),
        "single_pixel": str(create_test_image(extreme_dir / "single_pixel.png", kind="poly", width=1, height=1)),
        "wide_aspect": str(create_test_image(extreme_dir / "wide_aspect.png", kind="poly", width=100, height=10)),
        "tall_aspect": str(create_test_image(extreme_dir / "tall_aspect.png", kind="poly", width=10, height=100)),
    }
    return paths


# ============================================================================
# Metadata, Expression & State Fixtures
# ============================================================================


@pytest.fixture(scope="session")
def mock_8k_metadata() -> Dict[str, Any]:
    """Provides metadata describing an 8K (8192x8192) image for memory-aware tiling tests."""
    return {
        "width": 8192,
        "height": 8192,
        "aspect_ratio": 1.0,
        "channels": 3,
        "tile_size": 1024,
        "expected_tiles": 64,  # 8x8 grid
        "domain": {"x_min": -1.0, "x_max": 1.0, "y_min": -1.0, "y_max": 1.0},
        "bytes_per_pixel": 3,
        "total_uncompressed_bytes": 8192 * 8192 * 3,  # ~192 MB raw RGB
    }


@pytest.fixture
def mock_8k_image_path(e2e_temp_dir: Path, mock_8k_metadata: Dict[str, Any]) -> Path:
    """
    Creates a sparse/memory-mapped dummy 8K file for tiling and stream tests.
    Writes valid PNG header with 8192x8192 IHDR dimensions.
    """
    path = e2e_temp_dir / "synth_8k.png"
    # To keep disk footprint minimal, generate a tiled image using Pillow
    img = Image.new("RGB", (8192, 8192), color=(128, 128, 128))
    img.save(str(path))
    return path


@pytest.fixture
def sample_ast_expressions() -> Dict[str, Tuple[str, np.ndarray]]:
    """Returns sample valid AST expression strings with associated parameter arrays."""
    return {
        "polynomial": (
            "c0 + c1 * x + c2 * y + c3 * x^2 + c4 * y^2",
            np.array([0.5, 0.2, -0.1, 0.3, -0.2], dtype=np.float64),
        ),
        "fourier": ("c0 * sin(pi * x) * cos(pi * y) + c1 * cos(2 * pi * x)", np.array([0.8, -0.4], dtype=np.float64)),
        "rbf": ("c0 * exp(-((x - c1)^2 + (y - c2)^2) / (2 * c3^2))", np.array([1.0, 0.0, 0.0, 0.5], dtype=np.float64)),
        "rational": ("(c0 + c1 * x) / (1.0 + c2 * x^2 + c3 * y^2)", np.array([1.0, 0.5, 0.2, 0.2], dtype=np.float64)),
        "regularized_div": ("x / (y + 1e-12 * sign(y))", np.array([], dtype=np.float64)),
    }


@pytest.fixture
def sample_mathfield_project(e2e_temp_dir: Path) -> Path:
    """Generates a valid Draft-07 compliant .mathfield JSON project file."""
    path = e2e_temp_dir / "project.mathfield"
    data = {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "schema_version": "1.0.0",
        "project_name": "TestE2EProject",
        "coordinate_domain": {
            "x_min": -1.0,
            "x_max": 1.0,
            "y_min": -1.0,
            "y_max": 1.0,
            "pixel_width": 64,
            "pixel_height": 64,
            "aspect_ratio": 1.0,
        },
        "color_space": "RGB",
        "mode": "MODE_A_DISCOVERY",
        "global_model": {
            "channels": [
                {
                    "channel": "R",
                    "formula": "0.5 * x^2 + 0.1",
                    "parameters": [
                        {"name": "c0", "value": 0.5, "hex": (0.5).hex()},
                        {"name": "c1", "value": 0.1, "hex": (0.1).hex()},
                    ],
                },
                {
                    "channel": "G",
                    "formula": "0.4 * y^2 + 0.2",
                    "parameters": [
                        {"name": "c0", "value": 0.4, "hex": (0.4).hex()},
                        {"name": "c1", "value": 0.2, "hex": (0.2).hex()},
                    ],
                },
                {
                    "channel": "B",
                    "formula": "0.3 * x * y + 0.3",
                    "parameters": [
                        {"name": "c0", "value": 0.3, "hex": (0.3).hex()},
                        {"name": "c1", "value": 0.3, "hex": (0.3).hex()},
                    ],
                },
            ]
        },
        "local_patches": [],
        "final_metrics": {"mse": 0.0012, "psnr": 29.21, "ssim": 0.884, "complexity": 6.0},
        "random_seed": 42,
    }
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return path


@pytest.fixture
def sample_project_json(sample_mathfield_project: Path) -> Path:
    """Alias for sample_mathfield_project."""
    return sample_mathfield_project


@pytest.fixture
def sample_checkpoint_file(e2e_temp_dir: Path) -> Path:
    """
    Generates a valid .mfchk serialized checkpoint state file with IEEE-754 hex float strings.
    """
    path = e2e_temp_dir / "checkpoint.mfchk"
    data = {
        "schema_version": "1.0.0",
        "stage": "STAGE_4_OPTIMIZATION",
        "iteration": 25,
        "max_iter": 100,
        "loss": 0.003456,
        "mode": "discovery",
        "coordinate_domain": {"x_min": -1.0, "x_max": 1.0, "y_min": -1.0, "y_max": 1.0},
        "expressions": {"R": "c0 * sin(pi * x) + c1 * y", "G": "c2 * cos(pi * y) + c3", "B": "c4 * x^2 + c5"},
        "parameters": {
            "c0": {"float": 0.825, "hex": (0.825).hex()},
            "c1": {"float": -0.3333333333333333, "hex": float(-1.0 / 3.0).hex()},
            "c2": {"float": 0.65, "hex": (0.65).hex()},
            "c3": {"float": 0.12, "hex": (0.12).hex()},
            "c4": {"float": 0.45, "hex": (0.45).hex()},
            "c5": {"float": 0.05, "hex": (0.05).hex()},
        },
        "metrics": {"mse": 0.003456, "psnr": 24.61, "ssim": 0.795},
        "rng_state": {
            "bit_generator": "PCG64",
            "state": {"state": 123456789, "inc": 987654321},
            "has_uint32": 0,
            "uinteger": 0,
        },
    }
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return path
