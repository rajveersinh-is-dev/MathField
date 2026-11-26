"""
mathfield/core/verify/metrics.py

Pure NumPy Reference Quality Metrics:
- MSE, MAE, L_infinity
- PSNR (with zero-MSE division protection capped at 100.0 dB)
- SSIM (Wang et al. 2004 11x11 circular Gaussian window, sigma=1.5)
- CIEDE2000 (Delta E_00* full standard ISO/CIE 11664-6 formulation)
"""

from __future__ import annotations

import math
from typing import Dict, Optional

import numpy as np
from scipy.ndimage import convolve1d


def _broadcast_weights(weights: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    """Broadcast weights to the full data shape before reduction."""
    w = np.asarray(weights, dtype=np.float64)
    try:
        return np.broadcast_to(w, shape)
    except ValueError:
        expanded = w.reshape(w.shape + (1,) * max(0, len(shape) - w.ndim))
        return np.broadcast_to(expanded, shape)


def compute_mse(
    a: np.ndarray,
    b: np.ndarray,
    weights: Optional[np.ndarray] = None,
) -> float:
    """Mean Squared Error."""
    arr_a = np.asarray(a, dtype=np.float64)
    arr_b = np.asarray(b, dtype=np.float64)
    if arr_a.size == 0 or arr_b.size == 0:
        return 0.0
    diff_sq = (arr_a - arr_b) ** 2
    if weights is not None:
        w = _broadcast_weights(weights, diff_sq.shape)
        return float(np.sum(w * diff_sq) / (np.sum(w) + 1e-12))
    return float(np.mean(diff_sq))


def compute_mae(
    a: np.ndarray,
    b: np.ndarray,
    weights: Optional[np.ndarray] = None,
) -> float:
    """Mean Absolute Error."""
    arr_a = np.asarray(a, dtype=np.float64)
    arr_b = np.asarray(b, dtype=np.float64)
    if arr_a.size == 0 or arr_b.size == 0:
        return 0.0
    diff = np.abs(arr_a - arr_b)
    if weights is not None:
        w = _broadcast_weights(weights, diff.shape)
        return float(np.sum(w * diff) / (np.sum(w) + 1e-12))
    return float(np.mean(diff))


def compute_linf(a: np.ndarray, b: np.ndarray) -> float:
    """Chebyshev L_infinity (maximum absolute error)."""
    arr_a = np.asarray(a, dtype=np.float64)
    arr_b = np.asarray(b, dtype=np.float64)
    if arr_a.size == 0 or arr_b.size == 0:
        return 0.0
    return float(np.max(np.abs(arr_a - arr_b)))


def compute_psnr(
    a: np.ndarray,
    b: np.ndarray,
    max_val: float = 1.0,
) -> float:
    """
    Peak Signal-to-Noise Ratio (PSNR) in dB.
    If MSE < 1e-15, returns 100.0 dB (capped infinity).
    """
    if np.asarray(a).size == 0 or np.asarray(b).size == 0:
        return 100.0
    mse = compute_mse(a, b)
    if mse < 1e-15:
        return 100.0
    return float(20.0 * math.log10(max_val / math.sqrt(mse)))


def _gaussian_kernel_2d(size: int = 11, sigma: float = 1.5) -> np.ndarray:
    """Generate normalized 2D Gaussian kernel."""
    coords = np.arange(size) - (size - 1) / 2.0
    x, y = np.meshgrid(coords, coords)
    kernel = np.exp(-(x * x + y * y) / (2.0 * sigma * sigma))
    return kernel / np.sum(kernel)


def _convolve_2d_reflect(img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """2D spatial convolution with symmetric reflection padding."""
    kh, kw = kernel.shape
    pad_h = kh // 2
    pad_w = kw // 2
    padded = np.pad(img, ((pad_h, pad_h), (pad_w, pad_w)), mode="reflect")

    # Fast convolution using sliding window views
    from numpy.lib.stride_tricks import sliding_window_view

    windows = sliding_window_view(padded, (kh, kw))
    return np.einsum("ijkl,kl->ij", windows, kernel)


def compute_ssim(
    a: np.ndarray,
    b: np.ndarray,
    max_val: float = 1.0,
    window_size: int = 11,
    sigma: float = 1.5,
) -> float:
    """
    Structural Similarity Index Measure (SSIM) according to Wang et al. (2004).
    Uses 11x11 circular Gaussian window (sigma=1.5).
    """
    arr_a = np.asarray(a, dtype=np.float64)
    arr_b = np.asarray(b, dtype=np.float64)

    if arr_a.size == 0 or arr_b.size == 0:
        return 1.0 if arr_a.size == arr_b.size else 0.0

    if arr_a.ndim == 3:
        # Multi-channel mean
        scores = [
            compute_ssim(arr_a[:, :, c], arr_b[:, :, c], max_val, window_size, sigma) for c in range(arr_a.shape[2])
        ]
        return float(np.mean(scores))

    h, w = arr_a.shape
    if h < window_size or w < window_size:
        # For very small test images fallback to global formula
        c1 = (0.01 * max_val) ** 2
        c2 = (0.03 * max_val) ** 2
        mu_a = np.mean(arr_a)
        mu_b = np.mean(arr_b)
        sigma_a_sq = np.var(arr_a)
        sigma_b_sq = np.var(arr_b)
        sigma_ab = np.mean((arr_a - mu_a) * (arr_b - mu_b))
        num = (2.0 * mu_a * mu_b + c1) * (2.0 * sigma_ab + c2)
        den = (mu_a * mu_a + mu_b * mu_b + c1) * (sigma_a_sq + sigma_b_sq + c2)
        return float(num / den)

    c1 = (0.01 * max_val) ** 2
    c2 = (0.03 * max_val) ** 2

    if window_size % 2 == 0:
        # Even windows include an extra half-pixel border in the reference path.
        kernel_2d = _gaussian_kernel_2d(window_size, sigma)

        def smooth(image: np.ndarray) -> np.ndarray:
            return _convolve_2d_reflect(image, kernel_2d)

    else:
        coords = np.arange(window_size) - (window_size - 1) / 2.0
        kernel = np.exp(-(coords * coords) / (2.0 * sigma * sigma))
        kernel /= np.sum(kernel)

        def smooth(image: np.ndarray) -> np.ndarray:
            horizontal = convolve1d(image, kernel, axis=1, mode="mirror")
            return convolve1d(horizontal, kernel, axis=0, mode="mirror")

    mu1 = smooth(arr_a)
    mu2 = smooth(arr_b)

    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = smooth(arr_a * arr_a) - mu1_sq
    sigma2_sq = smooth(arr_b * arr_b) - mu2_sq
    sigma12 = smooth(arr_a * arr_b) - mu1_mu2

    num = (2.0 * mu1_mu2 + c1) * (2.0 * sigma12 + c2)
    den = (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)

    ssim_map = num / den
    return float(np.mean(ssim_map))


def compute_ciede2000(
    lab1: np.ndarray,
    lab2: np.ndarray,
    k_L: float = 1.0,
    k_C: float = 1.0,
    k_H: float = 1.0,
) -> float:
    """
    CIEDE2000 Total Color Difference Delta E_00* (ISO/CIE 11664-6:2014).
    Evaluated in CIELAB (L*, a*, b*) space.
    """
    if np.asarray(lab1).size == 0 or np.asarray(lab2).size == 0:
        return 0.0

    arr1 = np.asarray(lab1, dtype=np.float64).reshape(-1, 3)
    arr2 = np.asarray(lab2, dtype=np.float64).reshape(-1, 3)

    L1, a1, b1 = arr1[:, 0], arr1[:, 1], arr1[:, 2]
    L2, a2, b2 = arr2[:, 0], arr2[:, 1], arr2[:, 2]

    # Chroma
    C1 = np.sqrt(a1 * a1 + b1 * b1)
    C2 = np.sqrt(a2 * a2 + b2 * b2)
    C_bar = 0.5 * (C1 + C2)

    # a' correction
    C_bar7 = C_bar**7.0
    G = 0.5 * (1.0 - np.sqrt(C_bar7 / (C_bar7 + (25.0**7.0))))
    a1_prime = (1.0 + G) * a1
    a2_prime = (1.0 + G) * a2

    C1_prime = np.sqrt(a1_prime * a1_prime + b1 * b1)
    C2_prime = np.sqrt(a2_prime * a2_prime + b2 * b2)

    # Hue angles in degrees [0, 360)
    h1_prime = (np.degrees(np.arctan2(b1, a1_prime))) % 360.0
    h2_prime = (np.degrees(np.arctan2(b2, a2_prime))) % 360.0

    # Delta L', Delta C'
    delta_L_prime = L2 - L1
    delta_C_prime = C2_prime - C1_prime

    # Delta h'
    h_diff = h2_prime - h1_prime
    delta_h_prime = np.zeros_like(h_diff)
    nonzero_c = (C1_prime * C2_prime) > 1e-12

    m1 = nonzero_c & (np.abs(h_diff) <= 180.0)
    delta_h_prime[m1] = h_diff[m1]

    m2 = nonzero_c & (h_diff > 180.0)
    delta_h_prime[m2] = h_diff[m2] - 360.0

    m3 = nonzero_c & (h_diff < -180.0)
    delta_h_prime[m3] = h_diff[m3] + 360.0

    delta_H_prime = 2.0 * np.sqrt(C1_prime * C2_prime) * np.sin(np.radians(delta_h_prime / 2.0))

    # Mean L', C', h'
    L_bar_prime = 0.5 * (L1 + L2)
    C_bar_prime = 0.5 * (C1_prime + C2_prime)

    h_bar_prime = np.zeros_like(h_diff)
    h_sum = h1_prime + h2_prime

    h_m1 = nonzero_c & (np.abs(h_diff) <= 180.0)
    h_bar_prime[h_m1] = 0.5 * h_sum[h_m1]

    h_m2 = nonzero_c & (np.abs(h_diff) > 180.0) & (h_sum < 360.0)
    h_bar_prime[h_m2] = 0.5 * (h_sum[h_m2] + 360.0)

    h_m3 = nonzero_c & (np.abs(h_diff) > 180.0) & (h_sum >= 360.0)
    h_bar_prime[h_m3] = 0.5 * (h_sum[h_m3] - 360.0)

    h_bar_prime[~nonzero_c] = h_sum[~nonzero_c]

    # Weighting factors
    T = (
        1.0
        - 0.17 * np.cos(np.radians(h_bar_prime - 30.0))
        + 0.24 * np.cos(np.radians(2.0 * h_bar_prime))
        + 0.32 * np.cos(np.radians(3.0 * h_bar_prime + 6.0))
        - 0.20 * np.cos(np.radians(4.0 * h_bar_prime - 63.0))
    )

    L_shift_sq = (L_bar_prime - 50.0) ** 2.0
    S_L = 1.0 + (0.015 * L_shift_sq) / np.sqrt(20.0 + L_shift_sq)
    S_C = 1.0 + 0.045 * C_bar_prime
    S_H = 1.0 + 0.015 * C_bar_prime * T

    # Rotation factor R_T
    delta_theta = 30.0 * np.exp(-(((h_bar_prime - 275.0) / 25.0) ** 2.0))
    C_bar_p7 = C_bar_prime**7.0
    R_C = 2.0 * np.sqrt(C_bar_p7 / (C_bar_p7 + (25.0**7.0)))
    R_T = -np.sin(np.radians(2.0 * delta_theta)) * R_C

    term_L = delta_L_prime / (k_L * S_L)
    term_C = delta_C_prime / (k_C * S_C)
    term_H = delta_H_prime / (k_H * S_H)

    de00_sq = term_L**2.0 + term_C**2.0 + term_H**2.0 + R_T * term_C * term_H
    de00 = np.sqrt(np.maximum(de00_sq, 0.0))
    return float(np.mean(de00))


def compute_metrics(
    pred: np.ndarray,
    target: np.ndarray,
) -> Dict[str, float]:
    """
    Compute comprehensive verification metric battery between two arrays.
    """
    p = np.asarray(pred, dtype=np.float64)
    t = np.asarray(target, dtype=np.float64)

    mse_val = compute_mse(p, t)
    mae_val = compute_mae(p, t)
    linf_val = compute_linf(p, t)
    psnr_val = compute_psnr(p, t)
    ssim_val = compute_ssim(p, t)

    return {
        "mse": mse_val,
        "mae": mae_val,
        "linf": linf_val,
        "psnr": psnr_val,
        "ssim": ssim_val,
    }
