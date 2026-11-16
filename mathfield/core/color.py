"""
mathfield/core/color.py

Color Space Conversions, Gamut Mapping, and Alpha Transparency Support.
Provides:
- sRGB <-> Linear RGB (IEC 61966-2-1)
- RGB <-> Continuous Cartesian HSV (u, v, V) with zero jump singularity
- CIE 1931 CIELAB (delta = 6/29 eliminating infinite dark derivatives)
- Smooth algebraic soft-clipping and hard clipping
- Alpha-weighted reconstruction loss
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Union

import numpy as np


class ColorSpace(str, Enum):
    """Supported color representations."""
    SRGB = "srgb"
    LINEAR_RGB = "linear_rgb"
    HSV = "hsv"
    CIELAB = "cielab"

    @classmethod
    def from_str(cls, val: Union[str, ColorSpace]) -> ColorSpace:
        if isinstance(val, ColorSpace):
            return val
        s = str(val).strip().lower()
        for member in cls:
            if member.value == s or member.name.lower() == s:
                return member
        raise ValueError(f"Unknown ColorSpace: {val}")


@dataclass(frozen=True, slots=True)
class RGB:
    r: float
    g: float
    b: float


@dataclass(frozen=True, slots=True)
class HSV:
    u: float
    v: float
    V: float


@dataclass(frozen=True, slots=True)
class CIELAB:
    L: float
    a: float
    b: float


# CIE 1931 D65 Reference White
_D65_XN = 0.95047
_D65_YN = 1.00000
_D65_ZN = 1.08883

# Linear RGB <-> CIE XYZ (sRGB D65)
_RGB_TO_XYZ_MATRIX = np.array([
    [0.4124564, 0.3575761, 0.1804375],
    [0.2126729, 0.7151522, 0.0721750],
    [0.0193339, 0.1191920, 0.9503041],
], dtype=np.float64)

_XYZ_TO_RGB_MATRIX = np.array([
    [3.2404542, -1.5371385, -0.4985314],
    [-0.9692660, 1.8760108, 0.0415560],
    [0.0556434, -0.2040259, 1.0572252],
], dtype=np.float64)

# CIELAB linear threshold constants (delta = 6/29)
_DELTA = 6.0 / 29.0
_DELTA_CUBED = _DELTA ** 3  # 216 / 24389 ~= 0.00885645
_3_DELTA_SQUARED = 3.0 * (_DELTA ** 2)  # 108 / 841 ~= 0.12841855
_4_29 = 4.0 / 29.0  # 4 / 29 ~= 0.13793103


class ColorSpaceConverter:
    """Mathematical color space conversions and gamut mapping."""

    @staticmethod
    def srgb_to_linear(srgb: np.ndarray) -> np.ndarray:
        """
        Convert sRGB to Linear RGB according to IEC 61966-2-1.
        Preserves signs on linear toe and uses domain protection on power branch.
        """
        arr = np.asarray(srgb, dtype=np.float64)
        linear = np.zeros_like(arr)
        linear_mask = arr <= 0.04045

        # Linear toe: c / 12.92
        linear[linear_mask] = arr[linear_mask] / 12.92

        # Non-linear power curve: ((c + 0.055) / 1.055)^2.4
        non_linear_mask = ~linear_mask
        clipped_val = np.maximum(arr[non_linear_mask], 0.0)
        linear[non_linear_mask] = ((clipped_val + 0.055) / 1.055) ** 2.4

        return linear

    @staticmethod
    def linear_to_srgb(linear: np.ndarray) -> np.ndarray:
        """
        Convert Linear RGB to sRGB according to IEC 61966-2-1.
        Threshold tau = 0.04045 / 12.92 ~= 0.0031308.
        """
        arr = np.asarray(linear, dtype=np.float64)
        srgb = np.zeros_like(arr)
        tau = 0.04045 / 12.92
        linear_mask = arr <= tau

        # Linear toe: 12.92 * c
        srgb[linear_mask] = 12.92 * arr[linear_mask]

        # Non-linear power curve: 1.055 * c^(1 / 2.4) - 0.055
        non_linear_mask = ~linear_mask
        clipped_val = np.maximum(arr[non_linear_mask], 0.0)
        srgb[non_linear_mask] = 1.055 * (clipped_val ** (1.0 / 2.4)) - 0.055

        return srgb

    @staticmethod
    def rgb_to_cartesian_hsv(rgb: np.ndarray) -> np.ndarray:
        """
        Convert RGB [0, 1] to Continuous Cartesian HSV (u, v, V).
        u = S * cos(2 * pi * H)
        v = S * sin(2 * pi * H)
        V = max(R, G, B)
        Eliminates the branch cut jump singularity at H = 0 == 2*pi.
        """
        arr = np.asarray(rgb, dtype=np.float64)
        r = arr[..., 0]
        g = arr[..., 1]
        b = arr[..., 2]

        v = np.maximum(np.maximum(r, g), b)
        c_min = np.minimum(np.minimum(r, g), b)
        delta = v - c_min

        # Saturation S = delta / v (0 if v <= 1e-12)
        s = np.zeros_like(v)
        v_pos = v > 1e-12
        s[v_pos] = delta[v_pos] / v[v_pos]

        # Hue angle theta in radians [0, 2*pi)
        theta = np.zeros_like(v)
        nonzero_delta = delta > 1e-12

        # Mask for R == V
        mask_r = nonzero_delta & (v == r)
        theta[mask_r] = (np.pi / 3.0) * (((g[mask_r] - b[mask_r]) / delta[mask_r]) % 6.0)

        # Mask for G == V
        mask_g = nonzero_delta & (~mask_r) & (v == g)
        theta[mask_g] = (np.pi / 3.0) * (((b[mask_g] - r[mask_g]) / delta[mask_g]) + 2.0)

        # Mask for B == V
        mask_b = nonzero_delta & (~mask_r) & (~mask_g)
        theta[mask_b] = (np.pi / 3.0) * (((r[mask_b] - g[mask_b]) / delta[mask_b]) + 4.0)

        u = s * np.cos(theta)
        w = s * np.sin(theta)

        out = np.stack([u, w, v], axis=-1)
        return out

    @staticmethod
    def cartesian_hsv_to_rgb(uvV: np.ndarray) -> np.ndarray:
        """
        Convert Continuous Cartesian HSV (u, v, V) back to RGB [0, 1].
        """
        arr = np.asarray(uvV, dtype=np.float64)
        u = arr[..., 0]
        w = arr[..., 1]
        v = arr[..., 2]

        s = np.clip(np.sqrt(u * u + w * w), 0.0, 1.0)
        theta = np.arctan2(w, u) % (2.0 * np.pi)

        delta = s * v
        c_min = v - delta

        h_prime = theta / (np.pi / 3.0)  # in [0, 6)
        x_chroma = delta * (1.0 - np.abs((h_prime % 2.0) - 1.0))

        sector = np.floor(h_prime).astype(int) % 6

        r_prime = np.zeros_like(v)
        g_prime = np.zeros_like(v)
        b_prime = np.zeros_like(v)

        # Sector 0: delta, x_chroma, 0
        m0 = sector == 0
        r_prime[m0] = delta[m0]
        g_prime[m0] = x_chroma[m0]

        # Sector 1: x_chroma, delta, 0
        m1 = sector == 1
        r_prime[m1] = x_chroma[m1]
        g_prime[m1] = delta[m1]

        # Sector 2: 0, delta, x_chroma
        m2 = sector == 2
        g_prime[m2] = delta[m2]
        b_prime[m2] = x_chroma[m2]

        # Sector 3: 0, x_chroma, delta
        m3 = sector == 3
        g_prime[m3] = x_chroma[m3]
        b_prime[m3] = delta[m3]

        # Sector 4: x_chroma, 0, delta
        m4 = sector == 4
        r_prime[m4] = x_chroma[m4]
        b_prime[m4] = delta[m4]

        # Sector 5: delta, 0, x_chroma
        m5 = sector == 5
        r_prime[m5] = delta[m5]
        b_prime[m5] = x_chroma[m5]

        r = np.clip(r_prime + c_min, 0.0, 1.0)
        g = np.clip(g_prime + c_min, 0.0, 1.0)
        b = np.clip(b_prime + c_min, 0.0, 1.0)

        return np.stack([r, g, b], axis=-1)

    @staticmethod
    def srgb_to_cielab(srgb: np.ndarray) -> np.ndarray:
        """
        Convert sRGB to CIE 1931 CIELAB (L*, a*, b*) under D65 illuminant.
        Uses linear threshold delta = 6/29 to prevent infinite dark derivatives.
        """
        lin_rgb = ColorSpaceConverter.srgb_to_linear(srgb)
        # Reshape for matrix multiplication
        orig_shape = lin_rgb.shape
        flat_rgb = lin_rgb.reshape(-1, 3)

        # XYZ = flat_rgb @ M^T
        xyz = flat_rgb @ _RGB_TO_XYZ_MATRIX.T

        # Normalize by D65 reference white
        xr = xyz[:, 0] / _D65_XN
        yr = xyz[:, 1] / _D65_YN
        zr = xyz[:, 2] / _D65_ZN

        def f_trans(t: np.ndarray) -> np.ndarray:
            res = np.zeros_like(t)
            mask = t > _DELTA_CUBED
            res[mask] = np.cbrt(t[mask])
            res[~mask] = (t[~mask] / _3_DELTA_SQUARED) + _4_29
            return res

        fx = f_trans(xr)
        fy = f_trans(yr)
        fz = f_trans(zr)

        l_star = 116.0 * fy - 16.0
        a_star = 500.0 * (fx - fy)
        b_star = 200.0 * (fy - fz)

        lab_flat = np.stack([l_star, a_star, b_star], axis=-1)
        return lab_flat.reshape(orig_shape)

    @staticmethod
    def cielab_to_srgb(lab: np.ndarray) -> np.ndarray:
        """
        Convert CIE 1931 CIELAB (L*, a*, b*) back to sRGB [0, 1].
        """
        arr = np.asarray(lab, dtype=np.float64)
        orig_shape = arr.shape
        flat_lab = arr.reshape(-1, 3)

        l_star = flat_lab[:, 0]
        a_star = flat_lab[:, 1]
        b_star = flat_lab[:, 2]

        fy = (l_star + 16.0) / 116.0
        fx = fy + (a_star / 500.0)
        fz = fy - (b_star / 200.0)

        def f_inv(t: np.ndarray) -> np.ndarray:
            res = np.zeros_like(t)
            mask = t > _DELTA
            res[mask] = t[mask] ** 3.0
            res[~mask] = _3_DELTA_SQUARED * (t[~mask] - _4_29)
            return res

        xr = f_inv(fx)
        yr = f_inv(fy)
        zr = f_inv(fz)

        x = xr * _D65_XN
        y = yr * _D65_YN
        z = zr * _D65_ZN

        xyz = np.stack([x, y, z], axis=-1)
        flat_lin = xyz @ _XYZ_TO_RGB_MATRIX.T

        lin_rgb = flat_lin.reshape(orig_shape)
        srgb = ColorSpaceConverter.linear_to_srgb(lin_rgb)
        return np.clip(srgb, 0.0, 1.0)

    @staticmethod
    def soft_clamp(x: np.ndarray) -> np.ndarray:
        """
        Smooth algebraic soft-clipping function mapping (-inf, inf) -> (0, 1).
        sigma_soft(z) = 0.5 * (1 + (z - 0.5) / sqrt(0.25 + (z - 0.5)^2))
        Has strictly positive derivative everywhere, avoiding dead optimization zones.
        """
        arr = np.asarray(x, dtype=np.float64)
        shifted = arr - 0.5
        denom = np.sqrt(0.25 + shifted * shifted)
        return 0.5 * (1.0 + (shifted / denom))

    @staticmethod
    def hard_clamp(x: np.ndarray) -> np.ndarray:
        """Hard clamp to [0.0, 1.0] for raster export."""
        return np.clip(np.asarray(x, dtype=np.float64), 0.0, 1.0)

    @staticmethod
    def alpha_weighted_loss(
        pred: np.ndarray,
        target: np.ndarray,
        alpha: np.ndarray,
        eps: float = 1e-12,
    ) -> float:
        """
        Alpha-weighted reconstruction loss:
        L_alpha = sum(alpha * (pred - target)^2) / (sum(alpha) + eps)
        Completely ignores error on fully transparent pixels (alpha = 0).
        """
        p = np.asarray(pred, dtype=np.float64)
        t = np.asarray(target, dtype=np.float64)
        a = np.asarray(alpha, dtype=np.float64)

        # Normalize alpha from [0, 255] if necessary
        if np.max(a) > 1.0:
            a = a / 255.0

        diff_sq = (p - t) ** 2

        # If alpha has fewer dimensions (e.g. (H, W) vs (H, W, C)), expand it
        if a.ndim < diff_sq.ndim:
            a_expanded = np.expand_dims(a, axis=-1)
            # Broadcast weight
            weight = np.broadcast_to(a_expanded, diff_sq.shape)
        else:
            weight = a

        numerator = np.sum(weight * diff_sq)
        denominator = np.sum(weight) + eps
        return float(numerator / denominator)

    @classmethod
    def convert(
        cls,
        data: np.ndarray,
        src_space: ColorSpace | str,
        dst_space: ColorSpace | str,
    ) -> np.ndarray:
        """Arbitrary color space conversion between supported spaces."""
        src = ColorSpace.from_str(src_space)
        dst = ColorSpace.from_str(dst_space)

        if src == dst:
            return data.copy()

        # Route through sRGB as central hub
        if src == ColorSpace.SRGB:
            srgb = data
        elif src == ColorSpace.LINEAR_RGB:
            srgb = cls.linear_to_srgb(data)
        elif src == ColorSpace.HSV:
            rgb = cls.cartesian_hsv_to_rgb(data)
            srgb = rgb
        elif src == ColorSpace.CIELAB:
            srgb = cls.cielab_to_srgb(data)
        else:
            raise ValueError(f"Unsupported source space: {src}")

        if dst == ColorSpace.SRGB:
            return srgb
        elif dst == ColorSpace.LINEAR_RGB:
            return cls.srgb_to_linear(srgb)
        elif dst == ColorSpace.HSV:
            return cls.rgb_to_cartesian_hsv(srgb)
        elif dst == ColorSpace.CIELAB:
            return cls.srgb_to_cielab(srgb)
        else:
            raise ValueError(f"Unsupported destination space: {dst}")
