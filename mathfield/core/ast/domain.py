"""
Domain protection, singularity prevention, and numerical regularization functions.

Prevents NaN, Inf, and process crashes across CPU and GPU evaluation backends.
"""

from __future__ import annotations

import math
from typing import NamedTuple

import numpy as np


class PoleWarning(NamedTuple):
    """Warning record indicating a potential singularity or domain violation."""

    node_repr: str
    variable: str
    critical_value: float
    description: str


# ==============================================================================
# Vectorized Domain-Protected Operations (NumPy)
# ==============================================================================


def regularized_div(u: np.ndarray, v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    Regularized division: u / (v + copysign(eps, v)).
    Guarantees non-zero denominator while preserving the mathematical sign.
    """
    # np.copysign handles positive and negative zero correctly
    s = np.copysign(eps, v)
    denom = v + s
    return u / denom


def protected_log(x: np.ndarray, eps: float = 1e-15) -> np.ndarray:
    """
    Logarithm regularized for non-positive inputs: log(max(x, eps)).
    """
    return np.log(np.maximum(x, eps))


def protected_exp(x: np.ndarray, min_val: float = -80.0, max_val: float = 80.0) -> np.ndarray:
    """
    Exponential clamped to [-80, 80] to prevent floating-point overflow and subnormal underflow.
    """
    return np.exp(np.clip(x, min_val, max_val))


def protected_sqrt(x: np.ndarray) -> np.ndarray:
    """
    Square root protected against negative arguments: sqrt(max(x, 0.0)).
    """
    return np.sqrt(np.maximum(x, 0.0))


def protected_cbrt(x: np.ndarray) -> np.ndarray:
    """
    Cube root preserving real negative signs: sign(x) * (|x|)^(1/3).
    """
    return np.cbrt(x)


def protected_pow(u: np.ndarray, v: np.ndarray, eps: float = 1e-15) -> np.ndarray:
    """
    Domain-protected power function:
    - Handles negative bases by decomposing into sign parity and magnitude.
    - Clamps exponents to [-80.0, 80.0].
    - Resolves 0^0 to 1.0.
    """
    # Exponent clamped
    v_clamped = np.clip(v, -80.0, 80.0)
    abs_u = np.maximum(np.abs(u), eps)

    # Detect odd integers for negative base sign preservation
    # An exponent is effectively an odd integer if it is close to round(v) and round(v) % 2 != 0
    v_round = np.round(v_clamped)
    is_finite = np.isfinite(v_clamped)
    # Use an absolute tolerance with *zero* relative tolerance so the vector
    # backend makes exactly the same integer-parity decision as the scalar
    # reference in safe_scalar_pow (math.isclose default rel_tol is 1e-9).
    is_integer = is_finite & np.isclose(v_clamped, v_round, atol=1e-7, rtol=0.0)
    v_safe_int = np.nan_to_num(v_round, nan=0.0).astype(np.int64)
    is_odd_int = is_integer & (np.mod(np.abs(v_safe_int), 2) == 1)

    # Default sign factor: +1.0 unless base < 0 and exponent is odd integer
    sign = np.where((u < 0.0) & is_odd_int, -1.0, 1.0)

    # Magnitude computation via exp(v * log(|u|))
    log_abs_u = np.log(abs_u)
    exponent = np.clip(v_clamped * log_abs_u, -80.0, 80.0)
    result = sign * np.exp(exponent)

    # 0^0 special case: return 1.0
    zero_zero = (np.abs(u) < 1e-15) & (np.abs(v) < 1e-15)
    if np.any(zero_zero):
        result = np.where(zero_zero, 1.0, result)

    return result


def protected_sinc(x: np.ndarray) -> np.ndarray:
    """
    Normalized sinc function: sin(pi * x) / (pi * x) with lim x->0 = 1.0.
    """
    return np.sinc(x)


def protected_tan(x: np.ndarray, max_val: float = 1e6) -> np.ndarray:
    """
    Tangent clamped to [-1e6, 1e6] near odd pi/2 poles.
    """
    return np.clip(np.tan(x), -max_val, max_val)


def protected_asin(x: np.ndarray) -> np.ndarray:
    """
    Inverse sine clamped to valid domain [-1.0, 1.0].
    """
    return np.arcsin(np.clip(x, -1.0, 1.0))


def protected_acos(x: np.ndarray) -> np.ndarray:
    """
    Inverse cosine clamped to valid domain [-1.0, 1.0].
    """
    return np.arccos(np.clip(x, -1.0, 1.0))


def rbf_gaussian(r: np.ndarray, eps: np.ndarray) -> np.ndarray:
    """
    Gaussian radial basis function: exp(-(eps * r)^2).
    """
    scaled = eps * r
    return np.exp(-np.clip(scaled * scaled, 0.0, 80.0))


def rbf_multiquadric(r: np.ndarray, eps: np.ndarray) -> np.ndarray:
    """
    Multiquadric radial basis function: sqrt(1 + (eps * r)^2).
    """
    scaled = np.clip(eps * r, -1e80, 1e80)
    sc2 = np.minimum(scaled * scaled, 1e160)
    return np.sqrt(1.0 + sc2)


def rbf_wendland(r: np.ndarray, radius: np.ndarray) -> np.ndarray:
    """
    Wendland C2 compactly supported radial basis function:
    phi(r, R) = (1 - r/R)_+^4 * (4 * r/R + 1) for 0 <= r < R, 0 otherwise.
    """
    # Safe radius
    safe_r_denom = np.maximum(np.abs(radius), 1e-12)
    norm_r = np.maximum(r, 0.0) / safe_r_denom
    cutoff = np.clip(1.0 - norm_r, 0.0, 1.0)
    cutoff4 = cutoff * cutoff * cutoff * cutoff
    return cutoff4 * (4.0 * norm_r + 1.0)


def sigmoid_fn(x: np.ndarray) -> np.ndarray:
    """
    Sigmoid activation function: 1 / (1 + exp(-clamp(x, -80, 80))).
    """
    return 1.0 / (1.0 + np.exp(-np.clip(x, -80.0, 80.0)))


def softplus_fn(x: np.ndarray) -> np.ndarray:
    """
    Numerically stable softplus activation: log(1 + exp(x)).
    log1p(exp(-|x|)) + max(x, 0).
    """
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0.0)


def clamp_fn(x: np.ndarray, min_val: np.ndarray, max_val: np.ndarray) -> np.ndarray:
    """
    Clamps values to [min_val, max_val].
    """
    return np.clip(x, min_val, max_val)


# ==============================================================================
# Scalar Pure-Python Reference Operations
# ==============================================================================


def safe_scalar_div(u: float, v: float, eps: float = 1e-12) -> float:
    if math.isnan(u) or math.isnan(v):
        return float("nan")
    s = math.copysign(eps, v)
    return u / (v + s)


def safe_scalar_log(x: float, eps: float = 1e-15) -> float:
    if math.isnan(x):
        return float("nan")
    return math.log(max(x, eps))


def safe_scalar_exp(x: float) -> float:
    if math.isnan(x):
        return float("nan")
    clamped = max(-80.0, min(80.0, x))
    return math.exp(clamped)


def safe_scalar_sqrt(x: float) -> float:
    if math.isnan(x):
        return float("nan")
    return math.sqrt(max(x, 0.0))


def safe_scalar_cbrt(x: float) -> float:
    if math.isnan(x):
        return float("nan")
    return math.copysign(math.pow(abs(x), 1.0 / 3.0), x)


def safe_scalar_pow(u: float, v: float, eps: float = 1e-15) -> float:
    if math.isnan(u) or math.isnan(v):
        return float("nan")
    if abs(u) < 1e-15 and abs(v) < 1e-15:
        return 1.0
    v_clamped = max(-80.0, min(80.0, v))
    abs_u = max(abs(u), eps)
    v_round = round(v_clamped)
    is_odd_int = math.isclose(v_clamped, v_round, abs_tol=1e-7, rel_tol=0.0) and (int(abs(v_round)) % 2 == 1)
    sign = -1.0 if (u < 0.0 and is_odd_int) else 1.0
    exponent = max(-80.0, min(80.0, v_clamped * math.log(abs_u)))
    return sign * math.exp(exponent)


def safe_scalar_sinc(x: float) -> float:
    if math.isnan(x):
        return float("nan")
    if abs(x) < 1e-12:
        return 1.0
    pi_x = math.pi * x
    return math.sin(pi_x) / pi_x


def safe_scalar_tan(x: float, max_val: float = 1e6) -> float:
    if math.isnan(x):
        return float("nan")
    val = math.tan(x)
    return max(-max_val, min(max_val, val))


def safe_scalar_asin(x: float) -> float:
    if math.isnan(x):
        return float("nan")
    clamped = max(-1.0, min(1.0, x))
    return math.asin(clamped)


def safe_scalar_acos(x: float) -> float:
    if math.isnan(x):
        return float("nan")
    clamped = max(-1.0, min(1.0, x))
    return math.acos(clamped)


def safe_scalar_rbf_wendland(r: float, radius: float = 1.0) -> float:
    if math.isnan(r) or math.isnan(radius):
        return float("nan")
    safe_rad = max(abs(radius), 1e-12)
    norm_r = max(r, 0.0) / safe_rad
    if norm_r >= 1.0:
        return 0.0
    t = 1.0 - norm_r
    return (t**4) * (4.0 * norm_r + 1.0)


def safe_scalar_sigmoid(x: float) -> float:
    if math.isnan(x):
        return float("nan")
    clamped = max(-80.0, min(80.0, x))
    return 1.0 / (1.0 + math.exp(-clamped))


def safe_scalar_softplus(x: float) -> float:
    if math.isnan(x):
        return float("nan")
    ax = abs(x)
    max_x = max(x, 0.0)
    return math.log1p(math.exp(-ax)) + max_x
