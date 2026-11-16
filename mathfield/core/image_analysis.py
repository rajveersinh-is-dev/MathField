"""
mathfield/core/image_analysis.py

Stage 1: Image Analysis and Pre-characterization.
Inspects input rasters (up to 8K resolution):
- Geometry: width, height, aspect ratio
- Channels and alpha channel detection
- Spatial complexity: Total Variation (TV) and Shannon Information Entropy
- Frequency analysis: 2D FFT power spectrum, dominant frequencies, and radial roll-off alpha
- Inter-channel covariance and Pearson correlation matrix with zero-variance protection
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image


@dataclass(frozen=True, slots=True)
class ImageAnalysisResult:
    """Quantitative statistical characterization of an input raster."""
    width: int
    height: int
    aspect_ratio: float
    channels: int
    has_alpha: bool
    total_variation: float
    entropy: float
    spectral_rolloff: float
    dominant_frequencies: List[Tuple[float, float]]
    channel_correlations: np.ndarray

    def to_dict(self) -> Dict[str, Any]:
        return {
            "width": self.width,
            "height": self.height,
            "aspect_ratio": self.aspect_ratio,
            "channels": self.channels,
            "has_alpha": self.has_alpha,
            "total_variation": float(self.total_variation),
            "entropy": float(self.entropy),
            "spectral_rolloff": float(self.spectral_rolloff),
            "dominant_frequencies": self.dominant_frequencies,
            "channel_correlations": self.channel_correlations.tolist(),
        }


class ImageAnalyzer:
    """Analyzes image properties to configure downstream reconstruction stages."""

    def __init__(self, max_dimension: int = 8192) -> None:
        self.max_dimension = max_dimension

    def analyze(
        self,
        image_input: Union[str, Path, np.ndarray, Image.Image],
    ) -> ImageAnalysisResult:
        """
        Execute full Stage 1 image analysis pipeline.
        Accepts filepath, PIL Image, or NumPy array.
        """
        arr, has_alpha = self._load_and_validate(image_input)
        h, w = arr.shape[:2]
        c = arr.shape[2] if arr.ndim == 3 else 1
        aspect = float(w) / float(h)

        tv = self.compute_total_variation(arr)
        entropy = self.compute_shannon_entropy(arr)
        rolloff, dom_freqs = self.compute_spectral_rolloff(arr)
        corrs = self.compute_channel_correlations(arr)

        return ImageAnalysisResult(
            width=w,
            height=h,
            aspect_ratio=aspect,
            channels=c,
            has_alpha=has_alpha,
            total_variation=tv,
            entropy=entropy,
            spectral_rolloff=rolloff,
            dominant_frequencies=dom_freqs,
            channel_correlations=corrs,
        )

    def _load_and_validate(
        self,
        img: Union[str, Path, np.ndarray, Image.Image],
    ) -> Tuple[np.ndarray, bool]:
        """Convert input to normalized float/uint8 ndarray and validate dimensions."""
        has_alpha = False

        if isinstance(img, (str, Path)):
            path = Path(img)
            if not path.is_file():
                raise ValueError(f"Image file does not exist: {path}")
            # Let PIL raise UnidentifiedImageError or OSError on corrupt file
            with Image.open(path) as pil_img:
                pil_img.load()  # Force decode to detect corrupted data
                mode = pil_img.mode
                has_alpha = mode in ("RGBA", "LA", "PA")
                arr = np.array(pil_img)
        elif isinstance(img, Image.Image):
            mode = img.mode
            has_alpha = mode in ("RGBA", "LA", "PA")
            arr = np.array(img)
        elif isinstance(img, np.ndarray):
            arr = img
            if arr.ndim == 3 and arr.shape[2] == 4:
                has_alpha = True
        else:
            raise TypeError(f"Unsupported image input type: {type(img)}")

        if arr.ndim == 2:
            pass  # (H, W) grayscale
        elif arr.ndim == 3:
            if arr.shape[2] not in (1, 3, 4):
                raise ValueError(f"Unsupported number of image channels: {arr.shape[2]}")
        else:
            raise ValueError(f"Invalid array dimensionality: {arr.ndim}")

        h, w = arr.shape[:2]
        if w <= 0 or h <= 0:
            raise ValueError(f"Image dimensions must be positive, got {w}x{h}")
        if w > self.max_dimension or h > self.max_dimension:
            raise ValueError(f"Image exceeds maximum supported resolution {self.max_dimension}: {w}x{h}")

        return arr, has_alpha

    @staticmethod
    def compute_total_variation(image: np.ndarray, isotropic: bool = True) -> float:
        """
        Compute normalized spatial Total Variation across all color channels.
        Returns 0.0 for perfectly flat / constant images.
        """
        arr = np.asarray(image, dtype=np.float64)
        if arr.max() > 1.0:
            arr = arr / 255.0

        if arr.ndim == 2:
            arr = arr[:, :, np.newaxis]

        h, w, c = arr.shape
        if h <= 1 and w <= 1:
            return 0.0

        tv_sum = 0.0
        for ch in range(c):
            channel = arr[:, :, ch]
            dx = channel[:, 1:] - channel[:, :-1]
            dy = channel[1:, :] - channel[:-1, :]

            if isotropic:
                # Aligned grid TV: sqrt(dx^2 + dy^2) on overlapping region
                dx_sub = dx[:-1, :]
                dy_sub = dy[:, :-1]
                tv_sum += float(np.sum(np.sqrt(dx_sub * dx_sub + dy_sub * dy_sub)))
            else:
                tv_sum += float(np.sum(np.abs(dx)) + np.sum(np.abs(dy)))

        norm_factor = float(h * w * c)
        return float(tv_sum / norm_factor)

    @staticmethod
    def compute_shannon_entropy(image: np.ndarray) -> float:
        """
        Compute Shannon Information Entropy (in bits) over 256 quantization levels.
        Returns 0.0 for constant images.
        """
        arr = np.asarray(image)
        if np.issubdtype(arr.dtype, np.floating):
            # Scale to [0, 255] integer bins
            arr_norm = np.clip(arr, 0.0, 1.0) if arr.max() <= 1.0 else np.clip(arr, 0.0, 255.0) / 255.0
            quantized = (arr_norm * 255.0).astype(np.uint8)
        else:
            quantized = arr.astype(np.uint8)

        flat = quantized.ravel()
        if flat.size == 0:
            return 0.0

        counts = np.bincount(flat, minlength=256)
        probs = counts[counts > 0] / float(flat.size)

        entropy = float(max(-np.sum(probs * np.log2(probs)), 0.0))
        return entropy

    @staticmethod
    def compute_spectral_rolloff(
        image: np.ndarray,
        num_dominant: int = 5,
    ) -> Tuple[float, List[Tuple[float, float]]]:
        """
        Compute 2D FFT power spectrum, radial roll-off exponent alpha,
        and coordinates of dominant spatial frequencies.
        """
        arr = np.asarray(image, dtype=np.float32)
        if arr.ndim == 3:
            # Luminance: 0.299 R + 0.587 G + 0.114 B (or first channel if 1-channel)
            if arr.shape[2] >= 3:
                gray = 0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]
            else:
                gray = arr[:, :, 0]
        else:
            gray = arr

        h, w = gray.shape
        if h < 4 or w < 4:
            return 0.0, []

        # Subtract mean (DC removal)
        gray_centered = gray - np.mean(gray)
        fft_2d = np.fft.fftshift(np.fft.fft2(gray_centered))
        power = np.abs(fft_2d) ** 2

        cy, cx = h // 2, w // 2
        y_indices, x_indices = np.indices((h, w))
        r_grid = np.sqrt((y_indices - cy) ** 2 + (x_indices - cx) ** 2)

        max_radius = min(cy, cx)
        if max_radius < 2:
            return 0.0, []

        r_int = np.round(r_grid).astype(int)
        radial_profile = []
        radii = []

        for r in range(1, max_radius):
            mask = r_int == r
            if np.any(mask):
                radial_profile.append(float(np.mean(power[mask])))
                radii.append(float(r))

        if len(radii) < 2 or np.all(np.array(radial_profile) <= 1e-12):
            rolloff_alpha = 0.0
        else:
            r_arr = np.array(radii)
            p_arr = np.maximum(np.array(radial_profile), 1e-12)
            log_r = np.log(r_arr)
            log_p = np.log(p_arr)

            # Fit log(P) = -alpha * log(r) + const
            poly = np.polyfit(log_r, log_p, deg=1)
            rolloff_alpha = float(-poly[0])

        # Dominant non-DC frequencies
        power_no_dc = power.copy()
        power_no_dc[cy - 1 : cy + 2, cx - 1 : cx + 2] = 0.0

        dominant_coords: List[Tuple[float, float]] = []
        flat_sorted = np.argsort(power_no_dc.ravel())[::-1]
        for idx in flat_sorted[:num_dominant]:
            y_coord, x_coord = np.unravel_index(idx, (h, w))
            dominant_coords.append((float(x_coord - cx), float(y_coord - cy)))

        return rolloff_alpha, dominant_coords

    @staticmethod
    def compute_channel_correlations(image: np.ndarray) -> np.ndarray:
        """
        Compute Pearson correlation matrix between color channels.
        Guarantees 1.0 on diagonal and 0.0 cross-correlations when a channel has zero variance.
        """
        arr = np.asarray(image, dtype=np.float64)
        if arr.ndim == 2:
            return np.array([[1.0]], dtype=np.float64)

        h, w, c = arr.shape
        flat = arr.reshape(-1, c)

        stds = np.std(flat, axis=0)
        corr = np.eye(c, dtype=np.float64)

        for i in range(c):
            for j in range(i + 1, c):
                if stds[i] > 1e-12 and stds[j] > 1e-12:
                    val = np.corrcoef(flat[:, i], flat[:, j])[0, 1]
                    corr[i, j] = val
                    corr[j, i] = val
                else:
                    corr[i, j] = 0.0
                    corr[j, i] = 0.0

        return corr
