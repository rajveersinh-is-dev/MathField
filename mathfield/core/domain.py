"""
mathfield/core/domain.py

Continuous 2D Coordinate Domain Representation and Mapping.
Provides isotropic (aspect-ratio preserved) and normalized unit-square
continuous coordinate frames over arbitrary discrete image grids.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np


class DomainMode(str, Enum):
    """Spatial domain bounding mode."""

    ASPECT_PRESERVED = "aspect_preserved"
    UNIT_SQUARE = "unit_square"
    # Canonical aliases for E2E and UI compatibility:
    ISOTROPIC = ASPECT_PRESERVED
    SQUARE = UNIT_SQUARE

    @classmethod
    def from_str(cls, val: Union[str, DomainMode]) -> DomainMode:
        if isinstance(val, DomainMode):
            return val
        s = str(val).strip().lower()
        if s in ("aspect_preserved", "isotropic"):
            return cls.ASPECT_PRESERVED
        if s in ("unit_square", "square"):
            return cls.UNIT_SQUARE
        raise ValueError(f"Unknown DomainMode: {val}")


@dataclass
class CoordinateDomain2D:
    """
    Continuous 2D bounding domain mapped onto a discrete image raster.
    """

    width: int
    height: int
    mode: DomainMode | str = DomainMode.ASPECT_PRESERVED
    x_min: Optional[float] = None
    x_max: Optional[float] = None
    y_min: Optional[float] = None
    y_max: Optional[float] = None

    def __post_init__(self) -> None:
        if self.width <= 0:
            raise ValueError(f"width must be positive, got {self.width}")
        if self.height <= 0:
            raise ValueError(f"height must be positive, got {self.height}")

        self.mode = DomainMode.from_str(self.mode)

        if self.mode in (DomainMode.UNIT_SQUARE, DomainMode.SQUARE):
            default_x_min, default_x_max = -1.0, 1.0
            default_y_min, default_y_max = -1.0, 1.0
        else:
            r = float(self.width) / float(self.height)
            default_x_min, default_x_max = -r, r
            default_y_min, default_y_max = -1.0, 1.0

        if self.x_min is None:
            self.x_min = default_x_min
        if self.x_max is None:
            self.x_max = default_x_max
        if self.y_min is None:
            self.y_min = default_y_min
        if self.y_max is None:
            self.y_max = default_y_max

        if self.x_min >= self.x_max:
            raise ValueError(f"x_min ({self.x_min}) must be strictly less than x_max ({self.x_max})")
        if self.y_min >= self.y_max:
            raise ValueError(f"y_min ({self.y_min}) must be strictly less than y_max ({self.y_max})")

    @property
    def aspect_ratio(self) -> float:
        """Physical aspect ratio width / height."""
        return float(self.width) / float(self.height)

    @property
    def x_bounds(self) -> Tuple[float, float]:
        """Continuous X bounds (x_min, x_max)."""
        return (float(self.x_min), float(self.x_max))

    @property
    def y_bounds(self) -> Tuple[float, float]:
        """Continuous Y bounds (y_min, y_max)."""
        return (float(self.y_min), float(self.y_max))

    def cell_size(self) -> Tuple[float, float]:
        """
        Physical continuous cell spacing (dx, dy).
        In isotropic mode with x in [-r, r] and y in [-1, 1], dx == dy == 2 / height.
        """
        dx = (self.x_max - self.x_min) / float(self.width)
        dy = (self.y_max - self.y_min) / float(self.height)
        return (dx, dy)

    def to_discrete_grid(
        self,
        width: Optional[int] = None,
        height: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate continuous coordinate matrices X, Y each of shape (height, width).
        X varies across columns, Y varies across rows.
        """
        w = width if width is not None else self.width
        h = height if height is not None else self.height

        if w <= 0 or h <= 0:
            raise ValueError(f"Grid dimensions must be positive, got ({w}, {h})")

        x = np.linspace(self.x_min, self.x_max, w)
        y = np.linspace(self.y_min, self.y_max, h)
        X, Y = np.meshgrid(x, y, indexing="xy")
        return X, Y

    def to_continuous(
        self,
        i: Union[int, float, np.ndarray],
        j: Union[int, float, np.ndarray],
    ) -> Tuple[Union[float, np.ndarray], Union[float, np.ndarray]]:
        """
        Map discrete pixel indices (i: horizontal column [0, width-1],
        j: vertical row [0, height-1]) to continuous domain coordinates (x, y).
        """
        w_span = max(self.width - 1, 1)
        h_span = max(self.height - 1, 1)

        x = self.x_min + (np.asarray(i, dtype=float) / w_span) * (self.x_max - self.x_min)
        y = self.y_min + (np.asarray(j, dtype=float) / h_span) * (self.y_max - self.y_min)

        if np.isscalar(i) and np.isscalar(j):
            return float(x), float(y)
        return x, y

    def to_discrete(
        self,
        x: Union[float, np.ndarray],
        y: Union[float, np.ndarray],
    ) -> Tuple[Union[int, np.ndarray], Union[int, np.ndarray]]:
        """
        Map continuous domain coordinates (x, y) to discrete pixel indices (i, j).
        Guarantees exact bijectivity for integer pixel indices.
        """
        w_span = max(self.width - 1, 1)
        h_span = max(self.height - 1, 1)

        x_arr = np.asarray(x, dtype=float)
        y_arr = np.asarray(y, dtype=float)

        i_norm = (x_arr - self.x_min) / (self.x_max - self.x_min) * w_span
        j_norm = (y_arr - self.y_min) / (self.y_max - self.y_min) * h_span

        i_safe = np.nan_to_num(i_norm, nan=0.0, posinf=float(self.width - 1), neginf=0.0)
        j_safe = np.nan_to_num(j_norm, nan=0.0, posinf=float(self.height - 1), neginf=0.0)

        i_disc = np.clip(np.round(i_safe), 0, self.width - 1).astype(int)
        j_disc = np.clip(np.round(j_safe), 0, self.height - 1).astype(int)

        if np.isscalar(x) and np.isscalar(y):
            return int(i_disc), int(j_disc)
        return i_disc, j_disc

    @classmethod
    def unit_square(cls, width: int = 512, height: int = 512) -> CoordinateDomain2D:
        """Create a normalized unit square domain spanning [-1, 1] x [-1, 1]."""
        return cls(width=width, height=height, mode=DomainMode.UNIT_SQUARE)

    @classmethod
    def from_image_dimensions(
        cls,
        width: int,
        height: int,
        mode: DomainMode | str = DomainMode.ASPECT_PRESERVED,
        isotropic: bool = True,
    ) -> CoordinateDomain2D:
        """Construct domain from image dimensions, honoring isotropic flag if provided."""
        if not isotropic:
            mode = DomainMode.UNIT_SQUARE
        return cls(width=width, height=height, mode=mode)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize domain metadata to a dictionary."""
        return {
            "width": self.width,
            "height": self.height,
            "mode": self.mode.value,
            "x_min": self.x_min,
            "x_max": self.x_max,
            "y_min": self.y_min,
            "y_max": self.y_max,
            "aspect_ratio": self.aspect_ratio,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> CoordinateDomain2D:
        """Deserialize domain from dictionary."""
        return cls(
            width=int(data["width"]),
            height=int(data["height"]),
            mode=DomainMode.from_str(data.get("mode", DomainMode.ASPECT_PRESERVED)),
            x_min=float(data["x_min"]) if data.get("x_min") is not None else None,
            x_max=float(data["x_max"]) if data.get("x_max") is not None else None,
            y_min=float(data["y_min"]) if data.get("y_min") is not None else None,
            y_max=float(data["y_max"]) if data.get("y_max") is not None else None,
        )
