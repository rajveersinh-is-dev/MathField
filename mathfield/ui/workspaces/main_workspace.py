"""
mathfield/ui/workspaces/main_workspace.py

Feature 17: Main UI Workspace.
Drag-and-drop image loading, interactive pan/zoom viewport,
side-by-side curtain split comparison slider, and status ribbon.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Tuple, Union

import numpy as np
from PIL import Image


class MainWorkspace:
    """
    Main Workspace: High-performance image viewport and interactive comparison.
    """

    def __init__(self) -> None:
        self._width: int = 1024
        self._height: int = 768
        self.image_dimensions: Tuple[int, int] = (0, 0)  # (width, height)
        self.current_zoom: float = 1.0
        self.curtain_split_position: float = 0.5  # t in [0.0, 1.0]
        self._preview_buffer: np.ndarray = np.zeros((64, 64, 3), dtype=np.uint8)
        self._original_image: Optional[np.ndarray] = None
        self._status_text: str = "Ready"

    def resize(self, width: int, height: int) -> None:
        self._width = int(width)
        self._height = int(height)

    def width(self) -> int:
        return self._width

    def height(self) -> int:
        return self._height

    def load_image(self, path_or_array: Union[str, Path, np.ndarray]) -> bool:
        """
        Load source image from file path or array.
        Returns False and sets status text on invalid format.
        """
        try:
            if isinstance(path_or_array, (str, Path)):
                p = Path(path_or_array)
                if not p.exists():
                    self._status_text = f"File not found: {p}"
                    return False
                img = Image.open(p)
                w, h = img.size
                arr = np.array(img.convert("RGB"))
            elif isinstance(path_or_array, np.ndarray):
                arr = path_or_array
                h, w = arr.shape[:2]
            else:
                self._status_text = "Unsupported image format"
                return False

            self.image_dimensions = (int(w), int(h))
            self._original_image = arr
            self._status_text = f"Loaded image {w}x{h}"
            return True
        except Exception as exc:
            self._status_text = f"Error loading image: {exc}"
            return False

    def set_zoom(self, zoom: float) -> None:
        """Clamp zoom factor into safe range [0.1, 50.0]."""
        self.current_zoom = float(np.clip(zoom, 0.1, 50.0))

    def set_curtain_split(self, split: float) -> None:
        """Clamp split slider position to [0.0, 1.0]."""
        self.curtain_split_position = float(np.clip(split, 0.0, 1.0))

    def update_preview(self, preview: np.ndarray) -> None:
        """Update viewport preview buffer."""
        self._preview_buffer = np.asarray(preview)

    def get_preview_buffer(self) -> np.ndarray:
        """Retrieve current viewport preview buffer."""
        return self._preview_buffer

    def get_status_text(self) -> str:
        """Retrieve status ribbon text."""
        return self._status_text
