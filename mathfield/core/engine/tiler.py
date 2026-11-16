"""
mathfield/core/engine/tiler.py

Milestone 3: 8K High-Resolution Tiling, Memory Management & Streaming Engine.
Provides spatial domain partitioning, bit-exact coordinate subgrids,
disk-backed memory-mapped output buffers, hardware memory throttling,
and multi-tier tiled evaluation up to 8K (8192x8192) resolution.
"""

from __future__ import annotations

import gc
import math
import os
import tempfile
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, Generator, List, Optional, Sequence, Tuple, Union

import numpy as np

from mathfield.core.ast.nodes import ASTNode
from mathfield.core.domain import CoordinateDomain2D
from mathfield.core.engine.backend import MultiTierEvaluator


class InsufficientMemoryError(MemoryError):
    """Raised when available system memory or VRAM is insufficient for tile evaluation."""
    pass


# ============================================================================
# 1. TileInfo Data Structure
# ============================================================================

class TileInfo:
    """
    Metadata and coordinate subgrids for an individual spatial domain tile.
    Immutable representation supporting both direct subgrid caching and lazy 1D mesh generation.

    Fields:
        tile_index: Sequential 0-based index of the tile (row-major order).
        x_slice: Slice for indexing columns in the destination raster [x_start:x_end].
        y_slice: Slice for indexing rows in the destination raster [y_start:y_end].
        X_tile: 2D float64 coordinate array corresponding to columns (varying horizontally).
        Y_tile: 2D float64 coordinate array corresponding to rows (varying vertically).
        pixel_bounds: Tuple of (x_start, y_start, x_end, y_end) in pixel coordinates.
    """

    def __init__(
        self,
        tile_index: int,
        x_slice: slice,
        y_slice: slice,
        X_tile: Optional[np.ndarray] = None,
        Y_tile: Optional[np.ndarray] = None,
        pixel_bounds: Tuple[int, int, int, int] = (0, 0, 0, 0),
        _x_sub: Optional[np.ndarray] = None,
        _y_sub: Optional[np.ndarray] = None,
    ) -> None:
        object.__setattr__(self, "tile_index", int(tile_index))
        object.__setattr__(self, "x_slice", x_slice)
        object.__setattr__(self, "y_slice", y_slice)
        object.__setattr__(self, "pixel_bounds", pixel_bounds)
        object.__setattr__(self, "_X_tile", X_tile)
        object.__setattr__(self, "_Y_tile", Y_tile)
        object.__setattr__(self, "_x_sub", _x_sub)
        object.__setattr__(self, "_y_sub", _y_sub)
        object.__setattr__(self, "_initialized", True)

    def __setattr__(self, key: str, value: Any) -> None:
        if getattr(self, "_initialized", False):
            raise AttributeError(f"TileInfo is immutable, cannot modify attribute '{key}'")
        object.__setattr__(self, key, value)

    def __delattr__(self, key: str) -> None:
        if getattr(self, "_initialized", False):
            raise AttributeError(f"TileInfo is immutable, cannot delete attribute '{key}'")
        object.__delattr__(self, key)

    @property
    def width(self) -> int:
        """Tile width in pixels."""
        return self.pixel_bounds[2] - self.pixel_bounds[0]

    @property
    def height(self) -> int:
        """Tile height in pixels."""
        return self.pixel_bounds[3] - self.pixel_bounds[1]

    @property
    def shape(self) -> Tuple[int, int]:
        """Tile shape as (height, width)."""
        return (self.height, self.width)

    @property
    def pixel_count(self) -> int:
        """Total number of pixels in this tile."""
        return self.width * self.height

    @property
    def X_tile(self) -> np.ndarray:
        """2D coordinate grid for columns."""
        if self._X_tile is not None:
            return self._X_tile
        if self._x_sub is not None and self._y_sub is not None:
            return np.meshgrid(self._x_sub, self._y_sub, indexing="xy")[0]
        return np.empty((self.height, self.width), dtype=np.float64)

    @property
    def Y_tile(self) -> np.ndarray:
        """2D coordinate grid for rows."""
        if self._Y_tile is not None:
            return self._Y_tile
        if self._x_sub is not None and self._y_sub is not None:
            return np.meshgrid(self._x_sub, self._y_sub, indexing="xy")[1]
        return np.empty((self.height, self.width), dtype=np.float64)

    def __repr__(self) -> str:
        return (
            f"TileInfo(tile_index={self.tile_index}, x_slice={self.x_slice}, "
            f"y_slice={self.y_slice}, pixel_bounds={self.pixel_bounds})"
        )


# ============================================================================
# 2. DomainChunker
# ============================================================================

class DomainChunker:
    """
    Partitions continuous 2D coordinate domains into spatial tiles up to 8K (8192x8192).

    Guarantees:
    1. Bit-Exact Parity: Coordinate subgrids (X_tile, Y_tile) match the full-grid
       CoordinateDomain2D.to_discrete_grid() identically down to IEEE-754 bit-level (L_inf = 0.0).
    2. Remainder Handling: Arbitrary non-power-of-two dimensions (e.g. 1920x1080, 4096x2160, 101x97)
       are partitioned into pairwise disjoint rectangular tiles without gaps, clipping, or overlaps.
    3. Low Memory Footprint: Generates 1D coordinate vectors (128 KB for 8K) and materializes
       2D subgrids lazily on demand, holding at most one tile mesh in RAM at a time.
    """

    def __init__(
        self,
        tile_size: int = 1024,
        max_memory_mb: float = 300.0,
        min_tile_size: int = 64,
        domain: Optional[CoordinateDomain2D] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
    ) -> None:
        if tile_size <= 0:
            raise ValueError(f"tile_size must be positive, got {tile_size}")
        if max_memory_mb <= 0:
            raise ValueError(f"max_memory_mb must be positive, got {max_memory_mb}")
        if min_tile_size <= 0:
            raise ValueError(f"min_tile_size must be positive, got {min_tile_size}")

        self.tile_size = int(tile_size)
        self.max_memory_mb = float(max_memory_mb)
        self.min_tile_size = int(min_tile_size)
        self.domain = domain
        self.width = int(width) if width is not None else (domain.width if domain is not None else None)
        self.height = int(height) if height is not None else (domain.height if domain is not None else None)

    def estimate_working_set_mb(self, tile_size: Optional[int] = None, num_channels: int = 1) -> float:
        """
        Calculates working set memory in MB for a single tile.
        Accounts for X, Y coordinate grids plus destination channel buffer(s).
        For float64, each 1024x1024 buffer is exactly 8.0 MB.
        Total for 1 channel = 3 * 8.0 MB = 24.0 MB.
        """
        ts = int(tile_size) if tile_size is not None else self.tile_size
        single_buffer_mb = (ts * ts * 8.0) / (1024.0 * 1024.0)
        return float((2 + num_channels) * single_buffer_mb)

    def get_tiles(
        self,
        total_width: Optional[int] = None,
        total_height: Optional[int] = None,
        domain: Optional[CoordinateDomain2D] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
    ) -> List[TileInfo]:
        """
        Partitions the canvas into a list of non-overlapping TileInfo objects covering 100% of pixels.
        """
        w = total_width if total_width is not None else (width if width is not None else self.width)
        h = total_height if total_height is not None else (height if height is not None else self.height)

        if w is None or h is None or w <= 0 or h <= 0:
            raise ValueError(f"Canvas dimensions must be positive integers, got width={w}, height={h}")

        active_domain = domain if domain is not None else self.domain
        if active_domain is None:
            active_domain = CoordinateDomain2D.unit_square(width=w, height=h)

        n_tiles_x = math.ceil(w / self.tile_size)
        n_tiles_y = math.ceil(h / self.tile_size)

        # Precompute 1D coordinate vectors across canvas axes (only O(W + H) memory)
        x_1d = np.linspace(active_domain.x_min, active_domain.x_max, w, dtype=np.float64)
        y_1d = np.linspace(active_domain.y_min, active_domain.y_max, h, dtype=np.float64)

        tiles: List[TileInfo] = []
        tile_idx = 0

        for ty in range(n_tiles_y):
            y_start = ty * self.tile_size
            y_end = min((ty + 1) * self.tile_size, h)
            y_slice = slice(y_start, y_end)
            y_sub = y_1d[y_slice]

            for tx in range(n_tiles_x):
                x_start = tx * self.tile_size
                x_end = min((tx + 1) * self.tile_size, w)
                x_slice = slice(x_start, x_end)
                x_sub = x_1d[x_slice]

                tiles.append(
                    TileInfo(
                        tile_index=tile_idx,
                        x_slice=x_slice,
                        y_slice=y_slice,
                        pixel_bounds=(x_start, y_start, x_end, y_end),
                        _x_sub=x_sub,
                        _y_sub=y_sub,
                    )
                )
                tile_idx += 1

        return tiles

    def compute_tile_layout(
        self,
        width: int,
        height: int,
        domain: Optional[CoordinateDomain2D] = None,
    ) -> List[TileInfo]:
        """Alias for get_tiles compatible with TiledReconstructionManager interface."""
        return self.get_tiles(total_width=width, total_height=height, domain=domain)

    def evaluate_tiled(
        self,
        ast: ASTNode,
        domain: Optional[CoordinateDomain2D] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
        params: Optional[Union[np.ndarray, Sequence[float]]] = None,
        preferred_backend: str = "auto",
        out: Optional[Union[np.ndarray, np.memmap]] = None,
        out_memmap: Optional[Union[np.ndarray, np.memmap]] = None,
        progress_callback: Optional[Callable[[int, int, TileInfo], None]] = None,
        cancel_token: Optional[Callable[[], bool]] = None,
        dtype: Union[np.dtype, str] = np.float64,
        auto_flush: bool = True,
    ) -> Union[np.ndarray, np.memmap]:
        """Evaluates AST tiled over continuous domain using configured chunker settings."""
        return evaluate_tiled(
            ast=ast,
            domain=domain if domain is not None else self.domain,
            width=width if width is not None else self.width,
            height=height if height is not None else self.height,
            tile_size=self.tile_size,
            preferred_backend=preferred_backend,
            out=out,
            out_memmap=out_memmap,
            params=params,
            progress_callback=progress_callback,
            cancel_token=cancel_token,
            dtype=dtype,
            auto_flush=auto_flush,
        )


class TiledReconstructionManager(DomainChunker):
    """
    Manager class for high-resolution tiled reconstruction workflows.
    Compatible with E2E Tier 3 test suites.
    """
    pass


# ============================================================================
# 3. MemmapManager
# ============================================================================

class MemmapManager:
    """
    Manages disk-backed numpy.memmap buffer allocation, streaming, flushing,
    and safe temporary file lifecycle management on Windows.
    Ensures host RAM working set remains < 300 MB during 8K reconstruction.
    """

    def __init__(
        self,
        shape: Tuple[int, ...],
        dtype: Union[np.dtype, str] = np.float32,
        filename: Optional[Union[str, Path]] = None,
        filepath: Optional[Union[str, Path]] = None,
        auto_cleanup: bool = True,
        mode: str = "w+",
    ) -> None:
        self.shape = tuple(shape)
        self.dtype = np.dtype(dtype)
        self.auto_cleanup = auto_cleanup
        self.mode = mode

        chosen_path = filename if filename is not None else filepath
        if chosen_path is None:
            temp_dir = tempfile.gettempdir()
            unique_name = f"mathfield_memmap_{uuid.uuid4().hex[:12]}.dat"
            self.filepath = os.path.join(temp_dir, unique_name)
            self._is_temp_file = True
        else:
            self.filepath = os.path.abspath(str(chosen_path))
            self._is_temp_file = False
            os.makedirs(os.path.dirname(self.filepath), exist_ok=True)

        self._memmap: Optional[np.memmap] = np.memmap(
            self.filepath,
            dtype=self.dtype,
            mode=self.mode,
            shape=self.shape,
        )

    @property
    def buffer(self) -> np.memmap:
        """Active memory-mapped array."""
        if self._memmap is None:
            raise RuntimeError("MemmapManager buffer is closed.")
        return self._memmap

    @property
    def array(self) -> np.memmap:
        """Alias for buffer property."""
        return self.buffer

    @property
    def ndim(self) -> int:
        return self.buffer.ndim

    def __len__(self) -> int:
        return len(self.buffer)

    def __getitem__(self, item: Any) -> Any:
        return self.buffer[item]

    def __setitem__(self, item: Any, value: Any) -> None:
        self.buffer[item] = value

    def __array__(self, dtype: Optional[np.dtype] = None) -> np.ndarray:
        return np.asarray(self.buffer, dtype=dtype)

    def flush(self) -> None:
        """Synchronously flushes dirty pages in page cache to disk."""
        if self._memmap is not None:
            self._memmap.flush()

    def write_tile(self, y_slice: slice, x_slice: slice, data: np.ndarray, flush: bool = False) -> None:
        """Writes tile data into buffer with optional immediate disk flush."""
        self.buffer[y_slice, x_slice] = np.asarray(data, dtype=self.dtype)
        if flush:
            self.flush()

    def read_tile(self, y_slice: slice, x_slice: slice) -> np.ndarray:
        """Reads a tile slice from the memory-mapped buffer."""
        return np.asarray(self.buffer[y_slice, x_slice])

    def close(self) -> None:
        """
        Safely flushes and closes the memmap buffer.
        Releases Windows file handles via explicit garbage collection and removes temp file.
        """
        if self._memmap is not None:
            try:
                self._memmap.flush()
            except Exception:
                pass
            del self._memmap
            self._memmap = None

        gc.collect()

        if self.auto_cleanup and (self._is_temp_file or not self._is_temp_file) and os.path.exists(self.filepath):
            try:
                os.remove(self.filepath)
            except OSError:
                pass

    def __enter__(self) -> "MemmapManager":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()


# ============================================================================
# 4. MemoryMonitor & MemoryThrottle
# ============================================================================

class MemoryMonitor:
    """
    Hardware memory monitor and adaptive tile size throttle.
    Monitors host RAM via psutil and GPU VRAM via CuPy memory pool / CUDA device.
    """

    MIN_FREE_RAM_MB: float = 500.0
    MIN_FREE_VRAM_MB: float = 256.0
    MAX_PROCESS_RSS_MB: float = 300.0

    def __init__(
        self,
        ram_safety_threshold_bytes: float = 500.0 * 1024.0 * 1024.0,
        vram_safety_threshold_bytes: float = 256.0 * 1024.0 * 1024.0,
    ) -> None:
        self.ram_safety_threshold_bytes = float(ram_safety_threshold_bytes)
        self.vram_safety_threshold_bytes = float(vram_safety_threshold_bytes)

    def check_memory(self) -> Dict[str, Any]:
        """Collects current hardware memory telemetry."""
        import psutil
        vm = psutil.virtual_memory()
        proc = psutil.Process()

        telemetry: Dict[str, Any] = {
            "available_ram_bytes": int(vm.available),
            "total_ram_bytes": int(vm.total),
            "used_ram_bytes": int(vm.used),
            "ram_percent": float(vm.percent),
            "process_rss_bytes": int(proc.memory_info().rss),
            "ram_total_mb": float(vm.total) / (1024.0 ** 2),
            "ram_available_mb": float(vm.available) / (1024.0 ** 2),
            "process_rss_mb": float(proc.memory_info().rss) / (1024.0 ** 2),
            "cuda_available": False,
            "vram_available_bytes": None,
            "vram_free_mb": None,
            "vram_total_mb": None,
            "cupy_pool_used_mb": None,
        }

        try:
            import cupy as cp
            if cp.cuda.runtime.getDeviceCount() > 0:
                dev = cp.cuda.Device()
                free_b, total_b = dev.mem_info
                telemetry["cuda_available"] = True
                telemetry["vram_available_bytes"] = int(free_b)
                telemetry["vram_free_mb"] = float(free_b) / (1024.0 ** 2)
                telemetry["vram_total_mb"] = float(total_b) / (1024.0 ** 2)
                telemetry["cupy_pool_used_mb"] = float(cp.get_default_memory_pool().used_bytes()) / (1024.0 ** 2)
        except Exception:
            pass

        return telemetry

    def compute_safe_tile_size(
        self,
        current_tile_size: int = 1024,
        min_tile_size: int = 64,
        fail_on_exhaustion: bool = False,
        backend: str = "auto",
    ) -> int:
        """
        Dynamically adapts tile size based on available system memory.
        Halves tile size (1024 -> 512 -> 256 -> 128 -> 64) if available resources fall below safety thresholds.
        """
        telemetry = self.check_memory()
        avail_ram = telemetry["available_ram_bytes"]
        tile_size = int(current_tile_size)

        # Check host RAM constraints
        ram_constrained = avail_ram < self.ram_safety_threshold_bytes

        # Check GPU VRAM constraints if using CUDA
        vram_constrained = False
        if backend in ("auto", "cuda", "cupy") and telemetry["cuda_available"]:
            vram_avail = telemetry["vram_available_bytes"]
            if vram_avail is not None and vram_avail < self.vram_safety_threshold_bytes:
                vram_constrained = True

        if avail_ram < 100 * 1024 * 1024:
            tile_size = min_tile_size
        elif ram_constrained or vram_constrained:
            while tile_size > min_tile_size:
                tile_size //= 2
                if tile_size <= 512:
                    break

        final_size = max(tile_size, min_tile_size)

        if fail_on_exhaustion:
            min_mem_needed = 24.0 * final_size * final_size
            if avail_ram < min_mem_needed:
                raise InsufficientMemoryError(
                    f"Available RAM ({avail_ram / (1024**2):.1f} MB) is below minimum tile requirement "
                    f"({min_mem_needed / (1024**2):.1f} MB for {final_size}x{final_size} tile)."
                )

        return final_size

    @classmethod
    def get_telemetry(cls) -> Dict[str, Any]:
        """Classmethod returning telemetry from a default monitor instance."""
        return cls().check_memory()

    @classmethod
    def get_optimal_tile_size(
        cls,
        width: int,
        height: int,
        default_tile_size: int = 1024,
        backend: str = "auto",
        min_tile_size: int = 64,
    ) -> int:
        """Classmethod computing optimal tile size for given canvas dimensions."""
        monitor = cls()
        safe_size = monitor.compute_safe_tile_size(
            current_tile_size=default_tile_size,
            min_tile_size=min_tile_size,
            backend=backend,
        )
        return min(safe_size, max(width, height))

    @staticmethod
    def cleanup(backend: Optional[str] = None) -> None:
        """Proactively reclaims heap, Windows page cache, and GPU pool allocations."""
        gc.collect()
        if os.name == "nt":
            try:
                import ctypes
                ctypes.windll.psapi.EmptyWorkingSet(ctypes.windll.kernel32.GetCurrentProcess())
            except Exception:
                pass
        if backend in ("cuda", "cupy"):
            try:
                import cupy as cp
                cp.get_default_memory_pool().free_all_blocks()
                cp.get_default_pinned_memory_pool().free_all_blocks()
            except Exception:
                pass


# MemoryThrottle alias for backwards and blueprint compatibility
MemoryThrottle = MemoryMonitor


# ============================================================================
# 5. High-Level Tiled Evaluation Pipeline
# ============================================================================

def evaluate_tiled(
    ast: ASTNode,
    domain: Optional[CoordinateDomain2D] = None,
    width: Optional[int] = None,
    height: Optional[int] = None,
    tile_size: int = 1024,
    preferred_backend: str = "auto",
    out: Optional[Union[np.ndarray, np.memmap]] = None,
    out_memmap: Optional[Union[np.ndarray, np.memmap]] = None,
    params: Optional[Union[np.ndarray, Sequence[float]]] = None,
    progress_callback: Optional[Callable[[int, int, TileInfo], None]] = None,
    cancel_token: Optional[Callable[[], bool]] = None,
    dtype: Union[np.dtype, str] = np.float64,
    auto_flush: bool = True,
) -> Union[np.ndarray, np.memmap]:
    """
    Evaluates an AST expression across arbitrary canvas resolutions up to 8K (8192x8192).
    Streams tile evaluations directly into destination buffer, maintaining host RAM < 300 MB.
    """
    w = width if width is not None else (domain.width if domain is not None else None)
    h = height if height is not None else (domain.height if domain is not None else None)

    if w is None or h is None or w <= 0 or h <= 0:
        raise ValueError(f"Width and height must be positive integers, got ({w}, {h})")

    active_domain = domain if domain is not None else CoordinateDomain2D.unit_square(width=w, height=h)

    p = np.empty(0, dtype=np.float64) if params is None else np.asarray(params, dtype=np.float64)
    dt = np.dtype(dtype)

    # 1. Determine optimal tile size via memory monitor
    effective_tile_size = MemoryMonitor.get_optimal_tile_size(
        width=w,
        height=h,
        default_tile_size=tile_size,
        backend=preferred_backend,
    )

    # 2. Select destination buffer
    target_out = out_memmap if out_memmap is not None else out
    managed_memmap: Optional[MemmapManager] = None

    if target_out is None:
        buffer_bytes = w * h * dt.itemsize
        if buffer_bytes > 64 * 1024 * 1024:
            managed_memmap = MemmapManager(shape=(h, w), dtype=dt, auto_cleanup=False)
            dest_buffer = managed_memmap.buffer
        else:
            dest_buffer = np.empty((h, w), dtype=dt)
    else:
        if target_out.shape != (h, w):
            raise ValueError(f"Output buffer shape {target_out.shape} does not match canvas ({h}, {w})")
        dest_buffer = target_out

    # 3. Instantiate evaluator and chunker
    evaluator = MultiTierEvaluator(preferred_backend=preferred_backend)
    chunker = DomainChunker(tile_size=effective_tile_size)
    tiles = chunker.get_tiles(total_width=w, total_height=h, domain=active_domain)
    total_tiles = len(tiles)

    # 4. Stream tiles
    for idx, tile in enumerate(tiles):
        if cancel_token is not None and cancel_token():
            if isinstance(dest_buffer, np.memmap):
                dest_buffer.flush()
            raise InterruptedError("Tiled evaluation cancelled by cooperative token.")

        # Evaluate tile on active acceleration tier
        z_tile = evaluator.evaluate(
            ast=ast,
            X=tile.X_tile,
            Y=tile.Y_tile,
            params=p,
            preferred_backend=preferred_backend,
        )

        # Write directly into slice
        dest_buffer[tile.y_slice, tile.x_slice] = np.asarray(z_tile, dtype=dt)

        if auto_flush and isinstance(dest_buffer, np.memmap):
            dest_buffer.flush()

        # Clean up transient tile memory
        del z_tile
        if idx % 8 == 7 or idx == total_tiles - 1:
            MemoryMonitor.cleanup(backend=evaluator.last_backend_used)

        if progress_callback is not None:
            progress_callback(tile.tile_index + 1, total_tiles, tile)

    if isinstance(dest_buffer, np.memmap):
        dest_buffer.flush()

    return dest_buffer
