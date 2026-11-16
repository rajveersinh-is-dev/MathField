"""
tests/test_tiler.py

Comprehensive Unit & Integration Test Suite for Milestone 3 8K Tiling & Memory Management:
- TileInfo dataclass representation and immutability.
- Strict L_inf = 0.0 grid parity between tiled and monolithic full-grid evaluation.
- Remainder boundary handling and 100% pixel coverage across non-multiple dimensions.
- Disk-backed np.memmap buffer allocation, in-place slice writes, flush, and persistence.
- Hardware memory monitoring (psutil RAM, VRAM) and adaptive tile size shrinking.
- 8K high-resolution evaluation simulation with strict host RAM consumption < 300 MB.
"""

import math
from pathlib import Path
from unittest import mock

import numpy as np
import psutil
import pytest

from mathfield.core.ast.parser import parse
from mathfield.core.domain import CoordinateDomain2D, DomainMode
from mathfield.core.engine.backend import MultiTierEvaluator
from mathfield.core.engine.cpu_numpy import evaluate_numpy
from mathfield.core.engine.tiler import (
    DomainChunker,
    InsufficientMemoryError,
    MemmapManager,
    MemoryMonitor,
    TileInfo,
    evaluate_tiled,
)


class TestTileInfo:
    """Tests for TileInfo dataclass representation and geometry properties."""

    def test_tile_info_fields_and_properties(self) -> None:
        """TC-TILER-01: Verify TileInfo fields, slice representations, and width/height properties."""
        X_tile = np.zeros((256, 512), dtype=np.float64)
        Y_tile = np.zeros((256, 512), dtype=np.float64)
        tile = TileInfo(
            tile_index=0,
            x_slice=slice(100, 612),
            y_slice=slice(50, 306),
            X_tile=X_tile,
            Y_tile=Y_tile,
            pixel_bounds=(100, 50, 612, 306),
        )
        assert tile.tile_index == 0
        assert tile.x_slice == slice(100, 612)
        assert tile.y_slice == slice(50, 306)
        assert tile.width == 512
        assert tile.height == 256
        assert tile.shape == (256, 512)
        assert tile.pixel_count == 256 * 512
        assert tile.pixel_bounds == (100, 50, 612, 306)
        assert tile.X_tile.shape == (256, 512)
        assert tile.Y_tile.shape == (256, 512)


class TestGridParity:
    """Tests for strict L_inf = 0.0 grid parity between tiled and full evaluation."""

    @pytest.mark.parametrize(
        "expr_str, params",
        [
            ("x + y", np.array([], dtype=np.float64)),
            ("2.5 * x - 1.8 * y + 0.75", np.array([], dtype=np.float64)),
            ("sin(4.0 * pi * x) * cos(6.0 * pi * y)", np.array([], dtype=np.float64)),
            ("exp(- 5.0 * (x^2 + y^2))", np.array([], dtype=np.float64)),
            ("(x^2 - y^2) / (x^2 + y^2 + 0.05)", np.array([], dtype=np.float64)),
            ("clamp(x * 2.0, -0.5, 0.5) + abs(y)", np.array([], dtype=np.float64)),
            ("c0 * sin(c1 * x) + c2 * cos(c3 * y)", np.array([1.5, 2.0, -0.8, 3.5], dtype=np.float64)),
        ],
    )
    def test_grid_parity_various_expressions(self, expr_str: str, params: np.ndarray) -> None:
        """TC-TILER-02: Verify tiled evaluation matches full-grid evaluation with L_inf = 0.0 strictly."""
        width, height = 256, 256
        tile_size = 64
        domain = CoordinateDomain2D.unit_square(width=width, height=height)
        ast = parse(expr_str)

        # Monolithic full-grid evaluation
        X_full, Y_full = domain.to_discrete_grid(width, height)
        evaluator = MultiTierEvaluator(preferred_backend="numpy")
        full_res = evaluator.evaluate(ast, X_full, Y_full, params, preferred_backend="numpy")

        # Tiled evaluation
        chunker = DomainChunker(tile_size=tile_size)
        tiled_res = chunker.evaluate_tiled(
            ast, domain, width=width, height=height, params=params, preferred_backend="numpy"
        )

        # Assert strict L_inf = 0.0
        linf = float(np.max(np.abs(tiled_res - full_res)))
        assert linf == 0.0, f"Grid parity failure for '{expr_str}': L_inf = {linf}"

        # Boundary seam continuity check across tile boundaries (row/column 63-64, 127-128, 191-192)
        for seam in [63, 64, 127, 128, 191, 192]:
            assert np.array_equal(tiled_res[seam, :], full_res[seam, :])
            assert np.array_equal(tiled_res[:, seam], full_res[:, seam])

    @pytest.mark.parametrize(
        "domain",
        [
            CoordinateDomain2D.unit_square(width=300, height=300),
            CoordinateDomain2D(width=320, height=180, mode=DomainMode.ASPECT_PRESERVED),
            CoordinateDomain2D(width=200, height=150, mode=DomainMode.ASPECT_PRESERVED),
            CoordinateDomain2D(width=100, height=200, mode=DomainMode.ASPECT_PRESERVED),
            CoordinateDomain2D(width=200, height=200, x_min=-10.0, x_max=10.0, y_min=5.0, y_max=15.0),
        ],
    )
    def test_grid_parity_domain_configurations(self, domain: CoordinateDomain2D) -> None:
        """TC-TILER-03: Verify L_inf = 0.0 across varied domain aspect ratios and custom bounds."""
        ast = parse("sin(pi * x) * exp(-abs(y))")
        X_full, Y_full = domain.to_discrete_grid()
        full_res = evaluate_numpy(ast, X_full, Y_full, np.array([]))

        chunker = DomainChunker(tile_size=64)
        tiled_res = chunker.evaluate_tiled(
            ast, domain=domain, width=domain.width, height=domain.height, preferred_backend="numpy"
        )

        linf = float(np.max(np.abs(tiled_res - full_res)))
        assert linf == 0.0, f"Domain mode {domain.mode} failed parity: L_inf = {linf}"


class TestRemainderBoundaries:
    """Tests for arbitrary non-multiple canvas dimensions and boundary tile handling."""

    @pytest.mark.parametrize(
        "width, height, tile_size",
        [
            (1920, 1080, 512),  # 1080p with 512 tile
            (1920, 1080, 1024), # 1080p with 1024 tile
            (500, 300, 128),    # Non-power of two
            (1000, 1000, 256),  # Decimal dimension
            (4096, 2160, 1024), # 4K Cinema UHD
            (8192, 8192, 1024), # 8K Full
            (8192, 8192, 768),  # 8K with non-multiple tile
            (101, 97, 32),      # Prime/odd canvas
            (64, 48, 256),      # Canvas smaller than tile size
        ],
    )
    def test_remainder_coverage_and_no_overlap(self, width: int, height: int, tile_size: int) -> None:
        """TC-TILER-04: Verify 100% pixel coverage with zero overlap and zero gaps for arbitrary dimensions."""
        chunker = DomainChunker(tile_size=tile_size)
        domain = CoordinateDomain2D(width=width, height=height)
        tiles = chunker.get_tiles(total_width=width, total_height=height, domain=domain)

        expected_cols = math.ceil(width / tile_size)
        expected_rows = math.ceil(height / tile_size)
        assert len(tiles) == expected_cols * expected_rows

        hit_map = np.zeros((height, width), dtype=np.int32)
        total_pixel_count = 0

        for tile in tiles:
            x_start, y_start, x_end, y_end = tile.pixel_bounds
            assert 0 <= x_start < x_end <= width
            assert 0 <= y_start < y_end <= height
            assert tile.x_slice == slice(x_start, x_end)
            assert tile.y_slice == slice(y_start, y_end)
            assert tile.width == x_end - x_start
            assert tile.height == y_end - y_start
            assert tile.X_tile.shape == (tile.height, tile.width)
            assert tile.Y_tile.shape == (tile.height, tile.width)

            # Check boundary tile remainder sizing
            if x_end < width:
                assert tile.width == tile_size
            else:
                expected_rem_w = width % tile_size or tile_size
                assert tile.width == expected_rem_w

            if y_end < height:
                assert tile.height == tile_size
            else:
                expected_rem_h = height % tile_size or tile_size
                assert tile.height == expected_rem_h

            hit_map[tile.y_slice, tile.x_slice] += 1
            total_pixel_count += tile.width * tile.height

        # Assert every pixel visited exactly once (zero gaps, zero overlaps)
        assert np.all(hit_map == 1), f"Partition failed for ({width}x{height}, tile={tile_size})"
        assert total_pixel_count == width * height


class TestMemmapStreaming:
    """Tests for disk-backed numpy.memmap buffer allocation, streaming, and persistence."""

    def test_memmap_allocation_and_inplace_slice_writes(self, tmp_path: Path) -> None:
        """TC-TILER-05: Verify disk-backed memmap allocation, in-place slice writes, flush, and persistence."""
        mmap_path = tmp_path / "stream_test.mmap"
        width, height = 1000, 1000
        shape = (height, width)
        dtype = np.float32

        mgr = MemmapManager(shape=shape, dtype=dtype, filename=str(mmap_path), auto_cleanup=False)
        assert mmap_path.exists()
        assert mmap_path.stat().st_size == height * width * np.dtype(dtype).itemsize

        # In-place slice writes
        chunker = DomainChunker(tile_size=256)
        tiles = chunker.get_tiles(total_width=width, total_height=height)
        for tile in tiles:
            mgr.array[tile.y_slice, tile.x_slice] = np.full((tile.height, tile.width), float(tile.tile_index), dtype=dtype)

        mgr.flush()
        mgr.close()

        # Re-open in read-only mode and verify persistence
        persisted = np.memmap(str(mmap_path), dtype=dtype, mode="r", shape=shape)
        assert persisted[0, 0] == 0.0
        assert persisted[height - 1, width - 1] == float(len(tiles) - 1)
        assert not np.isnan(persisted).any()
        del persisted

    def test_memmap_context_manager_and_cleanup(self, tmp_path: Path) -> None:
        """TC-TILER-06: Verify MemmapManager context manager lifecycle and auto-cleanup on exit."""
        temp_file = tmp_path / "auto_cleaned.mmap"
        with MemmapManager(shape=(100, 100), dtype=np.float64, filename=str(temp_file), auto_cleanup=True) as mm:
            assert temp_file.exists()
            mm[0, 0] = 42.0
            assert mm[0, 0] == 42.0
        # Upon exit with auto_cleanup=True, file should be unlinked
        assert not temp_file.exists()

    def test_evaluate_tiled_streaming_into_destination_memmap(self, tmp_path: Path) -> None:
        """TC-TILER-07: Verify evaluate_tiled streams directly into destination buffer without RAM copies."""
        mmap_file = tmp_path / "direct_eval.mmap"
        width, height = 500, 300
        domain = CoordinateDomain2D(width=width, height=height)
        ast = parse("sin(2.0 * x) + cos(3.0 * y)")

        out_mmap = np.memmap(str(mmap_file), dtype=np.float64, mode="w+", shape=(height, width))
        out_mmap[:] = np.nan
        out_mmap.flush()

        evaluate_tiled(ast, domain=domain, width=width, height=height, out=out_mmap, preferred_backend="numpy")
        out_mmap.flush()

        # Verify canary NaNs completely overwritten
        assert not np.isnan(out_mmap).any()

        # Compare with reference evaluation
        X, Y = domain.to_discrete_grid(width, height)
        ref = evaluate_numpy(ast, X, Y, np.array([]))
        assert np.max(np.abs(np.array(out_mmap) - ref)) == 0.0
        del out_mmap


class TestMemoryMonitoringAndThrottling:
    """Tests for hardware telemetry and adaptive tile shrinking under simulated memory pressure."""

    def test_memory_monitor_telemetry(self) -> None:
        """TC-TILER-08: Verify psutil RAM and VRAM telemetry reporting."""
        monitor = MemoryMonitor()
        telemetry = monitor.check_memory()

        assert "available_ram_bytes" in telemetry
        assert "total_ram_bytes" in telemetry
        assert "used_ram_bytes" in telemetry
        assert "ram_percent" in telemetry
        assert telemetry["available_ram_bytes"] > 0
        assert 0.0 <= telemetry["ram_percent"] <= 100.0

    def test_adaptive_tile_shrinking_under_simulated_pressure(self) -> None:
        """TC-TILER-09: Verify adaptive tile size shrinking when memory pressure is simulated."""
        monitor = MemoryMonitor(ram_safety_threshold_bytes=500 * 1024 * 1024)

        # 1. Normal state: 8 GB available -> tile size stays 1024
        with mock.patch.object(monitor, "check_memory", return_value={
            "available_ram_bytes": 8 * 1024**3,
            "vram_available_bytes": 8 * 1024**3,
            "cuda_available": False,
        }):
            safe_size = monitor.compute_safe_tile_size(current_tile_size=1024)
            assert safe_size == 1024

        # 2. Moderate pressure: 300 MB available (< 500 MB threshold) -> tile size shrinks
        with mock.patch.object(monitor, "check_memory", return_value={
            "available_ram_bytes": 300 * 1024**2,
            "vram_available_bytes": 300 * 1024**2,
            "cuda_available": False,
        }):
            shrunk_size = monitor.compute_safe_tile_size(current_tile_size=1024)
            assert shrunk_size <= 512

        # 3. Severe pressure: 50 MB available -> clamps at minimum tile size (e.g. 64)
        with mock.patch.object(monitor, "check_memory", return_value={
            "available_ram_bytes": 50 * 1024**2,
            "vram_available_bytes": 50 * 1024**2,
            "cuda_available": False,
        }):
            clamped_size = monitor.compute_safe_tile_size(current_tile_size=1024, min_tile_size=64)
            assert clamped_size == 64

    def test_insufficient_memory_exception_raised(self) -> None:
        """TC-TILER-10: Verify InsufficientMemoryError raised when even minimum tile exceeds budget."""
        monitor = MemoryMonitor(ram_safety_threshold_bytes=100 * 1024 * 1024)
        with mock.patch.object(monitor, "check_memory", return_value={
            "available_ram_bytes": 1024,
            "vram_available_bytes": 1024,
            "cuda_available": False,
        }):
            with pytest.raises((InsufficientMemoryError, MemoryError)):
                monitor.compute_safe_tile_size(current_tile_size=1024, min_tile_size=64, fail_on_exhaustion=True)


class Test8KSimulation:
    """Tests for 8K evaluation working set memory ceiling (< 300 MB)."""

    def test_8k_evaluation_ram_limit_under_300mb(self, tmp_path: Path) -> None:
        """TC-TILER-11: Verify host RAM stays strictly < 300 MB during full 8K evaluation using MemmapManager."""
        width, height = 8192, 8192
        tile_size = 1024
        domain = CoordinateDomain2D.unit_square(width=width, height=height)
        ast = parse("sin(x) + cos(y)")
        mmap_file = tmp_path / "8k_sim.mmap"

        proc = psutil.Process()
        initial_rss = proc.memory_info().rss
        peak_rss = initial_rss

        with MemmapManager(shape=(height, width), dtype=np.float32, filename=str(mmap_file), auto_cleanup=True) as mm:
            chunker = DomainChunker(tile_size=tile_size)
            tiles = chunker.get_tiles(total_width=width, total_height=height, domain=domain)
            assert len(tiles) == 64

            for i, tile in enumerate(tiles):
                # Evaluate tile on CPU
                res_tile = evaluate_numpy(ast, tile.X_tile, tile.Y_tile, np.array([]))
                mm[tile.y_slice, tile.x_slice] = res_tile.astype(np.float32, copy=False)
                del res_tile
                if i % 8 == 7:
                    mm.flush()
                    MemoryMonitor.cleanup()
                current_rss = proc.memory_info().rss
                if current_rss > peak_rss:
                    peak_rss = current_rss

            mm.flush()
            MemoryMonitor.cleanup()

            # Verify spot check points
            # (0, 0): x = -1.0, y = -1.0
            expected_corner = ast.eval_scalar(-1.0, -1.0, [])
            assert math.isclose(float(mm[0, 0]), expected_corner, abs_tol=1e-6)

            # (4096, 4096): center
            center_x, center_y = domain.to_continuous(4096, 4096)
            expected_center = ast.eval_scalar(center_x, center_y, [])
            assert math.isclose(float(mm[4096, 4096]), expected_center, abs_tol=1e-6)

        # Assert host RAM increase strictly < 300 MB
        delta_rss_mb = (peak_rss - initial_rss) / (1024.0 * 1024.0)
        assert delta_rss_mb < 300.0, f"Host RAM increased by {delta_rss_mb:.2f} MB, exceeding 300 MB ceiling!"
