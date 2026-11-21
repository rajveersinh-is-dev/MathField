"""Regression tests for consistent parameter accounting and the benchmark runner."""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
from scipy.interpolate import RectBivariateSpline

from benchmarks.baselines import run_baseline
from benchmarks.run_benchmarks import BenchmarkRunner, run_benchmarks, summarize_times
from examples.generate_sample import build_sample
from mathfield.core.basis.fourier import Fourier2D
from mathfield.core.basis.polynomial import Chebyshev2D
from mathfield.core.domain import CoordinateDomain2D


def test_baseline_parameter_counts_are_totals_over_channels() -> None:
    image = build_sample(48, 32)
    h, w, c = image.shape
    domain = CoordinateDomain2D(width=w, height=h)
    X, Y = domain.to_discrete_grid()

    cheb = Chebyshev2D(deg_x=4, deg_y=4, truncation="total_degree")
    fourier = Fourier2D(freq_x=4, freq_y=4, truncation="rectangular")
    expected_cheb = cheb.basis_matrix(X, Y).shape[1] * c
    expected_fourier = fourier.basis_matrix(X, Y).shape[1] * c
    expected_rbf = 4 * 4 * c  # isqrt(rbf_count=16)^2 centres per channel

    n_levels = max(1, int(math.log2(min(h, w))) - 2)
    block = max(2, min(h, w) // (2**n_levels))
    expected_wavelets = (h // block) * (w // block) * c

    spline_coeffs = 0
    for channel in range(c):
        spline = RectBivariateSpline(
            np.linspace(-1, 1, h),
            np.linspace(-1, 1, w),
            image[:, :, channel],
            kx=3,
            ky=3,
            s=0.01,
        )
        spline_coeffs += int(spline.get_coeffs().size)

    expected = {
        "chebyshev": expected_cheb,
        "fourier": expected_fourier,
        "wavelets": expected_wavelets,
        "splines": spline_coeffs,
        "rbf": expected_rbf,
    }
    for name, count in expected.items():
        result = run_baseline(name, image, degree=4, freq=4, rbf_count=16)
        assert result.num_parameters == count, f"{name}: expected {count}, got {result.num_parameters}"
        assert result.num_parameters % c == 0


def test_summary_statistics_are_finite_and_positive() -> None:
    mean, std, ci, median = summarize_times([1.0, 2.0, 3.0, 4.0])
    assert mean == 2.5 and median == 2.5
    assert std > 0.0 and ci > 0.0
    assert summarize_times([5.0])[1:] == (0.0, 0.0, 5.0)


def test_runner_times_baselines_and_ablations_with_statistics(tmp_path: Path) -> None:
    image = build_sample(48, 32)
    runner = BenchmarkRunner()

    timing = runner.measure_function(lambda: time.sleep(0.001), warmup_runs=2, timed_runs=4)
    assert timing.warmup_count == 2 and timing.timed_count == 4
    assert timing.std_ms >= 0.0 and timing.ci95_ms >= 0.0
    assert timing.mean_ms > 0.5

    baselines = runner.benchmark_baselines(
        image,
        names=["chebyshev", "rbf"],
        warmup_runs=1,
        timed_runs=3,
        degree=3,
        rbf_count=9,
    )
    assert [record["name"] for record in baselines] == ["chebyshev", "rbf"]
    for record in baselines:
        assert record["mean_ms"] > 0.0
        assert record["std_ms"] >= 0.0 and record["ci95_ms"] >= 0.0
        assert len(record["raw_times_ms"]) == 3
        assert record["mse"] >= 0.0 and record["psnr"] > 0.0
        assert record["num_parameters"] % 3 == 0

    ablations = runner.benchmark_ablations(
        image,
        names=["no_quadtree", "no_varpro"],
        warmup_runs=1,
        timed_runs=2,
    )
    assert [record["name"] for record in ablations] == ["no_quadtree", "no_varpro"]
    for record in ablations:
        assert record["mean_ms"] > 0.0
        assert record["config"]["max_depth"] == 0 or record["config"]["enable_varpro"] is False
        assert len(record["raw_times_ms"]) == 2

    report = run_benchmarks(
        image,
        warmup_runs=1,
        timed_runs=2,
        baseline_names=["chebyshev"],
        ablation_names=["no_quadtree"],
    )
    assert set(report) >= {"baselines", "ablations", "warmup_count", "timed_count"}
    out = tmp_path / "benchmarks.json"
    runner.export_json(report, out)
    reloaded = json.loads(out.read_text(encoding="utf-8"))
    assert reloaded["baselines"][0]["ci95_ms"] >= 0.0
