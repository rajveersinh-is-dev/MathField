"""
benchmarks/run_benchmarks.py

Importable empirical benchmark runner for MathField.

The runner executes real workloads (classical baselines and pipeline
ablations) with discarded warmups and repeated timed runs, then reports the
mean, sample standard deviation and a 95% confidence interval for each case.
No precomputed scores or synthetic timing values are used anywhere.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

import numpy as np


@dataclass(frozen=True, slots=True)
class TimingResult:
    """High-resolution timing profile with repeated-run statistics."""

    mean_ms: float
    warmup_count: int
    timed_count: int
    raw_times: List[float]
    std_ms: float = 0.0
    ci95_ms: float = 0.0
    median_ms: float = 0.0


def summarize_times(samples: Sequence[float]) -> tuple[float, float, float, float]:
    """Return (mean, sample stdev, 95% CI half-width, median) in milliseconds."""
    values = list(samples)
    if not values:
        raise ValueError("Cannot summarize an empty timing sample")
    count = len(values)
    mean = statistics.fmean(values)
    median = statistics.median(values)
    std = statistics.stdev(values) if count > 1 else 0.0
    if count > 1:
        try:
            from scipy.stats import t

            critical = float(t.ppf(0.975, count - 1))
        except Exception:  # pragma: no cover - scipy is a core dependency
            critical = 1.96
        ci95 = critical * std / math.sqrt(count)
    else:
        ci95 = 0.0
    return mean, std, ci95, median


class BenchmarkRunner:
    """Coordinates baselines, ablations and repeated high-resolution timing."""

    def _time_calls(
        self,
        func: Callable[[], Any],
        warmup_runs: int,
        timed_runs: int,
    ) -> tuple[TimingResult, Any]:
        for _ in range(max(0, int(warmup_runs))):
            func()

        samples: List[float] = []
        last_result: Any = None
        for _ in range(max(1, int(timed_runs))):
            start = time.perf_counter()
            last_result = func()
            samples.append((time.perf_counter() - start) * 1000.0)

        mean, std, ci95, median = summarize_times(samples)
        return (
            TimingResult(
                mean_ms=mean,
                warmup_count=max(0, int(warmup_runs)),
                timed_count=max(1, int(timed_runs)),
                raw_times=samples,
                std_ms=std,
                ci95_ms=ci95,
                median_ms=median,
            ),
            last_result,
        )

    def measure_function(
        self,
        func: Callable[[], Any],
        warmup_runs: int = 5,
        timed_runs: int = 10,
    ) -> TimingResult:
        """Discard warmups and time the workload repeatedly."""
        timing, _ = self._time_calls(func, warmup_runs, timed_runs)
        return timing

    def _run_case(
        self,
        kind: str,
        name: str,
        func: Callable[[], Any],
        warmup_runs: int,
        timed_runs: int,
    ) -> Dict[str, Any]:
        timing, result = self._time_calls(func, warmup_runs, timed_runs)
        record: Dict[str, Any] = {
            "kind": kind,
            "name": name,
            "mean_ms": timing.mean_ms,
            "std_ms": timing.std_ms,
            "ci95_ms": timing.ci95_ms,
            "median_ms": timing.median_ms,
            "warmup_count": timing.warmup_count,
            "timed_count": timing.timed_count,
            "raw_times_ms": list(timing.raw_times),
        }
        if result is not None:
            for field in ("mse", "psnr", "runtime_ms", "num_parameters", "config"):
                if hasattr(result, field):
                    value = getattr(result, field)
                    record[field] = dict(value) if isinstance(value, dict) else value
        return record

    def benchmark_baselines(
        self,
        img: Any,
        names: Optional[Sequence[str]] = None,
        warmup_runs: int = 3,
        timed_runs: int = 7,
        **kwargs: Any,
    ) -> List[Dict[str, Any]]:
        """Time every requested classical baseline on the supplied image."""
        from benchmarks.baselines import BASELINE_NAMES, run_baseline

        selected = list(names) if names else list(BASELINE_NAMES)
        unknown = set(selected) - set(BASELINE_NAMES)
        if unknown:
            raise ValueError(f"Unknown baselines: {sorted(unknown)}")

        records: List[Dict[str, Any]] = []
        for name in selected:
            case = partial(run_baseline, name, img, **kwargs)
            records.append(self._run_case("baseline", name, case, warmup_runs, timed_runs))
        return records

    def benchmark_ablations(
        self,
        img: Any,
        names: Optional[Sequence[str]] = None,
        warmup_runs: int = 3,
        timed_runs: int = 7,
        **kwargs: Any,
    ) -> List[Dict[str, Any]]:
        """Time every requested Mode B ablation on the supplied image."""
        from benchmarks.ablations import ABLATION_NAMES, run_ablation

        selected = list(names) if names else list(ABLATION_NAMES)
        unknown = set(selected) - set(ABLATION_NAMES)
        if unknown:
            raise ValueError(f"Unknown ablations: {sorted(unknown)}")

        records: List[Dict[str, Any]] = []
        for name in selected:
            case = partial(run_ablation, name, img, **kwargs)
            records.append(self._run_case("ablation", name, case, warmup_runs, timed_runs))
        return records

    def run_all(
        self,
        img: Any,
        warmup_runs: int = 3,
        timed_runs: int = 7,
        baseline_names: Optional[Sequence[str]] = None,
        ablation_names: Optional[Sequence[str]] = None,
        baseline_kwargs: Optional[Dict[str, Any]] = None,
        ablation_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Run baselines and ablations and return one structured report."""
        return {
            "baselines": self.benchmark_baselines(
                img,
                names=baseline_names,
                warmup_runs=warmup_runs,
                timed_runs=timed_runs,
                **(baseline_kwargs or {}),
            ),
            "ablations": self.benchmark_ablations(
                img,
                names=ablation_names,
                warmup_runs=warmup_runs,
                timed_runs=timed_runs,
                **(ablation_kwargs or {}),
            ),
            "warmup_count": warmup_runs,
            "timed_count": timed_runs,
        }

    def export_json(self, data: Dict[str, Any], path: Union[str, Path]) -> None:
        """Export benchmark measurements to a JSON file."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)


def run_benchmarks(img: Any, **kwargs: Any) -> Dict[str, Any]:
    """Convenience entry point: run the complete benchmark suite."""
    return BenchmarkRunner().run_all(img, **kwargs)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="MathField empirical benchmark runner")
    parser.add_argument("--image", type=Path, default=None, help="Input image (default: bundled analytic sample)")
    parser.add_argument("--warmup-runs", type=int, default=3)
    parser.add_argument("--timed-runs", type=int, default=7)
    parser.add_argument("--output", type=Path, default=None, help="Optional JSON output path")
    args = parser.parse_args(argv)

    if args.image is None:
        from examples.generate_sample import build_sample

        image = build_sample(64, 48)
    else:
        from PIL import Image

        image = np.asarray(Image.open(args.image).convert("RGB"), dtype=np.float64) / 255.0

    report = run_benchmarks(
        image,
        warmup_runs=args.warmup_runs,
        timed_runs=args.timed_runs,
    )
    payload = json.dumps(report, indent=2)
    print(payload)
    if args.output is not None:
        BenchmarkRunner().export_json(report, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
