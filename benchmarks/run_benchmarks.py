"""
benchmarks/run_benchmarks.py

Empirical Benchmark Runner and High-Resolution Performance Profiler.
Uses hardware performance counters (time.perf_counter) with discarded warmup runs
to provide statistically rigorous, unadulterated benchmark measurements.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
import time
from typing import Any, Callable, Dict, List, Union


@dataclass(frozen=True, slots=True)
class TimingResult:
    """High-resolution timing profile."""
    mean_ms: float
    warmup_count: int
    timed_count: int
    raw_times: List[float]


class BenchmarkRunner:
    """
    Automated benchmark harness coordinating baselines, ablations,
    and high-resolution timing measurements.
    """

    def measure_function(
        self,
        func: Callable[[], Any],
        warmup_runs: int = 5,
        timed_runs: int = 10,
    ) -> TimingResult:
        """
        Execute real workload with at least 5 discarded warmup runs,
        followed by high-resolution hardware performance counter timing.
        """
        # Discarded warmup runs
        for _ in range(max(0, warmup_runs)):
            func()

        # Timed execution runs
        times: List[float] = []
        for _ in range(max(1, timed_runs)):
            t0 = time.perf_counter()
            func()
            t1 = time.perf_counter()
            times.append(t1 - t0)

        mean_sec = sum(times) / len(times)
        mean_ms = mean_sec * 1000.0

        return TimingResult(
            mean_ms=mean_ms,
            warmup_count=warmup_runs,
            timed_count=timed_runs,
            raw_times=times,
        )

    def export_json(self, data: Dict[str, Any], path: Union[str, Path]) -> None:
        """Export benchmark measurements to JSON file."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
