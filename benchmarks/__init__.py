"""
benchmarks package.
Rigorous empirical benchmarking suite comparing MathField against classical representations:
Chebyshev, Fourier, Wavelets, Splines, RBFs, and 4 structured ablation studies.
Adheres strictly to Computational Honesty: zero pre-computed lookup tables or shortcut heuristics.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "ABLATION_NAMES",
    "BASELINE_NAMES",
    "BenchmarkRunner",
    "TimingResult",
    "get_ablation_config",
    "list_ablations",
    "list_baselines",
    "run_ablation",
    "run_baseline",
    "run_benchmarks",
]

_BASELINE_EXPORTS = {"BASELINE_NAMES", "list_baselines", "run_baseline"}
_ABLATION_EXPORTS = {
    "ABLATION_NAMES",
    "get_ablation_config",
    "list_ablations",
    "run_ablation",
}
_RUNNER_EXPORTS = {"BenchmarkRunner", "TimingResult", "run_benchmarks"}


def __getattr__(name: str) -> Any:
    """Lazily resolve exports so `python -m benchmarks.run_benchmarks` is warning-free."""
    if name in _BASELINE_EXPORTS:
        return getattr(import_module("benchmarks.baselines"), name)
    if name in _ABLATION_EXPORTS:
        return getattr(import_module("benchmarks.ablations"), name)
    if name in _RUNNER_EXPORTS:
        module = import_module("benchmarks.run_benchmarks")
        value = getattr(module, name)
        # import_module binds the submodule under `run_benchmarks` on the
        # package; restore the public function export after that side effect,
        # regardless of which runner symbol was requested first.
        globals()["run_benchmarks"] = module.run_benchmarks
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
