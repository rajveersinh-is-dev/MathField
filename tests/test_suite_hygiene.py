"""Meta-tests protecting the honesty and observability of the test suite itself."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TIERS = [
    ROOT / "e2e_tests" / f"test_e2e_tier{n}.py"
    for n in (1, 2, 3, 4)
]


def test_e2e_suites_do_not_mask_import_regressions() -> None:
    for path in TIERS:
        source = path.read_text(encoding="utf-8")
        # Import failures must fail the suite, never silently skip it.
        assert not re.search(
            r"except [^\n]*ImportError[^\n]*:\s*\n\s*(?:pass|return|pytest\.skip)", source
        )
        # No vacuous existence-only assertions.
        assert "assert hasattr(" not in source
        # Only explicit hardware-capability skips remain (CUDA / native DLL).
        skips = source.count("pytest.skip(")
        assert skips <= 3, f"{path.name} still contains {skips} skip guards"
        for match in re.finditer(r"pytest\.skip\(([^\n]*)", source):
            message = match.group(1).lower()
            assert "cuda" in message or "native" in message or "dll" in message


def test_worker_signal_bus_is_documented_as_headless() -> None:
    source = (ROOT / "mathfield" / "ui" / "workers.py").read_text(encoding="utf-8")
    assert "not a Qt signal" in source
    assert "no queued" in source
    for signal in ("stage_changed", "progress", "iteration_completed", "finished", "error"):
        assert f"signals.{signal}.emit" in source
