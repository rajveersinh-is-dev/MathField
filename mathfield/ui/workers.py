"""
mathfield/ui/workers.py

Background Optimization Worker and Signal Architecture.
Provides non-blocking thread execution, cooperative cancellation (< 500ms latency),
safe exception emission, and thread lifecycle management.
"""

from __future__ import annotations

import sys
import threading
from typing import Any, Callable, List, Optional


class _Signal:
    """Lightweight signal emitter compatible with Qt signal semantics."""

    def __init__(self) -> None:
        self._slots: List[Callable[..., None]] = []

    def connect(self, slot: Callable[..., None]) -> None:
        if slot not in self._slots:
            self._slots.append(slot)

    def disconnect(self, slot: Optional[Callable[..., None]] = None) -> None:
        if slot is None:
            self._slots.clear()
        elif slot in self._slots:
            self._slots.remove(slot)

    def emit(self, *args: Any, **kwargs: Any) -> None:
        for slot in list(self._slots):
            try:
                slot(*args, **kwargs)
            except Exception as exc:
                print(f"Error in signal handler: {exc}", file=sys.stderr)


class WorkerSignals:
    """Cross-thread signal collection for reconstruction progress and status."""

    def __init__(self) -> None:
        self.iteration_completed = _Signal()
        self.stage_changed = _Signal()
        self.finished = _Signal()
        self.error = _Signal()
        self.progress = _Signal()


class OptimizationWorker(threading.Thread):
    """
    Dedicated worker thread running numerical equation discovery off the UI thread.
    Supports cooperative cancellation and clean teardown.
    """

    def __init__(self, task: Optional[Callable[[], Any]] = None) -> None:
        super().__init__(daemon=True)
        self.task: Optional[Callable[[], Any]] = task
        self.signals = WorkerSignals()
        self.is_cancelled: bool = False
        self._stop_requested: bool = False
        self.is_busy: bool = False
        self._result: Any = None
        self._error: Optional[Exception] = None

    def set_task(self, task: Callable[[], Any]) -> None:
        self.task = task

    def start_job(self, task: Optional[Callable[[], Any]] = None) -> bool:
        """Start a job if not already busy; cleanly reject concurrent jobs."""
        if self.is_busy or (self.is_alive() if hasattr(self, "is_alive") else False):
            return False
        if task is not None:
            self.task = task
        self.is_busy = True
        self.start()
        return True

    def execute_in_background(self, task: Callable[[], Any]) -> int:
        """Helper to run a task immediately in background and return its thread ID."""
        self.set_task(task)
        self.start()
        return self.ident or threading.get_ident()

    def request_cancel(self) -> None:
        """Set cooperative cancellation flag (< 500 ms target)."""
        self.is_cancelled = True
        self._stop_requested = True

    def stop(self) -> None:
        """Cooperative teardown: set cancel flag and wait up to 1000ms."""
        self.request_cancel()
        if self.is_alive():
            self.join(timeout=1.0)
        self.is_busy = False

    def isRunning(self) -> bool:
        """Qt compatibility alias for thread status."""
        return self.is_alive()

    def wait(self, timeout_ms: int = 1000) -> bool:
        """Qt compatibility alias for join(). Returns True if thread finished."""
        if not self.is_alive():
            return True
        self.join(timeout=max(0.001, timeout_ms / 1000.0))
        return not self.is_alive()

    def run(self) -> None:
        """Thread execution entry point with complete exception containment."""
        self.is_busy = True
        try:
            if self.task is not None:
                self._result = self.task()
            self.signals.finished.emit(self._result)
        except Exception as exc:
            self._error = exc
            self.signals.error.emit(str(exc))
        finally:
            self.is_busy = False
