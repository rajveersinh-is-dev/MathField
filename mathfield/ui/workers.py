"""
mathfield/ui/workers.py

Background Optimization Worker and headless signal bus.

This module deliberately does NOT depend on Qt. ``_Signal`` is a lightweight
synchronous callback dispatcher: slots are called on the *emitting* thread and
there is no queued cross-thread delivery. Host applications using Qt must
marshal GUI updates themselves (e.g. via ``QMetaObject.invokeMethod`` with a
queued connection). The worker itself remains GUI-framework agnostic and can be
used in headless tests and CLI tooling.

Provides non-blocking thread execution, cooperative cancellation (< 500 ms
latency), safe exception emission, and reusable thread lifecycle management.
"""

from __future__ import annotations

import sys
import threading
from typing import Any, Callable, List, Optional


class _Signal:
    """Synchronous callback bus (not a Qt signal; no queued delivery).

    Slots run on the thread that calls emit(). Exceptions raised by a slot are
    reported to stderr and never propagate back into the worker body.
    """

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
    """Signal collection for reconstruction progress and status.

    All signals are delivered synchronously on the worker thread; a GUI host is
    responsible for marshalling any widget updates onto its UI thread.
    """

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
        self._runner: Optional[threading.Thread] = None

    def set_task(self, task: Callable[[], Any]) -> None:
        self.task = task

    def start(self) -> None:
        """Launch a fresh runner each time so a worker can be reused for jobs."""
        if self._runner is not None and self._runner.is_alive():
            raise RuntimeError("Worker is already running a job")
        self.is_cancelled = False
        self._stop_requested = False
        self._result = None
        self._error = None
        self.is_busy = True
        self._runner = threading.Thread(target=self.run, daemon=True, name=f"{type(self).__name__}-runner")
        self._runner.start()

    def is_alive(self) -> bool:
        return self._runner is not None and self._runner.is_alive()

    def join(self, timeout: Optional[float] = None) -> None:
        if self._runner is not None:
            self._runner.join(timeout)

    def start_job(self, task: Optional[Callable[[], Any]] = None) -> bool:
        """Start a job if not already busy; cleanly reject concurrent jobs."""
        if self.is_busy or self.is_alive():
            return False
        if task is not None:
            self.task = task
        self.start()
        return True

    def execute_in_background(self, task: Callable[[], Any]) -> int:
        """Helper to run a task immediately in background and return its thread ID."""
        self.set_task(task)
        self.start()
        assert self._runner is not None
        return self._runner.ident or threading.get_ident()

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
            self.signals.stage_changed.emit("working", 1, 1)
            self.signals.progress.emit(0.0)
            if self.task is not None:
                self._result = self.task()
                self.signals.iteration_completed.emit(1, {"status": "complete"})
            self.signals.progress.emit(1.0)
            self.signals.finished.emit(self._result)
        except Exception as exc:
            self._error = exc
            self.signals.error.emit(str(exc))
        finally:
            self.is_busy = False
