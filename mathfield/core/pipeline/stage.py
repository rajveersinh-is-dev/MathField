"""
mathfield/core/pipeline/stage.py

Pipeline Stage Abstract Infrastructure and PipelineContext.
Maintains execution state, intermediate models, local decompositions,
verification metrics, and cooperative pause/cancel controls.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional

import numpy as np


class StageStatus(str, Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


@dataclass
class PipelineContext:
    """Shared state passed through sequential pipeline stages."""
    image: np.ndarray
    domain: Any
    config: Dict[str, Any] = field(default_factory=dict)
    analysis: Optional[Any] = None
    global_model: Optional[Any] = None
    local_decomposition: Optional[Any] = None
    metrics: Dict[str, float] = field(default_factory=dict)
    condition_report: Optional[Dict[str, Any]] = None
    rendered_image: Optional[np.ndarray] = None
    pause_flag: bool = False
    cancel_flag: bool = False

    def is_cancelled(self) -> bool:
        return self.cancel_flag

    def check_pause(self, poll_interval: float = 0.05) -> None:
        """Cooperative spin-wait if pause_flag is set."""
        while self.pause_flag and not self.cancel_flag:
            time.sleep(poll_interval)


class PipelineStage(ABC):
    """Abstract stage in the continuous reconstruction pipeline."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable stage title."""
        pass

    @abstractmethod
    def execute(self, context: PipelineContext) -> StageStatus:
        """Execute stage computation, updating context in-place."""
        pass
