"""
mathfield/core/pipeline/controller.py

Reconstruction Controller for 5-Stage Mathematical Field Pipeline.
Orchestrates Mode A (Discovery) and Mode B (Fidelity), cooperative thread execution
(pause/resume/cancel), and Qt UI progress callbacks.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Union

import numpy as np

from mathfield.core.domain import CoordinateDomain2D, DomainMode
from mathfield.core.pipeline.mode_a import ModeAPipeline
from mathfield.core.pipeline.mode_b import ModeBPipeline
from mathfield.core.pipeline.stage import PipelineContext, StageStatus


class ReconstructionController:
    """
    Coordinates end-to-end execution of mathematical field reconstruction.
    """

    def __init__(self) -> None:
        self.context: Optional[PipelineContext] = None
        self.current_pipeline: Optional[Union[ModeAPipeline, ModeBPipeline]] = None

    def start(
        self,
        image: np.ndarray,
        mode: str = "discovery",
        domain: Optional[CoordinateDomain2D] = None,
        config: Optional[Dict[str, Any]] = None,
        on_stage_changed: Optional[Callable[[str, int, int], None]] = None,
        on_iteration: Optional[Callable[[int, Dict[str, float]], None]] = None,
        on_preview: Optional[Callable[[np.ndarray], None]] = None,
        on_candidate: Optional[Callable[[Dict[str, Any]], None]] = None,
        on_finished: Optional[Callable[[PipelineContext], None]] = None,
    ) -> PipelineContext:
        """
        Initiate reconstruction pipeline execution synchronously or within a worker thread.
        """
        cfg = config or {}
        img_arr = np.asarray(image)

        if domain is None:
            h, w = img_arr.shape[:2]
            domain = CoordinateDomain2D(width=w, height=h, mode=DomainMode.ASPECT_PRESERVED)

        self.context = PipelineContext(
            image=img_arr,
            domain=domain,
            config=cfg,
        )

        norm_mode = mode.strip().lower()
        if norm_mode in ("discovery", "mode_a", "a"):
            pipeline = ModeAPipeline(config=cfg)
        elif norm_mode in ("fidelity", "mode_b", "b"):
            pipeline = ModeBPipeline(config=cfg)
        else:
            raise ValueError(f"Unknown reconstruction mode: {mode}")

        self.current_pipeline = pipeline

        def stage_callback(stage_name: str, stage_idx: int, total_stages: int) -> None:
            if on_stage_changed:
                on_stage_changed(stage_name, stage_idx, total_stages)
            if on_preview and self.context.rendered_image is not None:
                on_preview(self.context.rendered_image)
            if on_iteration and self.context.metrics:
                on_iteration(stage_idx, self.context.metrics)
            if on_candidate and self.context.global_model:
                on_candidate(self.context.global_model)

        status = pipeline.execute(self.context, progress_callback=stage_callback)

        if on_finished:
            on_finished(self.context)

        return self.context

    def pause(self) -> None:
        """Cooperatively pause running reconstruction."""
        if self.context:
            self.context.pause_flag = True

    def resume(self) -> None:
        """Resume paused reconstruction."""
        if self.context:
            self.context.pause_flag = False

    def cancel(self) -> None:
        """Cooperatively cancel running reconstruction."""
        if self.context:
            self.context.cancel_flag = True
            self.context.pause_flag = False
