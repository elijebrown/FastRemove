"""One remove_background call start to finish.

Four modes, depending on input params:

    chroma  fast   colour-difference ramp only; no model, no torch
    chroma  full   colour trimap + FeyNobg holdout + ViTMatte + despill
    subject fast   FeyNobg matte only
    subject full   FeyNobg trimap + ViTMatte + foreground unmixing
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from fastremove import pipeline

LOGGER = logging.getLogger(__name__)

ProgressCallback = Callable[[int, int, dict[str, Any]], None]
HOLDOUT_THRESHOLD = 0.95


@dataclass(frozen=True)
class ChromaSettings:

    key_color: tuple[int, int, int]
    fg_key_threshold: int = pipeline.DEFAULT_FG_KEY_THRESHOLD
    bg_key_threshold: int = pipeline.DEFAULT_BG_KEY_THRESHOLD
    despill: str = pipeline.DEFAULT_DESPILL


@dataclass(frozen=True)
class Job:

    inputs: list[Path]
    outputs: list[Path]
    chroma: ChromaSettings | None
    fast: bool
    band_radius: int = pipeline.DEFAULT_BAND_RADIUS
    output_size: int = 0

    @property
    def needs_subject_model(self) -> bool:
        return self.chroma is None or not self.fast

    @property
    def needs_vitmatte(self) -> bool:
        return not self.fast


@dataclass
class Stages:

    runtime: Any = None
    subject: tuple[Any, Any] | None = None
    vitmatte: tuple[Any, Any] | None = None
    names: list[str] = field(default_factory=list)

    @classmethod
    def load(cls, job: Job) -> Stages:
        if not (job.needs_subject_model or job.needs_vitmatte):
            return cls()

        from fastremove import models

        stages = cls(runtime=models.resolve_runtime())
        if job.needs_subject_model:
            stages.subject = models.load_subject_model(stages.runtime)
            stages.names.append(models.SUBJECT_MODEL)
        if job.needs_vitmatte:
            stages.vitmatte = models.load_vitmatte(stages.runtime)
            stages.names.append(models.VITMATTE_MODEL)
        return stages

    def release(self) -> None:
        if self.runtime is None:
            return
        from fastremove import models

        self.subject = None
        self.vitmatte = None
        models.release(self.runtime)

    def describe(self) -> dict[str, Any]:
        runtime = (
            self.runtime.describe()
            if self.runtime
            else {"device": "cpu", "dtype": "float32"}
        )
        return {**runtime, "models": list(self.names)}

    # -- model calls -------------------------------------------------------------

    def subject_matte(self, image: Image.Image) -> np.ndarray:
        from fastremove import inference

        model, processor = self.subject
        return inference.subject_matte(model, processor, image, self.runtime)

    def refine(self, image: Image.Image, trimap: np.ndarray) -> np.ndarray:
        from fastremove import inference, models

        model, processor = self.vitmatte
        alpha = inference.refine_with_vitmatte(
            model, processor, image, trimap, models.VITMATTE_MAX_SIDE, self.runtime
        )
        return pipeline.compose_alpha(alpha, trimap)

    def unmix_foreground(self, image: Image.Image, alpha: np.ndarray) -> np.ndarray:
        from fastremove import inference

        _, processor = self.subject
        return inference.unmix_foreground(processor, image, alpha)


def chroma_cutout(
    image: Image.Image, settings: ChromaSettings, job: Job, stages: Stages
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Key a flat chroma background. Returns float RGB, alpha, and details."""
    difference = pipeline.key_difference(image, settings.key_color)
    details: dict[str, Any] = {}

    if job.fast:
        alpha = pipeline.key_alpha_ramp(
            difference, settings.fg_key_threshold, settings.bg_key_threshold
        )
        details["alpha"] = "ramp"
    else:
        trimap = pipeline.build_trimap(
            difference, settings.fg_key_threshold, settings.bg_key_threshold
        )
        holdout = stages.subject_matte(image) > HOLDOUT_THRESHOLD
        trimap = pipeline.apply_holdout(trimap, holdout)
        trimap = pipeline.widen_unknown_band(trimap, job.band_radius, holdout)
        alpha = stages.refine(image, trimap)
        details["alpha"] = "vitmatte"
        details["unknown_percent"] = round(pipeline.unknown_share(trimap), 3)

    rgb = np.asarray(image, dtype=np.float32)
    rgb = pipeline.decontaminate(rgb, alpha, settings.key_color, settings.despill)
    return rgb, alpha, details


def subject_cutout(
    image: Image.Image, job: Job, stages: Stages
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Cut a subject out of a real background. Returns float RGB, alpha, details."""
    matte = stages.subject_matte(image)
    details: dict[str, Any] = {}

    if job.fast:
        alpha = matte
        rgb = np.asarray(image, dtype=np.float32)
        details["alpha"] = "feynobg"
    else:
        confident = matte >= HOLDOUT_THRESHOLD
        trimap = pipeline.widen_unknown_band(
            pipeline.trimap_from_matte(matte), job.band_radius, confident
        )
        alpha = stages.refine(image, trimap)
        rgb = stages.unmix_foreground(image, alpha)
        details["alpha"] = "vitmatte"
        details["unknown_percent"] = round(pipeline.unknown_share(trimap), 3)

    return rgb, alpha, details


def process_image(
    input_path: Path, output_path: Path, job: Job, stages: Stages
) -> dict[str, Any]:
    """Load, matte, decontaminate, and write one image."""
    with Image.open(input_path) as source:
        image = source.convert("RGB")

    if job.chroma is not None:
        rgb, alpha, details = chroma_cutout(image, job.chroma, job, stages)
    else:
        rgb, alpha, details = subject_cutout(image, job, stages)

    if job.output_size:
        rgb, alpha = pipeline.downscale_rgba(rgb, alpha, job.output_size)
        if job.chroma is not None and job.chroma.despill in ("limit", "both"):
            # Resampling can put a little key colour back on edge pixels.
            rgb = pipeline.despill_limit(rgb, job.chroma.key_color)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pipeline.to_rgba_image(rgb, alpha).save(output_path)

    height, width = alpha.shape
    return {
        "input": str(input_path),
        "output": str(output_path),
        "source_size": [image.width, image.height],
        "output_size": [width, height],
        **details,
    }


def run(job: Job, on_progress: ProgressCallback | None = None) -> dict[str, Any]:
    results: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    total = len(job.inputs)

    stages = Stages.load(job)
    LOGGER.info("remove_background starting: %d image(s), %s", total, stages.describe())
    try:
        for done, (input_path, output_path) in enumerate(
            zip(job.inputs, job.outputs, strict=True), start=1
        ):
            try:
                result = process_image(input_path, output_path, job, stages)
            except Exception as exc:
                LOGGER.exception("remove_background failed for %s", input_path)
                failures.append({"input": str(input_path), "error": str(exc)})
                continue
            results.append(result)
            if on_progress is not None:
                on_progress(done, total, result)
    finally:
        stages.release()

    return {
        "count": len(results),
        "failed": len(failures),
        "mode": "chroma" if job.chroma is not None else "subject",
        "fast": job.fast,
        "runtime": stages.describe(),
        "results": results,
        "failures": failures,
    }
