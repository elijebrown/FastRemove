from __future__ import annotations

import gc
import logging
from dataclasses import dataclass

import torch

LOGGER = logging.getLogger(__name__)

SUBJECT_MODEL = "feyninc/FeyNobg"
VITMATTE_MODEL = "hustvl/vitmatte-small-composition-1k"
VITMATTE_MAX_SIDE = 2048


@dataclass(frozen=True)
class Runtime:
    device: torch.device
    dtype: torch.dtype

    def describe(self) -> dict[str, str]:
        return {
            "device": str(self.device),
            "dtype": str(self.dtype).removeprefix("torch."),
        }


def resolve_runtime() -> Runtime:
    """CUDA, MPS, CPU pyTorch version selection."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        dtype = torch.float32
    else:
        device = torch.device("cpu")
        dtype = torch.float32
    return Runtime(device=device, dtype=dtype)


def load_subject_model(runtime: Runtime):
    from nobg import AutoModel, AutoProcessor

    LOGGER.info("loading %s on %s", SUBJECT_MODEL, runtime.device)
    model = (
        AutoModel.from_pretrained(SUBJECT_MODEL)
        .eval()
        .to(device=runtime.device, dtype=runtime.dtype)
    )
    processor = AutoProcessor.from_pretrained(SUBJECT_MODEL)
    return model, processor


def load_vitmatte(runtime: Runtime):
    from transformers import VitMatteForImageMatting, VitMatteImageProcessor

    LOGGER.info("loading %s on %s", VITMATTE_MODEL, runtime.device)
    model = (
        VitMatteForImageMatting.from_pretrained(VITMATTE_MODEL)
        .eval()
        .to(device=runtime.device, dtype=runtime.dtype)
    )
    processor = VitMatteImageProcessor.from_pretrained(VITMATTE_MODEL)
    return model, processor


def release(runtime: Runtime) -> None:
    gc.collect()
    if runtime.device.type == "cuda":
        torch.cuda.empty_cache()
    elif runtime.device.type == "mps":
        torch.mps.empty_cache()
