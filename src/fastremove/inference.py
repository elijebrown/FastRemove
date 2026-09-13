"""Model forward passes. Takes PIL images and numpy trimaps, returns numpy mattes."""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image

from fastremove.models import Runtime


def subject_matte(model, processor, image: Image.Image, runtime: Runtime) -> np.ndarray:
    """Return FeyNobg's alpha matte for one image, float32 in [0, 1]."""
    inputs = processor(image, return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(device=runtime.device, dtype=runtime.dtype)
    with torch.inference_mode():
        outputs = model(pixel_values=pixel_values)

    matte = processor.post_process_alpha_matting(
        outputs, target_sizes=[(image.height, image.width)]
    )[0]
    return np.clip(matte.detach().float().cpu().numpy(), 0.0, 1.0).astype(np.float32)


def refine_with_vitmatte(
    model,
    processor,
    image: Image.Image,
    trimap: np.ndarray,
    max_side: int,
    runtime: Runtime,
) -> np.ndarray:
    """Refine the unknown band into continuous alpha in [0, 1].

    Anything longer than `max_side` is solved at
    a reduced size and resampled back up.
    """
    height, width = trimap.shape
    scale = min(1.0, max_side / max(height, width))
    work_height, work_width = height, width
    work_image, work_trimap = image, trimap

    if scale < 1.0:
        work_height = max(1, round(height * scale))
        work_width = max(1, round(width * scale))
        work_image = image.resize((work_width, work_height), Image.LANCZOS)
        work_trimap = np.asarray(
            Image.fromarray(trimap, mode="L").resize(
                (work_width, work_height), Image.NEAREST
            )
        )

    inputs = processor(
        images=work_image,
        trimaps=Image.fromarray(work_trimap, mode="L"),
        return_tensors="pt",
    )
    inputs = {
        name: (
            value.to(device=runtime.device, dtype=runtime.dtype)
            if value.is_floating_point()
            else value.to(runtime.device)
        )
        for name, value in inputs.items()
    }
    with torch.inference_mode():
        alphas = model(**inputs).alphas

    alpha = alphas[0, 0].detach().float().cpu().numpy()[:work_height, :work_width]
    if scale < 1.0:
        alpha = np.asarray(
            Image.fromarray(alpha.astype(np.float32), mode="F").resize(
                (width, height), Image.BILINEAR
            )
        )
    return np.clip(alpha, 0.0, 1.0).astype(np.float32)


def unmix_foreground(processor, image: Image.Image, alpha: np.ndarray) -> np.ndarray:
    """Estimate the unmixed foreground colour under a soft matte, as float32 RGB."""
    refined = processor.refine_foreground(image, torch.from_numpy(alpha))
    return np.asarray(refined.convert("RGB"), dtype=np.float32)
