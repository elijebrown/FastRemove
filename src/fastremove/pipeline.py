"""Matting

Stages:
    1. Trimap. Either from a colour-difference score (chroma key) or from a
       segmentation matte (real background).
    2. Widen the unknown band so a matting model has something to solve.
    3. ViTMatte (inference.py) turns the unknown band into continuous alpha.
    4. Re-impose the trimap definites on the refined alpha.
    5. Decontamination (chroma key only). A partial-alpha edge pixel is a real
       mixture of subject and key colour; getting alpha right does not take the
       key colour back out, and the leftover reads as a coloured fringe.
    6. Optional downscale to delivery size, premultiplied and in linear light.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

NAMED_KEY_COLORS = {"green": "#00FF00", "blue": "#0047FF"}
DEFAULT_KEY_COLOR = "green"

# Colour-difference bounds. For foreground coverage a over the key colour the
# score is 255 * (1 - a), so 32 is a ~0.875 and 160 is a ~0.37.
DEFAULT_FG_KEY_THRESHOLD = 32
DEFAULT_BG_KEY_THRESHOLD = 160

DEFAULT_FG_MATTE_THRESHOLD = 0.9
DEFAULT_BG_MATTE_THRESHOLD = 0.1

TRIMAP_BACKGROUND = 0
TRIMAP_UNKNOWN = 128
TRIMAP_FOREGROUND = 255

# Unknown band radius for chroma backgrounds.
DEFAULT_BAND_RADIUS = 8

DESPILL_MODES = ("none", "unmix", "limit", "both")
DEFAULT_DESPILL = "both"

UNMIX_EPSILON = 1e-3

# sRGB transfer function breakpoints (IEC 61966-2-1).
SRGB_LINEAR_CUTOFF = 0.04045
SRGB_ENCODED_CUTOFF = 0.0031308


# --- Chroma background ----------------------------------------------------------


def parse_key_color(value: str) -> tuple[int, int, int]:
    """Return the chroma colour as RGB. Accepts a name, #RRGGBB, or 0xRRGGBB."""
    text = NAMED_KEY_COLORS.get(value.strip().lower(), value).strip()
    for prefix in ("#", "0x", "0X"):
        text = text.removeprefix(prefix)
    if len(text) != 6:
        raise ValueError(f"key colour must be 6 hex digits or a name, got {value!r}")
    try:
        packed = int(text, 16)
    except ValueError as error:
        raise ValueError(f"key colour is not valid hex: {value!r}") from error
    return (packed >> 16) & 0xFF, (packed >> 8) & 0xFF, packed & 0xFF


def key_index_and_others(key_color: tuple[int, int, int]) -> tuple[int, list[int]]:
    """Return the dominant channel of the key colour and the other two."""
    key_index = int(np.argmax(key_color))
    return key_index, [index for index in range(3) if index != key_index]


def key_difference(image, key_color: tuple[int, int, int]) -> np.ndarray:
    pixels = np.asarray(image, dtype=np.int16)
    key_index, others = key_index_and_others(key_color)
    return pixels[:, :, key_index] - np.maximum(
        pixels[:, :, others[0]], pixels[:, :, others[1]]
    )


def key_alpha_ramp(
    difference: np.ndarray, fg_threshold: int, bg_threshold: int
) -> np.ndarray:
    """Return a colour-derived matte in [0, 1], opaque where the key is absent."""
    span = max(bg_threshold - fg_threshold, 1)
    ramp = np.clip((bg_threshold - difference) / span, 0.0, 1.0)
    return (ramp * ramp * (3.0 - 2.0 * ramp)).astype(np.float32)


def build_trimap(
    difference: np.ndarray, fg_threshold: int, bg_threshold: int
) -> np.ndarray:
    """Return a three-level trimap from the colour-difference score. Unknown colour bands sent to matting model"""
    trimap = np.full(difference.shape, TRIMAP_UNKNOWN, dtype=np.uint8)
    trimap[difference <= fg_threshold] = TRIMAP_FOREGROUND
    trimap[difference >= bg_threshold] = TRIMAP_BACKGROUND
    return trimap


def apply_holdout(trimap: np.ndarray, holdout: np.ndarray) -> np.ndarray:
    """Lock the trimap to foreground wherever the segmentation model is confident."""
    locked = trimap.copy()
    locked[holdout] = TRIMAP_FOREGROUND
    return locked


# --- Real background ----------------------------------------------------------


def trimap_from_matte(
    matte: np.ndarray,
    fg_threshold: float = DEFAULT_FG_MATTE_THRESHOLD,
    bg_threshold: float = DEFAULT_BG_MATTE_THRESHOLD,
) -> np.ndarray:
    """Return a three-level trimap from a segmentation matte in [0, 1]."""
    trimap = np.full(matte.shape, TRIMAP_UNKNOWN, dtype=np.uint8)
    trimap[matte >= fg_threshold] = TRIMAP_FOREGROUND
    trimap[matte <= bg_threshold] = TRIMAP_BACKGROUND
    return trimap


# --- Trimap refinement --------------------------------------------------------


def dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    """Grow a boolean mask by `radius` px with a diamond (4-connected) kernel."""
    grown = mask.astype(bool)
    for _ in range(radius):
        padded = np.pad(grown, 1)
        grown = (
            padded[1:-1, 1:-1]
            | padded[:-2, 1:-1]
            | padded[2:, 1:-1]
            | padded[1:-1, :-2]
            | padded[1:-1, 2:]
        )
    return grown


def erode(mask: np.ndarray, radius: int) -> np.ndarray:
    """Shrink a boolean mask by `radius` px with a diamond kernel.

    Pixels outside the frame count as True, so a shape running off the edge is
    not eroded away along that edge.
    """
    return ~dilate(~mask.astype(bool), radius)


def widen_unknown_band(
    trimap: np.ndarray, radius: int, locked: np.ndarray | None = None
) -> np.ndarray:
    """Guarantee an unknown band of `radius` px on both sides of the foreground.

    A trimap built from hard thresholds can have no unknown region at all, which
    leaves a matting model nothing to refine. Dilating and eroding the definite
    foreground and marking the difference unknown forces a band of a workable
    width."""
    if radius <= 0:
        return trimap

    foreground = trimap == TRIMAP_FOREGROUND
    band = dilate(foreground, radius) & ~erode(foreground, radius)
    if locked is not None:
        band &= ~locked

    widened = trimap.copy()
    widened[band] = TRIMAP_UNKNOWN
    return widened


def compose_alpha(alpha: np.ndarray, trimap: np.ndarray) -> np.ndarray:
    """Re-impose the trimap definites on a refined matte."""
    composed = np.where(trimap == TRIMAP_FOREGROUND, 1.0, alpha)
    return np.clip(
        np.where(trimap == TRIMAP_BACKGROUND, 0.0, composed), 0.0, 1.0
    ).astype(np.float32)


def unknown_share(trimap: np.ndarray) -> float:
    """Percentage of the frame left for a matting model to decide."""
    return float(np.count_nonzero(trimap == TRIMAP_UNKNOWN)) / trimap.size * 100.0


# --- Decontamination ----------------------------------------------------------


def despill_unmix(
    rgb: np.ndarray, alpha: np.ndarray, key_color: tuple[int, int, int]
) -> np.ndarray:
    """Solve the matting equation I = a*F + (1-a)*B for F, with B known."""
    background = np.asarray(key_color, dtype=np.float32).reshape(1, 1, 3)
    coverage = np.clip(alpha, UNMIX_EPSILON, None).astype(np.float32)[:, :, None]
    foreground = np.clip((rgb - (1.0 - coverage) * background) / coverage, 0.0, 255.0)
    return np.where(alpha[:, :, None] > 0.0, foreground, rgb).astype(np.float32)


def despill_limit(rgb: np.ndarray, key_color: tuple[int, int, int]) -> np.ndarray:
    key_index, others = key_index_and_others(key_color)
    limited = rgb.astype(np.float32).copy()
    limited[:, :, key_index] = np.minimum(
        limited[:, :, key_index],
        np.maximum(limited[:, :, others[0]], limited[:, :, others[1]]),
    )
    return limited


def decontaminate(
    rgb: np.ndarray,
    alpha: np.ndarray,
    key_color: tuple[int, int, int],
    mode: str,
) -> np.ndarray:
    if mode not in DESPILL_MODES:
        raise ValueError(f"despill must be one of {DESPILL_MODES}, got {mode!r}")
    if mode in ("unmix", "both"):
        rgb = despill_unmix(rgb, alpha, key_color)
    if mode in ("limit", "both"):
        rgb = despill_limit(rgb, key_color)
    return rgb


# --- Output -------------------------------------------------------------------


def srgb_to_linear(values: np.ndarray) -> np.ndarray:
    """Decode sRGB in [0, 1] to linear light."""
    safe = np.clip(values, 0.0, 1.0)
    return np.where(
        safe <= SRGB_LINEAR_CUTOFF,
        safe / 12.92,
        np.power((safe + 0.055) / 1.055, 2.4),
    ).astype(np.float32)


def linear_to_srgb(values: np.ndarray) -> np.ndarray:
    """Encode linear light in [0, 1] back to sRGB."""
    safe = np.clip(values, 0.0, 1.0)
    return np.where(
        safe <= SRGB_ENCODED_CUTOFF,
        safe * 12.92,
        1.055 * np.power(safe, 1.0 / 2.4) - 0.055,
    ).astype(np.float32)


def resize_channel(channel: np.ndarray, width: int, height: int) -> np.ndarray:
    """Lanczos-resample one float32 plane. Pillow has no multi-channel float mode."""
    return np.asarray(
        Image.fromarray(channel.astype(np.float32), mode="F").resize(
            (width, height), Image.LANCZOS
        ),
        dtype=np.float32,
    )


def downscale_rgba(
    rgb: np.ndarray, alpha: np.ndarray, max_side: int
) -> tuple[np.ndarray, np.ndarray]:
    height, width = alpha.shape
    longest = max(height, width)
    if max_side <= 0 or longest <= max_side:
        return rgb, alpha

    scale = max_side / longest
    target_width = max(1, round(width * scale))
    target_height = max(1, round(height * scale))

    linear = srgb_to_linear(rgb / 255.0)
    premultiplied = linear * alpha[:, :, None]

    # Lanczos rings slightly, so clip both the premultiplied colour and alpha back into range
    small_premultiplied = np.clip(
        np.dstack(
            [
                resize_channel(premultiplied[:, :, index], target_width, target_height)
                for index in range(3)
            ]
        ),
        0.0,
        1.0,
    )
    small_alpha = np.clip(resize_channel(alpha, target_width, target_height), 0.0, 1.0)
    small_premultiplied = np.minimum(small_premultiplied, small_alpha[:, :, None])

    divisor = np.maximum(small_alpha, UNMIX_EPSILON)[:, :, None]
    small_linear = np.clip(small_premultiplied / divisor, 0.0, 1.0)
    small_rgb = linear_to_srgb(small_linear) * 255.0
    small_rgb = np.where(small_alpha[:, :, None] > 0.0, small_rgb, 0.0)
    return small_rgb.astype(np.float32), small_alpha.astype(np.float32)


def to_rgba_image(rgb: np.ndarray, alpha: np.ndarray) -> Image.Image:
    """Stack float RGB and alpha into an 8-bit RGBA image."""
    stacked = np.dstack(
        [np.clip(rgb, 0.0, 255.0), np.clip(alpha * 255.0, 0.0, 255.0)]
    ).astype(np.uint8)
    return Image.fromarray(stacked, mode="RGBA")
