import numpy as np
import pytest
from PIL import Image

from fastremove import pipeline

GREEN = (0, 255, 0)


def green_square_frame(size: int = 64, square: int = 32) -> Image.Image:
    """White square centred on flat green."""
    frame = np.zeros((size, size, 3), dtype=np.uint8)
    frame[:, :] = GREEN
    start = (size - square) // 2
    frame[start : start + square, start : start + square] = 255
    return Image.fromarray(frame, mode="RGB")


def test_parse_key_color_accepts_names_and_hex():
    assert pipeline.parse_key_color("green") == (0, 255, 0)
    assert pipeline.parse_key_color("#0047FF") == (0, 0x47, 0xFF)
    assert pipeline.parse_key_color("0x112233") == (0x11, 0x22, 0x33)


def test_parse_key_color_rejects_garbage():
    with pytest.raises(ValueError):
        pipeline.parse_key_color("chartreuse")


def test_key_alpha_ramp_is_opaque_on_subject_and_clear_on_key():
    difference = pipeline.key_difference(green_square_frame(), GREEN)
    alpha = pipeline.key_alpha_ramp(difference, 32, 160)
    assert alpha[32, 32] == 1.0
    assert alpha[0, 0] == 0.0


def test_dilate_and_erode_are_diamond_shaped_and_respect_borders():
    mask = np.zeros((7, 7), dtype=bool)
    mask[3, 3] = True
    grown = pipeline.dilate(mask, 1)
    assert grown.sum() == 5
    assert grown[3, 2] and grown[2, 3] and not grown[2, 2]

    # A shape touching the frame edge keeps that edge under erosion.
    edge = np.zeros((5, 5), dtype=bool)
    edge[:, :3] = True
    shrunk = pipeline.erode(edge, 1)
    assert shrunk[:, 0].all()
    assert shrunk[:, :2].all() and not shrunk[:, 2].any()


def test_widen_unknown_band_forces_a_band_around_a_hard_edge():
    difference = pipeline.key_difference(green_square_frame(), GREEN)
    trimap = pipeline.build_trimap(difference, 32, 160)
    assert not (trimap == pipeline.TRIMAP_UNKNOWN).any()

    widened = pipeline.widen_unknown_band(trimap, 2)
    unknown = widened == pipeline.TRIMAP_UNKNOWN
    # Square spans 16..47; band is 2px either side of that edge.
    assert unknown[32, 14:18].all()
    assert not unknown[32, 13] and not unknown[32, 18]
    assert widened[32, 32] == pipeline.TRIMAP_FOREGROUND


def test_widen_unknown_band_keeps_locked_pixels_foreground():
    difference = pipeline.key_difference(green_square_frame(), GREEN)
    trimap = pipeline.build_trimap(difference, 32, 160)
    locked = np.zeros(trimap.shape, dtype=bool)
    locked[32, 14:18] = True
    widened = pipeline.widen_unknown_band(
        pipeline.apply_holdout(trimap, locked), 2, locked
    )
    assert (widened[32, 14:18] == pipeline.TRIMAP_FOREGROUND).all()


def test_trimap_from_matte_uses_both_thresholds():
    matte = np.array([[0.0, 0.05, 0.5, 0.95, 1.0]], dtype=np.float32)
    trimap = pipeline.trimap_from_matte(matte, fg_threshold=0.9, bg_threshold=0.1)
    assert trimap.tolist() == [[0, 0, 128, 255, 255]]


def test_unknown_share_is_a_percentage():
    trimap = np.full((10, 10), pipeline.TRIMAP_FOREGROUND, dtype=np.uint8)
    trimap[:5] = pipeline.TRIMAP_UNKNOWN
    assert pipeline.unknown_share(trimap) == 50.0


def test_despill_unmix_recovers_white_from_a_half_covered_pixel():
    rgb = np.array([[[128.0, 255.0, 128.0]]], dtype=np.float32)
    alpha = np.array([[0.5]], dtype=np.float32)
    recovered = pipeline.despill_unmix(rgb, alpha, GREEN)
    assert np.allclose(recovered, 255.0, atol=1.0)


def test_compose_alpha_reimposes_trimap_definites():
    trimap = np.array([[0, 128, 255]], dtype=np.uint8)
    alpha = np.array([[0.7, 0.4, 0.2]], dtype=np.float32)
    assert pipeline.compose_alpha(alpha, trimap).tolist() == [
        [0.0, pytest.approx(0.4), 1.0]
    ]


def test_downscale_rgba_hits_the_requested_long_edge():
    rgb = np.full((40, 80, 3), 200.0, dtype=np.float32)
    alpha = np.ones((40, 80), dtype=np.float32)
    small_rgb, small_alpha = pipeline.downscale_rgba(rgb, alpha, 20)
    assert small_alpha.shape == (10, 20)
    assert small_rgb.shape == (10, 20, 3)
