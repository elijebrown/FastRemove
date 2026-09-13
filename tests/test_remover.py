import numpy as np
from PIL import Image

from fastremove import remover
from fastremove.pipeline import parse_key_color

GREEN = (0, 255, 0)


def write_green_frame(path, size: int = 64, square: int = 32) -> None:
    frame = np.zeros((size, size, 3), dtype=np.uint8)
    frame[:, :] = GREEN
    start = (size - square) // 2
    frame[start : start + square, start : start + square] = (255, 255, 255)
    Image.fromarray(frame, mode="RGB").save(path)


def test_fast_chroma_run_writes_a_transparent_cutout(tmp_path):
    source = tmp_path / "frame.png"
    write_green_frame(source)
    target = tmp_path / "frame-cut.png"

    seen = []
    job = remover.Job(
        inputs=[source],
        outputs=[target],
        chroma=remover.ChromaSettings(key_color=parse_key_color("green")),
        fast=True,
    )
    summary = remover.run(
        job, on_progress=lambda done, total, result: seen.append((done, total))
    )

    assert summary["count"] == 1 and summary["failed"] == 0
    assert summary["runtime"]["models"] == []
    assert seen == [(1, 1)]

    with Image.open(target) as cutout:
        rgba = np.asarray(cutout.convert("RGBA"))
    assert rgba[0, 0, 3] == 0
    assert rgba[32, 32, 3] == 255
    assert (rgba[32, 32, :3] == 255).all()


def test_unreadable_input_is_recorded_not_raised(tmp_path):
    broken = tmp_path / "broken.png"
    broken.write_bytes(b"not an image")
    job = remover.Job(
        inputs=[broken],
        outputs=[tmp_path / "out.png"],
        chroma=remover.ChromaSettings(key_color=GREEN),
        fast=True,
    )
    summary = remover.run(job, on_progress=None)
    assert summary["count"] == 0
    assert summary["failed"] == 1
    assert summary["failures"][0]["input"] == str(broken)
