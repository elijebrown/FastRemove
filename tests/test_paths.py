from pathlib import Path

import pytest
from fastmcp.exceptions import ToolError

from fastremove import paths


def touch(directory: Path, name: str) -> Path:
    path = directory / name
    path.write_bytes(b"")
    return path


def test_resolve_input_paths_requires_existing_supported_images(tmp_path):
    good = touch(tmp_path, "a.jpg")
    assert paths.resolve_input_paths([str(good)]) == [good]

    with pytest.raises(ToolError, match="at least one"):
        paths.resolve_input_paths([])
    with pytest.raises(ToolError, match="does not exist"):
        paths.resolve_input_paths([str(tmp_path / "missing.png")])
    with pytest.raises(ToolError, match="Unsupported"):
        paths.resolve_input_paths([str(touch(tmp_path, "notes.txt"))])


def test_default_outputs_sit_beside_inputs_with_a_tag(tmp_path):
    inputs = [touch(tmp_path, "a.jpg"), touch(tmp_path, "b.png")]
    outputs = paths.resolve_output_paths(inputs, None)
    assert outputs == [tmp_path / "a-nobg.png", tmp_path / "b-nobg.png"]


def test_single_string_is_a_file_for_one_input(tmp_path):
    inputs = [touch(tmp_path, "a.jpg")]
    outputs = paths.resolve_output_paths(inputs, str(tmp_path / "cutout.png"))
    assert outputs == [tmp_path / "cutout.png"]


def test_single_string_is_numbered_for_many_inputs(tmp_path):
    inputs = [touch(tmp_path, "a.jpg"), touch(tmp_path, "b.jpg")]
    outputs = paths.resolve_output_paths(inputs, str(tmp_path / "cat.png"))
    assert outputs == [tmp_path / "cat-01.png", tmp_path / "cat-02.png"]


def test_bare_string_lands_beside_the_first_input(tmp_path):
    inputs = [touch(tmp_path, "a.jpg"), touch(tmp_path, "b.jpg")]
    assert paths.resolve_output_paths(inputs[:1], "cat") == [tmp_path / "cat.png"]
    assert paths.resolve_output_paths(inputs, "cat") == [
        tmp_path / "cat-01.png",
        tmp_path / "cat-02.png",
    ]


def test_counter_widens_past_ninety_nine_inputs(tmp_path):
    inputs = [touch(tmp_path, f"{index}.jpg") for index in range(100)]
    outputs = paths.resolve_output_paths(inputs, "cat")
    assert outputs[0] == tmp_path / "cat-001.png"
    assert outputs[-1] == tmp_path / "cat-100.png"


def test_existing_directory_string_holds_one_file_per_input(tmp_path):
    inputs = [touch(tmp_path, "a.jpg"), touch(tmp_path, "b.jpg")]
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    outputs = paths.resolve_output_paths(inputs, str(out_dir))
    assert outputs == [out_dir / "a.png", out_dir / "b.png"]


def test_list_entries_are_names_beside_inputs_or_full_paths(tmp_path):
    other = tmp_path / "elsewhere"
    inputs = [touch(tmp_path, "a.jpg"), touch(tmp_path, "b.jpg")]
    outputs = paths.resolve_output_paths(
        inputs, ["hero.png", str(other / "second.jpg")]
    )
    assert outputs == [tmp_path / "hero.png", other / "second.png"]


def test_list_length_must_match(tmp_path):
    inputs = [touch(tmp_path, "a.jpg"), touch(tmp_path, "b.jpg")]
    with pytest.raises(ToolError, match="one output name per image"):
        paths.resolve_output_paths(inputs, ["only-one.png"])


def test_outputs_may_not_overwrite_inputs_or_each_other(tmp_path):
    a = touch(tmp_path, "a.png")
    with pytest.raises(ToolError, match="overwrite"):
        paths.resolve_output_paths([a], str(tmp_path))
    b = touch(tmp_path, "b.jpg")
    with pytest.raises(ToolError, match="unique"):
        paths.resolve_output_paths([a, b], ["same.png", "same.png"])
