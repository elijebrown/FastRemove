"""Input validation and output naming"""

from __future__ import annotations

from pathlib import Path

from fastmcp.exceptions import ToolError

SUPPORTED_IMAGE_SUFFIXES = frozenset(
    {".bmp", ".gif", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
)
OUTPUT_SUFFIX = ".png"
DEFAULT_OUTPUT_TAG = "-nobg"


def resolve_input_paths(images: list[str]) -> list[Path]:
    """Return absolute paths to existing, supported image files."""
    if not images:
        raise ToolError("images must list at least one file.")

    resolved: list[Path] = []
    for value in images:
        path = Path(value).expanduser().resolve()
        if not path.exists():
            raise ToolError(f"Input image does not exist: {path}")
        if not path.is_file():
            raise ToolError(f"Input path is not a file: {path}")
        if path.suffix.lower() not in SUPPORTED_IMAGE_SUFFIXES:
            suffixes = ", ".join(sorted(SUPPORTED_IMAGE_SUFFIXES))
            raise ToolError(
                f"Unsupported image type {path.suffix!r} for {path}. Supported: {suffixes}"
            )
        resolved.append(path)
    return resolved


def resolve_output_paths(
    inputs: list[Path], output_names: list[str] | str | None
) -> list[Path]:
    """Turn the tool's `output_names` argument into one PNG path per input.

    None            -> `<stem>-nobg.png` beside each input.
    str             -> an existing directory gets `<stem>.png` per input.
                       Otherwise it names the file: `cat.png` for one input,
                       `cat-01.png`, `cat-02.png`, ... for several.
    list[str]       -> one entry per input.

    Bare names (no directory component) land beside the input they belong to;
    a bare string for several inputs lands beside the first one.
    """
    if output_names is None:
        outputs = [default_output_path(path) for path in inputs]
    elif isinstance(output_names, str):
        outputs = outputs_from_single_value(inputs, output_names)
    else:
        outputs = outputs_from_list(inputs, output_names)

    validate_outputs(inputs, outputs)
    return outputs


def default_output_path(input_path: Path) -> Path:
    return input_path.with_name(f"{input_path.stem}{DEFAULT_OUTPUT_TAG}{OUTPUT_SUFFIX}")


def outputs_from_single_value(inputs: list[Path], value: str) -> list[Path]:
    target = anchor_to_input(Path(value), inputs[0])
    if target.is_dir():
        return [target / f"{path.stem}{OUTPUT_SUFFIX}" for path in inputs]
    if len(inputs) == 1:
        return [with_png_suffix(target)]
    return numbered_outputs(target, len(inputs))


def outputs_from_list(inputs: list[Path], names: list[str]) -> list[Path]:
    if len(names) != len(inputs):
        raise ToolError(
            f"output_names must give one output name per image: "
            f"got {len(names)} names for {len(inputs)} images."
        )
    return [
        with_png_suffix(anchor_to_input(Path(name), input_path))
        for input_path, name in zip(inputs, names, strict=True)
    ]


def anchor_to_input(candidate: Path, input_path: Path) -> Path:
    """Resolve a requested output path, placing bare names beside the input.

    The server's working directory is wherever the MCP client launched it, so
    a bare name resolved against it would land somewhere surprising.
    """
    candidate = candidate.expanduser()
    if candidate.parent == Path("."):
        candidate = input_path.parent / candidate.name
    return candidate.resolve()


def numbered_outputs(target: Path, count: int) -> list[Path]:
    """`cat.png` for three inputs becomes `cat-01.png`, `cat-02.png`, `cat-03.png`."""
    base = strip_image_suffix(target)
    width = max(2, len(str(count)))
    return [
        base.with_name(f"{base.name}-{index:0{width}d}{OUTPUT_SUFFIX}")
        for index in range(1, count + 1)
    ]


def with_png_suffix(path: Path) -> Path:
    """Cutouts carry alpha, so every output is a PNG whatever name was asked for."""
    base = strip_image_suffix(path)
    return base.with_name(base.name + OUTPUT_SUFFIX)


def strip_image_suffix(path: Path) -> Path:
    """Drop a trailing image suffix; other dots (`cat.v2`) are part of the name."""
    if path.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES:
        return path.with_suffix("")
    return path


def validate_outputs(inputs: list[Path], outputs: list[Path]) -> None:
    input_set = set(inputs)
    clobbered = [str(path) for path in outputs if path in input_set]
    if clobbered:
        raise ToolError(
            f"Output would overwrite an input image: {', '.join(clobbered)}"
        )

    seen: set[Path] = set()
    duplicates = sorted(
        {str(path) for path in outputs if path in seen or seen.add(path)}
    )
    if duplicates:
        raise ToolError(f"Output paths must be unique: {', '.join(duplicates)}")
