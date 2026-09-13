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
    str             -> a file path when there is one input and the value looks
                       like a file; otherwise a directory holding `<stem>.png`.
    list[str]       -> one entry per input. A bare name lands beside its input;
                       anything with a directory component is used as given.
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
    target = Path(value).expanduser().resolve()
    if len(inputs) == 1 and looks_like_file(target):
        return [with_png_suffix(target)]
    return [target / f"{path.stem}{OUTPUT_SUFFIX}" for path in inputs]


def outputs_from_list(inputs: list[Path], names: list[str]) -> list[Path]:
    if len(names) != len(inputs):
        raise ToolError(
            f"output_names must give one output name per image: "
            f"got {len(names)} names for {len(inputs)} images."
        )
    outputs = []
    for input_path, name in zip(inputs, names, strict=True):
        candidate = Path(name).expanduser()
        if candidate.parent == Path("."):
            candidate = input_path.parent / candidate.name
        outputs.append(with_png_suffix(candidate.resolve()))
    return outputs


def looks_like_file(path: Path) -> bool:
    """A path with an image suffix that is not already a directory."""
    return (
        bool(path.suffix)
        and path.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES
        and not path.is_dir()
    )


def with_png_suffix(path: Path) -> Path:
    """Cutouts carry alpha, so every output is a PNG whatever name was asked for."""
    if path.suffix.lower() == OUTPUT_SUFFIX:
        return path
    if path.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES:
        return path.with_suffix(OUTPUT_SUFFIX)
    return path.with_name(path.name + OUTPUT_SUFFIX)


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
