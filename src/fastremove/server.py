from __future__ import annotations

import asyncio
import logging
from typing import Any

from fastmcp import Context, FastMCP
from fastmcp.exceptions import ToolError

from fastremove import paths, pipeline, remover

LOGGER = logging.getLogger(__name__)

mcp = FastMCP(
    "FastRemove",
    instructions=("Removes image backgrounds locally."),
)


@mcp.tool
async def remove_background(
    images: list[str],
    output_names: list[str] | str | None = None,
    chroma_key: bool = False,
    key_color: str = pipeline.DEFAULT_KEY_COLOR,
    fast: bool = False,
    output_size: int | None = None,
    ctx: Context = None,
) -> dict[str, Any]:
    """Remove the background from one or more images, writing PNG cutouts.
    chroma_key: True when the images were shot or rendered against a flat colour;
        the background is then keyed by colour and the key colour is unmixed from
        the edges. False segments the subject out of a real background.
    key_color: "green", "blue", or a hex colour. Only used with chroma_key.
    fast: True skips the ViTMatte edge refinement. Chroma-key fast mode needs no
        model at all.
    output_size: optionally downscale each cutout to this long edge in pixels.
    """
    job = build_job(images, output_names, chroma_key, key_color, fast, output_size)

    loop = asyncio.get_running_loop()

    def on_progress(done: int, total: int, result: dict[str, Any]) -> None:
        # Called from the worker thread; hand the notification back to the loop.
        message = f"{done}/{total} {result['output']}"
        asyncio.run_coroutine_threadsafe(
            ctx.report_progress(progress=done, total=total, message=message), loop
        )

    summary = await asyncio.to_thread(remover.run, job, on_progress)
    return {"status": "ok" if not summary["failed"] else "partial", **summary}


def build_job(
    images: list[str],
    output_names: list[str] | str | None,
    chroma_key: bool,
    key_color: str,
    fast: bool,
    output_size: int | None,
) -> remover.Job:
    inputs = paths.resolve_input_paths(images)
    outputs = paths.resolve_output_paths(inputs, output_names)

    chroma = None
    if chroma_key:
        try:
            chroma = remover.ChromaSettings(
                key_color=pipeline.parse_key_color(key_color)
            )
        except ValueError as exc:
            raise ToolError(str(exc)) from exc

    if output_size is not None and output_size <= 0:
        raise ToolError("output_size must be a positive integer or omitted.")

    return remover.Job(
        inputs=inputs,
        outputs=outputs,
        chroma=chroma,
        fast=fast,
        output_size=output_size or 0,
    )


def main() -> None:
    mcp.run()
