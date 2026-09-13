---
name: remove-background
description: A local image background removal utility. Use when a user asks to remove the background from an image, remove backgrounds from images.
---

# remove-background

### Step 1: Gather Inputs

1. Infer or ask the user which image or images they would like to process.
2. Ask the user for a filename or filenames, or if they want this skill to maintain current filenames with -nobg suffixed to the output files.

### Step 2: Call MCP server:

1. Call the local stdio mcp tool: fastremove.remove_background

- images: array of image filepaths to be processed
- (optional) output_names: left blank, appends -nobg to current filename, array matched to each input, or template filename for 1-N files with counter.
- (optional) chroma_key: flag for chroma_key background images. More efficient processing if flag is set.
- (optional) key_color: background color for chroma key images
- (optional) fast: if true, disables model based matting and processing. Significantly faster at potential quality cost.
- (optional) output_size: downscale output image to a long-edge size of output_size
- (optional) ctx: Context object

```
fastremove.remove_background with parameters:

{
  images: list[str],
  output_names: list[str] | str | None,
  chroma_key: bool,
  key_color: str,
  fast: bool,
  output_size: int,
  ctx: Context,
}
```

2. Report tool output and any errors or warnings to user

- Number of files processed
- `input filename` -> `output filename`

### Common Errors

- fastremove MCP server is not connected:
  - fail gracefully, do not try to manually remove background, and report the error to the user
- OOM error:
  - abort tool call and report to the user that there is insufficient memory for the request and that they can try running the background removal in `fast` mode to use significantly less memory.
