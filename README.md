# fastremove

A fast, local background-removal MCP server. Handles both flat chroma-key and real backgrounds.

## Install

```bash
uv sync                 # torch from PyPI: CUDA on Linux, CPU on macOS/Windows
uv sync --extra cuda    # CUDA 13 (Linux, Windows) for supported nvidia GPUs
uv sync --extra cpu     # CPU-only (Linux, Windows)
```

At runtime the server picks CUDA, then Apple MPS, then CPU.

### Connecting the MCP server (Claude Code)

#### Command:

- In a terminal, paste:

```bash
claude mcp add --transport stdio --scope user fastremove -- uv run --directory path-to-fastremove fastremove
```

#### Config:

- For user and project level tool availability, edit `~/.claude.json`:
  - User-level: Add json to the top level `mcpServers` object
  - User-Project-level: Add json to `mcpServers` within the specific `project` object

```json
{
  "fastremove": {
    "type": "stdio",
    "command": "uv",
    "args": [
      "run",
      "--directory",
      "<absolute-path-to-fastremove>",
      "fastremove"
    ]
  }
}
```

### Connecting the MCP server (Codex, Mistral, dbt Wizard)

#### Codex Command:

- In a terminal, paste:

```bash
codex mcp add my_tools -- uv run --directory path-to-fastremove fastremove
```

#### Config:

- Edit `~/.codex/config.toml` (or ~/platform-of-your-choice/config.toml)

```toml
[mcp_servers.fastremove]
command = "uv"
args = [
"run",
"--directory",
"<absolute-path-to-fastremove>",
"fastremove",
]
```

- For _shared project level availability_ (any user, specific project), add the `fastremove` mcp json or .toml object to the project's local .mcp.json or config.toml configuration

## Using the tool

### Skill

For easier use, consider adding the included remove-background skill to your user config:

- **claude code CLI**: paste the `remove-background/` folder in `skill/` into `~/.claude/skills/`
- **codex CLI**: paste the `remove-background/` folder in `skill/` into `~/.agents/skills/`

Now you can just say something like: _remove the background from this image_ in your chat session and the tool call will be structured for you! To invoke the skill more directly, you can add $remove-background at the beginning of your prompt.

### Tool Specifics

```python
remove_background(images, output_names=None, chroma_key=False, key_color="green", fast=False, output_size=None)
```

| Argument       | Meaning                                                                                                                                                                                                                                                                                                     | Expects                  |
| -------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------ |
| `images`       | One or more image paths, processed in order in a single call.                                                                                                                                                                                                                                               | list[str]                |
| `output_names` | Omit for `<name>-nobg.png` beside each input. A string names the file: `cat.png` for one image, `cat-01.png`, `cat-02.png`, ... for several. A string that is an existing directory gets `<name>.png` per image. A list gives one name per image. Bare names land beside the input. Outputs are always PNG. | str \| list[str] \| None |
| `chroma_key`   | `True` keys a flat colour background and unmixes it from the edges. `False` segments the subject out of a real background with FeyNobg.                                                                                                                                                                     | bool                     |
| `key_color`    | `green`, `blue`, or a hex colour. Chroma key only.                                                                                                                                                                                                                                                          | str                      |
| `fast`         | `True` skips ViTMatte edge refinement. Chroma-key fast mode needs no model at all.                                                                                                                                                                                                                          | bool                     |
| `output_size`  | Downscale each cutout to this long edge, in premultiplied linear light.                                                                                                                                                                                                                                     | int                      |

Models used, downloaded on first use: `feyninc/FeyNobg` and `hustvl/vitmatte-small-composition-1k`.

## Development

```bash
uv sync --group dev
uv run pytest
```

Layout under `src/fastremove/`: `server.py` is the MCP surface, `remover.py` runs one call,
`paths.py` validates inputs and names outputs, `pipeline.py` is the pure-numpy matting maths,
and `models.py` plus `inference.py` contain matting model logic.
