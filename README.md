# llm-ui

Gradio chat UI for [KoboldCpp](https://github.com/LostRuins/koboldcpp) (OpenAI-compatible API) with MCP servers and Anthropic-style skills.

```bash
uv sync
uv run python -m llmui          # http://127.0.0.1:7860
```

- **Settings**: KoboldCpp URL (default `http://localhost:5001`), sampling, max tool iterations.
- **MCP**: edit a Claude-Desktop-style `mcp.json` (see `mcp.example.json`). stdio (`command`/`args`/`env`)
  and remote (`url`/`headers`; streamable HTTP, or SSE if the URL ends in `/sse` or `"type": "sse"`).
  Tools are exposed to the model as `server__tool`; calls auto-run and show as collapsible blocks.
- **Skills**: `./skills/<name>/SKILL.md` (frontmatter `name`, `description`) or `~/.config/llmui/skills/`.
  Descriptions go in the system prompt; the model calls `load_skill` / `read_skill_file` on demand.

Tool calling uses KoboldCpp's native `tools` support (run it with a tool-capable model; `--jinja` is
recommended). If the backend rejects `tools`, the request is retried without them.

Tests: `uv run pytest`.
