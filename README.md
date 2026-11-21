# llm-ui

Gradio chat UI with MCP servers and Anthropic-style skills. Providers: [KoboldCpp](https://github.com/LostRuins/koboldcpp) (local), OpenAI, DeepSeek, Google Gemini, and Anthropic Claude.

```bash
uv sync
uv run python -m llmui          # http://127.0.0.1:7860
```

- **Settings**: pick a provider, base URL, API key and model (type any id, or 🔄 refresh the list from the
  provider). Keys left blank are read from `OPENAI_API_KEY`, `DEEPSEEK_API_KEY`, `GEMINI_API_KEY`/`GOOGLE_API_KEY`
  or `ANTHROPIC_API_KEY` (Claude also works with `ant auth login`). Keys typed in the UI are saved in
  `config.json` (gitignored, mode 600). Also: sampling, max tool iterations.
- **MCP**: edit a Claude-Desktop-style `mcp.json` (see `mcp.example.json`). stdio (`command`/`args`/`env`)
  and remote (`url`/`headers`; streamable HTTP, or SSE if the URL ends in `/sse` or `"type": "sse"`).
  Tools are exposed to the model as `server__tool`; calls auto-run and show as collapsible blocks.
- **Skills**: `./skills/<name>/SKILL.md` (frontmatter `name`, `description`) or `~/.config/llmui/skills/`.
  Descriptions go in the system prompt; the model calls `load_skill` / `read_skill_file` on demand.

Provider notes:
- OpenAI / DeepSeek / Gemini use their OpenAI-compatible endpoints. If a model rejects `temperature`/`top_p`
  the request is retried without them. DeepSeek's `reasoning_content` is kept and echoed back as it requires.
- Claude uses the native Anthropic SDK (streaming, tool use, thinking blocks replayed unchanged). Sampling
  controls are not sent (current Claude models reject them). For models that support it, the server-side
  refusal-fallback beta is on by default; untick it in Settings if your account can't use it.
- Switching provider mid-conversation works; provider-specific data (thinking blocks, reasoning) is only
  replayed to the provider that produced it.

Tool calling on KoboldCpp uses its native `tools` support (run it with a tool-capable model; `--jinja` is
recommended). If the backend rejects `tools`, the request is retried without them.

Tests: `uv run pytest`.
