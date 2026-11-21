"""Settings + MCP config persistence."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path.cwd()
CONFIG_PATH = ROOT / "config.json"
MCP_PATH = ROOT / "mcp.json"

DEFAULTS: dict = {
    "provider": "kobold",
    "providers": {},  # name -> {"base_url", "api_key", "model"}
    "claude_fallback": True,
    "system_prompt": "You are a helpful assistant.",
    "temperature": 0.7,
    "top_p": 0.95,
    "top_k": 40,
    "min_p": 0.05,
    "rep_pen": 1.1,
    "max_tokens": 1024,
    "max_tool_iterations": 10,
    "disabled_skills": [],
}

DEFAULT_MCP = {"mcpServers": {}}


def load_settings() -> dict:
    s = dict(DEFAULTS)
    if CONFIG_PATH.exists():
        try:
            data = json.loads(CONFIG_PATH.read_text())
        except json.JSONDecodeError:
            data = {}
        # Migrate pre-provider configs (flat base_url/model were KoboldCpp's).
        if "providers" not in data and ("base_url" in data or "model" in data):
            data["providers"] = {"kobold": {k: data.pop(k) for k in ("base_url", "model") if k in data}}
        s.update(data)
    return s


def save_settings(s: dict) -> None:
    CONFIG_PATH.write_text(json.dumps(s, indent=2))
    CONFIG_PATH.chmod(0o600)  # may contain API keys


def load_mcp_text() -> str:
    if MCP_PATH.exists():
        return MCP_PATH.read_text()
    return json.dumps(DEFAULT_MCP, indent=2)


def parse_mcp(text: str) -> dict[str, dict]:
    """Return {name: server_config}. Raises ValueError on bad JSON/shape."""
    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON: {e}") from e
    servers = data.get("mcpServers") if isinstance(data, dict) else None
    if not isinstance(servers, dict):
        raise ValueError('Expected {"mcpServers": {...}}')
    for name, cfg in servers.items():
        if not isinstance(cfg, dict) or not ("command" in cfg or "url" in cfg):
            raise ValueError(f"Server '{name}' needs 'command' or 'url'")
    return servers


def save_mcp_text(text: str) -> dict[str, dict]:
    servers = parse_mcp(text)
    MCP_PATH.write_text(text)
    return servers
