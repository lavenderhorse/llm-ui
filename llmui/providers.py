"""Provider registry: KoboldCpp + hosted APIs (OpenAI-compatible or native Anthropic)."""
from __future__ import annotations

import os

PROVIDERS: dict[str, dict] = {
    "kobold": {"label": "KoboldCpp (local)", "kind": "openai", "base_url": "http://localhost:5001",
               "key_env": [], "model": "koboldcpp"},
    "openai": {"label": "OpenAI", "kind": "openai", "base_url": "https://api.openai.com/v1",
               "key_env": ["OPENAI_API_KEY"], "model": "gpt-5"},
    "deepseek": {"label": "DeepSeek", "kind": "openai", "base_url": "https://api.deepseek.com",
                 "key_env": ["DEEPSEEK_API_KEY"], "model": "deepseek-chat"},
    "gemini": {"label": "Google Gemini", "kind": "openai",
               "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
               "key_env": ["GEMINI_API_KEY", "GOOGLE_API_KEY"], "model": "gemini-2.5-flash"},
    "anthropic": {"label": "Anthropic Claude", "kind": "anthropic", "base_url": "",
                  "key_env": ["ANTHROPIC_API_KEY"], "model": "claude-opus-5-5"},
}

# Models that accept the server-side refusal `fallbacks` beta.
FALLBACK_MODELS = {"claude-fable-5-2", "claude-fable-5-1", "claude-opus-5-5", "claude-opus-5", "claude-sonnet-5-5"}


def choices() -> list[tuple[str, str]]:
    return [(p["label"], k) for k, p in PROVIDERS.items()]


def provider_cfg(settings: dict, name: str) -> dict:
    """Effective config for a provider: registry defaults < saved settings < env for the key."""
    meta = PROVIDERS[name]
    saved = settings.get("providers", {}).get(name, {})
    cfg = {"base_url": meta["base_url"], "model": meta["model"], "api_key": ""}
    cfg.update({k: v for k, v in saved.items() if v not in (None, "")})
    if not cfg["api_key"]:
        cfg["api_key"] = next((os.environ[e] for e in meta["key_env"] if os.environ.get(e)), "")
    return cfg
