"""Chat backends. OpenAI-compatible (KoboldCpp, OpenAI, DeepSeek, Gemini) share one class.

Every backend exposes:
  stream_chat(messages, tools, settings) -> async iterator of
      ("text", str) | ("meta", dict) | ("tool_calls", [{"id","name","arguments"[, "error"]}])
  ping() -> str,  list_models() -> list[str]
"""
from __future__ import annotations

import httpx
import openai
from openai import AsyncOpenAI

from .providers import PROVIDERS, provider_cfg

REASONING_KEY = "_reasoning_content"
_SAMPLING = ("temperature", "top_p")


def merge_tool_call_deltas(acc: dict[int, dict], deltas) -> None:
    """Accumulate streamed tool_call deltas (keyed by index) into acc."""
    for d in deltas or []:
        slot = acc.setdefault(d.index, {"id": "", "name": "", "arguments": ""})
        if d.id:
            slot["id"] = d.id
        if d.function:
            if d.function.name:
                slot["name"] += d.function.name
            if d.function.arguments:
                slot["arguments"] += d.function.arguments


def clean_messages(provider: str, messages: list[dict], has_tools: bool) -> list[dict]:
    """Drop private `_*` keys. DeepSeek needs earlier turns' reasoning_content echoed back when
    tools are sent (an empty string is accepted for turns that have none)."""
    out = []
    for m in messages:
        c = {k: v for k, v in m.items() if not k.startswith("_")}
        if provider == "deepseek" and has_tools and m["role"] == "assistant":
            c["reasoning_content"] = m.get(REASONING_KEY, "")
        out.append(c)
    return out


class OpenAICompatBackend:
    def __init__(self, provider: str, base_url: str, api_key: str):
        self.provider = provider
        self.base_url = base_url.rstrip("/")
        api_base = f"{self.base_url}/v1" if provider == "kobold" else self.base_url
        self.client = AsyncOpenAI(base_url=api_base, api_key=api_key or "none", timeout=600)

    async def ping(self) -> str:
        if self.provider == "kobold":
            async with httpx.AsyncClient(timeout=5) as c:
                r = await c.get(f"{self.base_url}/api/v1/model")
                r.raise_for_status()
                return r.json().get("result", "unknown")
        return f"{len(await self.list_models())} models available"

    async def list_models(self) -> list[str]:
        ids = [m.id.removeprefix("models/") async for m in self.client.models.list()]
        return sorted(ids)

    def _kwargs(self, messages, tools, s, with_sampling=True) -> dict:
        kw: dict = dict(model=s["model"], messages=clean_messages(self.provider, messages, bool(tools)),
                        stream=True)
        if with_sampling:
            kw.update(temperature=s["temperature"], top_p=s["top_p"])
        # OpenAI's newer models want max_completion_tokens; the rest still use max_tokens.
        kw["max_completion_tokens" if self.provider == "openai" else "max_tokens"] = int(s["max_tokens"])
        if self.provider == "kobold":
            kw["extra_body"] = {"top_k": int(s["top_k"]), "min_p": s["min_p"], "rep_pen": s["rep_pen"]}
        if tools:
            kw["tools"] = tools
        return kw

    async def stream_chat(self, messages: list[dict], tools: list[dict] | None, s: dict):
        try:
            stream = await self.client.chat.completions.create(**self._kwargs(messages, tools, s))
        except openai.BadRequestError as e:
            # Some models (e.g. reasoning models) reject non-default sampling params: retry without.
            if not any(k in str(e) for k in _SAMPLING):
                raise
            stream = await self.client.chat.completions.create(**self._kwargs(messages, tools, s, False))
        acc: dict[int, dict] = {}
        reasoning = ""
        async for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            if delta.content:
                yield "text", delta.content
            reasoning += getattr(delta, "reasoning_content", None) or ""
            merge_tool_call_deltas(acc, delta.tool_calls)
        if self.provider == "deepseek":
            yield "meta", {REASONING_KEY: reasoning}
        calls = [acc[i] for i in sorted(acc)]
        for n, c in enumerate(calls):
            c["id"] = c["id"] or f"call_{n}"
        yield "tool_calls", calls


def make_backend(settings: dict, provider: str | None = None, overrides: dict | None = None):
    """Build the backend for `provider` (default: the active one). `overrides` lets the UI test
    unsaved form values (base_url / api_key)."""
    provider = provider or settings["provider"]
    cfg = {**provider_cfg(settings, provider), **{k: v for k, v in (overrides or {}).items() if v}}
    if PROVIDERS[provider]["kind"] == "anthropic":
        from .anthropic_backend import AnthropicBackend
        return AnthropicBackend(cfg["api_key"], cfg["base_url"], settings.get("claude_fallback", True))
    return OpenAICompatBackend(provider, cfg["base_url"], cfg["api_key"])


def active_settings(settings: dict) -> dict:
    """Settings with the active provider's `model` resolved to the top level for stream_chat."""
    return {**settings, "model": provider_cfg(settings, settings["provider"])["model"]}
