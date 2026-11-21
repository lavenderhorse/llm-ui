"""Native Anthropic (Claude) backend using the official SDK, adapting OpenAI-style history."""
from __future__ import annotations

import json

import jsonschema
from anthropic import AsyncAnthropic

from .providers import FALLBACK_MODELS

RAW_KEY = "_anthropic_content"  # raw content blocks (thinking signatures etc.) kept on assistant turns
FALLBACK_BETA = "server-side-fallback-2026-07-01"


def to_anthropic(messages: list[dict]) -> tuple[str, list[dict]]:
    """OpenAI-format history -> (system, anthropic messages). Consecutive tool results are
    merged into one user turn, as the API requires."""
    system, out = [], []
    for m in messages:
        role = m["role"]
        if role == "system":
            system.append(m["content"])
        elif role == "user":
            out.append({"role": "user", "content": m["content"]})
        elif role == "assistant":
            if m.get(RAW_KEY):  # replay exactly what the model produced (keeps thinking blocks valid)
                out.append({"role": "assistant", "content": m[RAW_KEY]})
                continue
            blocks = []
            if m.get("content"):
                blocks.append({"type": "text", "text": m["content"]})
            for tc in m.get("tool_calls") or []:
                fn = tc["function"]
                blocks.append({"type": "tool_use", "id": tc["id"], "name": fn["name"],
                               "input": json.loads(fn["arguments"] or "{}")})
            if blocks:
                out.append({"role": "assistant", "content": blocks})
        elif role == "tool":
            block = {"type": "tool_result", "tool_use_id": m["tool_call_id"], "content": m["content"]}
            if str(m["content"]).startswith("Error"):
                block["is_error"] = True
            prev = out[-1] if out else None
            if prev and prev["role"] == "user" and isinstance(prev["content"], list) \
                    and all(b.get("type") == "tool_result" for b in prev["content"]):
                prev["content"].append(block)
            else:
                out.append({"role": "user", "content": [block]})
    return "\n\n".join(system), out


def to_anthropic_tools(tools: list[dict]) -> list[dict]:
    return [{"name": t["function"]["name"], "description": t["function"].get("description", ""),
             "input_schema": t["function"]["parameters"], "eager_input_streaming": True} for t in tools]


class AnthropicBackend:
    def __init__(self, api_key: str, base_url: str = "", fallback: bool = True):
        # No key -> SDK falls back to ANTHROPIC_AUTH_TOKEN / `ant auth login` credentials.
        self.client = AsyncAnthropic(api_key=api_key or None, base_url=base_url or None, timeout=600)
        self.fallback = fallback

    async def list_models(self) -> list[str]:
        return [m.id async for m in self.client.models.list()]

    async def ping(self) -> str:
        return f"{len(await self.list_models())} models available"

    async def stream_chat(self, messages: list[dict], tools: list[dict] | None, s: dict):
        system, msgs = to_anthropic(messages)
        params: dict = dict(model=s["model"], messages=msgs,
                            # thinking tokens count toward max_tokens, so leave headroom
                            max_tokens=max(int(s["max_tokens"]), 4096))
        if system:
            params["system"] = system
        if tools:
            params["tools"] = to_anthropic_tools(tools)
        # Sampling params are deliberately not sent: current Claude models reject them.
        if self.fallback and s["model"] in FALLBACK_MODELS:
            ctx = self.client.beta.messages.stream(**params, betas=[FALLBACK_BETA], fallbacks="default")
        else:
            ctx = self.client.messages.stream(**params)
        async with ctx as stream:
            async for text in stream.text_stream:
                yield "text", text
            final = await stream.get_final_message()

        schemas = {t["function"]["name"]: t["function"]["parameters"] for t in tools or []}
        calls = []
        for b in final.content:
            if b.type != "tool_use":
                continue
            call = {"id": b.id, "name": b.name, "arguments": json.dumps(b.input)}
            if final.stop_reason == "max_tokens":
                call["error"] = "tool input was truncated by max_tokens; raise Max tokens and retry."
            else:  # eager input streaming skips server-side validation, so check it here
                try:
                    jsonschema.validate(b.input, schemas.get(b.name, {}))
                except jsonschema.ValidationError as e:
                    call["error"] = f"tool input does not match its schema: {e.message}"
                except jsonschema.SchemaError:
                    pass
            calls.append(call)
        if final.stop_reason == "refusal":
            cat = getattr(final.stop_details, "category", None)
            yield "text", f"\n\n⚠️ Claude declined this request{f' ({cat})' if cat else ''}."
        yield "meta", {RAW_KEY: [b.model_dump(mode="json", exclude_none=True) for b in final.content]}
        yield "tool_calls", calls
