"""KoboldCpp client via its OpenAI-compatible API."""
from __future__ import annotations

import httpx
from openai import AsyncOpenAI


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


class Backend:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")
        self.client = AsyncOpenAI(base_url=f"{self.base_url}/v1", api_key="koboldcpp", timeout=600)

    async def ping(self) -> str:
        async with httpx.AsyncClient(timeout=5) as c:
            r = await c.get(f"{self.base_url}/api/v1/model")
            r.raise_for_status()
            return r.json().get("result", "unknown")

    async def stream_chat(self, messages: list[dict], tools: list[dict] | None, s: dict):
        """Yield ("text", delta) events, then a final ("tool_calls", [..]) event."""
        kwargs = dict(
            model=s.get("model") or "koboldcpp",
            messages=messages,
            stream=True,
            temperature=s["temperature"],
            top_p=s["top_p"],
            max_tokens=int(s["max_tokens"]),
            extra_body={"top_k": int(s["top_k"]), "min_p": s["min_p"], "rep_pen": s["rep_pen"]},
        )
        if tools:
            kwargs["tools"] = tools
        stream = await self.client.chat.completions.create(**kwargs)
        acc: dict[int, dict] = {}
        async for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            if delta.content:
                yield "text", delta.content
            merge_tool_call_deltas(acc, delta.tool_calls)
        calls = [acc[i] for i in sorted(acc)]
        for n, c in enumerate(calls):
            c["id"] = c["id"] or f"call_{n}"
        yield "tool_calls", calls
