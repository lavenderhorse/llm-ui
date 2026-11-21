"""Tool-calling loop: stream -> collect tool_calls -> execute -> repeat."""
from __future__ import annotations

import json
import time

from . import skills as skills_mod
from .mcp_manager import MCPManager


def build_messages(settings: dict, skills: dict, convo: list[dict]) -> list[dict]:
    system = settings["system_prompt"] + skills_mod.prompt_section(skills)
    return [{"role": "system", "content": system}] + convo


async def run_turn(backend, mcp: MCPManager, skills: dict, settings: dict,
                   convo: list[dict], display: list[dict]):
    """Async generator. Mutates `convo` (OpenAI format) and `display` (Gradio messages),
    yielding after every visible change."""
    tools = (mcp.tool_schemas() + (skills_mod.tool_schemas() if skills else [])) or None
    for _ in range(int(settings["max_tool_iterations"])):
        display.append({"role": "assistant", "content": ""})
        msg = display[-1]
        calls: list[dict] = []
        meta: dict = {}  # provider-specific data to keep on the assistant turn (thinking blocks etc.)
        text = ""
        try:
            try:
                async for kind, val in backend.stream_chat(build_messages(settings, skills, convo), tools, settings):
                    if kind == "text":
                        text += val
                        msg["content"] = text
                        yield
                    elif kind == "meta":
                        meta.update(val)
                    else:
                        calls = val
            except Exception:
                if not tools or text:
                    raise
                # Backend/model may not support tools: retry once without them.
                tools = None
                msg["content"] = "⚠️ Tool calling failed; retrying without tools."
                yield
                async for kind, val in backend.stream_chat(build_messages(settings, skills, convo), None, settings):
                    if kind == "text":
                        text += val
                        msg["content"] = text
                        yield
                    elif kind == "meta":
                        meta.update(val)
        except Exception as e:  # noqa: BLE001
            msg["content"] = (text + f"\n\n⚠️ Backend error: {e}").strip()
            yield
            return
        if not calls:
            if not text:
                display.pop()
            else:
                convo.append({"role": "assistant", "content": text, **meta})
            yield
            return
        if not text:
            display.pop()
        convo.append({"role": "assistant", "content": text or None, **meta, "tool_calls": [
            {"id": c["id"], "type": "function", "function": {"name": c["name"], "arguments": c["arguments"] or "{}"}}
            for c in calls]})
        for c in calls:
            try:
                args = json.loads(c["arguments"] or "{}")
            except json.JSONDecodeError:
                args = None
            block = {"role": "assistant", "content": f"```json\n{c['arguments']}\n```",
                     "metadata": {"title": f"🔧 {c['name']}", "status": "pending"}}
            display.append(block)
            yield
            t0 = time.time()
            if c.get("error"):
                result = f"Error: {c['error']}"
            elif args is None:
                result = "Error: tool arguments were not valid JSON."
            elif skills_mod.is_skill_tool(c["name"]):
                result = skills_mod.call_tool(skills, c["name"], args)
            elif mcp.owns(c["name"]):
                result = await mcp.call(c["name"], args)
            else:
                result = f"Error: unknown tool '{c['name']}'."
            block["content"] = f"```json\n{c['arguments']}\n```\n**Result:**\n```\n{result[:4000]}\n```"
            block["metadata"] = {"title": f"🔧 {c['name']}", "status": "done", "duration": round(time.time() - t0, 2)}
            convo.append({"role": "tool", "tool_call_id": c["id"], "content": result})
            yield
    display.append({"role": "assistant", "content": "⚠️ Stopped: reached max tool iterations."})
    yield
