import json
import socket
import sys
import threading
import time
from pathlib import Path

import pytest
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from llmui import agent, config
from llmui.anthropic_backend import RAW_KEY, to_anthropic
from llmui.backend import REASONING_KEY, clean_messages, make_backend
from llmui.mcp_manager import MCPManager


@pytest.fixture
def serve():
    servers = []

    def start(app) -> str:
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        srv = uvicorn.Server(uvicorn.Config(app, port=port, log_level="error"))
        threading.Thread(target=srv.run, daemon=True).start()
        while not srv.started:
            time.sleep(0.02)
        servers.append(srv)
        return f"http://127.0.0.1:{port}"

    yield start
    for s in servers:
        s.should_exit = True


def sse(events):
    return "".join(f"event: {e['type']}\ndata: {json.dumps(e)}\n\n" for e in events)


async def collect(gen):
    out = {"text": "", "meta": {}, "calls": []}
    async for kind, val in gen:
        if kind == "text":
            out["text"] += val
        elif kind == "meta":
            out["meta"].update(val)
        else:
            out["calls"] = val
    return out


# ---------- pure conversion ----------

def test_to_anthropic_merges_tool_results_and_replays_raw():
    raw = [{"type": "thinking", "thinking": "", "signature": "S"},
           {"type": "tool_use", "id": "a", "name": "x", "input": {}}]
    msgs = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": None, RAW_KEY: raw, "tool_calls": []},
        {"role": "tool", "tool_call_id": "a", "content": "ok"},
        {"role": "tool", "tool_call_id": "b", "content": "Error: boom"},
        {"role": "assistant", "content": "hello", "tool_calls": [
            {"id": "c", "type": "function", "function": {"name": "y", "arguments": '{"q":1}'}}]},
    ]
    system, out = to_anthropic(msgs)
    assert system == "sys"
    assert out[1]["content"] == raw  # thinking block signature preserved
    assert [b["tool_use_id"] for b in out[2]["content"]] == ["a", "b"]  # merged into one user turn
    assert out[2]["content"][1]["is_error"] is True
    assert out[3]["content"][1] == {"type": "tool_use", "id": "c", "name": "y", "input": {"q": 1}}


def test_clean_messages_deepseek_vs_others():
    msgs = [{"role": "assistant", "content": "a", REASONING_KEY: "think"}, {"role": "assistant", "content": "b"}]
    ds = clean_messages("deepseek", msgs, True)
    assert [m["reasoning_content"] for m in ds] == ["think", ""] and not any(k.startswith("_") for m in ds for k in m)
    assert all("reasoning_content" not in m for m in clean_messages("openai", msgs, True))
    assert all("reasoning_content" not in m for m in clean_messages("deepseek", msgs, False))


# ---------- OpenAI-compatible providers ----------

def make_openai_app(log, reject_sampling=False):
    app = FastAPI()

    @app.post("/v1/chat/completions")
    async def chat(r: Request):
        body = await r.json()
        log.append(body)
        if reject_sampling and "temperature" in body:
            return JSONResponse({"error": {"message": "Unsupported value: 'temperature' does not support 0.7",
                                           "type": "invalid_request_error"}}, status_code=400)

        async def gen():
            for d in [{"reasoning_content": "hmm"}, {"content": "hi"}]:
                yield "data: " + json.dumps({"choices": [{"delta": d}]}) + "\n\n"
            yield "data: [DONE]\n\n"
        return StreamingResponse(gen(), media_type="text/event-stream")
    return app


async def test_openai_retries_without_sampling_and_uses_max_completion_tokens(serve):
    log = []
    url = serve(make_openai_app(log, reject_sampling=True)) + "/v1"
    s = {**config.DEFAULTS, "provider": "openai", "providers": {"openai": {"base_url": url, "api_key": "k", "model": "m"}}}
    out = await collect(make_backend(s).stream_chat([{"role": "user", "content": "x"}], None, {**s, "model": "m"}))
    assert out["text"] == "hi" and out["meta"] == {}  # reasoning only kept for deepseek
    assert "temperature" in log[0] and "temperature" not in log[1]
    assert log[1]["max_completion_tokens"] == 1024 and "max_tokens" not in log[1]


async def test_deepseek_reasoning_roundtrip(serve):
    log = []
    url = serve(make_openai_app(log))
    s = {**config.DEFAULTS, "provider": "deepseek", "providers": {"deepseek": {"base_url": url + "/v1", "api_key": "k"}}}
    tools = [{"type": "function", "function": {"name": "t", "description": "", "parameters": {"type": "object"}}}]
    msgs = [{"role": "user", "content": "x"}, {"role": "assistant", "content": "a", REASONING_KEY: "earlier"}]
    out = await collect(make_backend(s).stream_chat(msgs, tools, {**s, "model": "deepseek-chat"}))
    assert out["meta"] == {REASONING_KEY: "hmm"}
    assert log[0]["messages"][1]["reasoning_content"] == "earlier" and log[0]["max_tokens"] == 1024


async def test_kobold_url_gets_v1_suffix_and_extra_params(serve):
    log = []
    url = serve(make_openai_app(log))
    s = {**config.DEFAULTS, "providers": {"kobold": {"base_url": url}}}
    await collect(make_backend(s).stream_chat([{"role": "user", "content": "x"}], None, {**s, "model": "koboldcpp"}))
    assert log[0]["top_k"] == 40 and log[0]["rep_pen"] == 1.1


# ---------- Anthropic native ----------

def make_anthropic_app(log):
    app = FastAPI()
    start = {"type": "message_start", "message": {
        "id": "msg_1", "type": "message", "role": "assistant", "model": "claude-opus-5-5", "content": [],
        "stop_reason": None, "stop_sequence": None, "usage": {"input_tokens": 1, "output_tokens": 1}}}

    @app.post("/v1/messages")
    async def messages(r: Request):
        body = await r.json()
        log.append({"body": body, "beta": r.headers.get("anthropic-beta", "")})
        has_result = any(isinstance(m["content"], list) and m["content"][0].get("type") == "tool_result"
                         for m in body["messages"])
        d = lambda i, delta: {"type": "content_block_delta", "index": i, "delta": delta}
        if not has_result:
            ev = [start,
                  {"type": "content_block_start", "index": 0, "content_block": {"type": "thinking", "thinking": "", "signature": ""}},
                  d(0, {"type": "signature_delta", "signature": "SIG123"}),
                  {"type": "content_block_stop", "index": 0},
                  {"type": "content_block_start", "index": 1, "content_block": {"type": "tool_use", "id": "tu_1", "name": "fake__add", "input": {}}},
                  d(1, {"type": "input_json_delta", "partial_json": '{"a":2,'}),
                  d(1, {"type": "input_json_delta", "partial_json": '"b":3}'}),
                  {"type": "content_block_stop", "index": 1},
                  {"type": "message_delta", "delta": {"stop_reason": "tool_use", "stop_sequence": None}, "usage": {"output_tokens": 5}},
                  {"type": "message_stop"}]
        else:
            ev = [start,
                  {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
                  d(0, {"type": "text_delta", "text": "The answer is 5"}),
                  {"type": "content_block_stop", "index": 0},
                  {"type": "message_delta", "delta": {"stop_reason": "end_turn", "stop_sequence": None}, "usage": {"output_tokens": 5}},
                  {"type": "message_stop"}]
        return StreamingResponse(iter([sse(ev)]), media_type="text/event-stream")
    return app


async def test_anthropic_agent_loop_with_real_mcp(serve):
    log = []
    url = serve(make_anthropic_app(log))
    s = {**config.DEFAULTS, "provider": "anthropic",
         "providers": {"anthropic": {"base_url": url, "api_key": "sk-test", "model": "claude-opus-5-5"}}}
    mgr = MCPManager()
    await mgr.start_all({"fake": {"command": sys.executable, "args": [str(Path(__file__).parent / "fake_mcp.py")]}})
    try:
        convo, display = [{"role": "user", "content": "2+3?"}], []
        run_s = {**s, "model": "claude-opus-5-5", "system_prompt": "Be brief."}
        async for _ in agent.run_turn(make_backend(s), mgr, {}, run_s, convo, display):
            pass
    finally:
        await mgr.stop_all()

    assert display[0]["metadata"]["status"] == "done" and "5" in display[0]["content"]
    assert display[-1]["content"] == "The answer is 5"
    first, second = log
    # request shape: system prompt, native tool schema, no sampling params, refusal-fallback beta
    assert first["body"]["system"] == "Be brief." and "temperature" not in first["body"]
    assert first["body"]["tools"][0]["name"] == "fake__add" and "input_schema" in first["body"]["tools"][0]
    assert first["body"]["fallbacks"] == "default" and "server-side-fallback-2026-07-01" in first["beta"]
    assert first["body"]["max_tokens"] >= 4096
    # second request replays the thinking block with its signature + a merged tool_result turn
    asst = second["body"]["messages"][1]["content"]
    assert asst[0]["type"] == "thinking" and asst[0]["signature"] == "SIG123"
    assert asst[1] == {"type": "tool_use", "id": "tu_1", "name": "fake__add", "input": {"a": 2, "b": 3}}
    assert second["body"]["messages"][2]["content"][0] == {"type": "tool_result", "tool_use_id": "tu_1", "content": "5"}


async def test_anthropic_fallback_disabled_uses_plain_endpoint(serve):
    log = []
    url = serve(make_anthropic_app(log))
    s = {**config.DEFAULTS, "provider": "anthropic", "claude_fallback": False,
         "providers": {"anthropic": {"base_url": url, "api_key": "k"}}}
    out = await collect(make_backend(s).stream_chat([{"role": "user", "content": "x"}], None, {**s, "model": "claude-haiku-5-5"}))
    assert "fallbacks" not in log[0]["body"] and log[0]["beta"] == ""
    assert [b["type"] for b in out["meta"][RAW_KEY]] == ["thinking", "tool_use"]
    assert out["calls"][0]["name"] == "fake__add" and json.loads(out["calls"][0]["arguments"]) == {"a": 2, "b": 3}
