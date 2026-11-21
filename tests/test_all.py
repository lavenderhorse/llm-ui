import sys
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from llmui import agent, config, skills as sk
from llmui.backend import merge_tool_call_deltas
from llmui.mcp_manager import MCPManager, to_openai_tool


def test_parse_skill(tmp_path):
    d = tmp_path / "s1"; d.mkdir()
    (d / "SKILL.md").write_text("---\nname: foo\ndescription: does foo\n---\nBODY")
    (d / "ref.txt").write_text("hello")
    skills = sk.discover([tmp_path])
    assert skills["foo"].description == "does foo" and skills["foo"].body == "BODY"
    assert "ref.txt" in sk.call_tool(skills, sk.LOAD_SKILL, {"name": "foo"})
    assert sk.call_tool(skills, sk.READ_SKILL_FILE, {"name": "foo", "path": "ref.txt"}) == "hello"
    assert sk.call_tool(skills, sk.READ_SKILL_FILE, {"name": "foo", "path": "../../etc/passwd"}).startswith("Error")


def test_delta_merge():
    acc = {}
    f = lambda n, a: NS(name=n, arguments=a)
    merge_tool_call_deltas(acc, [NS(index=0, id="c1", function=f("x__y", '{"a"'))])
    merge_tool_call_deltas(acc, [NS(index=0, id=None, function=f(None, ":1}"))])
    assert acc[0] == {"id": "c1", "name": "x__y", "arguments": '{"a":1}'}


def test_schema_conversion():
    t = NS(name="t", description="d", inputSchema={"type": "object", "properties": {}})
    assert to_openai_tool("srv", t)["function"]["name"] == "srv__t"


def test_parse_mcp():
    assert config.parse_mcp('{"mcpServers":{"a":{"command":"x"}}}')
    with pytest.raises(ValueError):
        config.parse_mcp('{"mcpServers":{"a":{}}}')


class FakeBackend:
    """First call requests a tool; second call answers."""
    def __init__(self): self.n = 0
    async def stream_chat(self, messages, tools, s):
        self.n += 1
        if self.n == 1:
            yield "tool_calls", [{"id": "c1", "name": "fake__add", "arguments": '{"a":2,"b":3}'}]
        else:
            assert messages[-1]["role"] == "tool" and messages[-1]["content"] == "5"
            yield "text", "The answer is 5"
            yield "tool_calls", []


async def test_agent_with_real_mcp_stdio():
    mgr = MCPManager()
    await mgr.start_all({"fake": {"command": sys.executable, "args": [str(Path(__file__).parent / "fake_mcp.py")]}})
    try:
        assert mgr.servers["fake"].status == "connected", mgr.status_rows()
        assert mgr.tool_schemas()[0]["function"]["name"] == "fake__add"
        convo, display = [{"role": "user", "content": "2+3?"}], []
        async for _ in agent.run_turn(FakeBackend(), mgr, {}, {**config.DEFAULTS}, convo, display):
            pass
        assert display[0]["metadata"]["status"] == "done" and "5" in display[0]["content"]
        assert display[-1]["content"] == "The answer is 5"
        assert [m["role"] for m in convo] == ["user", "assistant", "tool", "assistant"]
    finally:
        await mgr.stop_all()
