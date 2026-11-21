"""MCP client manager: one long-lived task per server (stdio / streamable HTTP / SSE)."""
from __future__ import annotations

import asyncio
import os
from contextlib import AsyncExitStack
from dataclasses import dataclass, field

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

SEP = "__"


def to_openai_tool(server: str, tool) -> dict:
    return {"type": "function", "function": {
        "name": f"{server}{SEP}{tool.name}",
        "description": tool.description or "",
        "parameters": (getattr(tool, "input_schema", None) or getattr(tool, "inputSchema", None)
                       or {"type": "object", "properties": {}}),
    }}


def flatten_result(result) -> str:
    parts = []
    for c in getattr(result, "content", None) or []:
        t = getattr(c, "text", None)
        parts.append(t if t is not None else f"[{getattr(c, 'type', 'content')} omitted]")
    text = "\n".join(parts) or "(no output)"
    return f"Error: {text}" if (getattr(result, "is_error", False) or getattr(result, "isError", False)) else text


@dataclass
class Server:
    name: str
    cfg: dict
    status: str = "connecting"
    error: str = ""
    tools: list = field(default_factory=list)
    session: ClientSession | None = None
    task: asyncio.Task | None = None
    stop: asyncio.Event = field(default_factory=asyncio.Event)
    ready: asyncio.Event = field(default_factory=asyncio.Event)


class MCPManager:
    def __init__(self):
        self.servers: dict[str, Server] = {}

    async def _transport(self, stack: AsyncExitStack, cfg: dict):
        if "command" in cfg:
            params = StdioServerParameters(
                command=cfg["command"], args=cfg.get("args", []),
                env={**os.environ, **cfg.get("env", {})})
            return await stack.enter_async_context(stdio_client(params))
        url, headers = cfg["url"], cfg.get("headers")
        if cfg.get("type") == "sse" or url.rstrip("/").endswith("/sse"):
            from mcp.client.sse import sse_client
            return await stack.enter_async_context(sse_client(url, headers=headers))
        from mcp.client.streamable_http import create_mcp_http_client, streamable_http_client
        http = create_mcp_http_client(headers=headers)
        await stack.enter_async_context(http)
        return await stack.enter_async_context(streamable_http_client(url, http_client=http))

    async def _run(self, srv: Server):
        try:
            async with AsyncExitStack() as stack:
                streams = await self._transport(stack, srv.cfg)
                session = await stack.enter_async_context(ClientSession(streams[0], streams[1]))
                await asyncio.wait_for(session.initialize(), 30)
                srv.tools = (await session.list_tools()).tools
                srv.session, srv.status = session, "connected"
                srv.ready.set()
                await srv.stop.wait()
        except BaseException as e:  # noqa: BLE001 - report any failure in the UI
            srv.status, srv.error = "error", f"{type(e).__name__}: {e}"
            if isinstance(e, asyncio.CancelledError):
                srv.status = "stopped"
        finally:
            srv.session = None
            srv.ready.set()

    async def start_all(self, configs: dict[str, dict]) -> None:
        await self.stop_all()
        for name, cfg in configs.items():
            srv = Server(name, cfg)
            srv.task = asyncio.create_task(self._run(srv))
            self.servers[name] = srv
        await asyncio.gather(*(asyncio.wait_for(s.ready.wait(), 40) for s in self.servers.values()),
                             return_exceptions=True)

    async def stop_all(self) -> None:
        for srv in self.servers.values():
            srv.stop.set()
        for srv in self.servers.values():
            if srv.task:
                try:
                    await asyncio.wait_for(srv.task, 5)
                except BaseException:  # noqa: BLE001
                    srv.task.cancel()
        self.servers.clear()

    def tool_schemas(self) -> list[dict]:
        return [to_openai_tool(s.name, t) for s in self.servers.values()
                if s.status == "connected" for t in s.tools]

    def owns(self, tool_name: str) -> bool:
        return SEP in tool_name and tool_name.split(SEP, 1)[0] in self.servers

    async def call(self, tool_name: str, args: dict) -> str:
        server, tool = tool_name.split(SEP, 1)
        srv = self.servers.get(server)
        if not srv or not srv.session:
            return f"Error: MCP server '{server}' is not connected."
        try:
            return flatten_result(await srv.session.call_tool(tool, args))
        except Exception as e:  # noqa: BLE001
            return f"Error calling {tool_name}: {e}"

    def status_rows(self) -> list[list]:
        return [[s.name, s.status + (f" ({s.error})" if s.error else ""),
                 ", ".join(t.name for t in s.tools)] for s in self.servers.values()]
