from mcp.server.mcpserver import MCPServer

m = MCPServer("fake")


@m.tool()
def add(a: int, b: int) -> str:
    """Add two numbers."""
    return str(a + b)


if __name__ == "__main__":
    m.run()
