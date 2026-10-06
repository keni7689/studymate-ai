"""Tiny MCP client: starts the server, lists its tools and calls one.

Run from the project root:   python -m mcp_server.client_demo "gradient descent"
"""
import asyncio
import os
import sys

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def main(query: str) -> None:
    params = StdioServerParameters(
        command=sys.executable, args=["-m", "mcp_server.server"], env=dict(os.environ)
    )
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            listing = await session.list_tools()
            print("Tools:", [tool.name for tool in listing.tools])
            result = await session.call_tool("search_notes", {"query": query, "top_k": 2})
            print(result.content[0].text)


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1] if len(sys.argv) > 1 else "gradient descent"))
