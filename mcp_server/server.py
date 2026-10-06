"""MCP server: exposes the study tools to any MCP client (Claude Desktop, MCP Inspector...).

Run from the project root:   python -m mcp_server.server
It talks over stdio, so it prints nothing on its own - a client starts and talks to it.
"""
import logging
import sys

from mcp.server.fastmcp import FastMCP

from mcp_server import tools

# stdout is reserved for the MCP protocol, so log to stderr only
logging.basicConfig(stream=sys.stderr, level=logging.WARNING)

mcp = FastMCP("studymate-ai")


@mcp.tool()
def search_notes(query: str, top_k: int = 3) -> str:
    """Search the student's notes and return the most relevant passages."""
    return tools.search_notes(query, top_k)


@mcp.tool()
def get_topic(topic: str) -> str:
    """Return what the student's notes say about a topic, with sources."""
    return tools.get_topic(topic)


@mcp.tool()
def generate_quiz(topic: str, num_questions: int = 5) -> str:
    """Generate quiz questions about a topic from the student's notes."""
    return tools.generate_quiz(topic, num_questions)


if __name__ == "__main__":
    mcp.run()  # stdio transport
