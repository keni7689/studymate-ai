import asyncio
import os
import sys

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from config import settings
from mcp_server import tools


def test_search_notes_tool(indexed_store):
    text = tools.search_notes("gradient descent learning rate", 2, store=indexed_store)
    assert "machine_learning.md" in text and "learning rate" in text


def test_get_topic_tool(indexed_store):
    text = tools.get_topic("backpropagation chain rule", store=indexed_store)
    assert "neural_networks.txt" in text and "Sources:" in text


def test_generate_quiz_tool(indexed_store, monkeypatch):
    monkeypatch.setattr(settings, "GENERATION_MODE", "retrieval_only")
    assert "Answer:" in tools.generate_quiz("overfitting", 2, store=indexed_store)


def test_tools_handle_empty_input(store):
    assert "empty" in tools.search_notes("x", store=store).lower()
    assert "empty" in tools.get_topic("x", store=store).lower()


def test_real_mcp_server_over_stdio(tmp_path):
    """Start the actual server process and talk to it like an MCP client would."""
    from src import rag_pipeline
    from src.vector_store import VectorStore

    chroma = tmp_path / "chroma"
    rag_pipeline.add_document(settings.SAMPLE_NOTES_DIR / "machine_learning.md", VectorStore(chroma))
    env = dict(os.environ, STUDYMATE_CHROMA_DIR=str(chroma), STUDYMATE_EMBEDDER="hashing", GENERATION_MODE="retrieval_only")

    async def run():
        params = StdioServerParameters(command=sys.executable, args=["-m", "mcp_server.server"], env=env, cwd=str(settings.BASE_DIR))
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                names = {t.name for t in (await session.list_tools()).tools}
                result = await session.call_tool("search_notes", {"query": "gradient descent learning rate"})
                return names, result.content[0].text

    names, text = asyncio.run(asyncio.wait_for(run(), 60))
    assert names == {"search_notes", "get_topic", "generate_quiz"}
    assert "machine_learning.md" in text
