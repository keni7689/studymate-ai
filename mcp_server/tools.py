"""The three study tools as plain Python functions.

They are used by the MCP server (server.py) AND can be called directly
(for example from the Streamlit page or from tests). Each returns text.
"""
from src import rag_pipeline
from src.prompts import NOT_FOUND_MESSAGE
from src.retriever import retrieve
from src.embeddings import EmbeddingError
from src.vector_store import VectorStoreError


def _format_chunks(chunks: list[dict]) -> str:
    lines = []
    for number, chunk in enumerate(chunks, start=1):
        page = f", page {chunk['page']}" if chunk["page"] else ""
        lines.append(f"[{number}] {chunk['source']}{page} (similarity {chunk['score']:.2f})\n{chunk['text']}")
    return "\n\n".join(lines)


def search_notes(query: str, top_k: int = 3, store=None) -> str:
    """Find the note passages most relevant to a query."""
    try:
        store = store or rag_pipeline.get_store()
        if store.count() == 0:
            return "The knowledge base is empty. Add notes first."
        chunks = retrieve(query, store, top_k=max(1, min(int(top_k), 10)))
    except ValueError as error:
        return str(error)
    except (EmbeddingError, VectorStoreError) as error:
        return f"Search failed: {error}"
    return _format_chunks(chunks) if chunks else NOT_FOUND_MESSAGE


def get_topic(topic: str, store=None) -> str:
    """Collect what the notes say about a topic (more passages, no LLM needed)."""
    try:
        store = store or rag_pipeline.get_store()
        if store.count() == 0:
            return "The knowledge base is empty. Add notes first."
        chunks = retrieve(topic, store, top_k=5)
    except ValueError as error:
        return str(error)
    except (EmbeddingError, VectorStoreError) as error:
        return f"Lookup failed: {error}"
    if not chunks:
        return NOT_FOUND_MESSAGE
    sources = ", ".join(f"{s['source']}" + (f" p.{s['page']}" if s["page"] else "") for s in rag_pipeline.unique_sources(chunks))
    return f"Topic: {topic}\n\n{_format_chunks(chunks)}\n\nSources: {sources}"


def generate_quiz(topic: str, num_questions: int = 5, store=None) -> str:
    """Make quiz questions about a topic using the notes."""
    result = rag_pipeline.make_quiz(topic, num_questions, store)
    sources = ", ".join(s["source"] for s in result["sources"])
    return result["answer"] + (f"\n\nSources: {sources}" if sources else "")
