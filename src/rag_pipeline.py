"""Phase 8: connect everything.

Indexing : file -> text -> chunks -> embeddings -> ChromaDB
Asking   : question -> query embedding -> search -> context -> prompt -> local LLM -> answer
"""
import re
from pathlib import Path

from config import settings
from src import generator
from src.chunker import chunk_pages
from src.document_loader import DocumentLoadError, file_hash, load_document
from src.embeddings import EmbeddingError, embed_texts
from src.prompts import NOT_FOUND_MESSAGE, build_messages, build_quiz_messages
from src.retriever import retrieve
from src.vector_store import VectorStore, VectorStoreError

_default_store: VectorStore | None = None


def get_store() -> VectorStore:
    """One shared persistent store for the app."""
    global _default_store
    if _default_store is None:
        _default_store = VectorStore()
    return _default_store


# ---------- Indexing ----------
def add_document(path: str | Path, store: VectorStore | None = None) -> dict:
    """Index one file. Returns {"status": added|duplicate|error, "message", "chunks"}."""
    path = Path(path)
    try:
        store = store or get_store()
        pages = load_document(path)
        chunks = chunk_pages(pages)
        if not chunks:
            return {"status": "error", "message": f"'{path.name}' has no usable text.", "chunks": 0}
        doc_hash = file_hash(path)
        if store.has_document(doc_hash):
            return {"status": "duplicate", "message": f"'{path.name}' is already in the knowledge base.", "chunks": 0}
        store.add_chunks(chunks, embed_texts([c["text"] for c in chunks]), doc_hash)
        return {"status": "added", "message": f"Added '{path.name}' ({len(chunks)} chunks).", "chunks": len(chunks)}
    except (DocumentLoadError, EmbeddingError, VectorStoreError) as error:
        return {"status": "error", "message": str(error), "chunks": 0}


def list_documents(store: VectorStore | None = None) -> dict[str, int]:
    return (store or get_store()).list_documents()


def reset_knowledge_base(store: VectorStore | None = None) -> None:
    (store or get_store()).reset()


# ---------- Asking ----------
def unique_sources(chunks: list[dict]) -> list[dict]:
    """Sources come from retrieval (never from the model), so they cannot be invented."""
    seen: dict[tuple, dict] = {}
    for chunk in chunks:
        key = (chunk["source"], chunk["page"])
        if key not in seen:
            seen[key] = {"source": chunk["source"], "page": chunk["page"], "score": chunk["score"]}
    return list(seen.values())


def _passages_text(chunks: list[dict]) -> str:
    lines = []
    for number, chunk in enumerate(chunks, start=1):
        page = f" (page {chunk['page']})" if chunk["page"] else ""
        lines.append(f"{number}. From {chunk['source']}{page}:\n{chunk['text']}")
    return "\n\n".join(lines)


def _result(answer: str, chunks: list[dict] | None = None) -> dict:
    chunks = chunks or []
    return {"answer": answer, "chunks": chunks, "sources": unique_sources(chunks)}


def ask(question: str, store: VectorStore | None = None, top_k: int | None = None) -> dict:
    """Answer a question from the notes. Returns {"answer", "chunks", "sources"}."""
    try:
        store = store or get_store()
        if store.count() == 0:
            return _result("Your knowledge base is empty. Upload some notes first.")
        chunks = retrieve(question, store, top_k)
    except ValueError as error:
        return _result(str(error))
    except (EmbeddingError, VectorStoreError) as error:
        return _result(f"Something went wrong while searching: {error}")
    if not chunks:
        return _result(NOT_FOUND_MESSAGE)

    if not generator.is_local_mode():
        return _result("Best matching passages from your notes:\n\n" + _passages_text(chunks), chunks)
    try:
        answer = generator.generate(build_messages(question, chunks))
    except generator.GenerationError as error:
        # Don't crash: still show the retrieved passages so the user gets value.
        return _result(f"{error}\n\nHere are the most relevant passages instead:\n\n{_passages_text(chunks)}", chunks)
    return _result(answer or NOT_FOUND_MESSAGE, chunks)


# ---------- Quiz ----------
def _fill_in_blank_quiz(chunks: list[dict], num_questions: int) -> str:
    """Model-free quiz: blank out one long word in sentences taken from the notes."""
    questions = []
    for chunk in chunks:
        for sentence in re.split(r"(?<=[.!?])\s+", chunk["text"].replace("\n", " ")):
            words = [w.strip(".,;:()") for w in sentence.split() if len(w.strip(".,;:()")) >= 6]
            if 40 <= len(sentence) <= 250 and words:
                answer = max(words, key=len)
                questions.append(f"Q{len(questions) + 1}. Fill in the blank: {sentence.replace(answer, '_____', 1)}\nAnswer: {answer}")
            if len(questions) >= num_questions:
                return "\n\n".join(questions)
    return "\n\n".join(questions)


def make_quiz(topic: str, num_questions: int = 5, store: VectorStore | None = None) -> dict:
    """Quiz questions built from retrieved notes. Returns {"answer", "chunks", "sources"}."""
    num_questions = max(1, min(int(num_questions), 10))
    try:
        store = store or get_store()
        if store.count() == 0:
            return _result("Your knowledge base is empty. Upload some notes first.")
        chunks = retrieve(topic, store, top_k=max(settings.TOP_K, 5))
    except ValueError as error:
        return _result(str(error))
    except (EmbeddingError, VectorStoreError) as error:
        return _result(f"Something went wrong while searching: {error}")
    if not chunks:
        return _result(NOT_FOUND_MESSAGE)

    if generator.is_local_mode():
        try:
            text = generator.generate(build_quiz_messages(topic, num_questions, chunks), max_new_tokens=500)
            return _result(text, chunks)
        except generator.GenerationError:
            pass  # fall back to the simple quiz below
    return _result(_fill_in_blank_quiz(chunks, num_questions) or NOT_FOUND_MESSAGE, chunks)
