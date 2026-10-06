"""Phase 6: question -> query embedding -> similarity search -> top-k chunks."""
from config import settings
from src.embeddings import embed_query
from src.vector_store import VectorStore


def retrieve(
    question: str, store: VectorStore, top_k: int | None = None, min_score: float | None = None
) -> list[dict]:
    """Return relevant chunks (best first). Chunks below min_score are dropped."""
    question = (question or "").strip()
    if not question:
        raise ValueError("Please type a question first.")
    top_k = settings.TOP_K if top_k is None else top_k
    min_score = settings.MIN_SCORE if min_score is None else min_score
    hits = store.search(embed_query(question), top_k)
    return [hit for hit in hits if hit["score"] >= min_score]
