"""Phase 5: store chunk vectors in a persistent ChromaDB folder."""
from pathlib import Path

import chromadb
from chromadb.config import Settings as ChromaSettings

from config import settings


class VectorStoreError(Exception):
    """Raised when the vector database cannot be opened or used."""


class VectorStore:
    def __init__(self, persist_dir: str | Path | None = None):
        persist_dir = Path(persist_dir or settings.CHROMA_DIR)
        persist_dir.mkdir(parents=True, exist_ok=True)
        try:
            self._client = chromadb.PersistentClient(
                path=str(persist_dir), settings=ChromaSettings(anonymized_telemetry=False)
            )
            self._collection = self._open_collection()
        except Exception as error:
            raise VectorStoreError(
                f"Could not open the vector database in '{persist_dir}'. "
                f"Try deleting that folder to rebuild it. Details: {error}"
            ) from error

    def _open_collection(self):
        # We always pass our own embeddings, so Chroma never downloads a model.
        return self._client.get_or_create_collection(
            name=settings.COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
        )

    def count(self) -> int:
        return self._collection.count()

    def has_document(self, doc_hash: str) -> bool:
        found = self._collection.get(where={"doc_hash": doc_hash}, limit=1)
        return len(found["ids"]) > 0

    def add_chunks(self, chunks: list[dict], embeddings: list[list[float]], doc_hash: str) -> None:
        if len(chunks) != len(embeddings):
            raise VectorStoreError("Number of chunks and embeddings must match.")
        try:
            self._collection.add(
                ids=[f"{doc_hash}-{chunk['chunk_index']}" for chunk in chunks],
                documents=[chunk["text"] for chunk in chunks],
                embeddings=embeddings,
                metadatas=[
                    {
                        "source": chunk["source"],
                        "page": chunk["page"],
                        "chunk_index": chunk["chunk_index"],
                        "doc_hash": doc_hash,
                    }
                    for chunk in chunks
                ],
            )
        except Exception as error:
            raise VectorStoreError(f"Could not save chunks: {error}") from error

    def search(self, query_embedding: list[float], top_k: int) -> list[dict]:
        """Return the top_k most similar chunks. score = cosine similarity (1 = identical)."""
        total = self.count()
        if total == 0:
            return []
        try:
            result = self._collection.query(
                query_embeddings=[query_embedding],
                n_results=min(top_k, total),
                include=["documents", "metadatas", "distances"],
            )
        except Exception as error:
            raise VectorStoreError(f"Search failed: {error}") from error
        hits = []
        for text, meta, distance in zip(
            result["documents"][0], result["metadatas"][0], result["distances"][0]
        ):
            hits.append(
                {"text": text, "source": meta["source"], "page": meta["page"], "score": 1.0 - distance}
            )
        return hits

    def list_documents(self) -> dict[str, int]:
        """Return {file name: number of chunks}."""
        metadatas = self._collection.get(include=["metadatas"])["metadatas"]
        summary: dict[str, int] = {}
        for meta in metadatas:
            summary[meta["source"]] = summary.get(meta["source"], 0) + 1
        return dict(sorted(summary.items()))

    def reset(self) -> None:
        """Delete everything (the original files in data/documents are kept)."""
        try:
            self._client.delete_collection(settings.COLLECTION_NAME)
            self._collection = self._open_collection()
        except Exception as error:
            raise VectorStoreError(f"Could not reset the database: {error}") from error
