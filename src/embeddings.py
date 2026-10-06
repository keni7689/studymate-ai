"""Phase 4: turn text into vectors (embeddings) with a local model.

The sentence-transformers model downloads once, then is cached on disk.
No API key, no internet needed after the first download.
"""
import hashlib
import os
import re

import numpy as np

from config import settings


class EmbeddingError(Exception):
    """Raised when the embedding model cannot be loaded."""


class HashingEmbedder:
    """Tiny offline stand-in (bag-of-words hashing). ONLY for automated tests.

    Enable with:  set STUDYMATE_EMBEDDER=hashing
    Real usage never needs it.
    """

    dim = 384

    def encode(self, texts, **_ignored) -> np.ndarray:
        vectors = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for word in re.findall(r"[a-z0-9]+", text.lower()):
                column = int(hashlib.md5(word.encode()).hexdigest(), 16) % self.dim
                vectors[row, column] += 1.0
            norm = np.linalg.norm(vectors[row])
            if norm == 0:
                vectors[row, 0], norm = 1.0, 1.0  # avoid zero vectors (cosine would break)
            vectors[row] /= norm
        return vectors


_embedder = None


def get_embedder():
    """Load the model once and reuse it (loading is the slow part)."""
    global _embedder
    if _embedder is None:
        if os.getenv("STUDYMATE_EMBEDDER") == "hashing":
            _embedder = HashingEmbedder()
        else:
            try:
                from sentence_transformers import SentenceTransformer

                _embedder = SentenceTransformer(settings.EMBEDDING_MODEL)
            except Exception as error:
                raise EmbeddingError(
                    f"Could not load the embedding model '{settings.EMBEDDING_MODEL}'. "
                    "The first run needs internet to download it once. "
                    f"Details: {error}"
                ) from error
    return _embedder


def embed_texts(texts: list[str], batch_size: int = 32) -> list[list[float]]:
    """Embed many texts. Vectors are normalised so cosine similarity works well."""
    if not texts:
        return []
    vectors = get_embedder().encode(
        texts, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=False
    )
    return np.asarray(vectors, dtype=np.float32).tolist()


def embed_query(question: str) -> list[float]:
    return embed_texts([question])[0]
