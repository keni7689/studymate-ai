"""Test setup: use the tiny offline embedder so tests need no model download."""
import os

os.environ["STUDYMATE_EMBEDDER"] = "hashing"
os.environ["MIN_SCORE"] = "0.05"  # the toy test embedder gives lower scores than a real model

import pytest

from config import settings
from src import generator, rag_pipeline
from src.vector_store import VectorStore


@pytest.fixture
def store(tmp_path):
    return VectorStore(tmp_path / "chroma")


@pytest.fixture
def indexed_store(store):
    for note in sorted(settings.SAMPLE_NOTES_DIR.iterdir()):
        assert rag_pipeline.add_document(note, store)["status"] == "added"
    return store


@pytest.fixture(autouse=True)
def reset_generator():
    yield
    generator.set_generation_backend(None)
