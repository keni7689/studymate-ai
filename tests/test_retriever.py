import pytest

from src.embeddings import embed_query, embed_texts
from src.retriever import retrieve
from src.vector_store import VectorStore


def test_embeddings_have_expected_shape():
    vectors = embed_texts(["hello world", "another text"])
    assert len(vectors) == 2 and len(vectors[0]) == len(embed_query("hi"))


def test_store_persists_between_sessions(tmp_path, indexed_store, store):
    reopened = VectorStore(store._client.get_settings().persist_directory)
    assert reopened.count() == indexed_store.count() > 0


def test_search_returns_relevant_chunk(indexed_store):
    hits = retrieve("What is gradient descent learning rate", indexed_store)
    assert hits and "gradient descent" in hits[0]["text"].lower()
    assert hits[0]["source"] == "machine_learning.md"


def test_search_finds_txt_note(indexed_store):
    hits = retrieve("backpropagation chain rule gradient", indexed_store)
    assert hits[0]["source"] == "neural_networks.txt"


def test_irrelevant_question_returns_nothing(indexed_store):
    assert retrieve("chocolate pancakes banana", indexed_store) == []


def test_empty_question_raises(indexed_store):
    with pytest.raises(ValueError):
        retrieve("   ", indexed_store)


def test_search_on_empty_store(store):
    assert store.search(embed_query("anything"), 3) == []
