from config import settings
from src import generator, rag_pipeline
from src.prompts import NOT_FOUND_MESSAGE


def test_duplicate_documents_are_skipped(store):
    note = settings.SAMPLE_NOTES_DIR / "machine_learning.md"
    assert rag_pipeline.add_document(note, store)["status"] == "added"
    count = store.count()
    assert rag_pipeline.add_document(note, store)["status"] == "duplicate"
    assert store.count() == count


def test_bad_file_does_not_crash(store, tmp_path):
    bad = tmp_path / "bad.pdf"
    bad.write_bytes(b"nope")
    assert rag_pipeline.add_document(bad, store)["status"] == "error"


def test_pipeline_sends_retrieved_context_to_model(indexed_store):
    seen = {}

    def fake_model(messages):
        seen["messages"] = messages
        return "Fake answer."

    generator.set_generation_backend(fake_model)
    result = rag_pipeline.ask("Explain gradient descent learning rate", indexed_store)
    assert result["answer"] == "Fake answer."
    user_prompt = seen["messages"][1]["content"]
    assert "CONTEXT" in user_prompt and "learning rate" in user_prompt
    assert seen["messages"][0]["role"] == "system"
    assert result["sources"][0]["source"] == "machine_learning.md"


def test_unrelated_question_skips_model(indexed_store):
    generator.set_generation_backend(lambda m: (_ for _ in ()).throw(AssertionError("model must not run")))
    result = rag_pipeline.ask("chocolate pancakes banana recipe", indexed_store)
    assert result["answer"] == NOT_FOUND_MESSAGE and result["sources"] == []


def test_empty_question_and_empty_store(tmp_path, indexed_store):
    from src.vector_store import VectorStore

    empty = VectorStore(tmp_path / "other")
    assert "empty" in rag_pipeline.ask("hello", empty)["answer"].lower()
    assert "question" in rag_pipeline.ask("  ", indexed_store)["answer"].lower()


def test_model_failure_falls_back_to_passages(indexed_store):
    def broken(messages):
        raise generator.GenerationError("model missing")

    generator.set_generation_backend(broken)
    result = rag_pipeline.ask("What is overfitting dropout regularization", indexed_store)
    assert "model missing" in result["answer"] and "Overfitting" in result["answer"]


def test_retrieval_only_mode(indexed_store, monkeypatch):
    monkeypatch.setattr(settings, "GENERATION_MODE", "retrieval_only")
    result = rag_pipeline.ask("What is overfitting dropout regularization", indexed_store)
    assert "Best matching passages" in result["answer"]


def test_quiz_fallback_without_model(indexed_store, monkeypatch):
    monkeypatch.setattr(settings, "GENERATION_MODE", "retrieval_only")
    result = rag_pipeline.make_quiz("gradient descent", 3, indexed_store)
    assert "Fill in the blank" in result["answer"] and "Answer:" in result["answer"]


def test_reset_clears_knowledge_base(indexed_store):
    rag_pipeline.reset_knowledge_base(indexed_store)
    assert indexed_store.count() == 0
