import pytest

from src.chunker import chunk_pages, chunk_text


def test_short_text_is_one_chunk():
    assert chunk_text("Short note.", 100, 10) == ["Short note."]


def test_empty_text_gives_no_chunks():
    assert chunk_text("   ", 100, 10) == []


def test_chunks_respect_size_and_cover_text():
    text = " ".join(f"word{i}." for i in range(400))
    chunks = chunk_text(text, 200, 40)
    assert len(chunks) > 5
    assert all(len(c) <= 200 for c in chunks)
    assert chunks[0].startswith("word0.") and chunks[-1].endswith("word399.")


def test_overlap_repeats_text():
    text = " ".join(f"w{i}" for i in range(300))
    chunks = chunk_text(text, 200, 50)
    for first, second in zip(chunks, chunks[1:]):
        assert second.split()[0] in first  # next chunk starts with text already seen


def test_no_infinite_loop_with_huge_overlap_and_no_spaces():
    assert len(chunk_text("x" * 1000, 100, 99)) >= 10


@pytest.mark.parametrize("size,overlap", [(0, 0), (100, 100), (100, -1)])
def test_invalid_settings(size, overlap):
    with pytest.raises(ValueError):
        chunk_text("text", size, overlap)


def test_chunk_pages_keeps_metadata():
    pages = [{"text": "a " * 300, "source": "f.pdf", "page": 2}]
    chunks = chunk_pages(pages, 200, 20)
    assert all(c["source"] == "f.pdf" and c["page"] == 2 for c in chunks)
    assert [c["chunk_index"] for c in chunks] == list(range(len(chunks)))
