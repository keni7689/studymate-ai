"""Phase 3: split long text into overlapping chunks.

Why overlap? A sentence cut at a chunk border would lose its context; the
overlap repeats the end of one chunk at the start of the next.
"""
from config import settings

# Preferred places to cut, best first
_BREAKS = ("\n\n", ". ", "\n", " ")


def _find_cut(text: str, start: int, end: int, chunk_size: int) -> int:
    """Move `end` back to a natural boundary (paragraph, sentence, word)."""
    earliest = start + int(chunk_size * 0.6)  # don't make tiny chunks
    for separator in _BREAKS:
        position = text.rfind(separator, earliest, end)
        if position != -1:
            return position + len(separator)
    return end


def chunk_text(text: str, chunk_size: int | None = None, overlap: int | None = None) -> list[str]:
    chunk_size = settings.CHUNK_SIZE if chunk_size is None else chunk_size
    overlap = settings.CHUNK_OVERLAP if overlap is None else overlap
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be between 0 and chunk_size - 1")

    text = text.strip()
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        if end < len(text):
            end = _find_cut(text, start, end, chunk_size)
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= len(text):
            break
        next_start = end - overlap if end - overlap > start else end
        while 0 < next_start < end and not text[next_start - 1].isspace():
            next_start += 1  # don't start a chunk in the middle of a word
        start = next_start
    return chunks


def chunk_pages(pages: list[dict], chunk_size: int | None = None, overlap: int | None = None) -> list[dict]:
    """Chunk every page, keeping source file and page number with each chunk."""
    chunks = []
    for page in pages:
        for piece in chunk_text(page["text"], chunk_size, overlap):
            chunks.append(
                {
                    "text": piece,
                    "source": page["source"],
                    "page": page["page"],
                    "chunk_index": len(chunks),
                }
            )
    return chunks
