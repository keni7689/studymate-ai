"""Phase 2: read .txt / .md / .pdf files and return clean text."""
import hashlib
import re
from pathlib import Path

from config import settings


class DocumentLoadError(Exception):
    """Raised with a beginner-friendly message when a file cannot be read."""


def clean_text(text: str) -> str:
    """Normalise line endings and whitespace so chunking behaves predictably."""
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\x00", "")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" ?\n ?", "\n", text)  # no stray spaces around line breaks
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def file_hash(path: Path) -> str:
    """Short fingerprint of the file CONTENT, used to detect duplicates."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]


def _read_text_file(path: Path) -> str:
    raw = path.read_bytes()
    try:
        return raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        return raw.decode("latin-1")  # never fails; good enough for old notes


def _read_pdf(path: Path) -> list[dict]:
    from pypdf import PdfReader  # imported here so txt/md work even if pypdf is broken

    try:
        reader = PdfReader(str(path))
        if reader.is_encrypted and not reader.decrypt(""):
            raise DocumentLoadError(f"'{path.name}' is password protected.")
        pages = []
        for number, page in enumerate(reader.pages, start=1):
            text = clean_text(page.extract_text() or "")
            if text:
                pages.append({"text": text, "source": path.name, "page": number})
    except DocumentLoadError:
        raise
    except Exception as error:  # pypdf raises many different error types
        raise DocumentLoadError(f"'{path.name}' is not a valid or readable PDF ({error}).") from error
    if not pages:
        raise DocumentLoadError(
            f"No text found in '{path.name}'. It may be a scanned PDF (images only)."
        )
    return pages


def load_document(path: str | Path) -> list[dict]:
    """Return a list of {"text", "source", "page"} dicts. page=0 means 'no pages'."""
    path = Path(path)
    if not path.is_file():
        raise DocumentLoadError(f"File not found: {path}")
    extension = path.suffix.lower()
    if extension not in settings.SUPPORTED_EXTENSIONS:
        raise DocumentLoadError(
            f"'{path.name}' has an unsupported type. Use: {', '.join(settings.SUPPORTED_EXTENSIONS)}"
        )
    if path.stat().st_size == 0:
        raise DocumentLoadError(f"'{path.name}' is empty.")
    if extension == ".pdf":
        return _read_pdf(path)
    text = clean_text(_read_text_file(path))
    if not text:
        raise DocumentLoadError(f"'{path.name}' contains no readable text.")
    return [{"text": text, "source": path.name, "page": 0}]
