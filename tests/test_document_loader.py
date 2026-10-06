import pytest
from fpdf import FPDF

from src.document_loader import DocumentLoadError, clean_text, file_hash, load_document


def make_pdf(path, pages):
    pdf = FPDF()
    for text in pages:
        pdf.add_page()
        pdf.set_font("Helvetica", size=12)
        pdf.multi_cell(0, 8, text)
    pdf.output(str(path))


def test_loads_txt_and_md(tmp_path):
    for name in ("a.txt", "b.md"):
        file = tmp_path / name
        file.write_text("Hello   world\r\n\r\n\r\n\r\nSecond", encoding="utf-8")
        pages = load_document(file)
        assert pages == [{"text": "Hello world\n\nSecond", "source": name, "page": 0}]


def test_loads_pdf_with_page_numbers(tmp_path):
    file = tmp_path / "notes.pdf"
    make_pdf(file, ["Gradient descent page one.", "Overfitting page two."])
    pages = load_document(file)
    assert [p["page"] for p in pages] == [1, 2]
    assert "Overfitting" in pages[1]["text"]


@pytest.mark.parametrize("name,content", [("missing.txt", None), ("e.txt", ""), ("x.docx", "hi"), ("ws.txt", "   \n ")])
def test_bad_files_raise_friendly_error(tmp_path, name, content):
    file = tmp_path / name
    if content is not None:
        file.write_text(content)
    with pytest.raises(DocumentLoadError):
        load_document(file)


def test_corrupted_pdf(tmp_path):
    file = tmp_path / "bad.pdf"
    file.write_bytes(b"this is not a pdf")
    with pytest.raises(DocumentLoadError):
        load_document(file)


def test_hash_is_content_based(tmp_path):
    a, b = tmp_path / "a.txt", tmp_path / "b.txt"
    a.write_text("same"), b.write_text("same")
    assert file_hash(a) == file_hash(b)
    b.write_text("different")
    assert file_hash(a) != file_hash(b)


def test_clean_text():
    assert clean_text("  a \t b \x00\n\n\n\nc ") == "a b\n\nc"
