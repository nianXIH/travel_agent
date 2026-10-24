import pytest

from app.ingestion.loader import DocumentLoadError, load_text


def _minimal_pdf(text: str) -> bytes:
    content = f"BT /F1 24 Tf 72 720 Td ({text}) Tj ET".encode("latin-1")
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        b"/Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >>",
        b"<< /Length %d >>\nstream\n" % len(content) + content + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    pdf = b"%PDF-1.4\n"
    offsets = []
    for index, obj in enumerate(objects, 1):
        offsets.append(len(pdf))
        pdf += f"{index} 0 obj\n".encode() + obj + b"\nendobj\n"

    xref = len(pdf)
    pdf += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode()
    for offset in offsets:
        pdf += f"{offset:010d} 00000 n \n".encode()
    pdf += (
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n"
    ).encode()
    return pdf


def test_load_text_extracts_docx_paragraphs(tmp_path):
    from docx import Document

    path = tmp_path / "travel.docx"
    document = Document()
    document.add_paragraph("杭州西湖适合清晨游览。")
    document.save(path)

    assert "杭州西湖适合清晨游览。" in load_text(path)


def test_load_text_extracts_pdf_text(tmp_path):
    path = tmp_path / "travel.pdf"
    path.write_bytes(_minimal_pdf("Travel Guide 2026"))

    assert "Travel Guide 2026" in load_text(path)


def test_load_text_rejects_pdf_without_text(tmp_path):
    from pypdf import PdfWriter

    path = tmp_path / "blank.pdf"
    writer = PdfWriter()
    writer.add_blank_page(width=612, height=792)
    with path.open("wb") as handle:
        writer.write(handle)

    with pytest.raises(DocumentLoadError, match="no extractable text"):
        load_text(path)


def test_load_text_rejects_unsupported_suffix(tmp_path):
    path = tmp_path / "travel.csv"
    path.write_text("a,b", encoding="utf-8")

    with pytest.raises(DocumentLoadError, match="unsupported file type"):
        load_text(path)
