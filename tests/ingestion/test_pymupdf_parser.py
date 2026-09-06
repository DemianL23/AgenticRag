import json
from pathlib import Path

import pymupdf

from agenticrag.ingestion.parse_pdf import _write_jsonl
from agenticrag.ingestion.parsers.pymupdf import parse_pdf


def _make_two_page_pdf(path: Path) -> None:
    with pymupdf.open() as pdf:
        first_page = pdf.new_page()
        first_page.insert_text((72, 72), "Revenue 2024: 100")
        pdf.new_page()
        pdf.save(path)


def test_parse_pdf_keeps_page_numbers_and_empty_pages(tmp_path: Path) -> None:
    pdf_path = tmp_path / "report.pdf"
    _make_two_page_pdf(pdf_path)

    documents = parse_pdf(pdf_path, doc_id="doc_test")

    assert len(documents) == 2
    assert documents[0].page_content == "Revenue 2024: 100"
    assert documents[0].metadata["page_number"] == 1
    assert documents[0].metadata["extraction_status"] == "ok"
    assert documents[1].page_content == ""
    assert documents[1].metadata["page_number"] == 2
    assert documents[1].metadata["extraction_status"] == "empty"
    assert all(doc.metadata["doc_id"] == "doc_test" for doc in documents)
    assert all(doc.metadata["total_pages"] == 2 for doc in documents)
    assert len(documents[0].metadata["source_sha256"]) == 64


def test_write_jsonl_preserves_unicode_content_and_metadata(tmp_path: Path) -> None:
    pdf_path = tmp_path / "财务报告.pdf"
    with pymupdf.open() as pdf:
        page = pdf.new_page()
        page.insert_font(fontname="china-s", fontfile="china-s")
        page.insert_text((72, 72), "净利润", fontname="china-s")
        pdf.save(pdf_path)

    documents = parse_pdf(pdf_path, doc_id="doc_cn")
    output_path = tmp_path / "parsed.jsonl"
    _write_jsonl(documents, output_path)

    record = json.loads(output_path.read_text(encoding="utf-8").strip())
    assert record["page_content"] == "净利润"
    assert record["metadata"]["filename"] == "财务报告.pdf"
