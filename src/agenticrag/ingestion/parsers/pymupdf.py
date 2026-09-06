"""Extract one LangChain document per physical PDF page with PyMuPDF."""

from __future__ import annotations

from hashlib import sha256
from importlib.metadata import version
from pathlib import Path

import pymupdf
from langchain_core.documents import Document


class PdfParseError(RuntimeError):
    """Raised when a PDF cannot be opened or requires a password."""


def parse_pdf(pdf_path: Path, *, doc_id: str) -> list[Document]:
    """Parse every physical page and retain enough metadata for citations.

    Empty or unreadable pages are represented by a Document with empty content
    and an ``extraction_status`` value instead of being silently discarded.
    """
    path = Path(pdf_path)
    clean_doc_id = doc_id.strip()
    _validate_input(path, clean_doc_id)

    source_hash = _sha256(path)
    try:
        pdf = pymupdf.open(path)
    except Exception as exc:
        raise PdfParseError(f"无法打开 PDF：{path}") from exc

    with pdf:
        if pdf.needs_pass:
            raise PdfParseError(f"PDF 需要密码：{path}")

        total_pages = pdf.page_count
        documents: list[Document] = []
        for page_index, page in enumerate(pdf):
            status = "ok"
            error_type = ""
            try:
                text = _normalize_text(page.get_text("text", sort=True))
                if not text:
                    status = "empty"
            except Exception as exc:
                text = ""
                status = "error"
                error_type = type(exc).__name__

            documents.append(
                Document(
                    page_content=text,
                    metadata={
                        "doc_id": clean_doc_id,
                        "source": path.as_posix(),
                        "filename": path.name,
                        "source_sha256": source_hash,
                        "page_number": page_index + 1,
                        "total_pages": total_pages,
                        "parser": "pymupdf",
                        "parser_version": version("pymupdf"),
                        "extraction_status": status,
                        "extraction_error_type": error_type,
                    },
                )
            )

    return documents


def _validate_input(path: Path, doc_id: str) -> None:
    if not doc_id:
        raise ValueError("doc_id 不能为空")
    if not path.is_file():
        raise FileNotFoundError(f"PDF 不存在：{path}")
    if path.suffix.lower() != ".pdf":
        raise ValueError(f"输入文件必须是 PDF：{path}")


def _sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as file:
        while block := file.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _normalize_text(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n").strip()
