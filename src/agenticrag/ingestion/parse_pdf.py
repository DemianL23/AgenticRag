"""Command-line entry point for inspecting one PDF parsing result."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from langchain_core.documents import Document

from agenticrag.ingestion.parsers import parse_pdf


def main() -> None:
    args = _parse_args()
    documents = parse_pdf(args.input, doc_id=args.doc_id)
    write_documents_jsonl(documents, args.output)

    statuses: dict[str, int] = {}
    for document in documents:
        status = str(document.metadata["extraction_status"])
        statuses[status] = statuses.get(status, 0) + 1

    print(
        json.dumps(
            {
                "doc_id": args.doc_id,
                "pages": len(documents),
                "statuses": statuses,
                "output": args.output.as_posix(),
            },
            ensure_ascii=False,
        )
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="使用 PyMuPDF 将一份 PDF 提取为逐页 JSONL。",
    )
    parser.add_argument("--input", type=Path, required=True, help="PDF 文件路径")
    parser.add_argument("--doc-id", required=True, help="稳定的文档标识，例如 doc_000")
    parser.add_argument("--output", type=Path, required=True, help="输出 JSONL 路径")
    return parser.parse_args()


def write_documents_jsonl(documents: list[Document], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(f".{output_path.name}.tmp")
    with temporary_path.open("w", encoding="utf-8", newline="\n") as file:
        for document in documents:
            record = {
                "page_content": document.page_content,
                "metadata": document.metadata,
            }
            file.write(json.dumps(record, ensure_ascii=False) + "\n")
    temporary_path.replace(output_path)


def _write_jsonl(documents: list[Document], output_path: Path) -> None:
    """Backward-compatible alias used by the first parser tests."""
    write_documents_jsonl(documents, output_path)
