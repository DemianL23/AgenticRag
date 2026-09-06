"""Split parsed page documents into stable, citation-preserving chunks."""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from agenticrag.ingestion.parse_pdf import write_documents_jsonl


@dataclass(frozen=True, slots=True)
class ChunkConfig:
    """V0 chunking parameters, recorded with every batch run."""

    chunk_size: int = 800
    chunk_overlap: int = 120
    separators: tuple[str, ...] = ("\n\n", "\n", "。", "！", "？", "；", "，", " ", "",)

    def validate(self) -> None:
        if self.chunk_size <= 0:
            raise ValueError("chunk_size 必须大于 0")
        if self.chunk_overlap < 0:
            raise ValueError("chunk_overlap 不能小于 0")
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap 必须小于 chunk_size")

    def to_record(self) -> dict[str, Any]:
        return asdict(self)


def chunk_documents(
    documents: Iterable[Document],
    *,
    config: ChunkConfig | None = None,
) -> list[Document]:
    """Split each page independently and preserve its physical-page metadata."""
    config = config or ChunkConfig()
    config.validate()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
        separators=list(config.separators),
        length_function=len,
    )

    chunks: list[Document] = []
    for page_document in documents:
        page_content = page_document.page_content.strip()
        status = str(page_document.metadata.get("extraction_status", "ok"))
        if not page_content or status != "ok":
            continue

        page_number = _positive_int(page_document.metadata.get("page_number"), "page_number")
        doc_id = str(page_document.metadata.get("doc_id", "")).strip()
        if not doc_id:
            raise ValueError("逐页文档缺少 doc_id")
        page_chunks = splitter.split_text(page_content)
        for chunk_index, chunk_text in enumerate(page_chunks):
            chunks.append(
                Document(
                    page_content=chunk_text,
                    metadata={
                        **page_document.metadata,
                        "chunk_id": f"{doc_id}:p{page_number:04d}:c{chunk_index:03d}",
                        "chunk_index": chunk_index,
                        "chunks_on_page": len(page_chunks),
                        "chunk_char_count": len(chunk_text),
                        "chunker": "recursive_character",
                        "chunk_size": config.chunk_size,
                        "chunk_overlap": config.chunk_overlap,
                    },
                )
            )
    return chunks


def load_documents_jsonl(input_path: Path) -> list[Document]:
    """Load the page-level JSONL produced by the PDF parser."""
    path = Path(input_path)
    if not path.is_file():
        raise FileNotFoundError(f"解析产物不存在：{path}")

    documents: list[Document] = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, raw_line in enumerate(file, start=1):
            if not raw_line.strip():
                continue
            try:
                record = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"解析产物第 {line_number} 行不是合法 JSON：{path}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"解析产物第 {line_number} 行必须是 JSON 对象：{path}")
            page_content = record.get("page_content")
            metadata = record.get("metadata")
            if not isinstance(page_content, str) or not isinstance(metadata, dict):
                raise ValueError(f"解析产物第 {line_number} 行缺少 page_content 或 metadata：{path}")
            documents.append(Document(page_content=page_content, metadata=metadata))
    return documents


def build_chunks(
    parsed_input_dir: Path,
    output_dir: Path,
    *,
    config: ChunkConfig | None = None,
    report_output: Path | None = None,
) -> dict[str, Any]:
    """Chunk every parsed ``doc_*.jsonl`` file and write a batch report."""
    config = config or ChunkConfig()
    config.validate()
    parsed_input_dir = Path(parsed_input_dir)
    if not parsed_input_dir.is_dir():
        raise FileNotFoundError(f"解析产物目录不存在：{parsed_input_dir}")
    input_files = sorted(parsed_input_dir.glob("doc_*.jsonl"))
    if not input_files:
        raise FileNotFoundError(f"解析产物目录中没有 doc_*.jsonl：{parsed_input_dir}")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = Path(report_output) if report_output is not None else output_dir / "report.json"
    document_reports: list[dict[str, Any]] = []
    for input_path in input_files:
        pages = load_documents_jsonl(input_path)
        chunks = chunk_documents(pages, config=config)
        output_path = output_dir / input_path.name
        write_documents_jsonl(chunks, output_path)
        skipped_empty = sum(
            1
            for page in pages
            if not page.page_content.strip() or page.metadata.get("extraction_status") == "empty"
        )
        skipped_error = sum(
            1 for page in pages if page.metadata.get("extraction_status") == "error"
        )
        document_reports.append(
            {
                "doc_id": input_path.stem,
                "input": input_path.as_posix(),
                "output": output_path.as_posix(),
                "pages": len(pages),
                "source_chars": sum(len(page.page_content) for page in pages),
                "chunks": len(chunks),
                "chunk_chars": sum(len(chunk.page_content) for chunk in chunks),
                "skipped_empty_pages": skipped_empty,
                "skipped_error_pages": skipped_error,
            }
        )

    report = {
        "parsed_input_dir": parsed_input_dir.as_posix(),
        "output_dir": output_dir.as_posix(),
        "report_output": report_path.as_posix(),
        "config": config.to_record(),
        "documents": document_reports,
        "totals": {
            "documents": len(document_reports),
            "pages": sum(item["pages"] for item in document_reports),
            "source_chars": sum(item["source_chars"] for item in document_reports),
            "chunks": sum(item["chunks"] for item in document_reports),
            "chunk_chars": sum(item["chunk_chars"] for item in document_reports),
            "skipped_empty_pages": sum(item["skipped_empty_pages"] for item in document_reports),
            "skipped_error_pages": sum(item["skipped_error_pages"] for item in document_reports),
        },
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = report_path.with_name(f".{report_path.name}.tmp")
    temporary_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary_path.replace(report_path)
    return report


def main() -> None:
    args = _parse_args()
    report = build_chunks(
        args.parsed_input_dir,
        args.output_dir,
        config=ChunkConfig(chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap),
        report_output=args.report_output,
    )
    print(json.dumps(report, ensure_ascii=False))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="将逐页解析结果切成保留页码的检索块。")
    parser.add_argument("--parsed-input-dir", type=Path, required=True, help="逐页 JSONL 所在目录")
    parser.add_argument("--output-dir", type=Path, required=True, help="chunk JSONL 输出目录")
    parser.add_argument("--chunk-size", type=int, default=800, help="每个 chunk 的最大字符数")
    parser.add_argument("--chunk-overlap", type=int, default=120, help="相邻 chunk 的重叠字符数")
    parser.add_argument("--report-output", type=Path, help="运行报告路径，默认写入输出目录/report.json")
    return parser.parse_args()


def _positive_int(value: object, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field_name} 必须是正整数")
    return value
