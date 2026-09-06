"""Build the V0 document manifest and parse every distinct PDF once."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from agenticrag.ingestion.manifest import build_document_manifest, write_manifest_jsonl
from agenticrag.ingestion.parse_pdf import write_documents_jsonl
from agenticrag.ingestion.parsers import parse_pdf


def build_corpus(
    qa_path: Path,
    manifest_output: Path,
    parsed_output_dir: Path,
    *,
    report_output: Path | None = None,
) -> dict[str, Any]:
    """Build manifest, parse all PDFs, and return a reproducible run report."""
    qa_path = Path(qa_path)
    manifest = build_document_manifest(qa_path)
    write_manifest_jsonl(manifest, manifest_output)
    parsed_output_dir = Path(parsed_output_dir)
    parsed_output_dir.mkdir(parents=True, exist_ok=True)
    report_path = Path(report_output) if report_output is not None else parsed_output_dir / "report.json"

    document_reports: list[dict[str, Any]] = []
    for item in manifest:
        pdf_path = qa_path.parent / Path(item.pdf_path)
        documents = parse_pdf(pdf_path, doc_id=item.doc_id)
        output_path = parsed_output_dir / f"{item.doc_id}.jsonl"
        write_documents_jsonl(documents, output_path)

        statuses = Counter(str(document.metadata["extraction_status"]) for document in documents)
        document_reports.append(
            {
                "doc_id": item.doc_id,
                "pdf_path": item.pdf_path,
                "qa_count": item.qa_count,
                "pages": len(documents),
                "content_chars": sum(len(document.page_content) for document in documents),
                "statuses": dict(sorted(statuses.items())),
                "output": output_path.as_posix(),
            }
        )

    report = {
        "qa_path": qa_path.as_posix(),
        "manifest_output": Path(manifest_output).as_posix(),
        "parsed_output_dir": parsed_output_dir.as_posix(),
        "report_output": report_path.as_posix(),
        "documents": document_reports,
        "totals": {
            "documents": len(document_reports),
            "qa_examples": sum(item["qa_count"] for item in document_reports),
            "pages": sum(item["pages"] for item in document_reports),
            "content_chars": sum(item["content_chars"] for item in document_reports),
        },
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = report_path.with_name(f".{report_path.name}.tmp")
    temporary_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary_path.replace(report_path)
    return report


def main() -> None:
    args = _parse_args()
    report = build_corpus(
        args.qa,
        args.manifest_output,
        args.parsed_output_dir,
        report_output=args.report_output,
    )
    print(json.dumps(report, ensure_ascii=False))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="按 QA 清单去重并批量解析全部 PDF。")
    parser.add_argument("--qa", type=Path, required=True, help="QA JSONL 文件路径")
    parser.add_argument("--manifest-output", type=Path, required=True, help="文档清单 JSONL 输出路径")
    parser.add_argument("--parsed-output-dir", type=Path, required=True, help="逐页 JSONL 输出目录")
    parser.add_argument("--report-output", type=Path, help="运行报告路径，默认写入解析输出目录/report.json")
    return parser.parse_args()
