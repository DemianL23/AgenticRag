"""Build a document-level registry from the QA JSONL file."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True, slots=True)
class DocumentManifest:
    """The deduplicated relationship between one PDF and its QA examples."""

    doc_id: str
    pdf_path: str
    original_pdf: str
    language: str
    qa_count: int
    finqa_ids: tuple[str, ...]
    task_types: tuple[str, ...]

    def to_record(self) -> dict[str, Any]:
        """Return a JSON-serializable manifest record."""
        record = asdict(self)
        record["finqa_ids"] = list(self.finqa_ids)
        record["task_types"] = list(self.task_types)
        return record


_REQUIRED_FIELDS = (
    "_doc_id",
    "_corpus_file",
    "_original_pdf",
    "_lang",
    "finqa_id",
    "task_type",
)


def build_document_manifest(qa_path: Path) -> list[DocumentManifest]:
    """Read QA JSONL and return one validated row per distinct document.

    A document ID must always identify exactly one corpus PDF and language.
    This catches a broken mapping before parsing or indexing begins.
    """
    path = Path(qa_path)
    if not path.is_file():
        raise FileNotFoundError(f"QA 文件不存在：{path}")

    grouped: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as file:
        for line_number, raw_line in enumerate(file, start=1):
            if not raw_line.strip():
                continue
            try:
                record = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"QA 第 {line_number} 行不是合法 JSON") from exc
            if not isinstance(record, dict):
                raise ValueError(f"QA 第 {line_number} 行必须是 JSON 对象")

            missing = [field for field in _REQUIRED_FIELDS if not _nonempty(record.get(field))]
            if missing:
                names = ", ".join(missing)
                raise ValueError(f"QA 第 {line_number} 行缺少必填字段：{names}")

            doc_id = str(record["_doc_id"]).strip()
            pdf_path = _normalise_relative_path(record["_corpus_file"], field="_corpus_file", line_number=line_number)
            original_pdf = _normalise_path(record["_original_pdf"])
            language = str(record["_lang"]).strip()
            pdf_on_disk = _resolve_source(path.parent, pdf_path)
            if not pdf_on_disk.is_file():
                raise FileNotFoundError(
                    f"QA 第 {line_number} 行引用的 PDF 不存在：{pdf_on_disk}"
                )
            if pdf_on_disk.suffix.lower() != ".pdf":
                raise ValueError(f"QA 第 {line_number} 行引用的文件不是 PDF：{pdf_on_disk}")

            entry = grouped.setdefault(
                doc_id,
                {
                    "pdf_path": pdf_path,
                    "original_pdf": original_pdf,
                    "language": language,
                    "finqa_ids": [],
                    "task_types": set(),
                },
            )
            for field, value in (
                ("_corpus_file", pdf_path),
                ("_original_pdf", original_pdf),
                ("_lang", language),
            ):
                if value != entry[_field_key(field)]:
                    raise ValueError(
                        f"文档 {doc_id} 的 {field} 映射不一致："
                        f"已有 {entry[_field_key(field)]!r}，第 {line_number} 行为 {value!r}"
                    )

            finqa_id = str(record["finqa_id"]).strip()
            entry["finqa_ids"].append(finqa_id)
            entry["task_types"].add(str(record["task_type"]).strip())

    return [
        DocumentManifest(
            doc_id=doc_id,
            pdf_path=entry["pdf_path"],
            original_pdf=entry["original_pdf"],
            language=entry["language"],
            qa_count=len(entry["finqa_ids"]),
            finqa_ids=tuple(entry["finqa_ids"]),
            task_types=tuple(sorted(entry["task_types"])),
        )
        for doc_id, entry in sorted(grouped.items())
    ]


def write_manifest_jsonl(manifest: Iterable[DocumentManifest], output_path: Path) -> None:
    """Write a manifest atomically so an interrupted run cannot leave a partial file."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.tmp")
    with temporary_path.open("w", encoding="utf-8", newline="\n") as file:
        for document in manifest:
            file.write(json.dumps(document.to_record(), ensure_ascii=False) + "\n")
    temporary_path.replace(path)


def _nonempty(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _normalise_path(value: object) -> str:
    return Path(str(value).strip()).as_posix()


def _normalise_relative_path(value: object, *, field: str, line_number: int) -> str:
    candidate = Path(str(value).strip())
    if candidate.is_absolute():
        raise ValueError(f"QA 第 {line_number} 行的 {field} 必须是相对路径")
    return candidate.as_posix()


def _resolve_source(root: Path, relative_path: str) -> Path:
    return (root / Path(relative_path)).resolve()


def _field_key(field: str) -> str:
    return {
        "_corpus_file": "pdf_path",
        "_original_pdf": "original_pdf",
        "_lang": "language",
    }[field]
