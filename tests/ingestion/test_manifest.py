import json
from pathlib import Path

import pymupdf
import pytest

from agenticrag.ingestion.batch_parse import build_corpus
from agenticrag.ingestion.manifest import build_document_manifest


def _write_pdf(path: Path, text: str) -> None:
    with pymupdf.open() as pdf:
        page = pdf.new_page()
        page.insert_text((72, 72), text)
        pdf.save(path)


def _qa_record(doc_id: str, question_id: str, task_type: str = "Comparison") -> dict[str, str]:
    return {
        "finqa_id": question_id,
        "task_type": task_type,
        "_lang": "CN",
        "_doc_id": doc_id,
        "_corpus_file": f"corpus/{doc_id}.pdf",
        "_original_pdf": f"Reference_documents/CN/{doc_id}.pdf",
    }


def test_manifest_deduplicates_questions_by_document(tmp_path: Path) -> None:
    (tmp_path / "corpus").mkdir()
    _write_pdf(tmp_path / "corpus/doc_007.pdf", "Netflix")
    qa_path = tmp_path / "qa.jsonl"
    qa_path.write_text(
        "\n".join(
            json.dumps(record, ensure_ascii=False)
            for record in (
                _qa_record("doc_007", "q1"),
                _qa_record("doc_007", "q2", "Summary"),
            )
        )
        + "\n",
        encoding="utf-8",
    )

    manifest = build_document_manifest(qa_path)

    assert len(manifest) == 1
    assert manifest[0].doc_id == "doc_007"
    assert manifest[0].qa_count == 2
    assert manifest[0].finqa_ids == ("q1", "q2")
    assert manifest[0].task_types == ("Comparison", "Summary")


def test_manifest_rejects_conflicting_pdf_mapping(tmp_path: Path) -> None:
    (tmp_path / "corpus").mkdir()
    _write_pdf(tmp_path / "corpus/doc_001.pdf", "one")
    _write_pdf(tmp_path / "corpus/doc_002.pdf", "two")
    qa_path = tmp_path / "qa.jsonl"
    records = [_qa_record("doc_001", "q1"), _qa_record("doc_001", "q2")]
    records[1]["_corpus_file"] = "corpus/doc_002.pdf"
    qa_path.write_text("\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="映射不一致"):
        build_document_manifest(qa_path)


def test_build_corpus_parses_each_manifest_document_once(tmp_path: Path) -> None:
    (tmp_path / "corpus").mkdir()
    _write_pdf(tmp_path / "corpus/doc_007.pdf", "Netflix revenue")
    qa_path = tmp_path / "qa.jsonl"
    qa_path.write_text(
        "\n".join(json.dumps(_qa_record("doc_007", question_id)) for question_id in ("q1", "q2")) + "\n",
        encoding="utf-8",
    )
    manifest_output = tmp_path / "artifacts/manifest.jsonl"
    parsed_dir = tmp_path / "artifacts/parsed"

    report = build_corpus(qa_path, manifest_output, parsed_dir)

    assert report["totals"] == {"documents": 1, "qa_examples": 2, "pages": 1, "content_chars": 15}
    assert (parsed_dir / "doc_007.jsonl").is_file()
    assert (parsed_dir / "report.json").is_file()
    assert json.loads((parsed_dir / "report.json").read_text(encoding="utf-8"))["report_output"] == (
        parsed_dir / "report.json"
    ).as_posix()
    assert len(manifest_output.read_text(encoding="utf-8").splitlines()) == 1
