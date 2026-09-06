import json
from pathlib import Path

from langchain_core.documents import Document

from agenticrag.ingestion.chunk import ChunkConfig, build_chunks, chunk_documents


def _page(doc_id: str, page_number: int, text: str, status: str = "ok") -> Document:
    return Document(
        page_content=text,
        metadata={
            "doc_id": doc_id,
            "page_number": page_number,
            "extraction_status": status,
            "filename": "report.pdf",
        },
    )


def test_chunking_preserves_page_metadata_and_stable_ids() -> None:
    chunks = chunk_documents(
        [_page("doc_test", 3, "甲乙丙丁戊己庚辛壬癸甲乙丙丁戊己庚辛壬癸")],
        config=ChunkConfig(chunk_size=10, chunk_overlap=2),
    )

    assert len(chunks) > 1
    assert all(chunk.metadata["doc_id"] == "doc_test" for chunk in chunks)
    assert all(chunk.metadata["page_number"] == 3 for chunk in chunks)
    assert [chunk.metadata["chunk_id"] for chunk in chunks] == [
        f"doc_test:p0003:c{index:03d}" for index in range(len(chunks))
    ]
    assert all(chunk.metadata["chunk_char_count"] == len(chunk.page_content) for chunk in chunks)


def test_chunking_skips_empty_and_failed_pages() -> None:
    chunks = chunk_documents(
        [
            _page("doc_test", 1, "有内容"),
            _page("doc_test", 2, "", "empty"),
            _page("doc_test", 3, "提取失败", "error"),
        ]
    )

    assert len(chunks) == 1
    assert chunks[0].metadata["page_number"] == 1


def test_build_chunks_writes_jsonl_and_report(tmp_path: Path) -> None:
    parsed_dir = tmp_path / "parsed"
    parsed_dir.mkdir()
    input_path = parsed_dir / "doc_test.jsonl"
    input_path.write_text(
        "\n".join(
            json.dumps(
                {"page_content": page.page_content, "metadata": page.metadata},
                ensure_ascii=False,
            )
            for page in [_page("doc_test", 1, "Revenue 2024")]
        )
        + "\n",
        encoding="utf-8",
    )

    report = build_chunks(
        parsed_dir,
        tmp_path / "chunks",
        config=ChunkConfig(chunk_size=5, chunk_overlap=1),
    )

    assert report["totals"]["documents"] == 1
    assert report["totals"]["chunks"] > 1
    assert (tmp_path / "chunks/doc_test.jsonl").is_file()
    assert (tmp_path / "chunks/report.json").is_file()
