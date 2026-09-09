import json
import sys
import types
from pathlib import Path

import pytest

from agenticrag.ingestion.index_bm25 import build_bm25_index
from agenticrag.rag.integrations.milvus_bm25 import BM25MilvusConfig


def _write_fixture_data(tmp_path: Path) -> tuple[Path, Path]:
    chunks_dir = tmp_path / "chunks"
    chunks_dir.mkdir()
    (chunks_dir / "doc_001.jsonl").write_text(
        json.dumps(
            {
                "page_content": "Revenue increased.",
                "metadata": {
                    "doc_id": "doc_001",
                    "source": "doc_001.pdf",
                    "page_number": 1,
                    "chunk_id": "doc_001:p0001:c000",
                    "chunk_index": 0,
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        json.dumps({"doc_id": "doc_001", "language": "EN"}) + "\n",
        encoding="utf-8",
    )
    return chunks_dir, manifest


def test_build_bm25_index_passes_builtin_function_and_schema(
    tmp_path: Path, monkeypatch
) -> None:
    chunks_dir, manifest = _write_fixture_data(tmp_path)
    milvus_calls: dict[str, object] = {}
    function_calls: dict[str, object] = {}

    class FakeBM25BuiltInFunction:
        def __init__(self, **kwargs: object) -> None:
            function_calls.update(kwargs)

    class FakeMilvus:
        @classmethod
        def from_documents(cls, **kwargs: object) -> "FakeMilvus":
            milvus_calls.update(kwargs)
            return cls()

    fake_module = types.ModuleType("langchain_milvus")
    fake_module.BM25BuiltInFunction = FakeBM25BuiltInFunction  # type: ignore[attr-defined]
    fake_module.Milvus = FakeMilvus  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "langchain_milvus", fake_module)
    monkeypatch.setattr("agenticrag.ingestion.index_bm25._collection_exists", lambda *_: False)

    report = build_bm25_index(
        chunks_dir,
        manifest_path=manifest,
        milvus_config=BM25MilvusConfig(collection_name="bm25_demo"),
        report_output=tmp_path / "report.json",
    )

    assert milvus_calls["embedding"] is None
    assert milvus_calls["ids"] == ["doc_001:p0001:c000"]
    assert milvus_calls["collection_name"] == "bm25_demo"
    assert milvus_calls["drop_old"] is False
    assert milvus_calls["vector_field"] == "sparse"
    assert milvus_calls["index_params"] == {
        "metric_type": "BM25",
        "index_type": "SPARSE_INVERTED_INDEX",
        "params": {"drop_ratio_build": 0.0},
    }
    assert milvus_calls["search_params"] == {
        "metric_type": "BM25",
        "params": {"drop_ratio_search": 0.0},
    }
    document = milvus_calls["documents"][0]  # type: ignore[index]
    assert document.metadata == {
        "doc_id": "doc_001",
        "source": "doc_001.pdf",
        "page_number": 1,
        "chunk_id": "doc_001:p0001:c000",
        "language": "en",
    }
    assert function_calls["multi_analyzer_params"]["by_field"] == "language"  # type: ignore[index]
    assert report["languages"] == {"en": 1}
    assert (tmp_path / "report.json").is_file()


def test_build_bm25_index_refuses_existing_collection(tmp_path: Path, monkeypatch) -> None:
    chunks_dir, manifest = _write_fixture_data(tmp_path)
    monkeypatch.setattr("agenticrag.ingestion.index_bm25._collection_exists", lambda *_: True)

    with pytest.raises(ValueError, match="BM25 collection 已存在"):
        build_bm25_index(
            chunks_dir,
            manifest_path=manifest,
            milvus_config=BM25MilvusConfig(collection_name="bm25_demo"),
            report_output=tmp_path / "report.json",
        )
