import json
import sys
import types
from pathlib import Path

from langchain_core.documents import Document

from agenticrag.ingestion.index_milvus import build_milvus_index, default_collection_name
from agenticrag.rag.integrations.embeddings import EmbeddingConfig
from agenticrag.rag.integrations.milvus import MilvusConfig


def test_default_collection_name_contains_model_and_chunk_settings(tmp_path: Path) -> None:
    chunks_dir = tmp_path / "chunks"
    chunks_dir.mkdir()
    (chunks_dir / "report.json").write_text(
        json.dumps({"config": {"chunk_size": 800, "chunk_overlap": 120}}),
        encoding="utf-8",
    )

    name = default_collection_name(EmbeddingConfig(), chunks_dir)

    assert name == "rag_v0_qwen_qwen3_embedding_0_6b_800_120"


def test_milvus_config_reads_environment(monkeypatch):
    monkeypatch.setenv("MILVUS_URI", "http://example:19530")
    monkeypatch.setenv("MILVUS_COLLECTION", "demo_collection")
    monkeypatch.setenv("MILVUS_DROP_OLD", "true")

    config = MilvusConfig.from_env()

    assert config.uri == "http://example:19530"
    assert config.collection_name == "demo_collection"
    assert config.drop_old is True


def test_build_milvus_index_passes_stable_ids_and_schema_settings(
    tmp_path: Path, monkeypatch
) -> None:
    chunks_dir = tmp_path / "chunks"
    chunks_dir.mkdir()
    records = [
        {
            "page_content": "Revenue increased.",
            "metadata": {
                "doc_id": "doc_001",
                "chunk_id": "doc_001:p0001:c000",
                "page_number": 1,
            },
        },
        {
            "page_content": "Profit increased.",
            "metadata": {
                "doc_id": "doc_001",
                "chunk_id": "doc_001:p0001:c001",
                "page_number": 1,
            },
        },
    ]
    (chunks_dir / "doc_001.jsonl").write_text(
        "\n".join(json.dumps(record) for record in records) + "\n",
        encoding="utf-8",
    )

    class FakeEmbeddings:
        def embed_query(self, text: str) -> list[float]:
            return [0.1, 0.2, 0.3]

    milvus_calls: dict[str, object] = {}

    class FakeMilvus:
        @classmethod
        def from_documents(cls, **kwargs: object) -> "FakeMilvus":
            milvus_calls.update(kwargs)
            return cls()

    monkeypatch.setattr("agenticrag.ingestion.index_milvus.create_embeddings", lambda _: FakeEmbeddings())
    monkeypatch.setattr("agenticrag.ingestion.index_milvus.embedding_dimension", lambda _: 3)
    fake_module = types.ModuleType("langchain_milvus")
    fake_module.Milvus = FakeMilvus  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "langchain_milvus", fake_module)

    report = build_milvus_index(
        chunks_dir,
        milvus_config=MilvusConfig(uri="http://localhost:19530", collection_name="demo"),
        embedding_config=EmbeddingConfig(),
    )

    assert milvus_calls["ids"] == ["doc_001:p0001:c000", "doc_001:p0001:c001"]
    assert milvus_calls["collection_name"] == "demo"
    assert milvus_calls["connection_args"] == {"uri": "http://localhost:19530"}
    assert milvus_calls["drop_old"] is False
    assert report["embedding"]["dimension"] == 3
    assert report["chunks"] == 2
    assert (chunks_dir / "milvus_report.json").is_file()
