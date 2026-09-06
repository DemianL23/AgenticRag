import sys
import types

import pytest
from langchain_core.documents import Document

from agenticrag.integrations.embeddings import EmbeddingConfig
from agenticrag.integrations.milvus import MilvusConfig
from agenticrag.retrieval.milvus_retriever import MilvusRetriever
from agenticrag.retrieval.schemas import RetrievedChunk


def test_retrieved_chunk_has_stable_record() -> None:
    chunk = RetrievedChunk(
        content="营业收入增长。",
        score=0.82,
        doc_id="doc_000",
        source="report.pdf",
        page=3,
        chunk_id="doc_000:p0003:c000",
    )

    assert chunk.to_record() == {
        "content": "营业收入增长。",
        "score": 0.82,
        "doc_id": "doc_000",
        "source": "report.pdf",
        "page": 3,
        "chunk_id": "doc_000:p0003:c000",
    }


def test_milvus_retriever_embeds_and_maps_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeEmbeddings:
        pass

    class FakeVectorStore:
        init_kwargs: dict[str, object] = {}

        def __init__(self, **kwargs: object) -> None:
            self.init_kwargs = kwargs

        def similarity_search_with_score(
            self, query: str, *, k: int
        ) -> list[tuple[Document, float]]:
            assert query == "主营业务是什么？"
            assert k == 2
            return [
                (
                    Document(
                        page_content="主营业务为工程承包。",
                        metadata={
                            "doc_id": "doc_000",
                            "source": "corpus/report.pdf",
                            "page_number": 3,
                            "chunk_id": "doc_000:p0003:c000",
                        },
                    ),
                    0.82,
                )
            ]

    fake_module = types.ModuleType("langchain_milvus")
    fake_module.Milvus = FakeVectorStore  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "langchain_milvus", fake_module)
    monkeypatch.setattr(
        "agenticrag.retrieval.milvus_retriever.create_embeddings",
        lambda _: FakeEmbeddings(),
    )

    retriever = MilvusRetriever(
        milvus_config=MilvusConfig(
            uri="http://localhost:19530",
            collection_name="demo",
        ),
        embedding_config=EmbeddingConfig(),
    )

    results = retriever.search("主营业务是什么？", k=2)

    assert results == [
        RetrievedChunk(
            content="主营业务为工程承包。",
            score=0.82,
            doc_id="doc_000",
            source="corpus/report.pdf",
            page=3,
            chunk_id="doc_000:p0003:c000",
        )
    ]
    assert retriever.vector_store.init_kwargs == {
        "embedding_function": retriever.embeddings,
        "collection_name": "demo",
        "connection_args": {"uri": "http://localhost:19530"},
    }


def test_milvus_retriever_requires_collection_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "agenticrag.retrieval.milvus_retriever.create_embeddings",
        lambda _: pytest.fail("模型不应在配置校验前加载"),
    )

    with pytest.raises(ValueError, match="MILVUS_COLLECTION"):
        MilvusRetriever(
            milvus_config=MilvusConfig(collection_name=None),
            embedding_config=EmbeddingConfig(),
        )


def test_milvus_retriever_validates_query_and_k(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeVectorStore:
        def __init__(self, **_: object) -> None:
            pass

        def similarity_search_with_score(
            self, query: str, *, k: int
        ) -> list[tuple[Document, float]]:
            return []

    fake_module = types.ModuleType("langchain_milvus")
    fake_module.Milvus = FakeVectorStore  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "langchain_milvus", fake_module)
    monkeypatch.setattr(
        "agenticrag.retrieval.milvus_retriever.create_embeddings",
        lambda _: object(),
    )
    retriever = MilvusRetriever(
        milvus_config=MilvusConfig(collection_name="demo"),
        embedding_config=EmbeddingConfig(),
    )

    with pytest.raises(ValueError, match="query"):
        retriever.search("   ")
    with pytest.raises(ValueError, match="k"):
        retriever.search("问题", k=0)
