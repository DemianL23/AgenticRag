import sys
import types

import pytest
from langchain_core.documents import Document

from agenticrag.rag.integrations.milvus_bm25 import BM25MilvusConfig
from agenticrag.retrieval.bm25_retriever import BM25Retriever, select_query_analyzer
from agenticrag.retrieval.schemas import RetrievedChunk


def test_select_query_analyzer_routes_chinese_english_and_icu() -> None:
    assert select_query_analyzer("营业收入是多少？") == "chinese"
    assert select_query_analyzer("What is revenue?") == "english"
    assert select_query_analyzer("2024 / 100") == "default"
    assert select_query_analyzer("Revenue 营业收入") == "chinese"


def test_bm25_retriever_uses_analyzer_and_maps_score(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeBM25BuiltInFunction:
        def __init__(self, **kwargs: object) -> None:
            self.kwargs = kwargs

    class FakeVectorStore:
        init_kwargs: dict[str, object] = {}
        search_kwargs: dict[str, object] = {}

        def __init__(self, **kwargs: object) -> None:
            self.init_kwargs = kwargs

        def similarity_search_with_score(
            self, query: str, *, k: int, param: dict[str, object]
        ) -> list[tuple[Document, float]]:
            assert query == "营业收入是多少？"
            assert k == 2
            self.search_kwargs = param
            return [
                (
                    Document(
                        page_content="营业收入为 438.00 百万元。",
                        metadata={
                            "doc_id": "doc_000",
                            "source": "report.pdf",
                            "page_number": 3,
                            "chunk_id": "doc_000:p0003:c000",
                        },
                    ),
                    12.5,
                )
            ]

    fake_module = types.ModuleType("langchain_milvus")
    fake_module.BM25BuiltInFunction = FakeBM25BuiltInFunction  # type: ignore[attr-defined]
    fake_module.Milvus = FakeVectorStore  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "langchain_milvus", fake_module)

    retriever = BM25Retriever(
        milvus_config=BM25MilvusConfig(collection_name="bm25_demo")
    )
    results = retriever.search("营业收入是多少？", k=2)

    assert results == [
        RetrievedChunk(
            content="营业收入为 438.00 百万元。",
            score=12.5,
            doc_id="doc_000",
            source="report.pdf",
            page=3,
            chunk_id="doc_000:p0003:c000",
        )
    ]
    assert retriever.vector_store.init_kwargs == {
        "embedding_function": None,
        "builtin_function": retriever.vector_store.init_kwargs["builtin_function"],
        "collection_name": "bm25_demo",
        "connection_args": {"uri": "http://localhost:19530"},
        "vector_field": "sparse",
        "search_params": {
            "metric_type": "BM25",
            "params": {"drop_ratio_search": 0.0},
        },
    }
    assert retriever.vector_store.search_kwargs == {
        "metric_type": "BM25",
        "params": {"drop_ratio_search": 0.0},
        "analyzer_name": "chinese",
    }


def test_bm25_retriever_returns_empty_without_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeVectorStore:
        def __init__(self, **_: object) -> None:
            pass

        def similarity_search_with_score(
            self, query: str, *, k: int, param: dict[str, object]
        ) -> list[tuple[Document, float]]:
            return []

    class FakeBM25BuiltInFunction:
        def __init__(self, **_: object) -> None:
            pass

    fake_module = types.ModuleType("langchain_milvus")
    fake_module.BM25BuiltInFunction = FakeBM25BuiltInFunction  # type: ignore[attr-defined]
    fake_module.Milvus = FakeVectorStore  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "langchain_milvus", fake_module)

    retriever = BM25Retriever(
        milvus_config=BM25MilvusConfig(collection_name="bm25_demo")
    )

    assert retriever.search("没有命中") == []


def test_bm25_retriever_validates_query_and_k(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeVectorStore:
        def __init__(self, **_: object) -> None:
            pass

        def similarity_search_with_score(
            self, query: str, *, k: int, param: dict[str, object]
        ) -> list[tuple[Document, float]]:
            return []

    class FakeBM25BuiltInFunction:
        def __init__(self, **_: object) -> None:
            pass

    fake_module = types.ModuleType("langchain_milvus")
    fake_module.BM25BuiltInFunction = FakeBM25BuiltInFunction  # type: ignore[attr-defined]
    fake_module.Milvus = FakeVectorStore  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "langchain_milvus", fake_module)

    retriever = BM25Retriever(
        milvus_config=BM25MilvusConfig(collection_name="bm25_demo")
    )

    with pytest.raises(ValueError, match="query"):
        retriever.search("   ")
    with pytest.raises(ValueError, match="k"):
        retriever.search("问题", k=0)
    with pytest.raises(ValueError, match="k"):
        retriever.search("问题", k=True)
