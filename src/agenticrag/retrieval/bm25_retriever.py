"""Milvus built-in BM25 retrieval for the V1.1 baseline."""

from __future__ import annotations

import re
from typing import Any

from agenticrag.rag.integrations.milvus_bm25 import (
    BM25_MULTI_ANALYZER_PARAMS,
    BM25_SEARCH_PARAMS,
    BM25MilvusConfig,
)
from agenticrag.retrieval.base import BaseRetriever
from agenticrag.retrieval.schemas import RetrievedChunk

DEFAULT_BM25_TOP_K = 20
_HAN_PATTERN = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
_LATIN_PATTERN = re.compile(r"[A-Za-z]")


class BM25Retriever(BaseRetriever):
    """Search an existing Milvus built-in BM25 collection.

    The collection must already have been built by
    ``agenticrag-index-bm25``. This retriever does not perform dense
    embedding, query rewriting, reranking, or fallback retrieval.
    """

    def __init__(self, *, milvus_config: BM25MilvusConfig | None = None) -> None:
        self.milvus_config = milvus_config or BM25MilvusConfig.from_env()
        self.milvus_config.validate()
        self.vector_store = self._create_vector_store()

    def search(self, query: str, k: int = DEFAULT_BM25_TOP_K) -> list[RetrievedChunk]:
        """Return up to ``k`` chunks ranked by the raw BM25 score."""
        clean_query = query.strip()
        if not clean_query:
            raise ValueError("query 不能为空")
        if isinstance(k, bool) or not isinstance(k, int) or k <= 0:
            raise ValueError("k 必须是正整数")

        analyzer_name = select_query_analyzer(clean_query)
        search_params = {
            **BM25_SEARCH_PARAMS,
            "analyzer_name": analyzer_name,
        }
        documents_and_scores = self.vector_store.similarity_search_with_score(
            clean_query,
            k=k,
            param=search_params,
        )
        return [
            _to_retrieved_chunk(document, score)
            for document, score in documents_and_scores
        ]

    def _create_vector_store(self) -> Any:
        try:
            from langchain_milvus import BM25BuiltInFunction, Milvus
        except ImportError as exc:
            raise RuntimeError(
                "Milvus BM25 检索需要可选依赖，请执行：uv sync --extra milvus"
            ) from exc

        bm25_function = BM25BuiltInFunction(
            multi_analyzer_params=BM25_MULTI_ANALYZER_PARAMS,
            function_name="v1_1_bm25_function",
        )
        return Milvus(
            embedding_function=None,
            builtin_function=bm25_function,
            collection_name=self.milvus_config.collection_name,
            connection_args={"uri": self.milvus_config.uri},
            vector_field="sparse",
            search_params=BM25_SEARCH_PARAMS,
        )


def select_query_analyzer(query: str) -> str:
    """Choose a Milvus analyzer from visible query language signals."""
    if _HAN_PATTERN.search(query):
        return "chinese"
    if _LATIN_PATTERN.search(query):
        return "english"
    return "default"


def _to_retrieved_chunk(document: Any, score: float) -> RetrievedChunk:
    metadata = getattr(document, "metadata", None)
    if not isinstance(metadata, dict):
        raise ValueError("Milvus BM25 返回结果缺少 metadata")

    return RetrievedChunk(
        content=_required_string(getattr(document, "page_content", None), "content"),
        score=float(score),
        doc_id=_required_string(metadata.get("doc_id"), "doc_id"),
        source=_required_string(metadata.get("source"), "source"),
        page=_positive_int(metadata.get("page_number"), "page_number"),
        chunk_id=_required_string(metadata.get("chunk_id"), "chunk_id"),
    )


def _required_string(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Milvus BM25 返回结果缺少合法 {field_name}")
    return value


def _positive_int(value: object, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"Milvus BM25 返回结果缺少合法 {field_name}")
    return value
