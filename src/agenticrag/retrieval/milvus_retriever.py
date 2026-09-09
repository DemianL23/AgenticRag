"""Dense retrieval against an existing LangChain-managed Milvus collection."""

from __future__ import annotations

from typing import Any

from agenticrag.rag.integrations.embeddings import EmbeddingConfig, create_embeddings
from agenticrag.rag.integrations.milvus import MilvusConfig
from agenticrag.retrieval.base import BaseRetriever
from agenticrag.retrieval.schemas import RetrievedChunk


class MilvusRetriever(BaseRetriever):
    """Use the configured embedding model and Milvus collection for V0 search.

    The collection must already have been built by ``agenticrag-index-milvus``.
    This class performs no indexing, reranking, filtering, query rewriting, or
    LLM calls.
    """

    def __init__(
        self,
        *,
        milvus_config: MilvusConfig | None = None,
        embedding_config: EmbeddingConfig | None = None,
    ) -> None:
        self.milvus_config = milvus_config or MilvusConfig.from_env()
        self.milvus_config.validate()
        if not self.milvus_config.collection_name:
            raise ValueError(
                "检索需要 MILVUS_COLLECTION，请将索引报告中的 collection_name 写入 .env"
            )

        self.embedding_config = embedding_config or EmbeddingConfig.from_env()
        self.embedding_config.validate()
        self.embeddings = create_embeddings(self.embedding_config)
        self.vector_store = self._create_vector_store()

    def search(self, query: str, k: int = 20) -> list[RetrievedChunk]:
        """Return the top ``k`` chunks and their original citation metadata."""
        clean_query = query.strip()
        if not clean_query:
            raise ValueError("query 不能为空")
        if k <= 0:
            raise ValueError("k 必须大于 0")

        documents_and_scores = self.vector_store.similarity_search_with_score(
            clean_query,
            k=k,
        )
        return [
            _to_retrieved_chunk(document, score)
            for document, score in documents_and_scores
        ]

    def _create_vector_store(self) -> Any:
        try:
            from langchain_milvus import Milvus
        except ImportError as exc:
            raise RuntimeError(
                "Milvus 检索需要可选依赖，请执行：uv sync --extra milvus"
            ) from exc

        return Milvus(
            embedding_function=self.embeddings,
            collection_name=self.milvus_config.collection_name,
            connection_args={"uri": self.milvus_config.uri},
        )


def _to_retrieved_chunk(document: Any, score: float) -> RetrievedChunk:
    metadata = getattr(document, "metadata", None)
    if not isinstance(metadata, dict):
        raise ValueError("Milvus 返回结果缺少 metadata")

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
        raise ValueError(f"Milvus 返回结果缺少合法 {field_name}")
    return value


def _positive_int(value: object, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"Milvus 返回结果缺少合法 {field_name}")
    return value
