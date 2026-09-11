"""Dense retrieval against an existing LangChain-managed Milvus collection."""

from __future__ import annotations

import time
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
        self.embeddings = _TimingEmbeddings(create_embeddings(self.embedding_config))
        self.last_query_embedding_seconds = 0.0
        self.last_embedding_request_seconds = 0.0
        self.vector_store = self._create_vector_store()

    def search(self, query: str, k: int = 20) -> list[RetrievedChunk]:
        """Return the top ``k`` chunks and their original citation metadata."""
        clean_query = query.strip()
        if not clean_query:
            raise ValueError("query 不能为空")
        if k <= 0:
            raise ValueError("k 必须大于 0")

        self.embeddings.reset_query_timing()
        try:
            documents_and_scores = self.vector_store.similarity_search_with_score(
                clean_query,
                k=k,
            )
        finally:
            self.last_query_embedding_seconds = (
                self.embeddings.last_query_embedding_seconds
            )
            self.last_embedding_request_seconds = (
                self.embeddings.last_request_seconds
            )
        return [
            _to_retrieved_chunk(document, score)
            for document, score in documents_and_scores
        ]

    def embedding_record(self) -> dict[str, Any]:
        """Return safe embedding backend and compatibility metadata."""
        record = self.embeddings.model_record()
        record.setdefault("backend_type", self.embedding_config.backend)
        record.setdefault("model", self.embedding_config.model_name)
        record["dimension"] = (
            1024 if "qwen3-embedding" in self.embedding_config.model_name.lower() else None
        )
        record["normalize_embeddings"] = self.embedding_config.normalize_embeddings
        record["query_prompt_name"] = self.embedding_config.effective_query_prompt_name()
        record["query_prompt_profile"] = (
            "qwen3_web_search_instruction_v1"
            if record["query_prompt_name"] == "query"
            else None
        )
        record["config"] = self.embedding_config.to_record()
        return record

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


class _TimingEmbeddings:
    """Transparent embedding proxy that records query encode wall time."""

    def __init__(self, embeddings: Any) -> None:
        self._embeddings = embeddings
        self.last_query_embedding_seconds = 0.0

    def reset_query_timing(self) -> None:
        self.last_query_embedding_seconds = 0.0
        self.last_request_seconds = 0.0

    def embed_query(self, text: str) -> list[float]:
        started = time.perf_counter()
        try:
            return self._embeddings.embed_query(text)
        finally:
            self.last_query_embedding_seconds += time.perf_counter() - started
            self.last_request_seconds = float(
                getattr(self._embeddings, "last_request_seconds", 0.0)
            )

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embeddings.embed_documents(texts)

    def model_record(self) -> dict[str, Any]:
        provider = "remote" if hasattr(self._embeddings, "remote_config") else "local"
        record: dict[str, Any] = {
            "backend": provider,
            "backend_type": provider,
            "provider": "vllm" if provider == "remote" else "langchain_huggingface",
        }
        if provider == "remote":
            backend_record = self._embeddings.model_record()
            record.update(backend_record)
        else:
            record["request_seconds"] = 0.0
            record["query_prompt_name"] = None
        return record
