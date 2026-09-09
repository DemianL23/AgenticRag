"""Hybrid Dense + BM25 retrieval with application-side RRF fusion."""

from __future__ import annotations

from agenticrag.retrieval.base import BaseRetriever
from agenticrag.retrieval.bm25_retriever import BM25Retriever
from agenticrag.retrieval.milvus_retriever import MilvusRetriever
from agenticrag.retrieval.rrf import DEFAULT_RRF_K, fuse_ranked_results
from agenticrag.retrieval.schemas import HybridRetrievedChunk

DEFAULT_HYBRID_TOP_K = 20


class HybridRetriever(BaseRetriever):
    """Fuse Dense and BM25 rankings while preserving both route ranks."""

    def __init__(
        self,
        *,
        dense_retriever: BaseRetriever | None = None,
        bm25_retriever: BaseRetriever | None = None,
        rrf_k: int = DEFAULT_RRF_K,
    ) -> None:
        self.dense_retriever = dense_retriever or MilvusRetriever()
        self.bm25_retriever = bm25_retriever or BM25Retriever()
        self.rrf_k = rrf_k

    def search(
        self,
        query: str,
        k: int = DEFAULT_HYBRID_TOP_K,
    ) -> list[HybridRetrievedChunk]:
        """Search both routes with the same query and fuse their rankings."""
        clean_query = query.strip()
        if not clean_query:
            raise ValueError("query 不能为空")
        if isinstance(k, bool) or not isinstance(k, int) or k <= 0:
            raise ValueError("k 必须是正整数")

        dense_results = self.dense_retriever.search(clean_query, k=k)
        bm25_results = self.bm25_retriever.search(clean_query, k=k)
        return fuse_ranked_results(
            dense_results,
            bm25_results,
            limit=k,
            rrf_k=self.rrf_k,
        )

