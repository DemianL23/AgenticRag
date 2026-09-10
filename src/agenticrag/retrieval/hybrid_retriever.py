"""Hybrid Dense + BM25 retrieval with application-side RRF fusion."""

from __future__ import annotations

from agenticrag.retrieval.base import BaseRetriever
from agenticrag.retrieval.bm25_retriever import BM25Retriever
from agenticrag.retrieval.milvus_retriever import MilvusRetriever
from agenticrag.retrieval.rrf import DEFAULT_RRF_K, fuse_ranked_results
from agenticrag.retrieval.schemas import HybridRetrievedChunk

DEFAULT_HYBRID_TOP_K = 20
DEFAULT_HYBRID_ROUTE_TOP_K = 20


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
        candidates = self.candidate_pool(query, route_k=k)
        return candidates[:k]

    def candidate_pool(
        self,
        query: str,
        *,
        route_k: int = DEFAULT_HYBRID_ROUTE_TOP_K,
    ) -> list[HybridRetrievedChunk]:
        """Return the complete deduplicated Dense + BM25 candidate pool."""
        clean_query = query.strip()
        if not clean_query:
            raise ValueError("query 不能为空")
        if isinstance(route_k, bool) or not isinstance(route_k, int) or route_k <= 0:
            raise ValueError("route_k 必须是正整数")

        dense_results = self.dense_retriever.search(clean_query, k=route_k)
        bm25_results = self.bm25_retriever.search(clean_query, k=route_k)
        return fuse_ranked_results(
            dense_results,
            bm25_results,
            limit=route_k * 2,
            rrf_k=self.rrf_k,
        )
