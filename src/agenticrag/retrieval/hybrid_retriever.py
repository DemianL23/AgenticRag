"""Hybrid Dense + BM25 retrieval with application-side RRF fusion."""

from __future__ import annotations

import time

from agenticrag.retrieval.base import BaseRetriever
from agenticrag.retrieval.bm25_retriever import BM25Retriever
from agenticrag.retrieval.milvus_retriever import MilvusRetriever
from agenticrag.retrieval.rrf import DEFAULT_RRF_K, fuse_ranked_results
from agenticrag.retrieval.schemas import CandidateGenerationTiming, HybridRetrievedChunk

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
        candidates, _ = self.candidate_pool_with_timing(query, route_k=route_k)
        return candidates

    def candidate_pool_with_timing(
        self,
        query: str,
        *,
        route_k: int = DEFAULT_HYBRID_ROUTE_TOP_K,
    ) -> tuple[list[HybridRetrievedChunk], CandidateGenerationTiming]:
        """Return the candidate pool and non-invasive stage timing diagnostics."""
        clean_query = query.strip()
        if not clean_query:
            raise ValueError("query 不能为空")
        if isinstance(route_k, bool) or not isinstance(route_k, int) or route_k <= 0:
            raise ValueError("route_k 必须是正整数")

        candidate_started = time.perf_counter()
        dense_started = time.perf_counter()
        dense_results = self.dense_retriever.search(clean_query, k=route_k)
        dense_elapsed = time.perf_counter() - dense_started
        query_embedding_seconds = _query_embedding_seconds(self.dense_retriever)
        dense_search_seconds = max(dense_elapsed - query_embedding_seconds, 0.0)

        bm25_started = time.perf_counter()
        bm25_results = self.bm25_retriever.search(clean_query, k=route_k)
        bm25_search_seconds = time.perf_counter() - bm25_started

        merge_started = time.perf_counter()
        candidates = fuse_ranked_results(
            dense_results,
            bm25_results,
            limit=route_k * 2,
            rrf_k=self.rrf_k,
        )
        merge_rrf_seconds = time.perf_counter() - merge_started
        timing = CandidateGenerationTiming(
            query_embedding_seconds=query_embedding_seconds,
            dense_search_seconds=dense_search_seconds,
            bm25_search_seconds=bm25_search_seconds,
            merge_rrf_seconds=merge_rrf_seconds,
            candidate_total_seconds=time.perf_counter() - candidate_started,
        )
        return candidates, timing


def _query_embedding_seconds(retriever: BaseRetriever) -> float:
    value = getattr(retriever, "last_query_embedding_seconds", 0.0)
    try:
        return max(float(value), 0.0)
    except (TypeError, ValueError):
        return 0.0
