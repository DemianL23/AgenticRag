"""Command-line smoke test for the V1.2-A local reranking pipeline."""

from __future__ import annotations

import argparse

from agenticrag.rag.integrations.milvus import MilvusConfig
from agenticrag.rag.integrations.milvus_bm25 import BM25MilvusConfig
from agenticrag.reranking.bge import BGEReranker
from agenticrag.reranking.config import RerankerConfig
from agenticrag.retrieval.bm25_retriever import BM25Retriever
from agenticrag.retrieval.hybrid_retriever import HybridRetriever
from agenticrag.retrieval.milvus_retriever import MilvusRetriever
from agenticrag.retrieval.reranking_retriever import (
    DEFAULT_RERANK_FINAL_TOP_K,
    RerankingRetriever,
)


def main() -> None:
    args = _parse_args()
    dense_base = MilvusConfig.from_env()
    bm25_base = BM25MilvusConfig.from_env()
    dense_config = MilvusConfig(
        uri=args.uri or dense_base.uri,
        collection_name=args.dense_collection or dense_base.collection_name,
    )
    bm25_config = BM25MilvusConfig(
        uri=args.uri or bm25_base.uri,
        collection_name=args.bm25_collection or bm25_base.collection_name,
    )
    retriever = RerankingRetriever(
        hybrid_retriever=HybridRetriever(
            dense_retriever=MilvusRetriever(milvus_config=dense_config),
            bm25_retriever=BM25Retriever(milvus_config=bm25_config),
        ),
        reranker=BGEReranker(RerankerConfig.from_env()),
    )
    trace = retriever.search_with_trace(args.query, k=args.k)

    print(f"Top {len(trace.results)} V1.2-A reranked results:")
    print(f"candidate_pool_size: {len(trace.candidate_pool)}")
    print(f"fallback_used: {trace.fallback_used}")
    print(f"fallback_reason: {trace.fallback_reason}")
    print(f"model_load_seconds: {trace.model_load_seconds:.6f}")
    print(f"rerank_seconds: {trace.rerank_seconds:.6f}")
    for result in trace.results:
        print(f"\n[{result.final_rank}]")
        print(f"score: {result.score:.6f}")
        print(f"rerank_score: {result.rerank_score}")
        print(f"rrf_rank: {result.rrf_rank}")
        print(f"rrf_score: {result.rrf_score:.6f}")
        print(f"dense_rank: {result.dense_rank}")
        print(f"bm25_rank: {result.bm25_rank}")
        print(f"source: {result.source}")
        print(f"page: {result.page}")
        print(f"chunk_id: {result.chunk_id}")
        print("\ncontent:")
        print(result.content)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="使用 Dense + BM25 + RRF + 本地 BGE Reranker 执行 V1.2-A Retrieval。"
    )
    parser.add_argument("query", help="要检索的自然语言问题")
    parser.add_argument(
        "--k",
        type=int,
        default=DEFAULT_RERANK_FINAL_TOP_K,
        help="最终返回结果数，默认 5",
    )
    parser.add_argument("--uri", help="Milvus 地址，默认读取 MILVUS_URI")
    parser.add_argument(
        "--dense-collection",
        help="Dense collection，默认读取 MILVUS_COLLECTION",
    )
    parser.add_argument(
        "--bm25-collection",
        help="BM25 collection，默认读取 BM25_MILVUS_COLLECTION",
    )
    return parser.parse_args()
