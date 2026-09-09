"""Command-line smoke test for the V1.1 BM25 retriever."""

from __future__ import annotations

import argparse

from agenticrag.rag.integrations.milvus_bm25 import BM25MilvusConfig
from agenticrag.retrieval.bm25_retriever import BM25Retriever


def main() -> None:
    args = _parse_args()
    base_config = BM25MilvusConfig.from_env()
    config = BM25MilvusConfig(
        uri=args.uri or base_config.uri,
        collection_name=args.collection_name or base_config.collection_name,
    )
    results = BM25Retriever(milvus_config=config).search(args.query, k=args.k)
    print(f"Top {len(results)} BM25 results:")
    for index, result in enumerate(results, start=1):
        print(f"\n[{index}]")
        print(f"score: {result.score:.6f}")
        print(f"source: {result.source}")
        print(f"page: {result.page}")
        print(f"chunk_id: {result.chunk_id}")
        print("\ncontent:")
        print(result.content)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="使用 Milvus 执行 V1.1 BM25 Retrieval。")
    parser.add_argument("query", help="要检索的自然语言问题")
    parser.add_argument("--k", type=int, default=20, help="返回结果数，默认 20")
    parser.add_argument("--uri", help="Milvus 地址，默认读取 MILVUS_URI")
    parser.add_argument(
        "--collection-name",
        help="BM25 collection，默认读取 BM25_MILVUS_COLLECTION",
    )
    return parser.parse_args()

