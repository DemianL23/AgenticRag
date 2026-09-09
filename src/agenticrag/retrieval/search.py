"""Command-line smoke test for the V0 dense retriever."""

from __future__ import annotations

import argparse

from agenticrag.rag.integrations.milvus import MilvusConfig
from agenticrag.retrieval.milvus_retriever import MilvusRetriever


def main() -> None:
    args = _parse_args()
    milvus_config = MilvusConfig.from_env()
    if args.uri:
        milvus_config = MilvusConfig(
            uri=args.uri,
            collection_name=milvus_config.collection_name,
        )
    if args.collection_name:
        milvus_config = MilvusConfig(
            uri=milvus_config.uri,
            collection_name=args.collection_name,
        )

    results = MilvusRetriever(milvus_config=milvus_config).search(args.query, k=args.k)
    print(f"Top {len(results)} results:")
    for index, result in enumerate(results, start=1):
        print(f"\n[{index}]")
        print(f"score: {result.score:.6f}")
        print(f"source: {result.source}")
        print(f"page: {result.page}")
        print(f"chunk_id: {result.chunk_id}")
        print("\ncontent:")
        print(result.content)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="使用 Milvus 执行 V0 Dense Retrieval。")
    parser.add_argument("query", help="要检索的自然语言问题")
    parser.add_argument("--k", type=int, default=20, help="返回结果数，默认 20")
    parser.add_argument("--uri", help="Milvus 地址，默认读取 MILVUS_URI")
    parser.add_argument(
        "--collection-name",
        help="Collection 名称，默认读取 MILVUS_COLLECTION",
    )
    return parser.parse_args()
