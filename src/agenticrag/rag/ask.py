"""Command-line smoke test for the complete V0 RAG answer path."""

from __future__ import annotations

import argparse
import json

from agenticrag.generation.generator import QwenAnswerGenerator
from agenticrag.rag.integrations.milvus import MilvusConfig
from agenticrag.rag.service import RagAnswerService
from agenticrag.retrieval.milvus_retriever import MilvusRetriever


def main() -> None:
    args = _parse_args()
    base_config = MilvusConfig.from_env()
    milvus_config = MilvusConfig(
        uri=args.uri or base_config.uri,
        collection_name=args.collection_name or base_config.collection_name,
    )
    service = RagAnswerService(
        retriever=MilvusRetriever(milvus_config=milvus_config),
        generator=QwenAnswerGenerator(),
    )
    result = service.answer(args.query, k=args.k)
    print(json.dumps(result.to_record(), ensure_ascii=False, indent=2))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="运行 V0 检索加 Qwen 生成流程。")
    parser.add_argument("query", help="要回答的自然语言问题")
    parser.add_argument("--k", type=int, default=5, help="检索 Top-K，默认 5")
    parser.add_argument("--uri", help="Milvus 地址，默认读取 MILVUS_URI")
    parser.add_argument(
        "--collection-name",
        help="Collection 名称，默认读取 MILVUS_COLLECTION",
    )
    return parser.parse_args()
