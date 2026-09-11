"""Run the V1.2 retrieval-to-generation control evaluation."""

from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
import uuid
from dataclasses import asdict
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from agenticrag.generation.config import GenerationConfig
from agenticrag.generation.generator import QwenAnswerGenerator
from agenticrag.rag.integrations.embeddings import EmbeddingConfig
from agenticrag.rag.integrations.milvus import MilvusConfig
from agenticrag.rag.integrations.milvus_bm25 import BM25MilvusConfig
from agenticrag.rag.v1_2_control import V12ControlAnswerService
from agenticrag.reranking.bge import BGEReranker
from agenticrag.reranking.config import RerankerConfig
from agenticrag.retrieval.bm25_retriever import BM25Retriever
from agenticrag.retrieval.hybrid_retriever import HybridRetriever
from agenticrag.retrieval.milvus_retriever import MilvusRetriever
from agenticrag.retrieval.reranking_retriever import (
    DEFAULT_RERANK_FINAL_TOP_K,
    RerankingRetriever,
)
from eval.ragas.config import RagasEvaluatorConfig
from eval.ragas.providers import create_ragas_evaluator
from eval.ragas.report import write_report
from eval.ragas.runner import evaluate_end_to_end


DEFAULT_OUTPUT_ROOT = Path("artifacts/eval/v1_2_end_to_end_control")


def main() -> None:
    args = _parse_args()
    asyncio.run(_run_cli(args))


async def _run_cli(args: argparse.Namespace) -> None:
    _validate_positive("top_k", args.top_k)
    if args.limit is not None:
        _validate_positive("limit", args.limit)

    run_id = str(uuid.uuid4())
    output = args.output or DEFAULT_OUTPUT_ROOT / run_id / "report.json"
    if output.exists():
        raise FileExistsError(
            f"拒绝覆盖已有 control report：{output}；请指定新的 --output 路径"
        )

    generation_config = GenerationConfig.from_env()
    embedding_config = EmbeddingConfig.from_env()
    evaluator_config = RagasEvaluatorConfig.from_env()
    reranker_config = RerankerConfig.from_env()

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
            dense_retriever=MilvusRetriever(
                embedding_config=embedding_config,
                milvus_config=dense_config,
            ),
            bm25_retriever=BM25Retriever(milvus_config=bm25_config),
        ),
        reranker=BGEReranker(reranker_config),
    )
    service = V12ControlAnswerService(
        retriever=retriever,
        generator=QwenAnswerGenerator(config=generation_config),
    )
    evaluator = create_ragas_evaluator(evaluator_config)
    report = await evaluate_end_to_end(
        args.dataset,
        service,
        evaluator,
        top_k=args.top_k,
        limit=args.limit,
        generation_model=generation_config.model,
        embedding_model=embedding_config.model_name,
        evaluator_model=evaluator_config.model,
        evaluator_embedding_model=evaluator_config.embedding_model,
        ragas_version=_ragas_version(),
        report_schema_version=2,
        report_name="v1_2_end_to_end_control",
        run_id=run_id,
        git_commit=_git_commit(),
        resolved_config={
            "generation": generation_config.to_record(),
            "embedding": embedding_config.to_record(),
            "evaluator": evaluator_config.to_record(),
            "reranker": reranker_config.to_record(),
            "retrieval_pipeline": {
                "dense": asdict(dense_config),
                "bm25": asdict(bm25_config),
                "route_top_k": retriever.route_k,
                "rrf_candidate_report_k": retriever.rrf_report_k,
                "rrf_k": retriever.hybrid_retriever.rrf_k,
                "final_top_k": args.top_k,
            },
        },
    )
    write_report(report, output)
    summary = {
        "report_name": report.report_name,
        "run_id": report.run_id,
        "dataset_size": report.dataset_size,
        "top_k": report.top_k,
        "aggregate_metrics": report.aggregate_metrics,
        "aggregate_metric_counts": report.aggregate_metric_counts,
        "successful_samples": report.successful_samples,
        "failed_samples": report.failed_samples,
        "retrieval_summary": report.retrieval_summary,
        "report": str(output),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def _git_commit() -> str | None:
    project_root = Path(__file__).resolve().parents[2]
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    commit = result.stdout.strip()
    return commit or None


def _ragas_version() -> str:
    try:
        return version("ragas")
    except PackageNotFoundError as exc:
        raise RuntimeError(
            "RAGAS 未安装，请执行：uv sync --extra evaluation"
        ) from exc


def _validate_positive(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} 必须是正整数")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "运行 V1.2 End-to-End Control："
            "Dense + BM25 + RRF + BGE Reranker Top-K + 现有 Answer Generator。"
        )
    )
    parser.add_argument(
        "--dataset", type=Path, default=Path("qa.jsonl"), help="QA JSONL 数据集"
    )
    parser.add_argument(
        "--top-k", "--k", dest="top_k", type=int,
        default=DEFAULT_RERANK_FINAL_TOP_K, help="送入 Generator 的最终 Top-K，默认 5"
    )
    parser.add_argument("--limit", type=int, help="只评测前 N 条，用于 smoke test")
    parser.add_argument(
        "--output", type=Path,
        help="control report 输出路径；默认写入带 UUID 的新目录且不覆盖历史报告",
    )
    parser.add_argument("--uri", help="Milvus 地址，默认读取 MILVUS_URI")
    parser.add_argument(
        "--dense-collection", help="Dense collection，默认读取 MILVUS_COLLECTION"
    )
    parser.add_argument(
        "--bm25-collection", help="BM25 collection，默认读取 BM25_MILVUS_COLLECTION"
    )
    return parser.parse_args()
