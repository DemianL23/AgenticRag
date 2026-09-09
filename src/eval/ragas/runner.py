"""Run the real RAG pipeline, score every sample, and write a JSON report."""

from __future__ import annotations

import argparse
import asyncio
import json
from collections.abc import Sequence
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Protocol

from agenticrag.generation.config import GenerationConfig
from agenticrag.generation.generator import QwenAnswerGenerator
from agenticrag.integrations.embeddings import EmbeddingConfig
from agenticrag.integrations.milvus import MilvusConfig
from agenticrag.rag.service import RagAnswerService, RagAnswerTrace
from agenticrag.retrieval.milvus_retriever import MilvusRetriever
from agenticrag.retrieval.schemas import RetrievedChunk

from .config import RagasEvaluatorConfig
from .dataset import load_qa_dataset
from .evaluator import RagasEvaluator, RagasMetricResult
from .providers import create_ragas_evaluator
from .report import RagasReport, SampleReport, aggregate_scores, write_report


class AnswerService(Protocol):
    def answer_with_trace(self, query: str, *, k: int = 5) -> RagAnswerTrace:
        ...


class SampleEvaluator(Protocol):
    @property
    def metric_names(self) -> tuple[str, ...]:
        ...

    async def evaluate(
        self,
        *,
        user_input: str,
        reference: str,
        response: str,
        retrieved_contexts: list[str],
    ) -> RagasMetricResult:
        ...


async def evaluate_end_to_end(
    dataset_path: Path,
    service: AnswerService,
    evaluator: SampleEvaluator,
    *,
    top_k: int,
    limit: int | None,
    generation_model: str,
    embedding_model: str,
    evaluator_model: str,
    evaluator_embedding_model: str,
    ragas_version: str,
) -> RagasReport:
    """Materialize and evaluate each RAG sample while isolating failures."""
    _validate_positive("top_k", top_k)
    if limit is not None:
        _validate_positive("limit", limit)
    qa_samples = load_qa_dataset(Path(dataset_path), limit=limit)
    sample_reports: list[SampleReport] = []

    for index, sample in enumerate(qa_samples, start=1):
        print(f"[{index}/{len(qa_samples)}] {sample.sample_id}: RAG + RAGAS")
        blank_scores = {name: None for name in evaluator.metric_names}
        try:
            trace = service.answer_with_trace(sample.question, k=top_k)
            contexts = retrieved_contexts_from_chunks(trace.retrieved_chunks)
            result = await evaluator.evaluate(
                user_input=sample.question,
                reference=sample.reference,
                response=trace.answer.answer,
                retrieved_contexts=contexts,
            )
            sample_reports.append(
                SampleReport(
                    sample_id=sample.sample_id,
                    question=sample.question,
                    reference=sample.reference,
                    generated_answer=trace.answer.answer,
                    retrieved_chunk_ids=tuple(
                        chunk.chunk_id for chunk in trace.retrieved_chunks
                    ),
                    retrieved_contexts=tuple(contexts),
                    metrics=result.scores,
                    metric_reasons=result.reasons,
                    evaluation_error=result.errors or None,
                )
            )
        except Exception as exc:  # noqa: BLE001 - one bad sample must not stop eval
            sample_reports.append(
                SampleReport(
                    sample_id=sample.sample_id,
                    question=sample.question,
                    reference=sample.reference,
                    generated_answer=None,
                    retrieved_chunk_ids=(),
                    retrieved_contexts=(),
                    metrics=blank_scores,
                    metric_reasons={},
                    evaluation_error={"pipeline": _format_error(exc)},
                )
            )

    aggregate_metrics, aggregate_counts = aggregate_scores(
        sample_reports, evaluator.metric_names
    )
    failed_samples = sum(
        sample.evaluation_error is not None for sample in sample_reports
    )
    return RagasReport(
        schema_version=1,
        created_at=datetime.now(UTC).isoformat(),
        dataset_path=str(Path(dataset_path)),
        dataset_size=len(sample_reports),
        top_k=top_k,
        generation_model=generation_model,
        embedding_model=embedding_model,
        evaluator_model=evaluator_model,
        evaluator_embedding_model=evaluator_embedding_model,
        ragas_version=ragas_version,
        metrics=evaluator.metric_names,
        context_entity_recall={
            "enabled": "context_entity_recall" in evaluator.metric_names,
            "rationale": (
                "金融 QA 的 reference 常包含公司、年份、金额等实体；该指标作为实体覆盖率的补充诊断，"
                "不能替代 Context Recall。"
            ),
        },
        aggregate_metrics=aggregate_metrics,
        aggregate_metric_counts=aggregate_counts,
        successful_samples=len(sample_reports) - failed_samples,
        failed_samples=failed_samples,
        samples=tuple(sample_reports),
    )


def retrieved_contexts_from_chunks(
    chunks: Sequence[RetrievedChunk],
) -> list[str]:
    """Expose exactly ``RetrievedChunk.content`` to RAGAS, in retrieval order."""
    return [chunk.content for chunk in chunks]


def main() -> None:
    args = _parse_args()
    asyncio.run(_run_cli(args))


async def _run_cli(args: argparse.Namespace) -> None:
    _validate_positive("top_k", args.top_k)
    if args.limit is not None:
        _validate_positive("limit", args.limit)

    generation_config = GenerationConfig.from_env()
    embedding_config = EmbeddingConfig.from_env()
    evaluator_config = RagasEvaluatorConfig.from_env()
    base_milvus_config = MilvusConfig.from_env()
    milvus_config = MilvusConfig(
        uri=args.uri or base_milvus_config.uri,
        collection_name=(
            args.collection_name or base_milvus_config.collection_name
        ),
    )

    service = RagAnswerService(
        retriever=MilvusRetriever(
            embedding_config=embedding_config,
            milvus_config=milvus_config,
        ),
        generator=QwenAnswerGenerator(config=generation_config),
    )
    evaluator: RagasEvaluator = create_ragas_evaluator(evaluator_config)
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
    )
    write_report(report, args.output)
    summary = {
        "dataset_size": report.dataset_size,
        "top_k": report.top_k,
        "aggregate_metrics": report.aggregate_metrics,
        "aggregate_metric_counts": report.aggregate_metric_counts,
        "successful_samples": report.successful_samples,
        "failed_samples": report.failed_samples,
        "report": str(args.output),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def _ragas_version() -> str:
    try:
        return version("ragas")
    except PackageNotFoundError as exc:
        raise RuntimeError(
            "RAGAS 未安装，请执行：uv sync --extra evaluation"
        ) from exc


def _format_error(exc: Exception) -> str:
    return f"{type(exc).__name__}: {exc}"


def _validate_positive(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} 必须是正整数")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="运行真实 RAG 链路并用 RAGAS 0.4 collections metrics 评测。"
    )
    parser.add_argument(
        "--dataset", type=Path, default=Path("qa.jsonl"), help="含 question + gold 的 QA JSONL"
    )
    parser.add_argument(
        "--top-k", "--k", dest="top_k", type=int, default=5, help="检索 Top-K，默认 5"
    )
    parser.add_argument("--limit", type=int, help="只评测前 N 条，用于 smoke test")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/eval/ragas_report.json"),
        help="JSON 报告输出路径",
    )
    parser.add_argument("--uri", help="Milvus 地址，默认读取 MILVUS_URI")
    parser.add_argument(
        "--collection-name", help="Milvus collection，默认读取 MILVUS_COLLECTION"
    )
    return parser.parse_args()
