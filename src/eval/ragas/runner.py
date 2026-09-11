"""Run the real RAG pipeline, score every sample, and write a JSON report."""

from __future__ import annotations

import argparse
import asyncio
import json
from collections.abc import Sequence
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Protocol

from agenticrag.generation.config import GenerationConfig
from agenticrag.generation.schemas import GeneratedAnswer
from agenticrag.generation.generator import QwenAnswerGenerator
from agenticrag.rag.integrations.embeddings import EmbeddingConfig
from agenticrag.rag.integrations.milvus import MilvusConfig
from agenticrag.rag.service import RagAnswerService
from agenticrag.retrieval.milvus_retriever import MilvusRetriever
from agenticrag.retrieval.schemas import RetrievedChunk

from .config import RagasEvaluatorConfig
from .dataset import load_qa_dataset
from .evaluator import RagasEvaluator, RagasMetricResult
from ..numeric_correctness import NUMERIC_CORRECTNESS, numeric_correctness
from .providers import create_ragas_evaluator
from .report import RagasReport, SampleReport, aggregate_scores, write_report


class AnswerTrace(Protocol):
    answer: GeneratedAnswer
    retrieved_chunks: Sequence[RetrievedChunk]


class AnswerService(Protocol):
    def answer_with_trace(self, query: str, *, k: int = 5) -> AnswerTrace:
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
    report_schema_version: int = 1,
    report_name: str = "ragas",
    run_id: str | None = None,
    git_commit: str | None = None,
    git_dirty: bool | None = None,
    resolved_config: dict[str, Any] | None = None,
) -> RagasReport:
    """Materialize and evaluate each RAG sample while isolating failures."""
    _validate_positive("top_k", top_k)
    if limit is not None:
        _validate_positive("limit", limit)
    qa_samples = load_qa_dataset(Path(dataset_path), limit=limit)
    sample_reports: list[SampleReport] = []
    report_metric_names = (*evaluator.metric_names, NUMERIC_CORRECTNESS)

    for index, sample in enumerate(qa_samples, start=1):
        print(f"[{index}/{len(qa_samples)}] {sample.sample_id}: RAG + RAGAS")
        blank_scores = {name: None for name in report_metric_names}
        try:
            trace = service.answer_with_trace(sample.question, k=top_k)
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
            continue

        contexts = retrieved_contexts_from_chunks(trace.retrieved_chunks)
        try:
            result = await evaluator.evaluate(
                user_input=sample.question,
                reference=sample.reference,
                response=trace.answer.answer,
                retrieved_contexts=contexts,
            )
        except Exception as exc:  # noqa: BLE001 - one bad sample must not stop eval
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
                    metrics={**blank_scores, NUMERIC_CORRECTNESS: _safe_numeric_score(
                        sample.reference,
                        trace.answer.answer,
                        task_type=sample.task_type,
                    )},
                    metric_reasons={},
                    evaluation_error={"evaluator": _format_error(exc)},
                    retrieval_trace=retrieval_trace_to_record(trace),
                )
            )
            continue

        try:
            numeric_score = numeric_correctness(
                sample.reference,
                trace.answer.answer,
                task_type=sample.task_type,
            )
        except Exception as exc:  # noqa: BLE001 - numeric score is diagnostic only
            evaluation_errors = dict(result.errors)
            evaluation_errors[NUMERIC_CORRECTNESS] = _format_error(exc)
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
                    metrics={**result.scores, NUMERIC_CORRECTNESS: None},
                    metric_reasons=result.reasons,
                    evaluation_error=evaluation_errors,
                    retrieval_trace=retrieval_trace_to_record(trace),
                )
            )
            continue

        scores = {**result.scores, NUMERIC_CORRECTNESS: numeric_score}
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
                metrics=scores,
                metric_reasons=result.reasons,
                evaluation_error=result.errors or None,
                retrieval_trace=retrieval_trace_to_record(trace),
            )
        )

    aggregate_metrics, aggregate_counts = aggregate_scores(
        sample_reports, report_metric_names
    )
    failed_samples = sum(
        sample.evaluation_error is not None for sample in sample_reports
    )
    retrieval_traces = [
        sample.retrieval_trace
        for sample in sample_reports
        if sample.retrieval_trace is not None
    ]
    return RagasReport(
        schema_version=report_schema_version,
        created_at=datetime.now(UTC).isoformat(),
        dataset_path=str(Path(dataset_path)),
        dataset_size=len(sample_reports),
        top_k=top_k,
        generation_model=generation_model,
        embedding_model=embedding_model,
        evaluator_model=evaluator_model,
        evaluator_embedding_model=evaluator_embedding_model,
        ragas_version=ragas_version,
        metrics=report_metric_names,
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
        report_name=report_name,
        run_id=run_id,
        git_commit=git_commit,
        git_dirty=git_dirty,
        resolved_config=resolved_config,
        retrieval_record=_service_retrieval_record(service),
        retrieval_summary=_retrieval_summary(retrieval_traces),
    )


def retrieval_trace_to_record(trace: AnswerTrace | None) -> dict[str, Any] | None:
    """Serialize optional retriever diagnostics without coupling V0 to V1.2."""
    if trace is None:
        return None
    retrieval_trace = getattr(trace, "retrieval_trace", None)
    if retrieval_trace is None:
        return None

    results = tuple(getattr(retrieval_trace, "results", ()))
    candidate_pool = tuple(getattr(retrieval_trace, "candidate_pool", ()))
    rrf_top20 = tuple(getattr(retrieval_trace, "rrf_top20", ()))
    return {
        "candidate_pool_size": len(candidate_pool),
        "candidate_pool_chunk_ids": [chunk.chunk_id for chunk in candidate_pool],
        "rrf_top20_size": len(rrf_top20),
        "rrf_top20_chunk_ids": [chunk.chunk_id for chunk in rrf_top20],
        "results": [chunk.to_record() for chunk in results],
        "fallback_used": bool(getattr(retrieval_trace, "fallback_used", False)),
        "fallback_reason": getattr(retrieval_trace, "fallback_reason", None),
        "invalid_scores": bool(getattr(retrieval_trace, "invalid_scores", False)),
        "candidate_seconds": float(getattr(retrieval_trace, "candidate_seconds", 0.0)),
        "model_load_seconds": float(getattr(retrieval_trace, "model_load_seconds", 0.0)),
        "rerank_seconds": float(getattr(retrieval_trace, "rerank_seconds", 0.0)),
        "total_seconds": float(getattr(retrieval_trace, "total_seconds", 0.0)),
    }


def _service_retrieval_record(service: AnswerService) -> dict[str, Any] | None:
    provider = getattr(service, "retrieval_record", None)
    if not callable(provider):
        return None
    try:
        record = provider()
    except Exception as exc:  # noqa: BLE001 - report creation must remain available
        return {"record_error": _format_error(exc)}
    return record if isinstance(record, dict) else {"record_error": "not_a_mapping"}


def _safe_numeric_score(
    reference: str,
    answer: str,
    *,
    task_type: str | None,
) -> float | None:
    """Keep evaluator failures from hiding an independently computed score."""
    try:
        return numeric_correctness(reference, answer, task_type=task_type)
    except Exception:  # noqa: BLE001 - numeric score is diagnostic only
        return None


def _retrieval_summary(traces: Sequence[dict[str, Any]]) -> dict[str, Any] | None:
    if not traces:
        return None
    fallback_reasons: dict[str, int] = {}
    for trace in traces:
        if trace["fallback_used"]:
            reason = trace["fallback_reason"] or "unknown"
            fallback_reasons[reason] = fallback_reasons.get(reason, 0) + 1

    def total(name: str) -> float:
        return sum(float(trace[name]) for trace in traces)

    return {
        "traced_queries": len(traces),
        "retrieval_degraded_queries": sum(
            bool(trace["fallback_used"]) for trace in traces
        ),
        "fallback_reason_counts": fallback_reasons,
        "total_candidate_seconds": total("candidate_seconds"),
        "total_model_load_seconds": total("model_load_seconds"),
        "total_rerank_seconds": total("rerank_seconds"),
        "total_retrieval_seconds": total("total_seconds"),
        "mean_candidate_seconds": total("candidate_seconds") / len(traces),
        "mean_model_load_seconds": total("model_load_seconds") / len(traces),
        "mean_rerank_seconds": total("rerank_seconds") / len(traces),
        "mean_retrieval_seconds": total("total_seconds") / len(traces),
    }


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
