"""Module 3 V1.2 retrieval adapter and bounded task fan-out."""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Protocol, Sequence

from agenticrag.retrieval.reranking_retriever import (
    RerankingRetriever,
    RerankingSearchTrace,
)
from agenticrag.retrieval.schemas import RerankedChunk

from .config import V2Config
from .ids import query_revision_id, retrieval_attempt_id
from .policies import capability_outcome
from .schemas import (
    Evidence,
    EvidenceOccurrence,
    ExecutionError,
    QueryRevision,
    RetrievalAttempt,
    RetrievalResult,
    RetrievalTask,
)
from .state import merge_evidence
from .types import RetrievalStrategy

FINAL_RETRIEVAL_TOP_K = 5


class V12RetrievalBackend(Protocol):
    """Stable retrieval boundary consumed by V2 orchestration."""

    def retrieve(
        self,
        *,
        task_id: str,
        query_revision_id: str,
        attempt_id: str,
        query: str,
        strategy: RetrievalStrategy = "original",
    ) -> RetrievalResult:
        """Return only the final V1.2 Top-5 as canonical Evidence."""


class RetrievalBackendError(RuntimeError):
    """A technical failure at the V1.2 retrieval boundary."""


class V12RetrievalAdapter:
    """Thin adapter over the existing V1.2 ``search_with_trace`` API."""

    def __init__(self, retriever: RerankingRetriever | None = None) -> None:
        self.retriever = retriever or RerankingRetriever()

    def retrieve(
        self,
        *,
        task_id: str,
        query_revision_id: str,
        attempt_id: str,
        query: str,
        strategy: RetrievalStrategy = "original",
    ) -> RetrievalResult:
        try:
            trace = self.retriever.search_with_trace(query, k=FINAL_RETRIEVAL_TOP_K)
            evidence = [
                _evidence_from_chunk(
                    chunk,
                    task_id=task_id,
                    query_revision_id=query_revision_id,
                    attempt_id=attempt_id,
                    strategy=strategy,
                )
                for chunk in trace.results
            ]
            return _retrieval_result(trace, evidence, attempt_id=attempt_id)
        except RetrievalBackendError:
            raise
        except Exception as exc:
            raise RetrievalBackendError(
                f"V1.2 retrieval failed for task {task_id}"
            ) from exc


@dataclass(frozen=True, slots=True)
class RetrievalFanoutResult:
    """Stable output of one bounded retrieval fan-out."""

    task_order: tuple[str, ...]
    tasks: tuple[RetrievalTask, ...]
    retrieval_results: dict[str, RetrievalResult]
    evidence: dict[str, Evidence]


class RetrievalFanoutService:
    """Execute independent retrieval tasks with bounded concurrency."""

    def __init__(
        self,
        backend: V12RetrievalBackend,
        config: V2Config | None = None,
    ) -> None:
        self.backend = backend
        self.config = config or V2Config.from_env()

    def retrieve_tasks(
        self, tasks: Sequence[RetrievalTask]
    ) -> RetrievalFanoutResult:
        ordered_tasks = _stable_task_order(tasks)
        if len(ordered_tasks) > self.config.budgets.max_subqueries:
            raise ValueError("RetrievalTask 数量超出 V2_MAX_SUBQUERIES")

        outcomes: dict[str, tuple[RetrievalTask, RetrievalResult | None]] = {}
        with ThreadPoolExecutor(
            max_workers=self.config.budgets.max_concurrent_subqueries
        ) as executor:
            futures: dict[Future[tuple[RetrievalTask, RetrievalResult | None]], str] = {
                executor.submit(self._retrieve_task, task): task.id
                for task in ordered_tasks
            }
            for future in as_completed(futures):
                task_id = futures[future]
                outcomes[task_id] = future.result()

        retrieval_results: dict[str, RetrievalResult] = {}
        evidence: dict[str, Evidence] = {}
        result_tasks: list[RetrievalTask] = []
        for task in ordered_tasks:
            result_task, retrieval_result = outcomes[task.id]
            result_tasks.append(result_task)
            if retrieval_result is None:
                continue
            retrieval_results[task.id] = retrieval_result
            for item in retrieval_result.evidence:
                evidence = merge_evidence(evidence, {item.evidence_id: item})

        return RetrievalFanoutResult(
            task_order=tuple(task.id for task in ordered_tasks),
            tasks=tuple(result_tasks),
            retrieval_results=retrieval_results,
            evidence=evidence,
        )

    def _retrieve_task(
        self, task: RetrievalTask
    ) -> tuple[RetrievalTask, RetrievalResult | None]:
        unsupported_outcome = capability_outcome(task.capability)
        if unsupported_outcome is not None:
            return (
                task.model_copy(
                    update={
                        "execution_status": "completed",
                        "answer_outcome": unsupported_outcome,
                        "terminal_reason": f"unsupported_capability:{task.capability}",
                    }
                ),
                None,
            )

        if task.query_revisions:
            return self._failed_task(
                task,
                code="invalid_retrieval_task_state",
                message="Module 3 只接受尚未创建 QueryRevision 的 RetrievalTask",
            )

        revision = QueryRevision(
            id=query_revision_id(task.id, 1),
            ordinal=1,
            source="original",
            query=task.query,
        )
        attempt = RetrievalAttempt(
            id=retrieval_attempt_id(task.id, revision.id, 1),
            ordinal=1,
            strategy="original",
            retrieval_query=task.query,
        )
        try:
            retrieval_result = self.backend.retrieve(
                task_id=task.id,
                query_revision_id=revision.id,
                attempt_id=attempt.id,
                query=task.query,
                strategy="original",
            )
            attempt = attempt.model_copy(
                update={
                    "evidence_ids": [
                        item.evidence_id for item in retrieval_result.evidence
                    ],
                    "retrieval_degraded": retrieval_result.retrieval_degraded,
                    "retrieval_degraded_reason": retrieval_result.degraded_reason,
                    "latency": retrieval_result.latency,
                    "trace_ref": retrieval_result.trace_ref,
                }
            )
            revision = revision.model_copy(update={"retrieval_attempts": [attempt]})
            return (
                task.model_copy(
                    update={
                        "query_revisions": [revision],
                        "execution_status": "running",
                    }
                ),
                retrieval_result,
            )
        except Exception as exc:
            failed_revision = revision.model_copy(update={"retrieval_attempts": [attempt]})
            return self._failed_task(
                task,
                code="retrieval_failed",
                message="V1.2 retrieval technical failure",
                revisions=[failed_revision],
                cause=exc,
            )

    def _failed_task(
        self,
        task: RetrievalTask,
        *,
        code: str,
        message: str,
        revisions: list[QueryRevision] | None = None,
        cause: Exception | None = None,
    ) -> tuple[RetrievalTask, None]:
        details = {"task_id": task.id}
        if cause is not None:
            details["exception_type"] = type(cause).__name__
        error = ExecutionError(
            code=code,
            message=message,
            stage="module3_retrieval",
            retryable=code == "retrieval_failed",
            details=details,
        )
        return (
            task.model_copy(
                update={
                    "query_revisions": revisions or task.query_revisions,
                    "execution_status": "failed",
                    "answer_outcome": None,
                    "error": error,
                }
            ),
            None,
        )


def _stable_task_order(tasks: Sequence[RetrievalTask]) -> list[RetrievalTask]:
    if len({task.id for task in tasks}) != len(tasks):
        raise ValueError("RetrievalTask ID 必须唯一")
    return sorted(tasks, key=lambda task: (task.ordinal, task.id))


def _evidence_from_chunk(
    chunk: RerankedChunk,
    *,
    task_id: str,
    query_revision_id: str,
    attempt_id: str,
    strategy: RetrievalStrategy,
) -> Evidence:
    occurrence = EvidenceOccurrence(
        task_id=task_id,
        query_revision_id=query_revision_id,
        retrieval_attempt_id=attempt_id,
        strategy=strategy,
        dense_rank=chunk.dense_rank,
        bm25_rank=chunk.bm25_rank,
        rrf_rank=chunk.rrf_rank,
        final_rank=chunk.final_rank,
    )
    return Evidence(
        evidence_id=chunk.chunk_id,
        chunk_id=chunk.chunk_id,
        content=chunk.content,
        doc_id=chunk.doc_id,
        source=chunk.source,
        page=chunk.page,
        occurrences=[occurrence],
    )


def _retrieval_result(
    trace: RerankingSearchTrace,
    evidence: list[Evidence],
    *,
    attempt_id: str,
) -> RetrievalResult:
    latency = {
        "query_embedding_seconds": trace.query_embedding_seconds,
        "dense_search_seconds": trace.dense_search_seconds,
        "bm25_search_seconds": trace.bm25_search_seconds,
        "merge_rrf_seconds": trace.merge_rrf_seconds,
        "candidate_total_seconds": trace.candidate_total_seconds,
        "rerank_seconds": trace.rerank_seconds,
        "total_seconds": trace.total_seconds,
    }
    diagnostics = {
        "candidate_pool": [item.to_record() for item in trace.candidate_pool],
        "rrf_top20": [item.to_record() for item in trace.rrf_top20],
        "final_results": [item.to_record() for item in trace.results],
        "reranker_backend": trace.reranker_backend,
        "reranker_endpoint": trace.reranker_endpoint,
        "fallback_used": trace.fallback_used,
        "fallback_reason": trace.fallback_reason,
        "invalid_scores": trace.invalid_scores,
        "embedding_backend": trace.embedding_backend,
        "embedding_endpoint": trace.embedding_endpoint,
    }
    return RetrievalResult(
        evidence=evidence,
        retrieval_degraded=trace.fallback_used,
        degraded_reason=trace.fallback_reason,
        latency=latency,
        trace_ref=attempt_id,
        diagnostics=diagnostics,
    )
