from __future__ import annotations

import time

from agenticrag.retrieval.reranking_retriever import RerankingSearchTrace
from agenticrag.retrieval.schemas import HybridRetrievedChunk, RerankedChunk
from agenticrag.v2.config import V2BudgetConfig, V2Config
from agenticrag.v2.retrieval import (
    RetrievalFanoutService,
    V12RetrievalAdapter,
)
from agenticrag.v2.schemas import RetrievalTask


def _chunk(
    chunk_id: str,
    *,
    final_rank: int = 1,
    fallback_used: bool = False,
    fallback_reason: str | None = None,
) -> RerankedChunk:
    return RerankedChunk(
        content=f"content-{chunk_id}",
        score=0.9,
        doc_id="doc-1",
        source="source.pdf",
        page=1,
        chunk_id=chunk_id,
        dense_rank=1,
        bm25_rank=1,
        rrf_score=0.5,
        rrf_rank=1,
        rerank_score=None if fallback_used else 0.9,
        final_rank=final_rank,
        fallback_used=fallback_used,
        fallback_reason=fallback_reason,
    )


def _trace(
    results: tuple[RerankedChunk, ...],
    *,
    fallback_used: bool = False,
    fallback_reason: str | None = None,
) -> RerankingSearchTrace:
    candidate_pool = tuple(
        HybridRetrievedChunk(
            content=f"candidate-{index}",
            score=0.1,
            doc_id="doc-1",
            source="source.pdf",
            page=1,
            chunk_id=f"candidate-{index}",
            rrf_score=0.1,
            rrf_rank=index,
        )
        for index in range(1, 7)
    )
    return RerankingSearchTrace(
        results=results,
        candidate_pool=candidate_pool,
        rrf_top20=candidate_pool,
        fallback_used=fallback_used,
        fallback_reason=fallback_reason,
        invalid_scores=False,
        candidate_seconds=0.02,
        model_load_seconds=0.0,
        rerank_seconds=0.03,
        total_seconds=0.05,
        reranker_backend="remote",
        reranker_endpoint="http://desktop:8001",
        rerank_request_seconds=0.03,
        rerank_candidate_count=6,
        query_embedding_seconds=0.01,
        dense_search_seconds=0.002,
        bm25_search_seconds=0.001,
        merge_rrf_seconds=0.0001,
        candidate_total_seconds=0.02,
        embedding_backend="remote",
        embedding_endpoint="http://desktop:8002",
        embedding_request_seconds=0.01,
        embedding_dimension=1024,
    )


class FakeRerankingRetriever:
    def __init__(self, traces: dict[str, RerankingSearchTrace]) -> None:
        self.traces = traces
        self.calls: list[tuple[str, int]] = []

    def search_with_trace(self, query: str, *, k: int = 5) -> RerankingSearchTrace:
        self.calls.append((query, k))
        if query == "slow":
            time.sleep(0.03)
        return self.traces[query]


def _task(identifier: str, query: str, capability: str = "retrieval_synthesis") -> RetrievalTask:
    ordinal = int(identifier.removeprefix("SQ_"))
    return RetrievalTask(
        id=identifier,
        ordinal=ordinal,
        query=query,
        intent=f"intent-{identifier}",
        capability=capability,
    )


def test_v12_adapter_returns_final_top5_and_keeps_trace_diagnostics() -> None:
    retriever = FakeRerankingRetriever(
        {"question": _trace((_chunk("chunk-1"),))}
    )
    result = V12RetrievalAdapter(retriever).retrieve(
        task_id="SQ_001",
        query_revision_id="QR_SQ001_001",
        attempt_id="ATT_SQ001_QR001_001",
        query="question",
    )

    assert retriever.calls == [("question", 5)]
    assert [item.evidence_id for item in result.evidence] == ["chunk-1"]
    assert result.evidence[0].occurrences[0].retrieval_attempt_id == "ATT_SQ001_QR001_001"
    assert len(result.diagnostics["candidate_pool"]) == 6
    assert len(result.diagnostics["final_results"]) == 1
    assert result.trace_ref == "ATT_SQ001_QR001_001"


def test_fanout_is_bounded_ordered_and_deduplicates_occurrences() -> None:
    retriever = FakeRerankingRetriever(
        {
            "slow": _trace((_chunk("shared"),)),
            "fast": _trace((_chunk("shared"),)),
        }
    )
    service = RetrievalFanoutService(
        V12RetrievalAdapter(retriever),
        V2Config(budgets=V2BudgetConfig(max_concurrent_subqueries=2)),
    )
    result = service.retrieve_tasks(
        [_task("SQ_002", "fast"), _task("SQ_001", "slow")]
    )

    assert result.task_order == ("SQ_001", "SQ_002")
    assert [task.id for task in result.tasks] == ["SQ_001", "SQ_002"]
    assert list(result.evidence) == ["shared"]
    assert len(result.evidence["shared"].occurrences) == 2
    assert [
        occurrence.task_id for occurrence in result.evidence["shared"].occurrences
    ] == ["SQ_001", "SQ_002"]
    first_task = result.tasks[0]
    assert first_task.query_revisions[0].id == "QR_SQ001_001"
    assert first_task.query_revisions[0].retrieval_attempts[0].id == (
        "ATT_SQ001_QR001_001"
    )


def test_unsupported_capability_skips_backend() -> None:
    retriever = FakeRerankingRetriever({"supported": _trace((_chunk("chunk-1"),))})
    result = RetrievalFanoutService(V12RetrievalAdapter(retriever)).retrieve_tasks(
        [
            _task("SQ_001", "calculation", "arithmetic"),
            _task("SQ_002", "supported"),
        ]
    )

    assert retriever.calls == [("supported", 5)]
    unsupported = result.tasks[0]
    assert unsupported.execution_status == "completed"
    assert unsupported.answer_outcome == "unsupported"
    assert unsupported.query_revisions == []


def test_degraded_trace_is_preserved_on_task_attempt() -> None:
    retriever = FakeRerankingRetriever(
        {
            "question": _trace(
                (_chunk("chunk-1", fallback_used=True, fallback_reason="inference_error"),),
                fallback_used=True,
                fallback_reason="inference_error",
            )
        }
    )
    result = RetrievalFanoutService(V12RetrievalAdapter(retriever)).retrieve_tasks(
        [_task("SQ_001", "question")]
    )

    retrieval = result.retrieval_results["SQ_001"]
    attempt = result.tasks[0].query_revisions[0].retrieval_attempts[0]
    assert retrieval.retrieval_degraded is True
    assert retrieval.degraded_reason == "inference_error"
    assert attempt.retrieval_degraded is True
    assert attempt.retrieval_degraded_reason == "inference_error"


def test_retrieval_error_marks_task_failed_not_no_knowledge() -> None:
    class FailingBackend:
        def retrieve(self, **kwargs: object) -> object:
            raise RuntimeError("backend unavailable")

    result = RetrievalFanoutService(FailingBackend()).retrieve_tasks(
        [_task("SQ_001", "question")]
    )

    failed = result.tasks[0]
    assert failed.execution_status == "failed"
    assert failed.answer_outcome is None
    assert failed.error is not None
    assert failed.error.code == "retrieval_failed"
    assert "SQ_001" not in result.retrieval_results
