from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from agenticrag.retrieval.reranking_retriever import RerankingSearchTrace
from agenticrag.retrieval.schemas import HybridRetrievedChunk, RerankedChunk
from eval.reranking_runner import evaluate_reranking


def _candidate(chunk_id: str, rank: int) -> HybridRetrievedChunk:
    return HybridRetrievedChunk(
        content=f"content-{chunk_id}",
        score=1 / (60 + rank),
        doc_id="doc_001",
        source="report.pdf",
        page=1,
        chunk_id=chunk_id,
        dense_rank=rank,
        bm25_rank=None,
        rrf_score=1 / (60 + rank),
        rrf_rank=rank,
    )


def _final(candidate: HybridRetrievedChunk, rank: int, score: float) -> RerankedChunk:
    return RerankedChunk(
        content=candidate.content,
        score=score,
        doc_id=candidate.doc_id,
        source=candidate.source,
        page=candidate.page,
        chunk_id=candidate.chunk_id,
        dense_rank=candidate.dense_rank,
        bm25_rank=candidate.bm25_rank,
        rrf_score=candidate.rrf_score,
        rrf_rank=candidate.rrf_rank,
        rerank_score=score,
        final_rank=rank,
    )


class FakeRerankingRetriever:
    route_k = 20
    rrf_report_k = 20
    hybrid_retriever = type("FakeHybrid", (), {"rrf_k": 60})()

    def __init__(self, traces: list[RerankingSearchTrace]) -> None:
        self.traces = traces
        self.index = 0

    def search_with_trace(self, query: str, *, k: int = 5) -> RerankingSearchTrace:
        trace = self.traces[self.index]
        self.index += 1
        return trace

    def model_record(self) -> dict[str, Any]:
        return {"model_name": "fake", "load_seconds": 0.2}


def _trace(
    pool: list[HybridRetrievedChunk],
    final: list[RerankedChunk],
    *,
    model_load: float = 0.0,
) -> RerankingSearchTrace:
    return RerankingSearchTrace(
        results=tuple(final),
        candidate_pool=tuple(pool),
        rrf_top20=tuple(pool[:20]),
        fallback_used=False,
        fallback_reason=None,
        invalid_scores=False,
        candidate_seconds=0.1,
        model_load_seconds=model_load,
        rerank_seconds=0.3,
        total_seconds=0.4 + model_load,
    )


def _write_dataset(path: Path) -> None:
    records = [
        {"id": "q1", "query": "one", "relevant_chunk_ids": ["gold1"]},
        {"id": "q2", "query": "two", "relevant_chunk_ids": ["gold2"]},
    ]
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )


def _write_baseline(path: Path) -> None:
    report = {
        "queries": [
            {
                "id": "q1",
                "retrieved": [{"chunk_id": "gold1"}],
            },
            {
                "id": "q2",
                "retrieved": [{"chunk_id": "other"}, {"chunk_id": "gold2"}],
            },
        ]
    }
    path.write_text(json.dumps(report), encoding="utf-8")


def test_reranking_report_separates_all_three_metric_stages(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset.jsonl"
    dense_report = tmp_path / "dense.json"
    hybrid_report = tmp_path / "hybrid.json"
    _write_dataset(dataset)
    _write_baseline(dense_report)
    _write_baseline(hybrid_report)

    pool1 = [_candidate("other1", 1), _candidate("gold1", 2)]
    pool2 = [_candidate("other2", 1), _candidate("gold2", 2)]
    retriever = FakeRerankingRetriever(
        [
            _trace(pool1, [_final(pool1[1], 1, 3.0)], model_load=0.2),
            _trace(pool2, [_final(pool2[0], 1, 2.0)]),
        ]
    )

    report = evaluate_reranking(
        dataset,
        retriever,
        dense_baseline_path=dense_report,
        hybrid_baseline_path=hybrid_report,
    )

    assert report["final_metrics"] == {
        "Recall@1": 0.5,
        "Recall@3": 0.5,
        "Recall@5": 0.5,
        "MRR@5": 0.5,
    }
    assert report["rrf_candidate_metrics"] == {"Recall@20": 1.0}
    assert report["union_pool_metrics"]["Recall@full_pool"] == 1.0
    assert report["latency_seconds"]["model_load"] == pytest.approx(0.2)
    assert report["latency_seconds"]["candidate"]["count"] == 2
    assert report["baseline_comparison"]["hybrid_v1_1"]["metrics"] == {
        "Recall@1": 0.5,
        "Recall@3": 1.0,
        "Recall@5": 1.0,
        "MRR@5": 0.75,
    }
    assert report["fallback_queries"] == 0
    assert report["freeze"]["eligible"] is False
    assert report["queries"][0]["final_results"][0]["rrf_rank"] == 2


def test_reranking_report_contains_candidate_generation_profiling(
    tmp_path: Path,
) -> None:
    dataset = tmp_path / "dataset.jsonl"
    _write_dataset(dataset)
    pool = [_candidate("gold1", 1)]
    trace = _trace(pool, [_final(pool[0], 1, 3.0)])

    report = evaluate_reranking(
        dataset,
        FakeRerankingRetriever(
            [
                replace(
                    trace,
                    query_embedding_seconds=0.1,
                    dense_search_seconds=0.2,
                    bm25_search_seconds=0.3,
                    merge_rrf_seconds=0.05,
                    candidate_total_seconds=0.7,
                ),
                replace(
                    trace,
                    query_embedding_seconds=0.2,
                    dense_search_seconds=0.3,
                    bm25_search_seconds=0.4,
                    merge_rrf_seconds=0.1,
                    candidate_total_seconds=1.0,
                ),
            ]
        ),
    )

    profiling = report["candidate_profiling"]
    assert profiling["stages"]["query_embedding_seconds"]["mean"] == pytest.approx(0.15)
    assert profiling["stages"]["candidate_total_seconds"]["median"] == pytest.approx(0.85)
    assert profiling["average_percentage_of_candidate_total"][
        "query_embedding_seconds"
    ] == pytest.approx((0.1 / 0.7 * 100.0 + 0.2 / 1.0 * 100.0) / 2.0)
    assert report["queries"][0]["candidate_timing_seconds"] == {
        "query_embedding_seconds": 0.1,
        "dense_search_seconds": 0.2,
        "bm25_search_seconds": 0.3,
        "merge_rrf_seconds": 0.05,
        "candidate_total_seconds": 0.7,
    }
