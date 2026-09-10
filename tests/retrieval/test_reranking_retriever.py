from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import pytest

from agenticrag.reranking.base import (
    BaseReranker,
    RerankerInferenceError,
    RerankerLoadError,
)
from agenticrag.retrieval.reranking_retriever import RerankingRetriever
from agenticrag.retrieval.schemas import HybridRetrievedChunk


def _candidate(chunk_id: str, rrf_rank: int) -> HybridRetrievedChunk:
    rrf_score = 1 / (60 + rrf_rank)
    return HybridRetrievedChunk(
        content=f"content-{chunk_id}",
        score=rrf_score,
        doc_id="doc_001",
        source="report.pdf",
        page=1,
        chunk_id=chunk_id,
        dense_rank=rrf_rank,
        bm25_rank=None,
        rrf_score=rrf_score,
        rrf_rank=rrf_rank,
    )


@dataclass
class FakeHybridRetriever:
    candidates: list[HybridRetrievedChunk]
    calls: list[tuple[str, int]]

    def candidate_pool(self, query: str, *, route_k: int = 20) -> list[HybridRetrievedChunk]:
        self.calls.append((query, route_k))
        return self.candidates


class FakeReranker(BaseReranker):
    def __init__(
        self,
        scores: Sequence[object] = (),
        *,
        load_error: bool = False,
        inference_error: bool = False,
    ) -> None:
        self.scores = scores
        self.load_error = load_error
        self.inference_error = inference_error
        self.load_calls = 0
        self.score_calls = 0
        self._loaded = False

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def load(self) -> None:
        self.load_calls += 1
        if self.load_error:
            raise RerankerLoadError("load failed")
        self._loaded = True

    def score(
        self,
        query: str,
        candidates: Sequence[HybridRetrievedChunk],
    ) -> Sequence[object]:
        self.score_calls += 1
        if self.inference_error:
            raise RerankerInferenceError("inference failed")
        return self.scores

    def model_record(self) -> dict[str, Any]:
        return {"backend": "fake"}


def test_reranking_uses_all_candidates_and_returns_final_top_k() -> None:
    candidates = [_candidate("a", 1), _candidate("b", 2), _candidate("c", 3)]
    hybrid = FakeHybridRetriever(candidates, [])
    reranker = FakeReranker([1.0, 3.0, 2.0])
    retriever = RerankingRetriever(
        hybrid_retriever=hybrid,  # type: ignore[arg-type]
        reranker=reranker,
    )

    trace = retriever.search_with_trace("query", k=2)

    assert hybrid.calls == [("query", 20)]
    assert reranker.score_calls == 1
    assert [result.chunk_id for result in trace.results] == ["b", "c"]
    assert [result.rerank_score for result in trace.results] == [3.0, 2.0]
    assert [result.score for result in trace.results] == [3.0, 2.0]
    assert [result.final_rank for result in trace.results] == [1, 2]
    assert trace.fallback_used is False
    assert len(trace.candidate_pool) == 3


def test_reranking_ties_use_rrf_rank_then_chunk_id() -> None:
    candidates = [_candidate("b", 1), _candidate("a", 1), _candidate("c", 3)]
    retriever = RerankingRetriever(
        hybrid_retriever=FakeHybridRetriever(candidates, []),  # type: ignore[arg-type]
        reranker=FakeReranker([2.0, 2.0, 2.0]),
    )

    results = retriever.search("query", k=3)

    assert [result.chunk_id for result in results] == ["a", "b", "c"]


@pytest.mark.parametrize("scores", [[1.0], [1.0, float("nan")], [1.0, "bad"]])
def test_invalid_scores_fall_back_atomically(scores: Sequence[object]) -> None:
    candidates = [_candidate("a", 1), _candidate("b", 2)]
    retriever = RerankingRetriever(
        hybrid_retriever=FakeHybridRetriever(candidates, []),  # type: ignore[arg-type]
        reranker=FakeReranker(scores),
    )

    trace = retriever.search_with_trace("query", k=2)

    assert [result.chunk_id for result in trace.results] == ["a", "b"]
    assert all(result.rerank_score is None for result in trace.results)
    assert all(result.score == result.rrf_score for result in trace.results)
    assert all(result.fallback_reason == "invalid_scores" for result in trace.results)
    assert trace.fallback_used is True
    assert trace.invalid_scores is True


@pytest.mark.parametrize(
    ("reranker", "reason"),
    [
        (FakeReranker(load_error=True), "model_load_error"),
        (FakeReranker(inference_error=True), "inference_error"),
    ],
)
def test_reranker_failures_fall_back_to_rrf(
    reranker: FakeReranker,
    reason: str,
) -> None:
    candidates = [_candidate("a", 1), _candidate("b", 2)]
    retriever = RerankingRetriever(
        hybrid_retriever=FakeHybridRetriever(candidates, []),  # type: ignore[arg-type]
        reranker=reranker,
    )

    trace = retriever.search_with_trace("query", k=2)

    assert [result.chunk_id for result in trace.results] == ["a", "b"]
    assert trace.fallback_used is True
    assert trace.fallback_reason == reason


def test_empty_candidate_pool_does_not_load_model() -> None:
    reranker = FakeReranker([])
    retriever = RerankingRetriever(
        hybrid_retriever=FakeHybridRetriever([], []),  # type: ignore[arg-type]
        reranker=reranker,
    )

    trace = retriever.search_with_trace("query")

    assert trace.results == ()
    assert reranker.load_calls == 0
