from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from agenticrag.reranking.base import RerankerInferenceError
from agenticrag.reranking.config import RemoteRerankerConfig
from agenticrag.reranking.remote import RemoteBGEReranker
from agenticrag.retrieval.reranking_retriever import RerankingRetriever
from agenticrag.retrieval.schemas import HybridRetrievedChunk


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


def _response(*items: tuple[int, float]) -> dict[str, Any]:
    return {
        "results": [
            {
                "index": index,
                "document": {"text": f"content-{chr(65 + index)}"},
                "relevance_score": score,
            }
            for index, score in items
        ]
    }


def _reranker() -> RemoteBGEReranker:
    return RemoteBGEReranker(
        RemoteRerankerConfig(
            url="http://192.168.31.238:8001",
            model="BAAI/bge-reranker-v2-m3",
            timeout_seconds=3.0,
        )
    )


def test_remote_reranker_sends_documents_and_maps_scores_by_index(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, dict[str, Any], float]] = []

    def fake_post(
        url: str, payload: dict[str, Any], *, timeout_seconds: float
    ) -> dict[str, Any]:
        calls.append((url, payload, timeout_seconds))
        return _response((2, 0.9), (0, 0.4), (1, 0.1))

    monkeypatch.setattr("agenticrag.reranking.remote._post_json", fake_post)
    reranker = _reranker()
    candidates = [_candidate("A", 1), _candidate("B", 2), _candidate("C", 3)]

    scores = reranker.score("query", candidates)

    assert scores == [0.4, 0.1, 0.9]
    assert calls == [
        (
            "http://192.168.31.238:8001/v1/rerank",
            {
                "model": "BAAI/bge-reranker-v2-m3",
                "query": "query",
                "documents": ["content-A", "content-B", "content-C"],
                "top_n": 3,
            },
            3.0,
        )
    ]
    record = reranker.model_record()
    assert record["backend"] == "remote"
    assert record["endpoint"] == "http://192.168.31.238:8001/v1/rerank"
    assert record["candidate_count"] == 3


@pytest.mark.parametrize(
    "response",
    [
        _response((0, 0.1), (0, 0.2), (2, 0.3)),  # duplicate index
        _response((0, 0.1), (2, 0.3)),  # missing index
        _response((0, 0.1), (1, 0.2), (3, 0.3)),  # out of range
        {"results": [{"index": 0, "relevance_score": float("nan")},
                      {"index": 1, "relevance_score": 0.2},
                      {"index": 2, "relevance_score": 0.3}]},
        {"results": [{"index": 0, "relevance_score": float("inf")},
                      {"index": 1, "relevance_score": 0.2},
                      {"index": 2, "relevance_score": 0.3}]},
        {"results": [{"index": 0, "relevance_score": 0.1},
                      {"index": 1, "relevance_score": 0.2},
                      {"index": 2}]},
        {"results": []},
    ],
)
def test_remote_reranker_rejects_invalid_responses(
    response: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "agenticrag.reranking.remote._post_json",
        lambda *args, **kwargs: response,
    )

    with pytest.raises(RerankerInferenceError):
        _reranker().score(
            "query",
            [_candidate("A", 1), _candidate("B", 2), _candidate("C", 3)],
        )


@dataclass
class FakeHybrid:
    candidates: list[HybridRetrievedChunk]

    rrf_k = 60

    def candidate_pool(
        self, query: str, *, route_k: int = 20
    ) -> list[HybridRetrievedChunk]:
        return self.candidates


@pytest.mark.parametrize(
    "failure",
    [TimeoutError("desktop reranker timed out"), OSError("connection refused")],
)
def test_remote_failure_uses_existing_atomic_rrf_fallback(
    failure: Exception,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def raise_timeout(*args: object, **kwargs: object) -> dict[str, Any]:
        raise failure

    monkeypatch.setattr("agenticrag.reranking.remote._post_json", raise_timeout)
    candidates = [_candidate("A", 1), _candidate("B", 2)]
    retriever = RerankingRetriever(
        hybrid_retriever=FakeHybrid(candidates),  # type: ignore[arg-type]
        reranker=_reranker(),
    )

    trace = retriever.search_with_trace("query", k=2)

    assert [result.chunk_id for result in trace.results] == ["A", "B"]
    assert all(result.rerank_score is None for result in trace.results)
    assert all(result.score == result.rrf_score for result in trace.results)
    assert trace.fallback_used is True
    assert trace.fallback_reason == "inference_error"
    assert trace.reranker_backend == "remote"
    assert trace.reranker_endpoint == "http://192.168.31.238:8001/v1/rerank"
    assert trace.rerank_candidate_count == 2
    assert trace.rerank_request_seconds >= 0.0
