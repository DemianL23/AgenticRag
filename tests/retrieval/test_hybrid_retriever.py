from dataclasses import dataclass

import pytest

from agenticrag.retrieval.hybrid_retriever import HybridRetriever
from agenticrag.retrieval.schemas import RetrievedChunk


def _chunk(chunk_id: str, score: float) -> RetrievedChunk:
    return RetrievedChunk(
        content=chunk_id,
        score=score,
        doc_id="doc_001",
        source="report.pdf",
        page=1,
        chunk_id=chunk_id,
    )


@dataclass
class FakeRetriever:
    results: list[RetrievedChunk]
    calls: list[tuple[str, int]]

    def search(self, query: str, k: int = 20) -> list[RetrievedChunk]:
        self.calls.append((query, k))
        return self.results


@dataclass
class TimedFakeDenseRetriever(FakeRetriever):
    last_query_embedding_seconds: float = 0.0


def test_hybrid_retriever_queries_both_routes_and_fuses() -> None:
    dense = FakeRetriever([_chunk("dense", 0.1), _chunk("shared", 0.2)], [])
    bm25 = FakeRetriever([_chunk("shared", 10.0), _chunk("bm25", 9.0)], [])
    retriever = HybridRetriever(dense_retriever=dense, bm25_retriever=bm25)

    results = retriever.search("原始问题", k=2)

    assert dense.calls == [("原始问题", 2)]
    assert bm25.calls == [("原始问题", 2)]
    assert [result.chunk_id for result in results] == ["shared", "dense"]
    assert results[0].dense_rank == 2
    assert results[0].bm25_rank == 1
    assert results[0].rrf_rank == 1


def test_candidate_pool_keeps_all_deduplicated_route_results() -> None:
    dense = FakeRetriever([_chunk("dense", 0.1), _chunk("shared", 0.2)], [])
    bm25 = FakeRetriever([_chunk("shared", 10.0), _chunk("bm25", 9.0)], [])
    retriever = HybridRetriever(dense_retriever=dense, bm25_retriever=bm25)

    results = retriever.candidate_pool("原始问题", route_k=20)

    assert dense.calls == [("原始问题", 20)]
    assert bm25.calls == [("原始问题", 20)]
    assert [result.chunk_id for result in results] == ["shared", "dense", "bm25"]
    assert [result.rrf_rank for result in results] == [1, 2, 3]


def test_candidate_pool_with_timing_preserves_results_and_records_stages() -> None:
    dense = TimedFakeDenseRetriever(
        [_chunk("dense", 0.1), _chunk("shared", 0.2)],
        [],
        last_query_embedding_seconds=0.001,
    )
    bm25 = FakeRetriever([_chunk("shared", 10.0), _chunk("bm25", 9.0)], [])
    retriever = HybridRetriever(dense_retriever=dense, bm25_retriever=bm25)

    results, timing = retriever.candidate_pool_with_timing("原始问题", route_k=2)

    assert [result.chunk_id for result in results] == ["shared", "dense", "bm25"]
    assert timing.query_embedding_seconds == 0.001
    assert timing.dense_search_seconds == 0.0
    assert timing.bm25_search_seconds >= 0.0
    assert timing.merge_rrf_seconds >= 0.0
    assert timing.candidate_total_seconds >= 0.0


def test_hybrid_retriever_validates_query_and_k() -> None:
    dense = FakeRetriever([], [])
    bm25 = FakeRetriever([], [])
    retriever = HybridRetriever(dense_retriever=dense, bm25_retriever=bm25)

    with pytest.raises(ValueError, match="query"):
        retriever.search(" ")
    with pytest.raises(ValueError, match="k"):
        retriever.search("问题", k=0)
    with pytest.raises(ValueError, match="k"):
        retriever.search("问题", k=True)
    with pytest.raises(ValueError, match="route_k"):
        retriever.candidate_pool("问题", route_k=0)
