import pytest

from agenticrag.retrieval.rrf import fuse_ranked_results
from agenticrag.retrieval.schemas import RetrievedChunk


def _chunk(chunk_id: str) -> RetrievedChunk:
    return RetrievedChunk(
        content=f"content-{chunk_id}",
        score=999.0,
        doc_id="doc_001",
        source="report.pdf",
        page=1,
        chunk_id=chunk_id,
    )


def test_rrf_deduplicates_and_preserves_both_route_ranks() -> None:
    results = fuse_ranked_results(
        [_chunk("dense_a"), _chunk("shared")],
        [_chunk("shared"), _chunk("bm25_c")],
    )

    assert [result.chunk_id for result in results] == ["shared", "dense_a", "bm25_c"]
    assert results[0].dense_rank == 2
    assert results[0].bm25_rank == 1
    assert results[0].rrf_score == pytest.approx(1 / 62 + 1 / 61)
    assert results[1].dense_rank == 1
    assert results[1].bm25_rank is None
    assert results[2].dense_rank is None
    assert results[2].bm25_rank == 2
    assert [result.rrf_rank for result in results] == [1, 2, 3]


def test_rrf_limit_and_deterministic_tie_break() -> None:
    results = fuse_ranked_results(
        [_chunk("a")],
        [_chunk("b")],
        limit=1,
    )

    assert len(results) == 1
    assert results[0].chunk_id == "a"


def test_rrf_rejects_duplicate_ids_in_one_route() -> None:
    with pytest.raises(ValueError, match="重复 chunk_id"):
        fuse_ranked_results([_chunk("same"), _chunk("same")], [])


def test_rrf_allows_one_route_to_be_empty() -> None:
    results = fuse_ranked_results([], [_chunk("bm25_only")])

    assert len(results) == 1
    assert results[0].dense_rank is None
    assert results[0].bm25_rank == 1
