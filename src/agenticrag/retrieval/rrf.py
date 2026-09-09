"""Application-side Reciprocal Rank Fusion for retrieval candidates."""

from __future__ import annotations

from dataclasses import dataclass
from math import inf

from agenticrag.retrieval.schemas import HybridRetrievedChunk, RetrievedChunk

DEFAULT_RRF_K = 60


@dataclass
class _FusionCandidate:
    chunk: RetrievedChunk
    dense_rank: int | None = None
    bm25_rank: int | None = None
    rrf_score: float = 0.0


def fuse_ranked_results(
    dense_results: list[RetrievedChunk],
    bm25_results: list[RetrievedChunk],
    *,
    limit: int = 20,
    rrf_k: int = DEFAULT_RRF_K,
) -> list[HybridRetrievedChunk]:
    """Fuse two ranked result lists by chunk ID using Reciprocal Rank Fusion."""
    _validate_positive_int(limit, "limit")
    _validate_positive_int(rrf_k, "rrf_k")

    candidates: dict[str, _FusionCandidate] = {}
    _add_route(candidates, dense_results, route="dense", rrf_k=rrf_k)
    _add_route(candidates, bm25_results, route="bm25", rrf_k=rrf_k)

    ranked_candidates = sorted(candidates.values(), key=_sort_key)
    return [
        HybridRetrievedChunk(
            content=candidate.chunk.content,
            score=candidate.rrf_score,
            doc_id=candidate.chunk.doc_id,
            source=candidate.chunk.source,
            page=candidate.chunk.page,
            chunk_id=candidate.chunk.chunk_id,
            dense_rank=candidate.dense_rank,
            bm25_rank=candidate.bm25_rank,
            rrf_score=candidate.rrf_score,
        )
        for candidate in ranked_candidates[:limit]
    ]


def _add_route(
    candidates: dict[str, _FusionCandidate],
    results: list[RetrievedChunk],
    *,
    route: str,
    rrf_k: int,
) -> None:
    seen: set[str] = set()
    for rank, chunk in enumerate(results, start=1):
        if chunk.chunk_id in seen:
            raise ValueError(f"{route} Retriever 返回重复 chunk_id：{chunk.chunk_id}")
        seen.add(chunk.chunk_id)

        candidate = candidates.setdefault(
            chunk.chunk_id,
            _FusionCandidate(chunk=chunk),
        )
        candidate.rrf_score += 1 / (rrf_k + rank)
        if route == "dense":
            candidate.dense_rank = rank
        else:
            candidate.bm25_rank = rank


def _sort_key(candidate: _FusionCandidate) -> tuple[float, int, int, int, str]:
    route_ranks = [
        rank for rank in (candidate.dense_rank, candidate.bm25_rank) if rank is not None
    ]
    best_rank = min(route_ranks, default=inf)
    return (
        -candidate.rrf_score,
        best_rank,
        candidate.dense_rank if candidate.dense_rank is not None else inf,
        candidate.bm25_rank if candidate.bm25_rank is not None else inf,
        candidate.chunk.chunk_id,
    )


def _validate_positive_int(value: int, field_name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field_name} 必须是正整数")

