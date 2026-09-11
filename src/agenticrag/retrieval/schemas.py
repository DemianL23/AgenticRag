"""Stable data structures returned by retrieval implementations."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class RetrievedChunk:
    """One chunk returned by a retriever, including citation metadata."""

    content: str
    score: float
    doc_id: str
    source: str
    page: int
    chunk_id: str

    def to_record(self) -> dict[str, Any]:
        """Return a JSON-serializable representation for APIs and CLI output."""
        return asdict(self)


@dataclass(frozen=True, slots=True)
class HybridRetrievedChunk(RetrievedChunk):
    """One RRF-fused chunk with its rank in each retrieval route."""

    dense_rank: int | None = None
    bm25_rank: int | None = None
    rrf_score: float = 0.0
    rrf_rank: int = 0


@dataclass(frozen=True, slots=True)
class CandidateGenerationTiming:
    """Wall-clock timing for one Dense + BM25 + RRF candidate generation."""

    query_embedding_seconds: float = 0.0
    dense_search_seconds: float = 0.0
    bm25_search_seconds: float = 0.0
    merge_rrf_seconds: float = 0.0
    candidate_total_seconds: float = 0.0

    def to_record(self) -> dict[str, float]:
        """Return the stable JSON report shape for profiling diagnostics."""
        return {
            "query_embedding_seconds": self.query_embedding_seconds,
            "dense_search_seconds": self.dense_search_seconds,
            "bm25_search_seconds": self.bm25_search_seconds,
            "merge_rrf_seconds": self.merge_rrf_seconds,
            "candidate_total_seconds": self.candidate_total_seconds,
        }


@dataclass(frozen=True, slots=True)
class RerankedChunk(HybridRetrievedChunk):
    """One final V1.2 chunk with both initial and final ranking evidence."""

    rerank_score: float | None = None
    final_rank: int = 0
    fallback_used: bool = False
    fallback_reason: str | None = None
