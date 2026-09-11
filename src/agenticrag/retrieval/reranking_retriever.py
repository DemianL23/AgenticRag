"""V1.2 Hybrid candidate generation followed by replaceable reranking."""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass
from numbers import Real
from typing import Any, Sequence

from agenticrag.reranking.base import (
    BaseReranker,
    RerankerInferenceError,
    RerankerLoadError,
)
from agenticrag.reranking.factory import create_reranker
from agenticrag.retrieval.base import BaseRetriever
from agenticrag.retrieval.hybrid_retriever import (
    DEFAULT_HYBRID_ROUTE_TOP_K,
    HybridRetriever,
)
from agenticrag.retrieval.schemas import (
    HybridRetrievedChunk,
    RerankedChunk,
)


DEFAULT_RERANK_FINAL_TOP_K = 5
DEFAULT_RRF_CANDIDATE_REPORT_K = 20
FALLBACK_MODEL_LOAD_ERROR = "model_load_error"
FALLBACK_INFERENCE_ERROR = "inference_error"
FALLBACK_INVALID_SCORES = "invalid_scores"

logger = logging.getLogger(__name__)


class InvalidRerankerScoresError(ValueError):
    """Reranker output cannot define one valid score per candidate."""


@dataclass(frozen=True, slots=True)
class RerankingSearchTrace:
    """Final ranking, intermediate rankings, and request-level diagnostics."""

    results: tuple[RerankedChunk, ...]
    candidate_pool: tuple[HybridRetrievedChunk, ...]
    rrf_top20: tuple[HybridRetrievedChunk, ...]
    fallback_used: bool
    fallback_reason: str | None
    invalid_scores: bool
    candidate_seconds: float
    model_load_seconds: float
    rerank_seconds: float
    total_seconds: float
    reranker_backend: str | None = None
    reranker_endpoint: str | None = None
    rerank_request_seconds: float = 0.0
    rerank_candidate_count: int = 0


class RerankingRetriever(BaseRetriever):
    """Rerank the complete V1.1 Hybrid pool and fall back atomically to RRF."""

    def __init__(
        self,
        *,
        hybrid_retriever: HybridRetriever | None = None,
        reranker: BaseReranker | None = None,
        route_k: int = DEFAULT_HYBRID_ROUTE_TOP_K,
        rrf_report_k: int = DEFAULT_RRF_CANDIDATE_REPORT_K,
    ) -> None:
        _validate_positive_int(route_k, "route_k")
        _validate_positive_int(rrf_report_k, "rrf_report_k")
        self.hybrid_retriever = hybrid_retriever or HybridRetriever()
        self.reranker = reranker or create_reranker()
        self.route_k = route_k
        self.rrf_report_k = rrf_report_k

    def search(
        self,
        query: str,
        k: int = DEFAULT_RERANK_FINAL_TOP_K,
    ) -> list[RerankedChunk]:
        """Return final Top-K chunks while retaining normal BaseRetriever shape."""
        return list(self.search_with_trace(query, k=k).results)

    def search_with_trace(
        self,
        query: str,
        *,
        k: int = DEFAULT_RERANK_FINAL_TOP_K,
    ) -> RerankingSearchTrace:
        """Return final results together with candidate and latency evidence."""
        clean_query = query.strip()
        if not clean_query:
            raise ValueError("query 不能为空")
        _validate_positive_int(k, "k")

        total_started = time.perf_counter()
        candidate_started = time.perf_counter()
        candidate_pool = self.hybrid_retriever.candidate_pool(
            clean_query,
            route_k=self.route_k,
        )
        candidate_seconds = time.perf_counter() - candidate_started
        rrf_top20 = candidate_pool[: self.rrf_report_k]

        if not candidate_pool:
            metadata = _reranker_trace_metadata(
                self.reranker,
                rerank_seconds=0.0,
                candidate_count=0,
            )
            return RerankingSearchTrace(
                results=(),
                candidate_pool=(),
                rrf_top20=(),
                fallback_used=False,
                fallback_reason=None,
                invalid_scores=False,
                candidate_seconds=candidate_seconds,
                model_load_seconds=0.0,
                rerank_seconds=0.0,
                total_seconds=time.perf_counter() - total_started,
                **metadata,
            )

        model_load_seconds = 0.0
        rerank_seconds = 0.0
        load_started: float | None = None
        rerank_started: float | None = None
        try:
            if not self.reranker.is_loaded:
                load_started = time.perf_counter()
                self.reranker.load()
                model_load_seconds = time.perf_counter() - load_started

            rerank_started = time.perf_counter()
            raw_scores = self.reranker.score(clean_query, candidate_pool)
            rerank_seconds = time.perf_counter() - rerank_started
            scores = _validated_scores(raw_scores, expected=len(candidate_pool))
            results = _reranked_results(candidate_pool, scores, k=k)
            fallback_used = False
            fallback_reason = None
            invalid_scores = False
        except RerankerLoadError:
            if model_load_seconds == 0.0 and load_started is not None:
                model_load_seconds = time.perf_counter() - load_started
            fallback_reason = FALLBACK_MODEL_LOAD_ERROR
            invalid_scores = False
            logger.warning(
                "Reranker 加载失败，当前 query 回退到 RRF：query=%r",
                clean_query,
                exc_info=True,
            )
            results = _fallback_results(candidate_pool, k=k, reason=fallback_reason)
            fallback_used = True
        except RerankerInferenceError:
            if rerank_seconds == 0.0 and rerank_started is not None:
                rerank_seconds = time.perf_counter() - rerank_started
            fallback_reason = FALLBACK_INFERENCE_ERROR
            invalid_scores = False
            logger.warning(
                "Reranker 推理失败，当前 query 回退到 RRF：query=%r",
                clean_query,
                exc_info=True,
            )
            results = _fallback_results(candidate_pool, k=k, reason=fallback_reason)
            fallback_used = True
        except InvalidRerankerScoresError:
            fallback_reason = FALLBACK_INVALID_SCORES
            invalid_scores = True
            logger.warning(
                "Reranker score 非法，当前 query 回退到 RRF：query=%r",
                clean_query,
                exc_info=True,
            )
            results = _fallback_results(candidate_pool, k=k, reason=fallback_reason)
            fallback_used = True
        except Exception:
            if rerank_started is not None:
                rerank_seconds = time.perf_counter() - rerank_started
            fallback_reason = FALLBACK_INFERENCE_ERROR
            invalid_scores = False
            logger.warning(
                "Reranker 发生未分类推理异常，当前 query 回退到 RRF：query=%r",
                clean_query,
                exc_info=True,
            )
            results = _fallback_results(candidate_pool, k=k, reason=fallback_reason)
            fallback_used = True

        metadata = _reranker_trace_metadata(
            self.reranker,
            rerank_seconds=rerank_seconds,
            candidate_count=len(candidate_pool),
        )
        return RerankingSearchTrace(
            results=tuple(results),
            candidate_pool=tuple(candidate_pool),
            rrf_top20=tuple(rrf_top20),
            fallback_used=fallback_used,
            fallback_reason=fallback_reason,
            invalid_scores=invalid_scores,
            candidate_seconds=candidate_seconds,
            model_load_seconds=model_load_seconds,
            rerank_seconds=rerank_seconds,
            total_seconds=time.perf_counter() - total_started,
            **metadata,
        )

    def model_record(self) -> dict[str, Any]:
        return self.reranker.model_record()


def _validated_scores(raw_scores: Sequence[object], *, expected: int) -> list[float]:
    if isinstance(raw_scores, (str, bytes, bytearray)):
        raise InvalidRerankerScoresError("score 返回值必须是序列")
    try:
        scores = list(raw_scores)
    except TypeError as exc:
        raise InvalidRerankerScoresError("score 返回值必须是序列") from exc
    if len(scores) != expected:
        raise InvalidRerankerScoresError(
            f"score 数量异常：expected={expected}, actual={len(scores)}"
        )

    validated: list[float] = []
    for index, score in enumerate(scores):
        if isinstance(score, bool) or not isinstance(score, Real):
            raise InvalidRerankerScoresError(f"第 {index} 个 score 不是数值")
        numeric_score = float(score)
        if not math.isfinite(numeric_score):
            raise InvalidRerankerScoresError(f"第 {index} 个 score 不是有限值")
        validated.append(numeric_score)
    return validated


def _reranked_results(
    candidates: Sequence[HybridRetrievedChunk],
    scores: Sequence[float],
    *,
    k: int,
) -> list[RerankedChunk]:
    ranked = sorted(
        zip(candidates, scores, strict=True),
        key=lambda item: (-item[1], item[0].rrf_rank, item[0].chunk_id),
    )
    return [
        _to_reranked_chunk(
            candidate,
            final_rank=final_rank,
            rerank_score=score,
            fallback_used=False,
            fallback_reason=None,
        )
        for final_rank, (candidate, score) in enumerate(ranked[:k], start=1)
    ]


def _fallback_results(
    candidates: Sequence[HybridRetrievedChunk],
    *,
    k: int,
    reason: str,
) -> list[RerankedChunk]:
    return [
        _to_reranked_chunk(
            candidate,
            final_rank=final_rank,
            rerank_score=None,
            fallback_used=True,
            fallback_reason=reason,
        )
        for final_rank, candidate in enumerate(candidates[:k], start=1)
    ]


def _to_reranked_chunk(
    candidate: HybridRetrievedChunk,
    *,
    final_rank: int,
    rerank_score: float | None,
    fallback_used: bool,
    fallback_reason: str | None,
) -> RerankedChunk:
    final_score = candidate.rrf_score if fallback_used else rerank_score
    if final_score is None:  # pragma: no cover - guarded by callers
        raise ValueError("正常重排序结果必须有 rerank_score")
    return RerankedChunk(
        content=candidate.content,
        score=final_score,
        doc_id=candidate.doc_id,
        source=candidate.source,
        page=candidate.page,
        chunk_id=candidate.chunk_id,
        dense_rank=candidate.dense_rank,
        bm25_rank=candidate.bm25_rank,
        rrf_score=candidate.rrf_score,
        rrf_rank=candidate.rrf_rank,
        rerank_score=rerank_score,
        final_rank=final_rank,
        fallback_used=fallback_used,
        fallback_reason=fallback_reason,
    )


def _validate_positive_int(value: int, field_name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field_name} 必须是正整数")


def _reranker_trace_metadata(
    reranker: BaseReranker,
    *,
    rerank_seconds: float,
    candidate_count: int,
) -> dict[str, Any]:
    """Expose backend diagnostics without changing the ranking contract."""
    try:
        record = reranker.model_record()
    except Exception:  # noqa: BLE001 - diagnostics must not break retrieval
        record = {}
    request_seconds = record.get("request_seconds", rerank_seconds)
    try:
        request_seconds = float(request_seconds or 0.0)
    except (TypeError, ValueError):
        request_seconds = rerank_seconds
    return {
        "reranker_backend": record.get("backend_type", record.get("backend")),
        "reranker_endpoint": record.get("endpoint"),
        "rerank_request_seconds": request_seconds,
        "rerank_candidate_count": int(record.get("candidate_count", candidate_count)),
    }
