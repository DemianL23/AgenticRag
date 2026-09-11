"""Stage-aware evaluation for the V1.2-A reranking retrieval pipeline."""

from __future__ import annotations

import json
import math
import statistics
import time
from collections import Counter
from pathlib import Path
from typing import Any, Protocol, Sequence

from agenticrag.retrieval.reranking_retriever import RerankingSearchTrace
from agenticrag.retrieval.schemas import HybridRetrievedChunk, RerankedChunk

from .retrieval_metrics import mean_reciprocal_rank, recall_at_k, reciprocal_rank
from .retrieval_runner import load_retrieval_dataset


class RerankingRetrieverProtocol(Protocol):
    """Minimum pipeline shape needed by the V1.2 evaluator."""

    def search_with_trace(self, query: str, *, k: int = 5) -> RerankingSearchTrace:
        ...

    def model_record(self) -> dict[str, Any]:
        ...


def evaluate_reranking(
    dataset_path: Path,
    retriever: RerankingRetrieverProtocol,
    *,
    final_k: int = 5,
    dense_baseline_path: Path | None = Path("artifacts/eval/retrieval_report.json"),
    hybrid_baseline_path: Path | None = Path(
        "artifacts/eval/retrieval_hybrid_v1_1_report.json"
    ),
) -> dict[str, Any]:
    """Evaluate final ranking, RRF Top-20, and full union-pool coverage."""
    _validate_positive_int(final_k, "final_k")
    samples = load_retrieval_dataset(Path(dataset_path))
    if not samples:
        raise ValueError(f"评测集不能为空：{dataset_path}")
    candidate_route_k = int(getattr(retriever, "route_k", 20))
    rrf_candidate_k = int(getattr(retriever, "rrf_report_k", 20))
    hybrid_retriever = getattr(retriever, "hybrid_retriever", None)
    rrf_k = int(getattr(hybrid_retriever, "rrf_k", 60))

    run_started = time.perf_counter()
    final_rankings: list[list[str]] = []
    rrf_rankings: list[list[str]] = []
    pool_rankings: list[list[str]] = []
    relevant_rankings: list[list[str]] = []
    query_reports: list[dict[str, Any]] = []
    fallback_reasons: Counter[str] = Counter()
    candidate_seconds: list[float] = []
    model_load_seconds: list[float] = []
    rerank_seconds: list[float] = []
    total_seconds: list[float] = []
    warm_total_seconds: list[float] = []
    candidate_timing_values: dict[str, list[float]] = {
        name: []
        for name in (
            "query_embedding_seconds",
            "dense_search_seconds",
            "bm25_search_seconds",
            "merge_rrf_seconds",
            "candidate_total_seconds",
        )
    }

    for sample in samples:
        trace = retriever.search_with_trace(sample["query"], k=final_k)
        final_ids = [result.chunk_id for result in trace.results]
        rrf_ids = [result.chunk_id for result in trace.rrf_top20]
        pool_ids = [result.chunk_id for result in trace.candidate_pool]
        relevant_ids = sample["relevant_chunk_ids"]

        final_rankings.append(final_ids)
        rrf_rankings.append(rrf_ids)
        pool_rankings.append(pool_ids)
        relevant_rankings.append(relevant_ids)
        if trace.fallback_reason is not None:
            fallback_reasons[trace.fallback_reason] += 1

        candidate_seconds.append(trace.candidate_seconds)
        model_load_seconds.append(trace.model_load_seconds)
        rerank_seconds.append(trace.rerank_seconds)
        total_seconds.append(trace.total_seconds)
        warm_total_seconds.append(trace.total_seconds - trace.model_load_seconds)
        candidate_timing = _candidate_timing_record(trace)
        for name, value in candidate_timing.items():
            candidate_timing_values[name].append(value)

        query_reports.append(
            {
                "id": sample["id"],
                "query": sample["query"],
                "relevant_chunk_ids": relevant_ids,
                "final_results": [
                    _final_result_record(result) for result in trace.results
                ],
                "rrf_top20": [
                    _candidate_record(result) for result in trace.rrf_top20
                ],
                "union_candidate_pool": [
                    _candidate_record(result) for result in trace.candidate_pool
                ],
                "metrics": {
                    **_single_final_metrics(final_ids, relevant_ids, final_k=final_k),
                    f"RRF Recall@{rrf_candidate_k}": recall_at_k(
                        rrf_ids,
                        relevant_ids,
                        k=rrf_candidate_k,
                    ),
                    "Union Pool Recall": recall_at_k(
                        pool_ids,
                        relevant_ids,
                        k=max(1, len(pool_ids)),
                    ),
                },
                "fallback_used": trace.fallback_used,
                "fallback_reason": trace.fallback_reason,
                "invalid_scores": trace.invalid_scores,
                "candidate_timing_seconds": candidate_timing,
                "timing_seconds": {
                    "candidate": trace.candidate_seconds,
                    **candidate_timing,
                    "model_load": trace.model_load_seconds,
                    "rerank_inference": trace.rerank_seconds,
                    "end_to_end": trace.total_seconds,
                    "end_to_end_excluding_model_load": (
                        trace.total_seconds - trace.model_load_seconds
                    ),
                },
                "metadata": sample.get("metadata", {}),
                "revision": sample.get("revision", {}),
            }
        )

    final_metrics = _aggregate_final_metrics(
        final_rankings,
        relevant_rankings,
        final_k=final_k,
    )
    rrf_candidate_metrics = {
        f"Recall@{rrf_candidate_k}": statistics.fmean(
            recall_at_k(retrieved, relevant, k=rrf_candidate_k)
            for retrieved, relevant in zip(
                rrf_rankings, relevant_rankings, strict=True
            )
        )
    }
    pool_recalls = [
        recall_at_k(retrieved, relevant, k=max(1, len(retrieved)))
        for retrieved, relevant in zip(pool_rankings, relevant_rankings, strict=True)
    ]
    pool_sizes = [len(ranking) for ranking in pool_rankings]
    union_pool_metrics = {
        "Recall@full_pool": statistics.fmean(pool_recalls),
        "pool_size": _size_summary(pool_sizes),
    }
    fallback_queries = sum(fallback_reasons.values())
    invalid_score_queries = fallback_reasons.get("invalid_scores", 0)
    run_seconds = time.perf_counter() - run_started

    baseline_comparison = _baseline_comparison(
        samples=samples,
        current_metrics=final_metrics,
        dense_baseline_path=dense_baseline_path,
        hybrid_baseline_path=hybrid_baseline_path,
    )
    freeze_eligible = (
        len(samples) == 47
        and len(query_reports) == 47
        and final_k == 5
        and candidate_route_k == 20
        and rrf_k == 60
        and fallback_queries == 0
        and invalid_score_queries == 0
    )
    return {
        "baseline": "V1.2-A Local BGE Reranker",
        "retriever": "hybrid_rrf_bge_reranker",
        "dataset": str(dataset_path),
        "dataset_size": len(samples),
        "evaluated_queries": len(query_reports),
        "final_k": final_k,
        "candidate_route_k": candidate_route_k,
        "rrf_k": rrf_k,
        "rrf_candidate_k": rrf_candidate_k,
        "final_metrics": final_metrics,
        "rrf_candidate_metrics": rrf_candidate_metrics,
        "union_pool_metrics": union_pool_metrics,
        "candidate_profiling": _candidate_profiling_summary(candidate_timing_values),
        "fallback_queries": fallback_queries,
        "invalid_score_queries": invalid_score_queries,
        "fallback_reasons": dict(sorted(fallback_reasons.items())),
        "latency_seconds": {
            "model_load": sum(model_load_seconds),
            "candidate": _latency_summary(candidate_seconds),
            "rerank_inference": _latency_summary(rerank_seconds),
            "end_to_end": _latency_summary(total_seconds),
            "end_to_end_excluding_model_load": _latency_summary(warm_total_seconds),
            "run_total": run_seconds,
        },
        "model": retriever.model_record(),
        "baseline_comparison": baseline_comparison,
        "optimization_outcome": _optimization_outcome(baseline_comparison),
        "freeze": {
            "eligible": freeze_eligible,
            "label": "V1.2-A experiment baseline",
            "criteria": {
                "all_47_queries_evaluated": (
                    len(samples) == 47 and len(query_reports) == 47
                ),
                "final_k_is_5": final_k == 5,
                "candidate_route_k_is_20": candidate_route_k == 20,
                "rrf_k_is_60": rrf_k == 60,
                "fallback_queries_is_0": fallback_queries == 0,
                "invalid_score_queries_is_0": invalid_score_queries == 0,
            },
        },
        "queries": query_reports,
    }


def _single_final_metrics(
    retrieved: Sequence[str],
    relevant: Sequence[str],
    *,
    final_k: int,
) -> dict[str, float]:
    cutoffs = sorted({cutoff for cutoff in (1, 3, 5) if cutoff <= final_k} | {final_k})
    metrics = {
        f"Final Recall@{cutoff}": recall_at_k(retrieved, relevant, k=cutoff)
        for cutoff in cutoffs
    }
    metrics[f"Final MRR@{final_k}"] = reciprocal_rank(
        retrieved,
        relevant,
        k=final_k,
    )
    return metrics


def _aggregate_final_metrics(
    rankings: Sequence[Sequence[str]],
    relevant_rankings: Sequence[Sequence[str]],
    *,
    final_k: int,
) -> dict[str, float]:
    cutoffs = sorted({cutoff for cutoff in (1, 3, 5) if cutoff <= final_k} | {final_k})
    metrics = {
        f"Recall@{cutoff}": statistics.fmean(
            recall_at_k(retrieved, relevant, k=cutoff)
            for retrieved, relevant in zip(rankings, relevant_rankings, strict=True)
        )
        for cutoff in cutoffs
    }
    metrics[f"MRR@{final_k}"] = mean_reciprocal_rank(
        reciprocal_rank(retrieved, relevant, k=final_k)
        for retrieved, relevant in zip(rankings, relevant_rankings, strict=True)
    )
    return metrics


def _candidate_record(result: HybridRetrievedChunk) -> dict[str, Any]:
    return {
        "rrf_rank": result.rrf_rank,
        "chunk_id": result.chunk_id,
        "score": result.score,
        "doc_id": result.doc_id,
        "source": result.source,
        "page": result.page,
        "dense_rank": result.dense_rank,
        "bm25_rank": result.bm25_rank,
        "rrf_score": result.rrf_score,
    }


def _final_result_record(result: RerankedChunk) -> dict[str, Any]:
    return {
        "final_rank": result.final_rank,
        "chunk_id": result.chunk_id,
        "score": result.score,
        "doc_id": result.doc_id,
        "source": result.source,
        "page": result.page,
        "dense_rank": result.dense_rank,
        "bm25_rank": result.bm25_rank,
        "rrf_rank": result.rrf_rank,
        "rrf_score": result.rrf_score,
        "rerank_score": result.rerank_score,
        "fallback_used": result.fallback_used,
        "fallback_reason": result.fallback_reason,
    }


def _candidate_timing_record(trace: RerankingSearchTrace) -> dict[str, float]:
    """Read the stable candidate-generation timing fields from one trace."""
    return {
        name: float(getattr(trace, name, 0.0))
        for name in (
            "query_embedding_seconds",
            "dense_search_seconds",
            "bm25_search_seconds",
            "merge_rrf_seconds",
            "candidate_total_seconds",
        )
    }


def _candidate_profiling_summary(
    values: dict[str, Sequence[float]],
) -> dict[str, Any]:
    """Aggregate candidate stages and mean per-query time percentages."""
    summaries = {name: _latency_summary(stage_values) for name, stage_values in values.items()}
    totals = values["candidate_total_seconds"]
    percentages = {
        name: _mean_percentage(stage_values, totals)
        for name, stage_values in values.items()
        if name != "candidate_total_seconds"
    }
    return {
        "stages": summaries,
        "average_percentage_of_candidate_total": percentages,
    }


def _mean_percentage(values: Sequence[float], totals: Sequence[float]) -> float:
    if not values:
        return 0.0
    ratios = [
        value / total * 100.0
        for value, total in zip(values, totals, strict=True)
        if total > 0.0
    ]
    return statistics.fmean(ratios) if ratios else 0.0


def _baseline_comparison(
    *,
    samples: Sequence[dict[str, Any]],
    current_metrics: dict[str, float],
    dense_baseline_path: Path | None,
    hybrid_baseline_path: Path | None,
) -> dict[str, Any]:
    comparison: dict[str, Any] = {
        "dense_v0": _load_comparable_baseline(dense_baseline_path, samples),
        "hybrid_v1_1": _load_comparable_baseline(hybrid_baseline_path, samples),
        "reranker_v1_2_a": {"metrics": current_metrics},
    }
    hybrid_metrics = comparison["hybrid_v1_1"].get("metrics")
    comparison["delta_vs_hybrid_v1_1"] = (
        {
            metric: current_metrics[metric] - hybrid_metrics[metric]
            for metric in ("Recall@1", "Recall@3", "Recall@5", "MRR@5")
        }
        if isinstance(hybrid_metrics, dict)
        else None
    )
    return comparison


def _load_comparable_baseline(
    path: Path | None,
    samples: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    if path is None:
        return {"status": "not_configured"}
    if not path.is_file():
        return {"status": "missing", "report": str(path)}

    report = json.loads(path.read_text(encoding="utf-8"))
    queries = report.get("queries")
    if not isinstance(queries, list):
        return {"status": "invalid", "report": str(path)}
    by_id = {query.get("id"): query for query in queries if isinstance(query, dict)}
    expected_ids = [sample["id"] for sample in samples]
    if set(by_id) != set(expected_ids):
        return {
            "status": "dataset_mismatch",
            "report": str(path),
            "expected_queries": len(expected_ids),
            "report_queries": len(by_id),
        }

    rankings: list[list[str]] = []
    relevant_rankings: list[list[str]] = []
    for sample in samples:
        query = by_id[sample["id"]]
        retrieved = query.get("retrieved", [])
        rankings.append(
            [
                item["chunk_id"]
                for item in retrieved
                if isinstance(item, dict) and isinstance(item.get("chunk_id"), str)
            ]
        )
        relevant_rankings.append(sample["relevant_chunk_ids"])
    return {
        "status": "ok",
        "report": str(path),
        "metrics": _aggregate_final_metrics(
            rankings,
            relevant_rankings,
            final_k=5,
        ),
    }


def _optimization_outcome(comparison: dict[str, Any]) -> dict[str, bool | None]:
    deltas = comparison.get("delta_vs_hybrid_v1_1")
    if not isinstance(deltas, dict):
        return {
            "recall_at_5_improved": None,
            "mrr_at_5_improved": None,
            "both_primary_metrics_improved": None,
        }
    recall_improved = deltas["Recall@5"] > 0
    mrr_improved = deltas["MRR@5"] > 0
    return {
        "recall_at_5_improved": recall_improved,
        "mrr_at_5_improved": mrr_improved,
        "both_primary_metrics_improved": recall_improved and mrr_improved,
    }


def _latency_summary(values: Sequence[float]) -> dict[str, float | int]:
    if not values:
        return {"count": 0, "mean": 0.0, "median": 0.0, "p95": 0.0}
    ordered = sorted(values)
    p95_index = max(0, math.ceil(len(ordered) * 0.95) - 1)
    return {
        "count": len(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "p95": ordered[p95_index],
    }

def _size_summary(values: Sequence[int]) -> dict[str, float | int]:
    if not values:
        return {"min": 0, "max": 0, "mean": 0.0, "median": 0.0}
    return {
        "min": min(values),
        "max": max(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
    }


def _validate_positive_int(value: int, field_name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field_name} 必须是正整数")
