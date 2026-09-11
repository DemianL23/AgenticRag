"""Benchmark query embedding cold and warm latency without changing retrieval."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from pathlib import Path
from typing import Any, Sequence

from agenticrag.rag.integrations.embeddings import EmbeddingConfig, create_embeddings

DEFAULT_QUERY = "什么是 Agentic RAG？"
DEFAULT_WARM_ITERATIONS = 20
DEFAULT_OUTPUT = Path("artifacts/profiling/query_embedding_qwen3_0_6b.json")


def benchmark_query_embedding(
    *,
    config: EmbeddingConfig | None = None,
    query: str = DEFAULT_QUERY,
    warm_iterations: int = DEFAULT_WARM_ITERATIONS,
) -> dict[str, Any]:
    """Measure model construction, first query, and subsequent warm queries."""
    if not query.strip():
        raise ValueError("query 不能为空")
    _validate_positive_int(warm_iterations, "warm_iterations")

    config = config or EmbeddingConfig.from_env()
    config.validate()

    load_started = time.perf_counter()
    embeddings = create_embeddings(config)
    model_load_seconds = time.perf_counter() - load_started

    cold_started = time.perf_counter()
    embeddings.embed_query(query)
    cold_query_seconds = time.perf_counter() - cold_started

    warm_seconds: list[float] = []
    for _ in range(warm_iterations):
        warm_started = time.perf_counter()
        embeddings.embed_query(query)
        warm_seconds.append(time.perf_counter() - warm_started)

    return {
        "benchmark": "query_embedding_latency",
        "embedding_model": config.model_name,
        "embedding_config": config.to_record(),
        "query": query,
        "warm_iterations": warm_iterations,
        "timing_seconds": {
            "model_load_seconds": model_load_seconds,
            "cold_first_query_embedding_seconds": cold_query_seconds,
            "cold_start_seconds": model_load_seconds + cold_query_seconds,
            "warm_query_embedding_seconds": _latency_summary(warm_seconds),
        },
        "warm_latency_excludes_model_load": True,
    }


def main() -> None:
    args = _parse_args()
    report = benchmark_query_embedding(
        query=args.query,
        warm_iterations=args.warm_iterations,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"报告已写入：{args.output}")


def _latency_summary(values: Sequence[float]) -> dict[str, float | int]:
    ordered = sorted(values)
    p95_index = max(0, math.ceil(len(ordered) * 0.95) - 1)
    return {
        "count": len(ordered),
        "mean": statistics.fmean(ordered),
        "median": statistics.median(ordered),
        "p95": ordered[p95_index],
    }


def _validate_positive_int(value: int, field_name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field_name} 必须是正整数")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="测量当前 Embedding 模型的 query cold/warm latency。"
    )
    parser.add_argument("--query", default=DEFAULT_QUERY, help="测试 query")
    parser.add_argument(
        "--warm-iterations",
        type=int,
        default=DEFAULT_WARM_ITERATIONS,
        help="warm 调用次数，默认 20",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()
