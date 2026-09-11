"""Benchmark local and remote document embedding on real corpus chunks."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

from agenticrag.ingestion.chunk import load_documents_jsonl
from agenticrag.rag.integrations.embeddings import EmbeddingConfig, create_embeddings

DEFAULT_CHUNKS_DIR = Path("artifacts/chunks/pymupdf/v0")
DEFAULT_SAMPLE_SIZE = 100
DEFAULT_OUTPUT = Path("artifacts/profiling/document_embedding_benchmark_100chunks.json")


def benchmark_document_embeddings(
    *,
    chunks_dir: Path = DEFAULT_CHUNKS_DIR,
    sample_size: int = DEFAULT_SAMPLE_SIZE,
) -> dict[str, Any]:
    if isinstance(sample_size, bool) or not isinstance(sample_size, int) or sample_size <= 0:
        raise ValueError("sample_size 必须是正整数")
    documents = _load_chunks(chunks_dir)[:sample_size]
    if len(documents) < sample_size:
        raise ValueError(f"chunk 数量不足：requested={sample_size}, available={len(documents)}")
    texts = [document.page_content for document in documents]
    base_config = EmbeddingConfig.from_env()
    results = {
        "local": _benchmark_backend(replace(base_config, backend="local"), texts),
        "remote": _benchmark_backend(replace(base_config, backend="remote"), texts),
    }
    local_total = results["local"]["total_seconds"]
    remote_total = results["remote"]["total_seconds"]
    results["speedup"] = local_total / remote_total if remote_total else None
    return {
        "benchmark": "local_vs_remote_document_embedding",
        "chunks_dir": str(chunks_dir),
        "sample_size": sample_size,
        "batch_size": {
            "local": base_config.batch_size,
            "remote": results["remote"]["config"]["batch_size"],
        },
        "results": results,
    }


def _benchmark_backend(config: EmbeddingConfig, texts: list[str]) -> dict[str, Any]:
    started = time.perf_counter()
    embeddings = create_embeddings(config)
    model_load_seconds = time.perf_counter() - started
    batch_size = config.batch_size
    if config.backend == "remote":
        batch_size = int(embeddings.remote_config.batch_size)
        # Warm the tokenizer used by the fail-fast max-length guard outside timing.
        embeddings.document_token_lengths(texts)

    batch_seconds: list[float] = []
    started_total = time.perf_counter()
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        started_batch = time.perf_counter()
        embeddings.embed_documents(batch)
        batch_seconds.append(time.perf_counter() - started_batch)
    total_seconds = time.perf_counter() - started_total
    backend_record = getattr(embeddings, "model_record", None)
    resolved_config = config.to_record()
    if callable(backend_record):
        resolved_config.update(backend_record())
    return {
        "config": {
            **resolved_config,
            "batch_size": batch_size,
        },
        "model_load_seconds": model_load_seconds,
        "total_seconds": total_seconds,
        "throughput_chunks_per_second": len(texts) / total_seconds if total_seconds else None,
        "batch_count": len(batch_seconds),
        "mean_batch_latency_seconds": statistics.fmean(batch_seconds),
        "median_batch_latency_seconds": statistics.median(batch_seconds),
        "p95_batch_latency_seconds": _percentile(batch_seconds, 0.95),
    }


def _load_chunks(chunks_dir: Path) -> list[Any]:
    files = sorted(Path(chunks_dir).glob("doc_*.jsonl"))
    if not files:
        raise FileNotFoundError(f"chunk 目录中没有 doc_*.jsonl：{chunks_dir}")
    return [document for path in files for document in load_documents_jsonl(path)]


def _percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def main() -> None:
    args = _parse_args()
    report = benchmark_document_embeddings(
        chunks_dir=args.chunks_dir,
        sample_size=args.sample_size,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["results"], ensure_ascii=False))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="比较本地和远程 document embedding 性能。")
    parser.add_argument("--chunks-dir", type=Path, default=DEFAULT_CHUNKS_DIR)
    parser.add_argument("--sample-size", type=int, default=DEFAULT_SAMPLE_SIZE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


if __name__ == "__main__":
    main()
