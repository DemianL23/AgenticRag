"""Machine-readable report schema and serialization."""

from __future__ import annotations

import json
import math
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class SampleReport:
    sample_id: str
    question: str
    reference: str
    generated_answer: str | None
    retrieved_chunk_ids: tuple[str, ...]
    retrieved_contexts: tuple[str, ...]
    metrics: dict[str, float | None]
    metric_reasons: dict[str, str]
    evaluation_error: dict[str, str] | None
    retrieval_trace: dict[str, Any] | None = None

    def to_record(self) -> dict[str, Any]:
        record = asdict(self)
        record["user_input"] = self.question
        record["response"] = self.generated_answer
        record["retrieved_chunk_ids"] = list(self.retrieved_chunk_ids)
        record["retrieved_contexts"] = list(self.retrieved_contexts)
        return record


@dataclass(frozen=True, slots=True)
class RagasReport:
    schema_version: int
    created_at: str
    dataset_path: str
    dataset_size: int
    top_k: int
    generation_model: str
    embedding_model: str
    evaluator_model: str
    evaluator_embedding_model: str
    ragas_version: str
    metrics: tuple[str, ...]
    context_entity_recall: dict[str, Any]
    aggregate_metrics: dict[str, float | None]
    aggregate_metric_counts: dict[str, int]
    successful_samples: int
    failed_samples: int
    samples: tuple[SampleReport, ...]
    report_name: str = "ragas"
    run_id: str | None = None
    git_commit: str | None = None
    resolved_config: dict[str, Any] | None = None
    retrieval_record: dict[str, Any] | None = None
    retrieval_summary: dict[str, Any] | None = None

    def to_record(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "created_at": self.created_at,
            "dataset_path": self.dataset_path,
            "dataset_size": self.dataset_size,
            "top_k": self.top_k,
            "generation_model": self.generation_model,
            "embedding_model": self.embedding_model,
            "evaluator_model": self.evaluator_model,
            "evaluator_embedding_model": self.evaluator_embedding_model,
            "ragas_version": self.ragas_version,
            "metrics": list(self.metrics),
            "context_entity_recall": self.context_entity_recall,
            "aggregate_metrics": self.aggregate_metrics,
            "aggregate_metric_counts": self.aggregate_metric_counts,
            "successful_samples": self.successful_samples,
            "failed_samples": self.failed_samples,
            "samples": [sample.to_record() for sample in self.samples],
            "report_name": self.report_name,
            "run_id": self.run_id,
            "git_commit": self.git_commit,
            "resolved_config": self.resolved_config,
            "retrieval_record": self.retrieval_record,
            "retrieval_summary": self.retrieval_summary,
        }


def aggregate_scores(
    samples: Iterable[SampleReport], metric_names: Iterable[str]
) -> tuple[dict[str, float | None], dict[str, int]]:
    """Average only successful finite scores and report each denominator."""
    samples = tuple(samples)
    aggregates: dict[str, float | None] = {}
    counts: dict[str, int] = {}
    for metric_name in metric_names:
        values = [
            value
            for sample in samples
            if (value := sample.metrics.get(metric_name)) is not None
            and math.isfinite(value)
        ]
        aggregates[metric_name] = sum(values) / len(values) if values else None
        counts[metric_name] = len(values)
    return aggregates, counts


def write_report(report: RagasReport, path: Path) -> None:
    """Atomically serialize a report as UTF-8 JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.tmp")
    temporary_path.write_text(
        json.dumps(report.to_record(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(path)
