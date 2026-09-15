"""V2 answer/outcome evaluation over the frozen 19-QA corpus.

The evaluator reuses the existing RAGAS sample evaluator and numeric metric,
but scores only annotation-supported questions as ordinary RAG answers.
Unsupported computation questions are scored on capability/outcome abstention.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from agenticrag.v2.config import V2Config
from agenticrag.v2.module6 import Module6Service
from eval.numeric_correctness import NUMERIC_CORRECTNESS, numeric_correctness
from eval.ragas.config import RagasEvaluatorConfig
from eval.ragas.dataset import gold_to_reference
from eval.ragas.providers import create_ragas_evaluator
from eval.ragas.report import SampleReport, aggregate_scores

from .planning import DEFAULT_ANNOTATION_PATH, DEFAULT_QA_PATH, load_planning_samples


ANSWER_REPORT_SCHEMA_VERSION = 1
ANSWER_REPORT_PRODUCER = "v2_answer_ragas_evaluator"


async def evaluate_v2_answers(
    *,
    qa_path: Path = DEFAULT_QA_PATH,
    annotation_path: Path = DEFAULT_ANNOTATION_PATH,
    config: V2Config | None = None,
    service: Module6Service | None = None,
    evaluator: Any | None = None,
    evaluator_config: RagasEvaluatorConfig | None = None,
    run_id: str | None = None,
) -> dict[str, Any]:
    config = config or V2Config.from_env()
    service = service or Module6Service(config)
    evaluator_config = evaluator_config or RagasEvaluatorConfig.from_env()
    evaluator = evaluator or create_ragas_evaluator(evaluator_config)
    samples, digests = load_planning_samples(qa_path, annotation_path)
    metric_names = (*evaluator.metric_names, NUMERIC_CORRECTNESS)
    ragas_samples: list[SampleReport] = []
    per_sample: list[dict[str, Any]] = []
    outcome_correct = unsupported_correct = abstention_correct = 0
    unsupported_total = abstention_total = 0

    for sample in samples:
        annotation = sample.annotation
        item: dict[str, Any] = {
            "sample_id": sample.finqa_id,
            "expected_outcome": annotation.expected_outcome,
            "actual_outcome": None,
            "actual_capabilities": [],
            "ragas": None,
            "error": None,
        }
        try:
            result = service.run(sample.question)
            actual_outcome = result.stage_result.answer_outcome
            capabilities = [task.capability for task in result.tasks]
            item["actual_outcome"] = actual_outcome
            item["actual_capabilities"] = capabilities
            outcome_correct += int(actual_outcome == annotation.expected_outcome)
            if annotation.expected_outcome == "unsupported":
                unsupported_total += 1
                abstention_total += 1
                expected_capabilities = {
                    annotation.capability,
                    *(unit.expected_capability for unit in annotation.required_information_units),
                } - {None}
                safe = (
                    actual_outcome == "unsupported"
                    and bool(expected_capabilities)
                    and bool(capabilities)
                    and set(capabilities) <= expected_capabilities
                )
                unsupported_correct += int(safe)
                abstention_correct += int(actual_outcome == "unsupported")
            elif annotation.expected_outcome in {"no_knowledge", "unresolved"}:
                abstention_total += 1
                abstention_correct += int(actual_outcome == annotation.expected_outcome)
            if annotation.expected_outcome in {"complete", "partial"}:
                final = result.stage_result.final_answer
                if final is None:
                    raise ValueError("supported answer sample has no final answer")
                contexts = [
                    result.evidence[evidence_id].content
                    for evidence_id in final.citation_evidence_ids
                ]
                judged = await evaluator.evaluate(
                    user_input=sample.question,
                    reference=gold_to_reference(sample.qa_record["gold"]),
                    response=final.answer,
                    retrieved_contexts=contexts,
                )
                scores = {
                    **judged.scores,
                    NUMERIC_CORRECTNESS: numeric_correctness(
                        gold_to_reference(sample.qa_record["gold"]),
                        final.answer,
                        task_type=sample.qa_record.get("task_type"),
                    ),
                }
                ragas_sample = SampleReport(
                    sample_id=sample.finqa_id,
                    question=sample.question,
                    reference=gold_to_reference(sample.qa_record["gold"]),
                    generated_answer=final.answer,
                    retrieved_chunk_ids=tuple(final.citation_evidence_ids),
                    retrieved_contexts=tuple(contexts),
                    metrics=scores,
                    metric_reasons=judged.reasons,
                    evaluation_error=judged.errors or None,
                )
                ragas_samples.append(ragas_sample)
                item["ragas"] = ragas_sample.to_record()
                if judged.errors:
                    item["error"] = {
                        "type": "ragas_metric_error",
                        "message": json.dumps(judged.errors, ensure_ascii=False)[:1000],
                    }
        except Exception as exc:  # one failed sample must remain auditable
            item["error"] = {
                "type": type(exc).__name__,
                "message": str(exc).replace("\n", " ")[:1000],
            }
        per_sample.append(item)

    aggregate, counts = aggregate_scores(ragas_samples, metric_names)
    payload: dict[str, Any] = {
        "report_schema_version": ANSWER_REPORT_SCHEMA_VERSION,
        "producer": ANSWER_REPORT_PRODUCER,
        "run_id": run_id or str(uuid4()),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "git_dirty": _git_dirty(),
        "dataset": {
            "qa_path": str(qa_path),
            "annotation_path": str(annotation_path),
            "qa_sha256": digests["qa_sha256"],
            "annotation_sha256": digests["annotation_sha256"],
            "sample_count": len(samples),
        },
        "resolved_config": config.resolved_record(),
        "metrics": {
            "supported_subset_ragas": aggregate,
            "supported_subset_metric_counts": counts,
            "outcome_accuracy": outcome_correct / len(samples) if samples else None,
            "unsupported_computation_recall": (
                unsupported_correct / unsupported_total if unsupported_total else None
            ),
            "abstention_correctness": (
                abstention_correct / abstention_total if abstention_total else None
            ),
            "technical_failure_count": sum(item["error"] is not None for item in per_sample),
        },
        "evaluation_incomplete": any(item["error"] is not None for item in per_sample),
        "per_sample": per_sample,
        "artifact_digest": "",
    }
    payload["artifact_digest"] = _digest(payload)
    return payload


def write_answer_report(report: dict[str, Any], path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite V2 answer report: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _digest(payload: dict[str, Any]) -> str:
    canonical = {**payload, "artifact_digest": ""}
    return hashlib.sha256(
        json.dumps(canonical, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _git_dirty() -> bool | None:
    try:
        return bool(subprocess.check_output(["git", "status", "--short"], text=True).strip())
    except (OSError, subprocess.CalledProcessError):
        return None
