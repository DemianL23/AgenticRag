"""V2.2 Stage Report runner for one non-persistent application request."""

from __future__ import annotations

import hashlib
import json
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from agenticrag.v2.config import V2Config
from agenticrag.v2.module6 import Module6Service

from .audit import audit_v2_result


DEFAULT_OUTPUT_ROOT = Path("artifacts/eval/v2/module6_v2_2")


def evaluate_module6(
    question: str,
    *,
    service: Module6Service | None = None,
    config: V2Config | None = None,
    run_id: str | None = None,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    response_language: str | None = None,
) -> dict[str, Any]:
    config = config or V2Config.from_env()
    run_id = run_id or str(uuid4())
    report_dir = output_root / run_id
    if report_dir.exists():
        raise FileExistsError(f"拒绝覆盖 V2.2 report：{report_dir}")
    service = service or Module6Service(config)
    result = service.run(question, response_language=response_language)
    violations = check_module6_invariants(result)
    audit = audit_v2_result(
        list(result.tasks), result.evidence, result.stage_result, config=config
    )
    tasks = list(result.tasks)
    routes = Counter(
        task.routing_decisions[-1].route
        for task in tasks
        if task.routing_decisions
    )
    recovery_count = sum(
        1
        for task in tasks
        for revision in task.query_revisions
        if len(revision.retrieval_attempts) == 2
    )
    report = {
        "report_schema_version": 1,
        "producer": "v2_2_stage_evaluator",
        "run_id": run_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "git_dirty": _git_dirty(),
        "target_stage": "v2_2",
        "question": question,
        "resolved_config": config.resolved_record(),
        "model_configs": {
            "router": config.decision_models.router.model_dump(mode="json"),
            "decomposer": config.decision_models.decomposer.model_dump(mode="json"),
            "grader": config.decision_models.grader.model_dump(mode="json"),
            "rewrite": config.decision_models.rewrite.model_dump(mode="json"),
            "hitl": config.decision_models.hitl.model_dump(mode="json"),
            "simple_answer": config.answer_models.simple_answer.model_dump(mode="json"),
            "finding": config.answer_models.finding.model_dump(mode="json"),
            "synthesis": config.answer_models.synthesis.model_dump(mode="json"),
        },
        "metrics": {
            "task_count": len(tasks),
            "route_counts": dict(routes),
            "recovery_count": recovery_count,
            "finding_count": sum(task.grounded_finding is not None for task in tasks),
            "global_execution_status": result.stage_result.execution_status,
            "global_answer_outcome": result.stage_result.answer_outcome,
            "technical_failure_count": sum(task.execution_status == "failed" for task in tasks)
            or int(result.stage_result.execution_status == "failed"),
            "degraded_retrieval_count": sum(
                attempt.retrieval_degraded
                for task in tasks
                for revision in task.query_revisions
                for attempt in revision.retrieval_attempts
            ),
            "invariant_violation_count": max(
                len(violations), audit.schema_invariant_violation_count
            ),
            "schema_invariant_violation_count": max(
                len(violations), audit.schema_invariant_violation_count
            ),
            "provenance_violation_count": audit.provenance_violation_count,
            "citation_violation_count": audit.citation_violation_count,
            "budget_violation_count": audit.budget_violation_count,
        },
        "invariant_violations": violations,
        "artifact_digest": "",
    }
    report["artifact_digest"] = _report_digest(report)
    predictions = {
        "run_id": run_id,
        "question": question,
        "stage_result": result.stage_result.model_dump(mode="json"),
        "tasks": [task.model_dump(mode="json") for task in tasks],
        "evidence": [item.model_dump(mode="json") for item in result.evidence.values()],
        "retrieval_results": {
            task_id: item.model_dump(mode="json")
            for task_id, item in result.retrieval_results.items()
        },
    }
    report_dir.mkdir(parents=True, exist_ok=False)
    (report_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (report_dir / "predictions.jsonl").write_text(
        json.dumps(predictions, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return report


def _report_digest(report: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            {**report, "artifact_digest": ""},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()


def check_module6_invariants(result: Any) -> list[str]:
    tasks = list(result.tasks)
    violations: list[str] = []
    if [task.id for task in tasks] != [task.id for task in sorted(tasks, key=lambda item: (item.ordinal, item.id))]:
        violations.append("task order is not deterministic")
    evidence_ids = set(result.evidence)
    for task in tasks:
        if task.execution_status == "failed" and task.answer_outcome is not None:
            violations.append(f"{task.id}: failed task has answer_outcome")
        if task.answer_outcome == "unsupported":
            if task.grade_records:
                violations.append(f"{task.id}: unsupported task has GradeRecord")
            if task.query_revisions:
                violations.append(f"{task.id}: unsupported task called retrieval")
        if task.grounded_finding is not None:
            if task.grounded_finding.task_id != task.id:
                violations.append(f"{task.id}: finding task_id mismatch")
            if not set(task.grounded_finding.evidence_ids) <= evidence_ids:
                violations.append(f"{task.id}: finding cites unknown Evidence")
    final = result.stage_result.final_answer
    if final is not None:
        valid_finding_ids = {
            evidence_id
            for task in tasks
            if task.grounded_finding
            for evidence_id in task.grounded_finding.evidence_ids
        }
        if not set(final.citation_evidence_ids) <= valid_finding_ids:
            violations.append("final citations exceed Finding citation union")
    return violations


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(prog="agenticrag-eval-v2-module6")
    parser.add_argument("question")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--response-language", choices=("zh", "en"))
    args = parser.parse_args()
    report = evaluate_module6(
        args.question,
        output_root=args.output_root,
        response_language=args.response_language,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


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
