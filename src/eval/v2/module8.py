"""V2.3 durable HITL stage report.

This is an evaluation/reporting adapter around DurableV23Service. It does
not own persistence or workflow semantics and never overwrites a previous
run directory.
"""

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
from agenticrag.v2.durable import DurableRun, DurableV23Service


DEFAULT_OUTPUT_ROOT = Path("artifacts/eval/v2/module8_v2_3")


def evaluate_module8_start(
    question: str,
    *,
    service: DurableV23Service | None = None,
    config: V2Config | None = None,
    run_id: str | None = None,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    response_language: str | None = None,
) -> dict[str, Any]:
    config = config or V2Config.from_env()
    run_id = run_id or str(uuid4())
    report_dir = output_root / run_id
    if report_dir.exists():
        raise FileExistsError(f"拒绝覆盖 V2.3 report：{report_dir}")
    owns_service = service is None
    service = service or DurableV23Service(config)
    result = service.start(question, response_language=response_language)
    report = build_module8_report(result, config=config, run_id=run_id, question=question)
    predictions = {
        "run_id": run_id,
        "request_id": result.request_id,
        "thread_id": result.thread_id,
        "stage_result": result.stage_result.model_dump(mode="json"),
        "tasks": [
            task.model_dump(mode="json")
            for task in sorted(
                result.state.get("tasks", {}).values(),
                key=lambda item: (item.ordinal, item.id),
            )
        ],
        "evidence": [
            item.model_dump(mode="json") for item in result.state.get("evidence", {}).values()
        ],
    }
    report_dir.mkdir(parents=True, exist_ok=False)
    (report_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (report_dir / "predictions.jsonl").write_text(
        json.dumps(predictions, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    if owns_service:
        service.close()
    return report


def build_module8_report(
    result: DurableRun,
    *,
    config: V2Config,
    run_id: str,
    question: str,
) -> dict[str, Any]:
    tasks = sorted(
        result.state.get("tasks", {}).values(),
        key=lambda item: (item.ordinal, item.id),
    )
    attempts = [
        attempt
        for task in tasks
        for revision in task.query_revisions
        for attempt in revision.retrieval_attempts
    ]
    routes = Counter(
        task.routing_decisions[-1].route
        for task in tasks
        if task.routing_decisions
    )
    metadata = None
    if result.interrupted:
        metadata = {
            "execution_status": "waiting_user",
            "resumable": True,
            "pending_hitl_request_id": (
                result.stage_result.pending_hitl_request.id
                if result.stage_result.pending_hitl_request
                else None
            ),
        }
    report = {
        "report_schema_version": 1,
        "producer": "v2_3_stage_evaluator",
        "run_id": run_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "git_dirty": _git_dirty(),
        "target_stage": "v2_3",
        "request_id": result.request_id,
        "thread_id": result.thread_id,
        "question": question,
        "persistence_backend": "langgraph_sqlite + sqlite:v2_requests",
        "persistence": config.persistence.model_dump(mode="json"),
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
        "start": {
            "interrupted": result.interrupted,
            "metadata": metadata,
            "checkpoint_write_latency_seconds": None,
        },
        "metrics": {
            "task_count": len(tasks),
            "route_counts": dict(routes),
            "interrupt_count": int(result.interrupted),
            "resume_count": 0,
            "accepted_resume_count": 0,
            "rejected_resume_count": 0,
            "affected_task_count": 0,
            "new_query_revision_count": sum(
                max(0, len(task.query_revisions) - 1) for task in tasks
            ),
            "user_clarified_retrieval_count": sum(
                attempt.strategy == "user_clarified" for attempt in attempts
            ),
            "duplicate_resume_count": 0,
            "lease_conflict_count": 0,
            "checkpoint_load_latency_seconds": None,
            "resume_latency_seconds": None,
            "finding_count": sum(task.grounded_finding is not None for task in tasks),
            "final_execution_status": result.stage_result.execution_status,
            "final_answer_outcome": result.stage_result.answer_outcome,
            "technical_failure_count": sum(task.execution_status == "failed" for task in tasks)
            or int(result.stage_result.execution_status == "failed"),
            "degraded_retrieval_count": sum(
                attempt.retrieval_degraded for attempt in attempts
            ),
            "invariant_violation_count": 0,
        },
        "artifact_digest": "",
    }
    report["artifact_digest"] = _report_digest(report)
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


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(prog="agenticrag-eval-v2-module8")
    parser.add_argument("question")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--response-language", choices=("zh", "en"))
    args = parser.parse_args()
    report = evaluate_module8_start(
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
