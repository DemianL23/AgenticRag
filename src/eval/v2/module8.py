"""V2.3 durable HITL stage report.

This is an evaluation/reporting adapter around DurableV23Service. It does
not own persistence or workflow semantics and never overwrites a previous
run directory.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

from pydantic import Field

from agenticrag.v2.config import V2Config
from agenticrag.v2.durable import DurableRun, DurableV23Service
from agenticrag.v2.schemas import V2Model

from .audit import audit_v2_result


DEFAULT_OUTPUT_ROOT = Path("artifacts/eval/v2/module8_v2_3")


class InvocationTelemetry(V2Model):
    retrieval_calls: int = Field(default=0, ge=0)
    grader_calls: int = Field(default=0, ge=0)
    finding_calls: int = Field(default=0, ge=0)
    synthesis_calls: int = Field(default=0, ge=0)
    hitl_calls: int = Field(default=0, ge=0)


class CrossProcessInvocation(V2Model):
    name: str
    passed: bool
    request_id: str
    thread_id: str
    invocation_index: int = Field(ge=1)
    command: list[str] = Field(min_length=1)
    exit_code: int
    pid: int | None = None
    business_state_before_digest: str | None = Field(
        default=None, pattern=r"^[0-9a-f]{64}$"
    )
    business_state_after_digest: str | None = Field(
        default=None, pattern=r"^[0-9a-f]{64}$"
    )
    telemetry: InvocationTelemetry = Field(default_factory=InvocationTelemetry)
    result: dict[str, Any]


class PersistenceRuntimeOverride(V2Model):
    sqlite_path: str


class EvaluationRuntimeOverrides(V2Model):
    persistence: PersistenceRuntimeOverride


class CrossProcessEvidence(V2Model):
    schema_version: Literal[1] = 1
    producer: Literal["module8_cross_process_acceptance"] = (
        "module8_cross_process_acceptance"
    )
    run_id: str
    git_commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    request_id: str
    thread_id: str
    steps: list[CrossProcessInvocation] = Field(min_length=4)
    negative_contracts: list[CrossProcessInvocation] = Field(min_length=5)
    artifact_digest: str = Field(pattern=r"^[0-9a-f]{64}$")


def evaluate_module8_cross_process_acceptance(
    *,
    config: V2Config | None = None,
    run_id: str | None = None,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    sqlite_path: Path | None = None,
) -> dict[str, Any]:
    """Produce immutable cross-process evidence and its V2.3 stage report."""

    base_config = config or V2Config.from_env()
    run_id = run_id or str(uuid4())
    report_dir = output_root / run_id
    if report_dir.exists():
        raise FileExistsError(f"拒绝覆盖 V2.3 cross-process report：{report_dir}")
    report_dir.mkdir(parents=True, exist_ok=False)
    database = sqlite_path or report_dir / "acceptance.sqlite3"
    runtime_config = base_config.model_copy(
        update={
            "persistence": base_config.persistence.model_copy(
                update={"sqlite_path": str(database)}
            )
        }
    )
    config_path = report_dir / "worker_config.json"
    config_path.write_text(
        json.dumps(runtime_config.model_dump(mode="json"), ensure_ascii=False),
        encoding="utf-8",
    )
    request_id = str(uuid4())
    invocation_index = 0

    def invoke(name: str, operation: str, target: str) -> CrossProcessInvocation:
        nonlocal invocation_index
        invocation_index += 1
        command = [
            sys.executable,
            "-m",
            "eval.v2.module8_worker",
            operation,
            "--db",
            str(database),
            "--config-json",
            str(config_path),
            "--request-id",
            target,
        ]
        completed = subprocess.run(command, capture_output=True, text=True, check=False)
        try:
            result = json.loads(completed.stdout)
        except json.JSONDecodeError:
            result = {
                "parse_error": completed.stdout[-1000:],
                "stderr": completed.stderr[-1000:],
            }
        return CrossProcessInvocation(
            name=name,
            passed=_cross_invocation_passed(name, completed.returncode, result),
            request_id=str(result.get("request_id", target)),
            thread_id=str(result.get("thread_id", target)),
            invocation_index=invocation_index,
            command=command,
            exit_code=completed.returncode,
            pid=result.get("pid"),
            business_state_before_digest=result.get(
                "business_state_before_digest"
            ),
            business_state_after_digest=result.get("business_state_after_digest"),
            telemetry=InvocationTelemetry.model_validate(
                result.get("telemetry", {})
            ),
            result=result,
        )

    start = invoke("start", "start", request_id)
    waiting = invoke("status_waiting", "status", request_id)
    invalid = invoke("invalid_payload", "invalid", request_id)
    resume = invoke("resume", "resume", request_id)
    duplicate = invoke("duplicate_resume", "duplicate", request_id)
    stale = invoke("stale_resume", "stale", request_id)
    completed = invoke("status_completed", "status", request_id)
    expired = invoke("expired_checkpoint", "expired", str(uuid4()))
    lease = invoke("lease_recovery", "lease_recovery", str(uuid4()))
    steps = [start, waiting, resume, completed]
    negative_contracts = [invalid, duplicate, stale, expired, lease]
    evidence_payload = CrossProcessEvidence(
        run_id=run_id,
        git_commit=_git_commit() or "0" * 40,
        request_id=request_id,
        thread_id=request_id,
        steps=steps,
        negative_contracts=negative_contracts,
        artifact_digest="0" * 64,
    ).model_dump(mode="json")
    evidence_payload["artifact_digest"] = hashlib.sha256(
        json.dumps(
            {
                key: value
                for key, value in evidence_payload.items()
                if key != "artifact_digest"
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    evidence = CrossProcessEvidence.model_validate(evidence_payload)
    cross_path = report_dir / "cross_process_acceptance.json"
    cross_path.write_text(
        json.dumps(evidence.model_dump(mode="json"), ensure_ascii=False, indent=2)
        + "\n",
        encoding="utf-8",
    )
    written_evidence = CrossProcessEvidence.model_validate_json(
        cross_path.read_text(encoding="utf-8")
    )
    if written_evidence.artifact_digest != hashlib.sha256(
        json.dumps(
            {
                key: value
                for key, value in written_evidence.model_dump(mode="json").items()
                if key != "artifact_digest"
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest():
        raise ValueError("written cross-process artifact digest verification failed")
    invocation_contracts_passed = all(
        item.passed for item in [*steps, *negative_contracts]
    )
    try:
        final_audit = _audit_from_completed_invocation(completed)
    except (TypeError, ValueError):
        final_audit = None
    audit_clear = bool(
        final_audit is not None
        and not any(final_audit.model_dump(mode="json").values())
    )
    all_passed = invocation_contracts_passed and audit_clear
    runtime_overrides = EvaluationRuntimeOverrides(
        persistence=PersistenceRuntimeOverride(sqlite_path=str(database))
    )
    stage_report = {
        "report_schema_version": 1,
        "producer": "v2_3_stage_evaluator",
        "run_id": run_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "git_dirty": _git_dirty(),
        "target_stage": "v2_3",
        "request_id": request_id,
        "thread_id": request_id,
        # Baseline identity excludes the ephemeral acceptance database.  The
        # worker still executes with ``runtime_config`` below, while this
        # report remains directly consumable by the final baseline loader.
        "resolved_config": base_config.resolved_record(),
        "runtime_overrides": runtime_overrides.model_dump(mode="json"),
        "cross_process_acceptance_ref": str(cross_path),
        "cross_process_acceptance_digest": _sha256(cross_path),
        "metrics": {
            "interrupt_count": int(start.passed),
            "resume_count": 1,
            "accepted_resume_count": int(resume.passed),
            "rejected_resume_count": sum(
                item.passed for item in (invalid, stale)
            ),
            "affected_task_count": int(resume.result.get("affected_task_count", 0)),
            "new_query_revision_count": int(
                resume.result.get("new_query_revision_count", 0)
            ),
            "user_clarified_retrieval_count": int(
                resume.result.get("user_clarified_retrieval_count", 0)
            ),
            "duplicate_resume_count": int(duplicate.passed),
            "lease_conflict_count": 0,
            "process_invocation_count": invocation_index,
            "final_execution_status": completed.result.get("execution_status"),
            "final_answer_outcome": completed.result.get("answer_outcome"),
            "technical_failure_count": (
                int(completed.result.get("technical_failure_count", 0))
                if invocation_contracts_passed
                else 1
            ),
            "degraded_retrieval_count": (
                final_audit.retrieval_degraded_queries_count
                if final_audit is not None
                else None
            ),
            "invariant_violation_count": (
                final_audit.schema_invariant_violation_count
                if final_audit is not None
                else None
            ),
            "schema_invariant_violation_count": (
                final_audit.schema_invariant_violation_count
                if final_audit is not None
                else None
            ),
            "provenance_violation_count": (
                final_audit.provenance_violation_count
                if final_audit is not None
                else None
            ),
            "citation_violation_count": (
                final_audit.citation_violation_count
                if final_audit is not None
                else None
            ),
            "budget_violation_count": (
                final_audit.budget_violation_count
                if final_audit is not None
                else None
            ),
        },
        "evaluation_incomplete": not (
            invocation_contracts_passed and final_audit is not None
        ),
        "artifact_digest": "",
    }
    stage_report["artifact_digest"] = _report_digest(stage_report)
    stage_path = report_dir / "stage_v2_3.json"
    stage_path.write_text(
        json.dumps(stage_report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    written_stage = json.loads(stage_path.read_text(encoding="utf-8"))
    if written_stage.get("artifact_digest") != _report_digest(written_stage):
        raise ValueError("written V2.3 stage artifact digest verification failed")
    return {
        "cross_process_evidence": evidence.model_dump(mode="json"),
        "cross_process_path": str(cross_path),
        "stage_report": stage_report,
        "stage_report_path": str(stage_path),
        "all_passed": all_passed,
    }


def _audit_from_completed_invocation(
    completed: CrossProcessInvocation,
) -> "AuditCounts":
    from .audit import AuditCounts

    if completed.name != "status_completed":
        raise ValueError("final audit must come from status_completed")
    if completed.result.get("audit_source") != "durable_checkpoint_state":
        raise ValueError("final audit is not derived from durable checkpoint state")
    return AuditCounts.model_validate(completed.result.get("audit_counts"))


def _cross_invocation_passed(
    name: str, exit_code: int, result: dict[str, Any]
) -> bool:
    if exit_code != 0:
        return False
    if name == "start":
        return result.get("execution_status") == "waiting_user" and result.get("resumable") is True
    if name == "status_waiting":
        return result.get("execution_status") == "waiting_user" and result.get("resumable") is True
    if name == "resume":
        return result.get("execution_status") == "completed" and result.get("answer_outcome") == "complete"
    if name == "status_completed":
        return result.get("execution_status") == "completed" and result.get("resumable") is False
    if name == "invalid_payload":
        return (
            result.get("error_code") == "resume_payload_invalid"
            and result.get("execution_status") == "waiting_user"
            and _negative_invocation_has_zero_side_effects(result)
        )
    if name == "duplicate_resume":
        return (
            result.get("execution_status") == "completed"
            and result.get("new_query_revision_count") == 1
            and result.get("hitl_rounds") == 1
            and _negative_invocation_has_zero_side_effects(result)
        )
    if name == "stale_resume":
        return (
            result.get("error_code") == "resume_conflict"
            and _negative_invocation_has_zero_side_effects(result)
        )
    if name == "expired_checkpoint":
        return result.get("error_code") == "checkpoint_expired"
    if name == "lease_recovery":
        return result.get("execution_status") == "completed" and result.get("answer_outcome") == "complete"
    return False


def _negative_invocation_has_zero_side_effects(result: dict[str, Any]) -> bool:
    telemetry = result.get("telemetry")
    before = result.get("business_state_before_digest")
    after = result.get("business_state_after_digest")
    return bool(
        isinstance(telemetry, dict)
        and not any(telemetry.values())
        and isinstance(before, str)
        and before
        and before == after
    )


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
    audit = audit_v2_result(
        tasks,
        result.state.get("evidence", {}),
        result.stage_result,
        config=config,
        state=result.state,
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
            "invariant_violation_count": audit.schema_invariant_violation_count,
            "schema_invariant_violation_count": audit.schema_invariant_violation_count,
            "provenance_violation_count": audit.provenance_violation_count,
            "citation_violation_count": audit.citation_violation_count,
            "budget_violation_count": audit.budget_violation_count,
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


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(prog="agenticrag-eval-v2-module8")
    parser.add_argument("question", nargs="?")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--response-language", choices=("zh", "en"))
    parser.add_argument("--cross-process", action="store_true")
    parser.add_argument("--sqlite-path", type=Path)
    args = parser.parse_args()
    if args.cross_process:
        result = evaluate_module8_cross_process_acceptance(
            output_root=args.output_root,
            sqlite_path=args.sqlite_path,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
    if not args.question:
        parser.error("question is required unless --cross-process is used")
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
