"""Final V2 baseline candidate evaluator.

The evaluator deliberately separates real service runs from deterministic
contract scenarios.  Contract scenarios exercise the frozen terminal and
failure vocabulary without pretending that a mock model is a production
quality measurement.  Every output is written to a fresh run directory.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

from pydantic import Field, ValidationError

from agenticrag.v2.config import V2Config, V2PersistenceConfig
from agenticrag.v2.module4 import Module4Service
from agenticrag.v2.module6 import Module6Service
from agenticrag.v2.types import (
    GlobalAnswerOutcome,
    GlobalExecutionStatus,
    RecoveryStrategy,
    Route,
    TargetStage,
    TaskCapability,
)
from agenticrag.v2.durable import DurableV23Service
from agenticrag.v2.schemas import V2Model


DEFAULT_DATASET = Path("eval/datasets/v2_workflow_scenarios.jsonl")
DEFAULT_OUTPUT_ROOT = Path("artifacts/eval/v2/final_baseline_candidate")
DEFAULT_RETRIEVAL_REPORT = Path("artifacts/eval/retrieval_reranker_v1_2_a_report.json")

ScenarioMode = Literal["real", "contract"]
ScenarioLanguage = Literal["zh", "en", "mixed"]
FaultKind = Literal["none", "provider", "timeout", "schema", "retrieval"]


class ScenarioExpected(V2Model):
    execution_status: GlobalExecutionStatus
    answer_outcome: GlobalAnswerOutcome | None = None
    route: Route | None = None
    recovery_strategy: RecoveryStrategy | None = None
    resumable: bool = False
    technical_failure: bool = False

    def validate_contract(self, *, stage: TargetStage) -> None:
        if self.route == "recover" and self.recovery_strategy is None:
            raise ValueError("recover scenario must declare recovery_strategy")
        if self.route != "recover" and self.recovery_strategy is not None:
            raise ValueError("only recover scenario may declare recovery_strategy")
        if self.technical_failure:
            if self.execution_status != "failed" or self.answer_outcome is not None:
                raise ValueError("technical failure must be failed with null outcome")
            if self.resumable:
                raise ValueError("technical failure cannot be resumable")
        if self.execution_status == "waiting_user":
            if stage != "v2_3" or not self.resumable:
                raise ValueError("durable waiting scenario must be resumable v2_3")
            if self.answer_outcome is not None:
                raise ValueError("waiting scenario must have null outcome")
        elif self.resumable:
            raise ValueError("only waiting v2_3 scenarios may be resumable")


class WorkflowScenario(V2Model):
    scenario_id: str = Field(min_length=1)
    stage: TargetStage
    mode: ScenarioMode
    question: str = Field(min_length=1)
    language: ScenarioLanguage
    complexity: Literal["simple", "complex"]
    capability: TaskCapability | Literal["mixed"]
    tags: list[str] = Field(min_length=1)
    fault: FaultKind = "none"
    expected: ScenarioExpected

    def validate_record(self) -> None:
        if len(self.tags) != len(set(self.tags)):
            raise ValueError(f"{self.scenario_id}: duplicate tags")
        self.expected.validate_contract(stage=self.stage)
        if self.fault == "none" and self.expected.technical_failure:
            raise ValueError(f"{self.scenario_id}: fault is required for technical failure")
        if self.fault != "none" and not self.expected.technical_failure:
            raise ValueError(f"{self.scenario_id}: fault requires technical failure expectation")


REQUIRED_COVERAGE_TAGS = (
    "recovery_direct_rewrite",
    "recovery_step_back",
    "recovery_hyde",
    "hitl_clarify",
    "hitl_scope_select",
    "terminal_complete",
    "terminal_partial",
    "terminal_no_knowledge",
    "terminal_unsupported",
    "terminal_unresolved",
    "technical_provider",
    "technical_timeout",
    "technical_schema",
    "technical_retrieval",
    "language_zh",
    "language_en",
    "language_mixed",
    "complexity_simple",
    "complexity_complex",
    "supported",
    "unsupported",
    "supported_unsupported_mixed",
    "cross_process",
)

TWO_CASE_COVERAGE_TAGS = {
    "recovery_direct_rewrite",
    "recovery_step_back",
    "recovery_hyde",
    "hitl_clarify",
    "hitl_scope_select",
    "terminal_complete",
    "terminal_partial",
    "terminal_no_knowledge",
    "terminal_unsupported",
    "terminal_unresolved",
}


def load_workflow_scenarios(path: Path = DEFAULT_DATASET) -> list[WorkflowScenario]:
    """Load and validate the fixed Module 9 JSONL dataset."""
    if not path.is_file():
        raise FileNotFoundError(f"workflow scenario dataset does not exist: {path}")
    records: list[WorkflowScenario] = []
    seen: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, 1):
            if not raw.strip():
                continue
            try:
                record = WorkflowScenario.model_validate(json.loads(raw))
                record.validate_record()
            except (json.JSONDecodeError, ValidationError, ValueError) as exc:
                raise ValueError(f"invalid workflow scenario line {line_number}: {exc}") from exc
            if record.scenario_id in seen:
                raise ValueError(f"duplicate workflow scenario: {record.scenario_id}")
            seen.add(record.scenario_id)
            records.append(record)
    if not records:
        raise ValueError(f"workflow scenario dataset is empty: {path}")
    _validate_coverage(records)
    return records


def _validate_coverage(records: list[WorkflowScenario]) -> None:
    counts = Counter(tag for record in records for tag in record.tags)
    missing = [tag for tag in REQUIRED_COVERAGE_TAGS if counts[tag] < (2 if tag in TWO_CASE_COVERAGE_TAGS else 1)]
    if missing:
        raise ValueError("workflow scenario coverage requires at least two records: " + ", ".join(missing))
    stages = {record.stage for record in records}
    if stages != {"v2_1", "v2_2", "v2_3"}:
        raise ValueError("workflow scenarios must cover v2_1, v2_2, and v2_3")


def scenario_coverage(records: list[WorkflowScenario]) -> dict[str, Any]:
    return {
        "scenario_count": len(records),
        "mode_counts": dict(Counter(record.mode for record in records)),
        "stage_counts": dict(Counter(record.stage for record in records)),
        "tag_counts": dict(sorted(Counter(tag for record in records for tag in record.tags).items())),
    }


def evaluate_baseline(
    *,
    dataset_path: Path = DEFAULT_DATASET,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    run_id: str | None = None,
    mode: Literal["all", "real", "contract"] = "all",
    config: V2Config | None = None,
    retrieval_report: Path = DEFAULT_RETRIEVAL_REPORT,
    cross_process_report: Path | None = None,
) -> dict[str, Any]:
    """Run the V2 final baseline candidate evaluation.

    ``contract`` records are checked deterministically; ``real`` records are
    dispatched to the existing V2.1/V2.2/V2.3 application services.  The
    output directory is immutable: an existing run ID is rejected.
    """
    config = config or V2Config.from_env()
    run_id = run_id or str(uuid4())
    report_dir = output_root / run_id
    if report_dir.exists():
        raise FileExistsError(f"refusing to overwrite baseline candidate: {report_dir}")
    records = load_workflow_scenarios(dataset_path)
    selected = [record for record in records if mode == "all" or record.mode == mode]
    predictions: list[dict[str, Any]] = []
    for scenario in selected:
        if scenario.mode == "contract":
            predictions.append(_contract_prediction(scenario))
        else:
            predictions.append(_real_prediction(scenario, config))

    dataset_digest = _sha256(dataset_path)
    config_digest = _json_digest(config.resolved_record())
    report = _build_report(
        run_id=run_id,
        dataset_path=dataset_path,
        dataset_digest=dataset_digest,
        config=config,
        config_digest=config_digest,
        records=records,
        selected=selected,
        predictions=predictions,
        retrieval_report=retrieval_report,
        cross_process_report=cross_process_report,
    )
    report_dir.mkdir(parents=True, exist_ok=False)
    predictions_path = report_dir / "predictions.jsonl"
    predictions_path.write_text(
        "".join(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n" for item in predictions),
        encoding="utf-8",
    )
    report["digests"]["predictions_sha256"] = _sha256(predictions_path)
    report["digests"]["artifact_manifest_sha256"] = _json_digest(
        {
            "dataset_sha256": report["digests"]["dataset_sha256"],
            "resolved_config_sha256": report["digests"]["resolved_config_sha256"],
            "predictions_sha256": report["digests"]["predictions_sha256"],
        }
    )
    report_path = report_dir / "report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def _contract_prediction(scenario: WorkflowScenario) -> dict[str, Any]:
    scenario.expected.validate_contract(stage=scenario.stage)
    return {
        "scenario_id": scenario.scenario_id,
        "mode": "contract",
        "status": "contract_validated",
        "passed": True,
        "expected": scenario.expected.model_dump(mode="json"),
    }


def _real_prediction(scenario: WorkflowScenario, config: V2Config) -> dict[str, Any]:
    started = datetime.now(timezone.utc)
    try:
        # V2.3 uses a temporary database for candidate evaluation; no runtime
        # SQLite file is ever created in the repository or report directory.
        if scenario.stage == "v2_1":
            result = Module4Service(config).run(scenario.question, response_language=_language(scenario))
        elif scenario.stage == "v2_2":
            result = Module6Service(config).run(scenario.question, response_language=_language(scenario))
        else:
            with tempfile.TemporaryDirectory(prefix="agenticrag-v23-baseline-") as temp_dir:
                durable_config = config.model_copy(
                    update={
                        "persistence": V2PersistenceConfig(
                            **{
                                **config.persistence.model_dump(),
                                "sqlite_path": str(Path(temp_dir) / "v2.sqlite3"),
                            }
                        )
                    }
                )
                service = DurableV23Service(durable_config)
                try:
                    result = service.start(scenario.question, response_language=_language(scenario))
                finally:
                    service.close()
        prediction = _observe_real_result(scenario, result)
    except Exception as exc:  # real providers are an explicit report error
        prediction = {
            "scenario_id": scenario.scenario_id,
            "mode": "real",
            "stage": scenario.stage,
            "status": "technical_failure",
            "passed": False,
            "error": _safe_error(exc),
        }
    prediction["elapsed_seconds"] = (datetime.now(timezone.utc) - started).total_seconds()
    return prediction


def _observe_real_result(scenario: WorkflowScenario, result: Any) -> dict[str, Any]:
    tasks_value = getattr(result, "tasks", None)
    if tasks_value is None:
        tasks_value = result.state.get("tasks", {}).values()
    tasks = list(tasks_value)
    stage = result.stage_result
    evidence_map = getattr(result, "evidence", None)
    if evidence_map is None:
        evidence_map = result.state.get("evidence", {})
    routes = [task.routing_decisions[-1].route for task in tasks if task.routing_decisions]
    failures = [task for task in tasks if task.execution_status == "failed"]
    degraded = sum(
        attempt.retrieval_degraded
        for task in tasks
        for revision in task.query_revisions
        for attempt in revision.retrieval_attempts
    )
    route = routes[0] if len(set(routes)) == 1 and routes else None
    expected = scenario.expected
    invariant_violations = _result_invariant_violations(tasks, evidence_map, stage)
    passed = (
        stage.execution_status == expected.execution_status
        and stage.answer_outcome == expected.answer_outcome
        and (expected.route is None or route == expected.route)
        and (expected.recovery_strategy is None or any(
            decision.recovery_strategy == expected.recovery_strategy
            for task in tasks for decision in task.routing_decisions
        ))
        and not failures
        and not invariant_violations
    )
    stage_error = stage.error.model_dump(mode="json") if stage.error is not None else None
    return {
        "scenario_id": scenario.scenario_id,
        "mode": "real",
        "stage": scenario.stage,
        "status": "completed" if stage.execution_status != "failed" else "technical_failure",
        "passed": passed,
        "observed": {
            "execution_status": stage.execution_status,
            "answer_outcome": stage.answer_outcome,
            "route": route,
            "route_counts": dict(Counter(routes)),
            "task_count": len(tasks),
            "finding_count": sum(task.grounded_finding is not None for task in tasks),
            "degraded_retrieval_count": degraded,
            "technical_failure_count": len(failures) or int(stage.execution_status == "failed"),
            "invariant_violation_count": len(invariant_violations),
            "invariant_violations": invariant_violations,
            "error": stage_error,
        },
        "expected": expected.model_dump(mode="json"),
    }


def _result_invariant_violations(tasks: list[Any], evidence: dict[str, Any], stage: Any) -> list[str]:
    """Check only cross-record contracts observable at the final stage."""
    violations: list[str] = []
    evidence_ids = set(evidence)
    finding_ids: set[str] = set()
    for task in tasks:
        if task.execution_status == "failed" and task.answer_outcome is not None:
            violations.append(f"{task.id}: failed task has answer_outcome")
        if len(task.query_revisions) > 2:
            violations.append(f"{task.id}: query revision budget exceeded")
        for revision in task.query_revisions:
            if len(revision.retrieval_attempts) > 2:
                violations.append(f"{task.id}/{revision.id}: retrieval attempt budget exceeded")
        if task.grounded_finding is not None:
            finding_ids.update(task.grounded_finding.evidence_ids)
            if not set(task.grounded_finding.evidence_ids) <= evidence_ids:
                violations.append(f"{task.id}: finding cites unknown Evidence")
    final = stage.final_answer
    if final is not None and not set(final.citation_evidence_ids) <= finding_ids:
        violations.append("final citations exceed Finding citation union")
    return violations


def _build_report(
    *,
    run_id: str,
    dataset_path: Path,
    dataset_digest: str,
    config: V2Config,
    config_digest: str,
    records: list[WorkflowScenario],
    selected: list[WorkflowScenario],
    predictions: list[dict[str, Any]],
    retrieval_report: Path,
    cross_process_report: Path | None,
) -> dict[str, Any]:
    real_predictions = [item for item in predictions if item["mode"] == "real"]
    contract_predictions = [item for item in predictions if item["mode"] == "contract"]
    real_failures = [item for item in real_predictions if item.get("status") == "technical_failure"]
    scenario_pass = all(item.get("passed") is True for item in predictions)
    v1_gate = _check_retrieval_gate(retrieval_report)
    cross_process_gate = _check_cross_process_gate(cross_process_report)
    hard_gates = {
        "dataset_schema_and_coverage": True,
        "v1_2_retrieval_regression": v1_gate["passed"],
        "baseline_scenarios_terminal_contract": scenario_pass,
        "v2_3_cross_process": cross_process_gate["passed"],
        "schema_provenance_citation_budget_invariants": not any(
            item.get("observed", {}).get("invariant_violation_count", 0)
            for item in real_predictions
        ),
        "ordinary_baseline_zero_unexpected_technical_failures": not real_failures,
    }
    route_counts = Counter(
        item["observed"]["route"]
        for item in real_predictions
        if item.get("observed", {}).get("route")
    )
    stage_metrics: dict[str, dict[str, Any]] = {}
    for stage in ("v2_1", "v2_2", "v2_3"):
        stage_items = [item for item in real_predictions if item.get("stage") == stage]
        stage_metrics[stage] = {
            "scenario_count": len(stage_items),
            "passed_count": sum(item.get("passed") is True for item in stage_items),
            "technical_failure_count": sum(
                item.get("status") == "technical_failure" for item in stage_items
            ),
            "route_counts": dict(
                Counter(
                    item.get("observed", {}).get("route")
                    for item in stage_items
                    if item.get("observed", {}).get("route")
                )
            ),
        }
    return {
        "run_id": run_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "git_dirty": _git_dirty(),
        "target": "v2_final_baseline_candidate",
        "dataset": {
            "path": str(dataset_path),
            "sha256": dataset_digest,
            "scenario_count": len(records),
            "selected_count": len(selected),
            "coverage": scenario_coverage(records),
        },
        "model_identifiers": {
            "decision": {role: getattr(config.decision_models, role).model for role in ("router", "decomposer", "grader", "rewrite", "hitl")},
            "answer": {role: getattr(config.answer_models, role).model for role in ("simple_answer", "finding", "synthesis")},
        },
        "resolved_config": config.resolved_record(),
        "metrics": {
            "scenario_count": len(selected),
            "real_scenario_count": len(real_predictions),
            "contract_scenario_count": len(contract_predictions),
            "scenario_pass_count": sum(item.get("passed") is True for item in predictions),
            "scenario_failure_count": sum(item.get("passed") is not True for item in predictions),
            "technical_failure_count": len(real_failures),
            "route_counts": dict(route_counts),
            "degraded_retrieval_count": sum(item.get("observed", {}).get("degraded_retrieval_count", 0) for item in real_predictions),
            "invariant_violation_count": sum(
                item.get("observed", {}).get("invariant_violation_count", 0)
                for item in real_predictions
            ),
            "planning": {
                "real_stage_counts": {
                    stage: values["scenario_count"] for stage, values in stage_metrics.items()
                }
            },
            "evidence_routing": {
                "route_counts": dict(route_counts),
                "real_stage_metrics": stage_metrics,
            },
            "answer": {
                "finding_count": sum(
                    item.get("observed", {}).get("finding_count", 0)
                    for item in real_predictions
                ),
                "completed_answer_count": sum(
                    item.get("observed", {}).get("answer_outcome") == "complete"
                    for item in real_predictions
                ),
            },
            "runtime": {
                "active_scenario_seconds": sum(
                    float(item.get("elapsed_seconds", 0.0)) for item in real_predictions
                ),
                "degraded_retrieval_count": sum(
                    item.get("observed", {}).get("degraded_retrieval_count", 0)
                    for item in real_predictions
                ),
            },
        },
        "hard_gates": hard_gates,
        "hard_gate_details": {
            "v1_2_retrieval_regression": v1_gate,
            "v2_3_cross_process": cross_process_gate,
        },
        "errors": [
            item.get("error") or item.get("observed", {}).get("error")
            for item in real_predictions
            if item.get("error") or item.get("observed", {}).get("error")
        ],
        "digests": {
            "dataset_sha256": dataset_digest,
            "resolved_config_sha256": config_digest,
        },
    }


def _check_retrieval_gate(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"passed": False, "status": "not_found", "path": str(path)}
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
        passed = (
            report.get("dataset_size") == 47
            and report.get("evaluated_queries") == 47
            and report.get("fallback_queries", 0) == 0
            and report.get("invalid_score_queries", 0) == 0
        )
        return {"passed": passed, "status": "checked", "path": str(path), "sha256": _sha256(path)}
    except (OSError, json.JSONDecodeError) as exc:
        return {"passed": False, "status": "invalid", "path": str(path), "error": _safe_error(exc)}


def _check_cross_process_gate(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {"passed": False, "status": "not_provided"}
    if not path.is_file():
        return {"passed": False, "status": "not_found", "path": str(path)}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return {"passed": payload.get("passed") is True, "status": "checked", "path": str(path), "sha256": _sha256(path)}
    except (OSError, json.JSONDecodeError) as exc:
        return {"passed": False, "status": "invalid", "path": str(path), "error": _safe_error(exc)}


def _language(scenario: WorkflowScenario) -> str | None:
    return scenario.language if scenario.language in {"zh", "en"} else None


def _json_digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _safe_error(exc: Exception) -> dict[str, str]:
    message = str(exc).replace("\n", " ")[:1500]
    return {"exception_type": type(exc).__name__, "message": message}


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


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(prog="agenticrag-eval-v2-final-baseline")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--mode", choices=("all", "real", "contract"), default="all")
    parser.add_argument("--retrieval-report", type=Path, default=DEFAULT_RETRIEVAL_REPORT)
    parser.add_argument("--cross-process-report", type=Path)
    args = parser.parse_args()
    report = evaluate_baseline(
        dataset_path=args.dataset,
        output_root=args.output_root,
        mode=args.mode,
        retrieval_report=args.retrieval_report,
        cross_process_report=args.cross_process_report,
    )
    print(json.dumps({"run_id": report["run_id"], "output": str(args.output_root / report["run_id"]), "hard_gates": report["hard_gates"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
