"""Final V2 baseline candidate evaluator.

Contract scenarios execute the frozen V2 graphs with deterministic
collaborators.  This module is evaluation-only and does not add scenario
branches to the production runtime.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import tempfile
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

from pydantic import Field, model_validator

from agenticrag.v2.config import V2Config, V2PersistenceConfig
from agenticrag.v2.durable import DurableV23Service
from agenticrag.v2.module4 import Module4Service
from agenticrag.v2.module6 import Module6Service
from agenticrag.v2.schemas import V2Model
from agenticrag.v2.types import (
    GlobalAnswerOutcome,
    GlobalExecutionStatus,
    HITLAction,
    RecoveryStrategy,
    Route,
    TargetStage,
    TaskCapability,
)
from .contract_harness import run_contract_scenario
from .planning import load_planning_samples
from eval.retrieval_runner import load_retrieval_dataset

DEFAULT_DATASET = Path("eval/datasets/v2_workflow_scenarios.jsonl")
DEFAULT_OUTPUT_ROOT = Path("artifacts/eval/v2/final_baseline_candidate")
DEFAULT_RETRIEVAL_REPORT = Path("artifacts/eval/retrieval_reranker_v1_2_a_report.json")
RETRIEVAL_DATASET = Path("eval/datasets/retrieval_eval_v2.jsonl")
QA_DATASET = Path("qa.jsonl")
ANNOTATION_DATASET = Path("eval/datasets/v2_qa_annotations.jsonl")
MODULE4_GOLD_DATASET = Path("eval/datasets/v2_module4_grade_route_gold.jsonl")
FROZEN_RETRIEVAL_DATASET_SHA256 = "61b366a7cac645df75b4549e6bd940b62757755974c0afcfa28a20e67196e2e5"
FROZEN_RETRIEVAL_METRICS = {
    "Recall@1": 0.28900709219858156,
    "Recall@3": 0.549645390070922,
    "Recall@5": 0.6028368794326241,
    "Recall@20": 0.8404255319148937,
    "MRR@5": 0.5273049645390071,
}

ScenarioMode = Literal["real", "contract"]
ScenarioLanguage = Literal["zh", "en", "mixed"]
FaultKind = Literal["none", "provider", "timeout", "schema", "retrieval"]


class ScenarioFixture(V2Model):
    deterministic_service: Literal["frozen_graph"] = "frozen_graph"
    fault: FaultKind = "none"
    route_sequence: list[Route] = Field(default_factory=list)
    recovery_strategies: list[RecoveryStrategy] = Field(default_factory=list)
    hitl_action: HITLAction | None = None
    resume_request: dict[str, Any] | None = None
    cross_process_reference: str | None = None


class ScenarioExpected(V2Model):
    execution_status: GlobalExecutionStatus
    answer_outcome: GlobalAnswerOutcome | None = None
    route: Route | None = None
    route_sequence: list[Route] = Field(default_factory=list)
    required_routes: list[Route] = Field(default_factory=list)
    recovery_strategy: RecoveryStrategy | None = None
    recovery_strategies: list[RecoveryStrategy] = Field(default_factory=list)
    hitl_action: HITLAction | None = None
    error_code: str | None = None
    resumable: bool = False
    technical_failure: bool = False
    query_revision_count: int | None = None
    query_revision_min: int | None = Field(default=None, ge=0)
    query_revision_max: int | None = Field(default=None, ge=0)
    retrieval_attempt_count: int | None = None
    retrieval_attempt_min: int | None = Field(default=None, ge=0)
    retrieval_attempt_max: int | None = Field(default=None, ge=0)
    hitl_rounds: int | None = None
    hitl_rounds_min: int | None = Field(default=None, ge=0)
    hitl_rounds_max: int | None = Field(default=None, ge=0)
    citation_valid: bool | None = None
    provenance_valid: bool | None = None

    @model_validator(mode="after")
    def validate_contract(self) -> "ScenarioExpected":
        sequence = self.route_sequence or self.required_routes or ([self.route] if self.route else [])
        if self.route == "recover" and self.recovery_strategy is None and not self.recovery_strategies:
            raise ValueError("recover scenario must declare recovery_strategy")
        if self.route != "recover" and self.recovery_strategy is not None:
            raise ValueError("only recover route may declare recovery_strategy")
        if "recover" in sequence and not (self.recovery_strategies or self.recovery_strategy):
            raise ValueError("route history containing recover must declare recovery strategy")
        if self.technical_failure and (self.execution_status != "failed" or self.answer_outcome is not None):
            raise ValueError("technical failure must be failed with null outcome")
        if self.execution_status == "waiting_user" and self.answer_outcome is not None:
            raise ValueError("waiting scenario must have null outcome")
        if self.execution_status != "waiting_user" and self.resumable:
            raise ValueError("only waiting scenarios may be resumable")
        if self.required_routes and not set(self.required_routes).issubset(set(sequence)):
            raise ValueError("required_routes must be present in route_sequence")
        return self


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
    fixture: ScenarioFixture = Field(default_factory=ScenarioFixture)
    expected: ScenarioExpected

    def validate_record(self) -> None:
        if len(self.tags) != len(set(self.tags)):
            raise ValueError(f"{self.scenario_id}: duplicate tags")
        if self.fixture.fault not in {"none", self.fault}:
            raise ValueError(f"{self.scenario_id}: fixture.fault conflicts with scenario fault")
        if self.fault == "none" and self.expected.technical_failure:
            raise ValueError(f"{self.scenario_id}: fault is required for technical failure")
        if self.fault != "none" and not self.expected.technical_failure:
            raise ValueError(f"{self.scenario_id}: fault requires technical failure expectation")


class DatasetRecord(V2Model):
    name: str
    path: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    available: bool
    record_count: int | None = Field(default=None, ge=0)
    role: str
    error: str | None = None


class ScenarioResult(V2Model):
    scenario_id: str
    mode: ScenarioMode
    stage: TargetStage
    passed: bool
    status: str
    expected: dict[str, Any]
    observed: dict[str, Any] = Field(default_factory=dict)
    violations: list[str] = Field(default_factory=list)
    error: dict[str, Any] | None = None
    elapsed_seconds: float = Field(default=0.0, ge=0.0)


class InvariantCounts(V2Model):
    schema_invariant_violation_count: int = Field(default=0, ge=0)
    provenance_violation_count: int = Field(default=0, ge=0)
    citation_violation_count: int = Field(default=0, ge=0)
    budget_violation_count: int = Field(default=0, ge=0)
    retrieval_degraded_queries_count: int = Field(default=0, ge=0)


class HardGateResult(V2Model):
    name: str
    passed: bool
    evaluated: bool
    reason: str
    evidence_ref: str | None = None


class StageReportReference(V2Model):
    target_stage: TargetStage
    run_id: str
    path: str
    digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    available: bool = True


class CrossProcessStep(V2Model):
    name: str
    passed: bool
    request_id: str
    thread_id: str
    pid: int | None = None


class CrossProcessEvidence(V2Model):
    run_id: str
    git_commit: str
    request_id: str
    thread_id: str
    steps: list[CrossProcessStep] = Field(min_length=4)
    duplicate_resume_passed: bool
    stale_resume_passed: bool
    invalid_payload_passed: bool
    expired_checkpoint_passed: bool
    lease_recovery_passed: bool
    artifact_digest: str = Field(pattern=r"^[0-9a-f]{64}$")


class BaselineCandidateReport(V2Model):
    run_id: str
    timestamp: str
    git_commit: str | None
    git_dirty: bool | None
    target: Literal["v2_final_baseline_candidate"]
    dataset_records: list[DatasetRecord]
    stage_reports: list[StageReportReference]
    resolved_config: dict[str, Any]
    metrics: dict[str, Any]
    invariant_counts: InvariantCounts
    hard_gate_results: list[HardGateResult]
    hard_gates: dict[str, bool]
    failed_gates: list[str]
    baseline_status: Literal["eligible", "not_eligible"]
    freeze_eligible: bool
    evaluation_incomplete: bool
    incompleteness_reasons: list[str] = Field(default_factory=list)
    predictions_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    artifact_digest: str = Field(default="0" * 64, pattern=r"^[0-9a-f]{64}$")
    errors: list[dict[str, Any]] = Field(default_factory=list)
    digests: dict[str, str] = Field(default_factory=dict)


REQUIRED_COVERAGE_TAGS = (
    "recovery_direct_rewrite", "recovery_step_back", "recovery_hyde", "hitl_clarify",
    "hitl_scope_select", "terminal_complete", "terminal_partial", "terminal_no_knowledge",
    "terminal_unsupported", "terminal_unresolved", "technical_provider", "technical_timeout",
    "technical_schema", "technical_retrieval", "language_zh", "language_en", "language_mixed",
    "complexity_simple", "complexity_complex", "supported", "unsupported",
    "supported_unsupported_mixed", "cross_process",
)
TWO_CASE_COVERAGE_TAGS = {
    "recovery_direct_rewrite", "recovery_step_back", "recovery_hyde", "hitl_clarify",
    "hitl_scope_select", "terminal_complete", "terminal_partial", "terminal_no_knowledge",
    "terminal_unsupported", "terminal_unresolved",
}


def load_workflow_scenarios(path: Path = DEFAULT_DATASET) -> list[WorkflowScenario]:
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
            except Exception as exc:
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
        raise ValueError("workflow scenario coverage requires: " + ", ".join(missing))
    if {record.stage for record in records} != {"v2_1", "v2_2", "v2_3"}:
        raise ValueError("workflow scenarios must cover v2_1, v2_2, and v2_3")


def scenario_coverage(records: list[WorkflowScenario]) -> dict[str, Any]:
    return {
        "scenario_count": len(records),
        "mode_counts": dict(Counter(record.mode for record in records)),
        "stage_counts": dict(Counter(record.stage for record in records)),
        "tag_counts": dict(sorted(Counter(tag for record in records for tag in record.tags).items())),
    }


def evaluate_baseline(
    *, dataset_path: Path = DEFAULT_DATASET, output_root: Path = DEFAULT_OUTPUT_ROOT,
    run_id: str | None = None, mode: Literal["all", "real", "contract"] = "all",
    config: V2Config | None = None, retrieval_report: Path = DEFAULT_RETRIEVAL_REPORT,
    cross_process_report: Path | None = None,
) -> dict[str, Any]:
    config = config or V2Config.from_env()
    run_id = run_id or str(uuid4())
    report_dir = output_root / run_id
    if report_dir.exists():
        raise FileExistsError(f"refusing to overwrite baseline candidate: {report_dir}")
    records = load_workflow_scenarios(dataset_path)
    selected = [record for record in records if mode == "all" or record.mode == mode]
    raw_predictions = [_contract_prediction(record, config) if record.mode == "contract" else _real_prediction(record, config) for record in selected]
    predictions = [ScenarioResult.model_validate(item).model_dump(mode="json") for item in raw_predictions]
    report_dir.mkdir(parents=True, exist_ok=False)
    predictions_path = report_dir / "predictions.jsonl"
    predictions_path.write_text("".join(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n" for item in predictions), encoding="utf-8")
    stage_refs = _write_stage_reports(report_dir, run_id, predictions)
    report = _build_report(run_id=run_id, dataset_path=dataset_path, config=config, records=records, selected=selected, predictions=predictions, predictions_path=predictions_path, stage_refs=stage_refs, retrieval_report=retrieval_report, cross_process_report=cross_process_report)
    report_path = report_dir / "report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    loaded = BaselineCandidateReport.model_validate_json(report_path.read_text(encoding="utf-8"))
    if loaded.artifact_digest != _report_digest(loaded):
        raise ValueError("written baseline report artifact_digest verification failed")
    return loaded.model_dump(mode="json")


def _contract_prediction(scenario: WorkflowScenario, config: V2Config) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        harness = run_contract_scenario(scenario, config)
        prediction = _observe_result(scenario, harness.result, mode="contract", config=config, telemetry=harness.telemetry)
    except Exception as exc:
        prediction = {"scenario_id": scenario.scenario_id, "mode": "contract", "stage": scenario.stage, "status": "technical_failure", "passed": False, "expected": scenario.expected.model_dump(mode="json"), "error": _safe_error(exc), "violations": [f"harness exception: {type(exc).__name__}"]}
    prediction["elapsed_seconds"] = time.perf_counter() - started
    return prediction


def _real_prediction(scenario: WorkflowScenario, config: V2Config) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        if scenario.stage == "v2_1":
            result = Module4Service(config).run(scenario.question, response_language=_language(scenario))
        elif scenario.stage == "v2_2":
            result = Module6Service(config).run(scenario.question, response_language=_language(scenario))
        else:
            with tempfile.TemporaryDirectory(prefix="agenticrag-v23-baseline-") as temp_dir:
                durable_config = config.model_copy(update={"persistence": V2PersistenceConfig(**{**config.persistence.model_dump(), "sqlite_path": str(Path(temp_dir) / "v2.sqlite3")})})
                service = DurableV23Service(durable_config)
                try:
                    result = service.start(scenario.question, response_language=_language(scenario))
                finally:
                    service.close()
        prediction = _observe_result(scenario, result, mode="real", config=config)
    except Exception as exc:
        prediction = {"scenario_id": scenario.scenario_id, "mode": "real", "stage": scenario.stage, "status": "technical_failure", "passed": False, "expected": scenario.expected.model_dump(mode="json"), "error": _safe_error(exc), "violations": ["real service exception"]}
    prediction["elapsed_seconds"] = time.perf_counter() - started
    return prediction


def _observe_result(scenario: WorkflowScenario, result: Any, *, mode: str, config: V2Config, telemetry: Any | None = None) -> dict[str, Any]:
    task_value = getattr(result, "tasks", None)
    tasks = list(task_value if task_value is not None else result.state.get("tasks", {}).values())
    stage = result.stage_result
    evidence_value = getattr(result, "evidence", None)
    evidence = evidence_value if evidence_value is not None else result.state.get("evidence", {})
    state = getattr(result, "state", None)
    observed, counts = _audit_tasks(tasks, evidence, stage, config=config, state=state)
    routes = [decision.route for task in tasks for decision in task.routing_decisions]
    strategies = [decision.recovery_strategy for task in tasks for decision in task.routing_decisions if decision.recovery_strategy]
    violations = _compare_expected(scenario, stage, tasks, routes, strategies, counts)
    revisions = [revision for task in tasks for revision in task.query_revisions]
    attempts = [attempt for revision in revisions for attempt in revision.retrieval_attempts]
    pending_actions = [item.action for item in stage.pending_hitl_request.items] if stage.pending_hitl_request else []
    observed.update({
        "route": routes[-1] if routes else None,
        "route_sequence": routes,
        "recovery_strategies": strategies,
        "task_count": len(tasks),
        "capability": scenario.capability,
        "finding_count": sum(task.grounded_finding is not None for task in tasks),
        "query_revision_count": len(revisions),
        "retrieval_attempt_count": len(attempts),
        "retrieval_attempt_ids": [attempt.id for attempt in attempts],
        "grade_record_count": sum(len(task.grade_records) for task in tasks),
        "grade_input_attempt_ids": [attempt_id for task in tasks for record in task.grade_records for attempt_id in record.input_attempt_ids],
        "routing_decision_count": sum(len(task.routing_decisions) for task in tasks),
        "hitl_action": pending_actions[0] if pending_actions else None,
        "telemetry": _telemetry(telemetry),
    })
    return {"scenario_id": scenario.scenario_id, "mode": mode, "stage": scenario.stage, "status": "technical_failure" if stage.execution_status == "failed" else stage.execution_status, "passed": not violations, "expected": scenario.expected.model_dump(mode="json"), "observed": observed, "violations": violations, "error": stage.error.model_dump(mode="json") if stage.error else None}


def _compare_expected(scenario: WorkflowScenario, stage: Any, tasks: list[Any], routes: list[str], strategies: list[str], counts: InvariantCounts) -> list[str]:
    expected = scenario.expected
    violations: list[str] = []
    if stage.execution_status != expected.execution_status:
        violations.append("execution_status mismatch")
    if stage.answer_outcome != expected.answer_outcome:
        violations.append("answer_outcome mismatch")
    if stage.resumable != expected.resumable:
        violations.append("resumable mismatch")
    if expected.route is not None and (not routes or routes[-1] != expected.route):
        violations.append("final route mismatch")
    sequence = expected.route_sequence or ([expected.route] if expected.route else [])
    if sequence and routes[-len(sequence):] != sequence:
        violations.append("route history mismatch")
    if expected.required_routes and not set(expected.required_routes).issubset(set(routes)):
        violations.append("required route missing")
    wanted = expected.recovery_strategies or ([expected.recovery_strategy] if expected.recovery_strategy else [])
    if wanted and not set(wanted).issubset(set(strategies)):
        violations.append("recovery strategy history mismatch")
    if expected.error_code is not None and expected.error_code not in {task.error.code for task in tasks if task.error}:
        violations.append("error_code mismatch")
    revisions = sum(len(task.query_revisions) for task in tasks)
    attempts = sum(len(revision.retrieval_attempts) for task in tasks for revision in task.query_revisions)
    if expected.query_revision_count is not None and revisions != expected.query_revision_count:
        violations.append("query_revision_count mismatch")
    if expected.query_revision_min is not None and revisions < expected.query_revision_min:
        violations.append("query_revision_min violated")
    if expected.query_revision_max is not None and revisions > expected.query_revision_max:
        violations.append("query_revision_max violated")
    if expected.retrieval_attempt_count is not None and attempts != expected.retrieval_attempt_count:
        violations.append("retrieval_attempt_count mismatch")
    if expected.retrieval_attempt_min is not None and attempts < expected.retrieval_attempt_min:
        violations.append("retrieval_attempt_min violated")
    if expected.retrieval_attempt_max is not None and attempts > expected.retrieval_attempt_max:
        violations.append("retrieval_attempt_max violated")
    if expected.hitl_action is not None and expected.hitl_action not in {
        item.action for item in (stage.pending_hitl_request.items if stage.pending_hitl_request else [])
    }:
        violations.append("hitl_action mismatch")
    if expected.citation_valid is True and counts.citation_violation_count:
        violations.append("citation validity mismatch")
    if expected.provenance_valid is True and counts.provenance_violation_count:
        violations.append("provenance validity mismatch")
    if any((counts.schema_invariant_violation_count, counts.provenance_violation_count, counts.citation_violation_count, counts.budget_violation_count)):
        violations.append("audit violation")
    return violations


def _audit_tasks(tasks: list[Any], evidence: dict[str, Any], stage: Any, *, config: V2Config, state: dict[str, Any] | None = None) -> tuple[dict[str, Any], InvariantCounts]:
    schema = provenance = citation = budget = degraded = 0
    finding_ids: set[str] = set()
    evidence_ids = set(evidence)
    for task in tasks:
        if len(tasks) > config.budgets.max_subqueries:
            budget += 1
        if task.execution_status == "failed" and task.answer_outcome is not None:
            schema += 1
        if len(task.query_revisions) > 2 or any(len(revision.retrieval_attempts) > 2 for revision in task.query_revisions):
            budget += 1
        if len(task.query_revisions) > 2 or any(attempt.ordinal >= 3 for revision in task.query_revisions for attempt in revision.retrieval_attempts):
            budget += 1
        for revision in task.query_revisions:
            degraded += sum(int(attempt.retrieval_degraded) for attempt in revision.retrieval_attempts)
        if task.grounded_finding is not None:
            finding = task.grounded_finding
            finding_ids.update(finding.evidence_ids)
            support = set(task.grade_records[-1].grade.supporting_evidence_ids) if task.grade_records else set()
            latest_revision = task.query_revisions[-1].id if task.query_revisions else None
            latest_grade_revision = task.grade_records[-1].query_revision_id if task.grade_records else None
            if finding.task_id != task.id or not finding.evidence_ids or len(finding.evidence_ids) > config.budgets.max_evidence_per_finding or not set(finding.evidence_ids) <= evidence_ids or not set(finding.evidence_ids) <= support or latest_revision != latest_grade_revision:
                provenance += 1
    if state is not None and state.get("hitl_rounds", 0) > config.budgets.max_hitl_rounds:
        budget += 1
    pending = getattr(stage, "pending_hitl_request", None)
    if pending is not None and any(len(item.scope_options) > config.budgets.max_scope_options for item in pending.items):
        budget += 1
    final = stage.final_answer
    if final is not None and not set(final.citation_evidence_ids) <= finding_ids:
        citation += 1
    return {"execution_status": stage.execution_status, "answer_outcome": stage.answer_outcome, "route": next((task.routing_decisions[-1].route for task in reversed(tasks) if task.routing_decisions), None), "route_counts": dict(Counter(decision.route for task in tasks for decision in task.routing_decisions)), "technical_failure_count": sum(task.execution_status == "failed" for task in tasks) + int(stage.execution_status == "failed" and not any(task.execution_status == "failed" for task in tasks)), "degraded_retrieval_count": degraded, "schema_invariant_violation_count": schema, "provenance_violation_count": provenance, "citation_violation_count": citation, "budget_violation_count": budget}, InvariantCounts(schema_invariant_violation_count=schema, provenance_violation_count=provenance, citation_violation_count=citation, budget_violation_count=budget, retrieval_degraded_queries_count=degraded)


def _build_report(*, run_id: str, dataset_path: Path, config: V2Config, records: list[WorkflowScenario], selected: list[WorkflowScenario], predictions: list[dict[str, Any]], predictions_path: Path, stage_refs: list[StageReportReference], retrieval_report: Path, cross_process_report: Path | None) -> dict[str, Any]:
    dataset_records = _all_dataset_records(dataset_path)
    counts = InvariantCounts(
        schema_invariant_violation_count=sum(item.get("observed", {}).get("schema_invariant_violation_count", 0) for item in predictions),
        provenance_violation_count=sum(item.get("observed", {}).get("provenance_violation_count", 0) for item in predictions),
        citation_violation_count=sum(item.get("observed", {}).get("citation_violation_count", 0) for item in predictions),
        budget_violation_count=sum(item.get("observed", {}).get("budget_violation_count", 0) for item in predictions),
        retrieval_degraded_queries_count=sum(item.get("observed", {}).get("degraded_retrieval_count", 0) for item in predictions),
    )
    v1 = _check_retrieval_gate(retrieval_report)
    cross = _check_cross_process_gate(cross_process_report)
    dataset_gate = all(record.available for record in dataset_records)
    gate_values = {
        "dataset_schema_and_coverage": dataset_gate,
        "v1_2_retrieval_regression": v1["passed"],
        "schema_invariant_zero": counts.schema_invariant_violation_count == 0,
        "provenance_violation_zero": counts.provenance_violation_count == 0,
        "citation_violation_zero": counts.citation_violation_count == 0,
        "budget_violation_zero": counts.budget_violation_count == 0,
        "retrieval_degraded_queries_zero": counts.retrieval_degraded_queries_count == 0,
        "computation_capability_safety": _computation_gate(predictions),
        "baseline_scenarios_terminal_contract": all(item.get("passed") is True for item in predictions),
        "v2_3_cross_process": cross["passed"],
        "ordinary_baseline_zero_unexpected_technical_failures": not any(item.get("status") == "technical_failure" for item in predictions if item.get("mode") == "real"),
    }
    incomplete_reasons = []
    if not all(ref.available for ref in stage_refs):
        incomplete_reasons.append("required stage report missing")
    if not cross["evaluated"]:
        incomplete_reasons.append("cross-process evidence missing")
    if not dataset_gate:
        incomplete_reasons.append("required dataset missing")
    evaluation_incomplete = bool(incomplete_reasons)
    dirty = _git_dirty()
    failed_gates = [name for name, passed in gate_values.items() if not passed]
    if dirty is not False:
        failed_gates.append("git_dirty")
    if incomplete_reasons:
        failed_gates.append("evaluation_incomplete")
    freeze_eligible = not evaluation_incomplete and dirty is False and all(gate_values.values())
    report = BaselineCandidateReport(
        run_id=run_id, timestamp=datetime.now(timezone.utc).isoformat(), git_commit=_git_commit(), git_dirty=dirty,
        target="v2_final_baseline_candidate", dataset_records=dataset_records, stage_reports=stage_refs,
        resolved_config=config.resolved_record(), metrics=_metrics(selected, predictions, config), invariant_counts=counts,
        hard_gate_results=[HardGateResult(name=name, passed=passed, evaluated=True, reason=_gate_reason(name, passed, v1, cross)) for name, passed in gate_values.items()],
        hard_gates=gate_values, failed_gates=failed_gates, baseline_status="eligible" if freeze_eligible else "not_eligible", freeze_eligible=freeze_eligible, evaluation_incomplete=evaluation_incomplete, incompleteness_reasons=incomplete_reasons,
        predictions_sha256=_sha256(predictions_path), errors=[item["error"] for item in predictions if item.get("error")],
        digests={"dataset_sha256": _sha256(dataset_path), "workflow_scenarios_sha256": _sha256(dataset_path), "resolved_config_sha256": _json_digest(config.resolved_record()), "predictions_sha256": _sha256(predictions_path), "retrieval_eval_v2_sha256": _digest_if_exists(RETRIEVAL_DATASET), "qa_sha256": _digest_if_exists(QA_DATASET), "v2_qa_annotations_sha256": _digest_if_exists(ANNOTATION_DATASET), "module4_gold_sha256": _digest_if_exists(MODULE4_GOLD_DATASET), "artifact_manifest_sha256": _json_digest({"dataset_sha256": _sha256(dataset_path), "resolved_config_sha256": _json_digest(config.resolved_record()), "predictions_sha256": _sha256(predictions_path)})},
    )
    return report.model_copy(update={"artifact_digest": _report_digest(report)}).model_dump(mode="json")


def _metrics(selected: list[WorkflowScenario], predictions: list[dict[str, Any]], config: V2Config) -> dict[str, Any]:
    routes = Counter(item.get("observed", {}).get("route") for item in predictions if item.get("observed", {}).get("route"))
    computation = [item for item in predictions if "unsupported" in item.get("scenario_id", "")]
    return {
        "scenario_count": len(selected), "real_scenario_count": sum(item["mode"] == "real" for item in predictions), "contract_scenario_count": sum(item["mode"] == "contract" for item in predictions), "scenario_pass_count": sum(item.get("passed") is True for item in predictions), "scenario_failure_count": sum(item.get("passed") is not True for item in predictions), "route_counts": dict(routes),
        "technical_failure_count": sum(item.get("status") == "technical_failure" for item in predictions if item.get("mode") == "real"),
        "contract_technical_failure_count": sum(item.get("status") == "technical_failure" for item in predictions if item.get("mode") == "contract"),
        "planning": {"complexity_accuracy": None, "capability_accuracy": None, "decomposition_requirement_coverage": None, "structural_violation_counts": None},
        "evidence_route": {name: None for name in ("relevance_accuracy", "answerability_accuracy", "ambiguity_accuracy", "recoverability_accuracy", "failure_reason_accuracy", "grade_exact_match", "route_accuracy", "recovery_strategy_accuracy", "recovery_success_rate")},
        "answer": {"supported_subset_ragas": None, "outcome_accuracy": None, "unsupported_computation_recall": (sum(item.get("observed", {}).get("answer_outcome") == "unsupported" for item in computation) / len(computation)) if computation else None, "abstention_correctness": None, "citation_violations": sum(item.get("observed", {}).get("citation_violation_count", 0) for item in predictions), "provenance_violations": sum(item.get("observed", {}).get("provenance_violation_count", 0) for item in predictions)},
        "runtime": {"active_runtime_seconds": sum(float(item.get("elapsed_seconds", 0)) for item in predictions), "hitl_waiting_seconds": None, "retrieval_latency_seconds": None, "role_calls": None, "role_retries": None, "tokens": None, "attempts": None, "checkpoint_resume_latency_seconds": None},
        "budget_limits": config.budgets.model_dump(mode="json"), "route_counts": dict(routes),
    }


def _computation_gate(predictions: list[dict[str, Any]]) -> bool:
    cases = [item for item in predictions if "unsupported" in item.get("scenario_id", "") or item.get("observed", {}).get("capability") in {"arithmetic", "statistical_computation", "sql", "other_unsupported"}]
    return bool(cases) and all(item.get("observed", {}).get("answer_outcome") == "unsupported" for item in cases)


def _write_stage_reports(report_dir: Path, run_id: str, predictions: list[dict[str, Any]]) -> list[StageReportReference]:
    refs = []
    for stage in ("v2_1", "v2_2", "v2_3"):
        path = report_dir / f"stage_{stage}.json"
        path.write_text(json.dumps({"run_id": run_id, "target_stage": stage, "predictions": [item for item in predictions if item.get("stage") == stage]}, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        refs.append(StageReportReference(target_stage=stage, run_id=run_id, path=str(path), digest=_sha256(path)))
    return refs


def _all_dataset_records(workflow_path: Path) -> list[DatasetRecord]:
    records = [_dataset_record(RETRIEVAL_DATASET, "retrieval_eval_v2", "retrieval regression"), _dataset_record(QA_DATASET, "qa", "planning/answer QA"), _dataset_record(ANNOTATION_DATASET, "v2_qa_annotations", "planning annotations"), _dataset_record(workflow_path, "v2_workflow_scenarios", "workflow contracts"), _dataset_record(MODULE4_GOLD_DATASET, "module4_grade_route_gold", "frozen Module 4 semantic gold")]
    if records[0].available:
        if len(load_retrieval_dataset(RETRIEVAL_DATASET)) != 47:
            raise ValueError("retrieval_eval_v2.jsonl must contain exactly 47 frozen queries")
    if records[1].available and records[2].available:
        load_planning_samples(QA_DATASET, ANNOTATION_DATASET)
    return records


def _dataset_record(path: Path, name: str, role: str) -> DatasetRecord:
    if not path.is_file():
        return DatasetRecord(name=name, path=str(path), sha256="0" * 64, available=False, role=role, error="file not found")
    return DatasetRecord(name=name, path=str(path), sha256=_sha256(path), available=True, record_count=sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip()), role=role)


def _check_retrieval_gate(path: Path) -> dict[str, Any]:
    if not path.is_file() or not RETRIEVAL_DATASET.is_file():
        return {"passed": False, "evaluated": False, "status": "not_found", "path": str(path)}
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
        metrics = {**report.get("final_metrics", {}), "Recall@20": report.get("rrf_candidate_metrics", {}).get("Recall@20")}
        dataset_matches = report.get("dataset") == str(RETRIEVAL_DATASET) and _sha256(RETRIEVAL_DATASET) == FROZEN_RETRIEVAL_DATASET_SHA256
        checks = {name: metrics.get(name) is not None and float(metrics[name]) >= threshold for name, threshold in FROZEN_RETRIEVAL_METRICS.items()}
        passed = dataset_matches and report.get("dataset_size") == 47 and report.get("evaluated_queries") == 47 and report.get("fallback_queries", 0) == 0 and report.get("invalid_score_queries", 0) == 0 and all(checks.values())
        return {"passed": passed, "evaluated": True, "status": "checked", "path": str(path), "sha256": _sha256(path), "dataset_sha256": _sha256(RETRIEVAL_DATASET), "metrics": metrics, "metric_checks": checks, "dataset_matches": dataset_matches}
    except Exception as exc:
        return {"passed": False, "evaluated": False, "status": "invalid", "path": str(path), "error": _safe_error(exc)}


def _check_cross_process_gate(path: Path | None) -> dict[str, Any]:
    if path is None or not path.is_file():
        return {"passed": False, "evaluated": False, "status": "not_provided" if path is None else "not_found"}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        evidence = CrossProcessEvidence.model_validate(payload)
        canonical = evidence.model_dump(mode="json")
        digest = canonical.pop("artifact_digest")
        digest_ok = digest == _json_digest({**canonical, "artifact_digest": ""})
        names = [step.name for step in evidence.steps]
        ids_ok = all(step.request_id == evidence.request_id and step.thread_id == evidence.thread_id for step in evidence.steps)
        negative_ok = all((evidence.duplicate_resume_passed, evidence.stale_resume_passed, evidence.invalid_payload_passed, evidence.expired_checkpoint_passed, evidence.lease_recovery_passed))
        passed = digest_ok and names == ["start", "status_waiting", "resume", "status_completed"] and ids_ok and negative_ok and all(step.passed for step in evidence.steps)
        return {"passed": passed, "evaluated": True, "status": "checked", "path": str(path), "sha256": _sha256(path), "digest_valid": digest_ok, "request_id": evidence.request_id, "thread_id": evidence.thread_id, "step_count": len(evidence.steps), "negative_contracts": negative_ok}
    except Exception as exc:
        return {"passed": False, "evaluated": False, "status": "invalid", "path": str(path), "error": _safe_error(exc)}


def _report_digest(report: BaselineCandidateReport) -> str:
    payload = report.model_dump(mode="json")
    payload["artifact_digest"] = ""
    return _json_digest(payload)


def _gate_reason(name: str, passed: bool, v1: dict[str, Any], cross: dict[str, Any]) -> str:
    if passed:
        return "validated"
    if name == "v1_2_retrieval_regression":
        return v1.get("status", "failed")
    if name == "v2_3_cross_process":
        return cross.get("status", "failed")
    return "contract or audit failure"


def _telemetry(telemetry: Any | None) -> dict[str, Any] | None:
    if telemetry is None:
        return None
    return {"retrieval_calls": len(telemetry.backend_calls), "grader_calls": len(telemetry.grader_calls), "finding_calls": len(telemetry.finding_calls), "synthesis_calls": telemetry.synthesis_calls, "hitl_calls": telemetry.hitl_calls}


def _language(scenario: WorkflowScenario) -> str:
    return scenario.language if scenario.language in {"zh", "en"} else "en"


def _json_digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _digest_if_exists(path: Path) -> str:
    return _sha256(path) if path.is_file() else "0" * 64


def _safe_error(exc: Exception) -> dict[str, str]:
    return {"exception_type": type(exc).__name__, "message": str(exc).replace("\n", " ")[:1500]}


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
    parser.add_argument("--cross-process-report", type=Path, default=None)
    args = parser.parse_args()
    report = evaluate_baseline(dataset_path=args.dataset, output_root=args.output_root, mode=args.mode, retrieval_report=args.retrieval_report, cross_process_report=args.cross_process_report)
    print(json.dumps({"run_id": report["run_id"], "output": str(args.output_root / report["run_id"]), "baseline_status": report["baseline_status"], "failed_gates": report["failed_gates"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
