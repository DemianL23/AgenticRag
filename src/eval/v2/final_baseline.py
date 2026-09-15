"""Final V2 baseline candidate evaluator.

Contract scenarios execute the frozen V2 graphs with deterministic
collaborators.  This module is evaluation-only and does not add scenario
branches to the production runtime.
"""

from __future__ import annotations

import hashlib
import asyncio
import json
import subprocess
import tempfile
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator

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
from .audit import AuditCounts, audit_v2_result
from .contract_harness import run_contract_scenario
from .module8 import CrossProcessEvidence
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
FROZEN_ANNOTATION_DATASET_SHA256 = "2c3a638a36797454af081b851f0d78df3419f2b6f1d00766e561ac98b4bab639"
FROZEN_MODULE4_GOLD_SHA256 = "b2204ec505d76c3fd57dd060378ea0e9e5946482e4987b6b0b6c6a58ff1e594c"
MODULE8_FROZEN_IMPLEMENTATION_SHA = "891ef20d218ed3e01d7b3a54ac5b50bde628dc52"
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
EvaluationProfile = Literal[
    "development_contract", "development_real", "full_baseline"
]


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
    expected_capability: TaskCapability | Literal["mixed"]
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
    run_id: str | None = None
    path: str | None = None
    digest: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    available: bool = False
    evaluation_complete: bool = False
    audit_counts: InvariantCounts = Field(default_factory=InvariantCounts)
    cross_process_acceptance_ref: str | None = None
    cross_process_acceptance_digest: str | None = Field(
        default=None, pattern=r"^[0-9a-f]{64}$"
    )
    error: str | None = None


class EvaluatorReportReference(V2Model):
    evaluator: Literal["planning", "module4_grade_route", "answer_ragas"]
    run_id: str | None = None
    path: str | None = None
    digest: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    available: bool = False
    evaluation_complete: bool = False
    audit_counts: InvariantCounts = Field(default_factory=InvariantCounts)
    error: str | None = None


class ExternalReportEnvelope(BaseModel):
    """Minimum common contract for immutable evaluator artifacts."""

    model_config = ConfigDict(extra="allow")

    report_schema_version: Literal[1]
    producer: str
    run_id: str = Field(min_length=1)
    git_commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    git_dirty: bool
    resolved_config: dict[str, Any]
    artifact_digest: str = Field(pattern=r"^[0-9a-f]{64}$")


class EvaluationCompleteness(V2Model):
    profile: EvaluationProfile
    retrieval_regression_evaluated: bool
    planning_evaluator_evaluated: bool
    module4_grade_route_evaluator_evaluated: bool
    answer_ragas_evaluator_evaluated: bool
    workflow_contract_scenarios_evaluated: bool
    required_real_model_scenarios_evaluated: bool
    v2_3_cross_process_acceptance_evaluated: bool
    required_datasets_integrity_validated: bool
    stage_reports_evaluated: bool
    complete: bool
    missing_components: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_profile_completeness(self) -> "EvaluationCompleteness":
        evaluated_components = (
            self.retrieval_regression_evaluated,
            self.planning_evaluator_evaluated,
            self.module4_grade_route_evaluator_evaluated,
            self.answer_ragas_evaluator_evaluated,
            self.workflow_contract_scenarios_evaluated,
            self.required_real_model_scenarios_evaluated,
            self.v2_3_cross_process_acceptance_evaluated,
            self.required_datasets_integrity_validated,
            self.stage_reports_evaluated,
        )
        if self.profile != "full_baseline" and self.complete:
            raise ValueError("development evaluation profile can never be complete")
        if self.complete and not all(evaluated_components):
            raise ValueError("complete evaluation requires every baseline component")
        if self.complete and self.missing_components:
            raise ValueError("complete evaluation cannot list missing components")
        return self


class BaselineCandidateReport(V2Model):
    run_id: str
    timestamp: str
    git_commit: str | None
    git_dirty: bool | None
    target: Literal["v2_final_baseline_candidate"]
    evaluation_profile: EvaluationProfile
    evaluation_completeness: EvaluationCompleteness
    dataset_records: list[DatasetRecord]
    stage_reports: list[StageReportReference]
    evaluator_reports: list[EvaluatorReportReference]
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

    @model_validator(mode="after")
    def validate_freeze_eligibility(self) -> "BaselineCandidateReport":
        if self.freeze_eligible and (
            self.evaluation_profile != "full_baseline"
            or not self.evaluation_completeness.complete
            or self.evaluation_incomplete
            or self.git_dirty is not False
            or not all(self.hard_gates.values())
        ):
            raise ValueError("freeze eligibility requires a complete clean full_baseline")
        if (self.baseline_status == "eligible") != self.freeze_eligible:
            raise ValueError("baseline_status and freeze_eligible disagree")
        return self


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
    unresolved = [record for record in records if "terminal_unresolved" in record.tags]
    if len(unresolved) < 2 or any(
        record.expected.execution_status != "completed"
        or record.expected.answer_outcome != "unresolved"
        or record.expected.resumable
        or record.fixture.resume_request is None
        or record.expected.hitl_rounds != 1
        for record in unresolved
    ):
        raise ValueError(
            "terminal_unresolved requires at least two completed/unresolved resumed HITL scenarios"
        )
    if any(
        "terminal_unresolved" in record.tags
        for record in records
        if record.expected.execution_status == "waiting_user"
    ):
        raise ValueError("waiting_user scenarios cannot claim terminal_unresolved")
    for strategy in ("direct_rewrite", "step_back", "hyde"):
        cases = [
            record
            for record in records
            if strategy in record.fixture.recovery_strategies
        ]
        success = any(record.expected.route == "answer" for record in cases)
        bounded = any(
            record.expected.route_sequence == ["recover", "no_knowledge"]
            and record.expected.retrieval_attempt_max is not None
            for record in cases
        )
        if not success or not bounded:
            raise ValueError(
                f"{strategy} requires one recovery success and one bounded no-ATT3 case"
            )
    cross_process = [record for record in records if "cross_process" in record.tags]
    if not cross_process or any(
        record.fixture.cross_process_reference is None
        or "durable_cross_process_evidence" not in record.tags
        or record.expected.execution_status != "waiting_user"
        for record in cross_process
    ):
        raise ValueError(
            "cross_process scenarios must reference durable external acceptance evidence"
        )
    computation_capabilities = {
        record.capability
        for record in records
        if record.capability
        in {"arithmetic", "statistical_computation", "sql", "other_unsupported"}
        and record.expected.answer_outcome == "unsupported"
    }
    if computation_capabilities != {
        "arithmetic",
        "statistical_computation",
        "sql",
        "other_unsupported",
    }:
        raise ValueError("workflow scenarios must cover every unsupported computation capability")


def scenario_coverage(records: list[WorkflowScenario]) -> dict[str, Any]:
    return {
        "scenario_count": len(records),
        "mode_counts": dict(Counter(record.mode for record in records)),
        "stage_counts": dict(Counter(record.stage for record in records)),
        "tag_counts": dict(sorted(Counter(tag for record in records for tag in record.tags).items())),
        "semantic": {
            "terminal_unresolved_completed": sum(
                "terminal_unresolved" in record.tags
                and record.expected.execution_status == "completed"
                and record.expected.answer_outcome == "unresolved"
                for record in records
            ),
            "bounded_recovery_by_strategy": {
                strategy: sum(
                    record.expected.route_sequence == ["recover", "no_knowledge"]
                    and strategy in record.fixture.recovery_strategies
                    for record in records
                )
                for strategy in ("direct_rewrite", "step_back", "hyde")
            },
            "durable_cross_process_references": sum(
                record.fixture.cross_process_reference is not None for record in records
            ),
        },
    }


def evaluate_baseline(
    *, dataset_path: Path = DEFAULT_DATASET, output_root: Path = DEFAULT_OUTPUT_ROOT,
    run_id: str | None = None, mode: Literal["all", "real", "contract"] = "all",
    profile: EvaluationProfile | None = None,
    config: V2Config | None = None, retrieval_report: Path = DEFAULT_RETRIEVAL_REPORT,
    cross_process_report: Path | None = None,
    planning_report: Path | None = None,
    module4_gold_report: Path | None = None,
    answer_ragas_report: Path | None = None,
    stage_v21_report: Path | None = None,
    stage_v22_report: Path | None = None,
    stage_v23_report: Path | None = None,
) -> dict[str, Any]:
    config = config or V2Config.from_env()
    run_id = run_id or str(uuid4())
    report_dir = output_root / run_id
    if report_dir.exists():
        raise FileExistsError(f"refusing to overwrite baseline candidate: {report_dir}")
    profile = profile or {
        "contract": "development_contract",
        "real": "development_real",
        "all": "full_baseline",
    }[mode]
    expected_mode = {
        "development_contract": "contract",
        "development_real": "real",
        "full_baseline": "all",
    }[profile]
    if mode != "all" and mode != expected_mode:
        raise ValueError("mode and evaluation profile select different scenario sets")
    mode = expected_mode
    records = load_workflow_scenarios(dataset_path)
    selected = [record for record in records if mode == "all" or record.mode == mode]
    raw_predictions = [_contract_prediction(record, config) if record.mode == "contract" else _real_prediction(record, config) for record in selected]
    predictions = [ScenarioResult.model_validate(item).model_dump(mode="json") for item in raw_predictions]
    report_dir.mkdir(parents=True, exist_ok=False)
    predictions_path = report_dir / "predictions.jsonl"
    predictions_path.write_text("".join(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n" for item in predictions), encoding="utf-8")
    if profile == "full_baseline":
        planning_report, module4_gold_report, answer_ragas_report = (
            _run_integrated_evaluators(
                report_dir=report_dir,
                config=config,
                planning_report=planning_report,
                module4_gold_report=module4_gold_report,
                answer_ragas_report=answer_ragas_report,
            )
        )
    evaluated_commit = _git_commit()
    stage_refs = [
        _load_stage_report(stage_v21_report, "v2_1", config, evaluated_commit),
        _load_stage_report(stage_v22_report, "v2_2", config, evaluated_commit),
        _load_stage_report(stage_v23_report, "v2_3", config, evaluated_commit),
    ]
    evaluator_reports, evaluator_metrics = _load_evaluator_reports(
        planning_report=planning_report,
        module4_gold_report=module4_gold_report,
        answer_ragas_report=answer_ragas_report,
        config=config,
        evaluated_commit=evaluated_commit,
    )
    evaluator_metrics["runtime"].update(
        _load_stage_runtime_metrics(
            [stage_v21_report, stage_v22_report, stage_v23_report]
        )
    )
    report = _build_report(
        run_id=run_id,
        profile=profile,
        dataset_path=dataset_path,
        config=config,
        records=records,
        selected=selected,
        predictions=predictions,
        predictions_path=predictions_path,
        stage_refs=stage_refs,
        evaluator_reports=evaluator_reports,
        evaluator_metrics=evaluator_metrics,
        retrieval_report=retrieval_report,
        cross_process_report=cross_process_report,
    )
    report_path = report_dir / "report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    loaded = BaselineCandidateReport.model_validate_json(report_path.read_text(encoding="utf-8"))
    if loaded.artifact_digest != _report_digest(loaded):
        raise ValueError("written baseline report artifact_digest verification failed")
    return loaded.model_dump(mode="json")


def _run_integrated_evaluators(
    *,
    report_dir: Path,
    config: V2Config,
    planning_report: Path | None,
    module4_gold_report: Path | None,
    answer_ragas_report: Path | None,
) -> tuple[Path | None, Path | None, Path | None]:
    """Run required evaluators when a full baseline did not provide artifacts."""

    integrated = report_dir / "integrated_evaluators"
    if planning_report is None:
        from .planning import PlanningCoverageJudge, evaluate_planning

        try:
            payload = evaluate_planning(
                QA_DATASET,
                ANNOTATION_DATASET,
                config=config,
                judge=PlanningCoverageJudge(),
            )
            planning_report = integrated / "planning.json"
            _write_integrated_report(
                planning_report, payload, producer="module2_planning_evaluator"
            )
        except Exception:
            planning_report = None
    if module4_gold_report is None:
        from .module4_gold import replay_module4_gold

        try:
            payload = replay_module4_gold(
                MODULE4_GOLD_DATASET,
                config=config,
                output_root=integrated / "module4_runs",
            )
            module4_gold_report = integrated / "module4_gold.json"
            _write_integrated_report(
                module4_gold_report,
                {**payload, "resolved_config": config.resolved_record()},
                producer="module4_gold_replay_evaluator",
            )
        except Exception:
            module4_gold_report = None
    if answer_ragas_report is None:
        from .answer_baseline import evaluate_v2_answers, write_answer_report

        try:
            payload = asyncio.run(
                evaluate_v2_answers(
                    qa_path=QA_DATASET,
                    annotation_path=ANNOTATION_DATASET,
                    config=config,
                )
            )
            answer_ragas_report = integrated / "answer_ragas.json"
            write_answer_report(payload, answer_ragas_report)
        except Exception:
            answer_ragas_report = None
    return planning_report, module4_gold_report, answer_ragas_report


def _write_integrated_report(
    path: Path, payload: dict[str, Any], *, producer: str
) -> None:
    report = {
        **payload,
        "report_schema_version": 1,
        "producer": producer,
        "artifact_digest": "",
    }
    report["artifact_digest"] = _json_digest(report)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _contract_prediction(scenario: WorkflowScenario, config: V2Config) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        harness = run_contract_scenario(scenario, config)
        prediction = _observe_result(scenario, harness.result, mode="contract", config=config, telemetry=harness.telemetry)
    except Exception as exc:
        prediction = {"scenario_id": scenario.scenario_id, "mode": "contract", "stage": scenario.stage, "expected_capability": scenario.capability, "status": "technical_failure", "passed": False, "expected": scenario.expected.model_dump(mode="json"), "error": _safe_error(exc), "violations": [f"harness exception: {type(exc).__name__}"]}
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
        prediction = {"scenario_id": scenario.scenario_id, "mode": "real", "stage": scenario.stage, "expected_capability": scenario.capability, "status": "technical_failure", "passed": False, "expected": scenario.expected.model_dump(mode="json"), "error": _safe_error(exc), "violations": ["real service exception"]}
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
    hitl_rounds = state.get("hitl_rounds", 0) if state is not None else 0
    violations = _compare_expected(
        scenario, stage, tasks, routes, strategies, counts, hitl_rounds=hitl_rounds
    )
    revisions = [revision for task in tasks for revision in task.query_revisions]
    attempts = [attempt for revision in revisions for attempt in revision.retrieval_attempts]
    pending_actions = [item.action for item in stage.pending_hitl_request.items] if stage.pending_hitl_request else []
    observed.update({
        "route": routes[-1] if routes else None,
        "route_sequence": routes,
        "recovery_strategies": strategies,
        "task_count": len(tasks),
        "task_capabilities": [task.capability for task in tasks],
        "finding_count": sum(task.grounded_finding is not None for task in tasks),
        "query_revision_count": len(revisions),
        "retrieval_attempt_count": len(attempts),
        "retrieval_attempt_ids": [attempt.id for attempt in attempts],
        "grade_record_count": sum(len(task.grade_records) for task in tasks),
        "grade_input_attempt_ids": [attempt_id for task in tasks for record in task.grade_records for attempt_id in record.input_attempt_ids],
        "routing_decision_count": sum(len(task.routing_decisions) for task in tasks),
        "hitl_action": pending_actions[0] if pending_actions else next(
            (route for route in reversed(routes) if route in {"clarify", "scope_select"}),
            None,
        ),
        "hitl_rounds": hitl_rounds,
        "telemetry": _telemetry(telemetry),
    })
    return {"scenario_id": scenario.scenario_id, "mode": mode, "stage": scenario.stage, "expected_capability": scenario.capability, "status": "technical_failure" if stage.execution_status == "failed" else stage.execution_status, "passed": not violations, "expected": scenario.expected.model_dump(mode="json"), "observed": observed, "violations": violations, "error": stage.error.model_dump(mode="json") if stage.error else None}


def _compare_expected(
    scenario: WorkflowScenario,
    stage: Any,
    tasks: list[Any],
    routes: list[str],
    strategies: list[str],
    counts: InvariantCounts,
    *,
    hitl_rounds: int,
) -> list[str]:
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
    } and expected.hitl_action not in routes:
        violations.append("hitl_action mismatch")
    if expected.hitl_rounds is not None and hitl_rounds != expected.hitl_rounds:
        violations.append("hitl_rounds mismatch")
    if expected.hitl_rounds_min is not None and hitl_rounds < expected.hitl_rounds_min:
        violations.append("hitl_rounds_min violated")
    if expected.hitl_rounds_max is not None and hitl_rounds > expected.hitl_rounds_max:
        violations.append("hitl_rounds_max violated")
    if expected.citation_valid is True and counts.citation_violation_count:
        violations.append("citation validity mismatch")
    if expected.provenance_valid is True and counts.provenance_violation_count:
        violations.append("provenance validity mismatch")
    if any((counts.schema_invariant_violation_count, counts.provenance_violation_count, counts.citation_violation_count, counts.budget_violation_count)):
        violations.append("audit violation")
    return violations


def _audit_tasks(tasks: list[Any], evidence: dict[str, Any], stage: Any, *, config: V2Config, state: dict[str, Any] | None = None) -> tuple[dict[str, Any], InvariantCounts]:
    audit = audit_v2_result(tasks, evidence, stage, config=config, state=state)
    task_failures = sum(
        getattr(task, "execution_status", None) == "failed" for task in tasks
    )
    request_failure = int(
        stage.execution_status == "failed" and task_failures == 0
    )
    observed = {
        "execution_status": stage.execution_status,
        "answer_outcome": stage.answer_outcome,
        "route": next(
            (
                task.routing_decisions[-1].route
                for task in reversed(tasks)
                if getattr(task, "routing_decisions", None)
            ),
            None,
        ),
        "route_counts": dict(
            Counter(
                decision.route
                for task in tasks
                for decision in getattr(task, "routing_decisions", [])
            )
        ),
        "technical_failure_count": task_failures + request_failure,
        "degraded_retrieval_count": audit.retrieval_degraded_queries_count,
        **audit.model_dump(mode="json"),
    }
    return observed, InvariantCounts.model_validate(audit.model_dump())


def _build_report(
    *,
    run_id: str,
    profile: EvaluationProfile,
    dataset_path: Path,
    config: V2Config,
    records: list[WorkflowScenario],
    selected: list[WorkflowScenario],
    predictions: list[dict[str, Any]],
    predictions_path: Path,
    stage_refs: list[StageReportReference],
    evaluator_reports: list[EvaluatorReportReference],
    evaluator_metrics: dict[str, Any],
    retrieval_report: Path,
    cross_process_report: Path | None,
) -> dict[str, Any]:
    dataset_records = _all_dataset_records(dataset_path)
    scenario_by_id = {record.scenario_id: record for record in selected}
    ordinary_predictions = [
        item
        for item in predictions
        if not scenario_by_id[item["scenario_id"]].expected.technical_failure
    ]
    workflow_counts = InvariantCounts(
        schema_invariant_violation_count=sum(item.get("observed", {}).get("schema_invariant_violation_count", 0) for item in ordinary_predictions),
        provenance_violation_count=sum(item.get("observed", {}).get("provenance_violation_count", 0) for item in ordinary_predictions),
        citation_violation_count=sum(item.get("observed", {}).get("citation_violation_count", 0) for item in ordinary_predictions),
        budget_violation_count=sum(item.get("observed", {}).get("budget_violation_count", 0) for item in ordinary_predictions),
        retrieval_degraded_queries_count=sum(item.get("observed", {}).get("degraded_retrieval_count", 0) for item in ordinary_predictions),
    )
    counts = _sum_invariant_counts(
        [
            workflow_counts,
            *(item.audit_counts for item in evaluator_reports),
            *(item.audit_counts for item in stage_refs),
        ]
    )
    v1 = _check_retrieval_gate(retrieval_report)
    current_commit = _git_commit()
    cross = _check_cross_process_gate(
        cross_process_report, current_git_commit=current_commit
    )
    dataset_gate = all(record.available for record in dataset_records)
    integrity = _check_frozen_dataset_integrity(dataset_records)
    contract_ids = {record.scenario_id for record in records if record.mode == "contract"}
    real_ids = {record.scenario_id for record in records if record.mode == "real"}
    selected_ids = {record.scenario_id for record in selected}
    workflow_evaluated = contract_ids <= selected_ids
    real_evaluated = real_ids <= selected_ids and bool(real_ids)
    evaluator_by_name = {item.evaluator: item for item in evaluator_reports}
    v23_stage = next(item for item in stage_refs if item.target_stage == "v2_3")
    v23_cross_linked = bool(
        cross["passed"]
        and v23_stage.cross_process_acceptance_ref == cross.get("path")
        and v23_stage.cross_process_acceptance_digest == cross.get("sha256")
    )
    stage_evaluated = all(item.evaluation_complete for item in stage_refs) and v23_cross_linked
    component_values = {
        "retrieval_regression": bool(v1["evaluated"]),
        "planning_evaluator": evaluator_by_name["planning"].evaluation_complete,
        "module4_grade_route_evaluator": evaluator_by_name[
            "module4_grade_route"
        ].evaluation_complete,
        "answer_ragas_evaluator": evaluator_by_name[
            "answer_ragas"
        ].evaluation_complete,
        "workflow_contract_scenarios": workflow_evaluated,
        "required_real_model_scenarios": real_evaluated,
        "v2_3_cross_process_acceptance": bool(cross["evaluated"]),
        "required_datasets_integrity": integrity["passed"],
        "stage_reports": stage_evaluated,
    }
    missing_components = [name for name, value in component_values.items() if not value]
    if profile != "full_baseline":
        missing_components.insert(0, f"profile {profile} is development-only")
    completeness = EvaluationCompleteness(
        profile=profile,
        retrieval_regression_evaluated=component_values["retrieval_regression"],
        planning_evaluator_evaluated=component_values["planning_evaluator"],
        module4_grade_route_evaluator_evaluated=component_values[
            "module4_grade_route_evaluator"
        ],
        answer_ragas_evaluator_evaluated=component_values["answer_ragas_evaluator"],
        workflow_contract_scenarios_evaluated=component_values[
            "workflow_contract_scenarios"
        ],
        required_real_model_scenarios_evaluated=component_values[
            "required_real_model_scenarios"
        ],
        v2_3_cross_process_acceptance_evaluated=component_values[
            "v2_3_cross_process_acceptance"
        ],
        required_datasets_integrity_validated=component_values[
            "required_datasets_integrity"
        ],
        stage_reports_evaluated=component_values["stage_reports"],
        complete=profile == "full_baseline" and not missing_components,
        missing_components=missing_components,
    )
    computation_evaluated = {
        "arithmetic", "statistical_computation", "sql", "other_unsupported"
    } <= {
        item.get("expected_capability")
        for item in predictions
    }
    answer_computation_recall = evaluator_metrics["answer"].get(
        "unsupported_computation_recall"
    )
    computation_evaluated = (
        computation_evaluated and answer_computation_recall is not None
    )
    contract_terminal_passed = workflow_evaluated and all(
        item.get("passed") is True
        for item in predictions
        if item.get("mode") == "contract"
    )
    real_terminal_passed = real_evaluated and all(
        item.get("passed") is True
        for item in predictions
        if item.get("mode") == "real"
    )
    gate_checks: dict[str, tuple[bool, bool]] = {
        "dataset_schema_and_coverage": (dataset_gate, True),
        "frozen_dataset_integrity": (integrity["passed"], True),
        "v1_2_retrieval_regression": (v1["passed"], v1["evaluated"]),
        "schema_invariant_zero": (
            counts.schema_invariant_violation_count == 0,
            workflow_evaluated,
        ),
        "provenance_violation_zero": (
            counts.provenance_violation_count == 0,
            workflow_evaluated,
        ),
        "citation_violation_zero": (
            counts.citation_violation_count == 0,
            workflow_evaluated,
        ),
        "budget_violation_zero": (
            counts.budget_violation_count == 0,
            workflow_evaluated,
        ),
        "retrieval_degraded_queries_zero": (
            counts.retrieval_degraded_queries_count == 0,
            workflow_evaluated,
        ),
        "computation_capability_safety": (
            _computation_gate(predictions)
            and answer_computation_recall == 1.0,
            computation_evaluated,
        ),
        "baseline_scenarios_terminal_contract": (
            contract_terminal_passed,
            workflow_evaluated,
        ),
        "required_real_model_scenarios_terminal_contract": (
            real_terminal_passed,
            real_evaluated,
        ),
        "v2_3_cross_process": (cross["passed"], cross["evaluated"]),
        "ordinary_baseline_zero_unexpected_technical_failures": (
            real_evaluated
            and not any(
                item.get("status") == "technical_failure"
                for item in predictions
                if item.get("mode") == "real"
            ),
            real_evaluated,
        ),
    }
    gate_values = {
        name: bool(value and evaluated)
        for name, (value, evaluated) in gate_checks.items()
    }
    evaluation_incomplete = not completeness.complete
    incomplete_reasons = completeness.missing_components
    dirty = _git_dirty()
    failed_gates = [name for name, passed in gate_values.items() if not passed]
    if dirty is not False:
        failed_gates.append("git_dirty")
    if incomplete_reasons:
        failed_gates.append("evaluation_incomplete")
    freeze_eligible = (
        profile == "full_baseline"
        and not evaluation_incomplete
        and dirty is False
        and all(gate_values.values())
    )
    report = BaselineCandidateReport(
        run_id=run_id, timestamp=datetime.now(timezone.utc).isoformat(), git_commit=current_commit, git_dirty=dirty,
        target="v2_final_baseline_candidate", evaluation_profile=profile,
        evaluation_completeness=completeness, dataset_records=dataset_records,
        stage_reports=stage_refs, evaluator_reports=evaluator_reports,
        resolved_config=config.resolved_record(),
        metrics=_metrics(selected, predictions, config, evaluator_metrics), invariant_counts=counts,
        hard_gate_results=[HardGateResult(name=name, passed=gate_values[name], evaluated=evaluated, reason=_gate_reason(name, gate_values[name], v1, cross)) for name, (_value, evaluated) in gate_checks.items()],
        hard_gates=gate_values, failed_gates=failed_gates, baseline_status="eligible" if freeze_eligible else "not_eligible", freeze_eligible=freeze_eligible, evaluation_incomplete=evaluation_incomplete, incompleteness_reasons=incomplete_reasons,
        predictions_sha256=_sha256(predictions_path), errors=[item["error"] for item in predictions if item.get("error")],
        digests={"dataset_sha256": _sha256(dataset_path), "workflow_scenarios_sha256": _sha256(dataset_path), "resolved_config_sha256": _json_digest(config.resolved_record()), "predictions_sha256": _sha256(predictions_path), "retrieval_eval_v2_sha256": _digest_if_exists(RETRIEVAL_DATASET), "qa_sha256": _digest_if_exists(QA_DATASET), "v2_qa_annotations_sha256": _digest_if_exists(ANNOTATION_DATASET), "module4_gold_sha256": _digest_if_exists(MODULE4_GOLD_DATASET), "artifact_manifest_sha256": _json_digest({"dataset_sha256": _sha256(dataset_path), "resolved_config_sha256": _json_digest(config.resolved_record()), "predictions_sha256": _sha256(predictions_path)})},
    )
    return report.model_copy(update={"artifact_digest": _report_digest(report)}).model_dump(mode="json")


def _sum_invariant_counts(items: list[InvariantCounts]) -> InvariantCounts:
    totals = AuditCounts()
    for item in items:
        totals = totals + AuditCounts.model_validate(item.model_dump())
    return InvariantCounts.model_validate(totals.model_dump())


def _metrics(
    selected: list[WorkflowScenario],
    predictions: list[dict[str, Any]],
    config: V2Config,
    evaluator_metrics: dict[str, Any],
) -> dict[str, Any]:
    routes = Counter(item.get("observed", {}).get("route") for item in predictions if item.get("observed", {}).get("route"))
    recovery = [
        item for item in predictions if "recover" in item.get("observed", {}).get("route_sequence", [])
    ]
    telemetry = [item.get("observed", {}).get("telemetry") for item in predictions]
    telemetry = [item for item in telemetry if item is not None]
    retrieval_attempts = [
        int(item.get("observed", {}).get("retrieval_attempt_count", 0))
        for item in predictions
    ]
    return {
        "scenario_count": len(selected), "real_scenario_count": sum(item["mode"] == "real" for item in predictions), "contract_scenario_count": sum(item["mode"] == "contract" for item in predictions), "scenario_pass_count": sum(item.get("passed") is True for item in predictions), "scenario_failure_count": sum(item.get("passed") is not True for item in predictions), "route_counts": dict(routes),
        "technical_failure_count": sum(item.get("status") == "technical_failure" for item in predictions if item.get("mode") == "real"),
        "contract_technical_failure_count": sum(item.get("status") == "technical_failure" for item in predictions if item.get("mode") == "contract"),
        "planning": evaluator_metrics["planning"],
        "evidence_route": {
            **evaluator_metrics["evidence_route"],
            "recovery_success_rate": (
                sum(item.get("observed", {}).get("route") == "answer" for item in recovery)
                / len(recovery)
                if recovery
                else None
            ),
            "retrieval_attempt_metrics": {
                "total": sum(retrieval_attempts),
                "mean_per_scenario": (
                    sum(retrieval_attempts) / len(retrieval_attempts)
                    if retrieval_attempts
                    else None
                ),
                "max_per_scenario": max(retrieval_attempts) if retrieval_attempts else None,
            },
        },
        "answer": {
            "supported_subset_ragas": evaluator_metrics["answer"]["supported_subset_ragas"],
            "outcome_accuracy": evaluator_metrics["answer"].get("outcome_accuracy"),
            "unsupported_computation_recall": evaluator_metrics["answer"].get(
                "unsupported_computation_recall"
            ),
            "abstention_correctness": evaluator_metrics["answer"].get(
                "abstention_correctness"
            ),
            "citation_violations": sum(item.get("observed", {}).get("citation_violation_count", 0) for item in predictions),
            "provenance_violations": sum(item.get("observed", {}).get("provenance_violation_count", 0) for item in predictions),
        },
        "runtime": {
            "active_runtime_seconds": sum(float(item.get("elapsed_seconds", 0)) for item in predictions),
            "hitl_waiting_seconds": evaluator_metrics["runtime"].get("hitl_waiting_seconds"),
            "retrieval_latency_seconds": evaluator_metrics["runtime"].get("retrieval_latency_seconds"),
            "role_calls": {
                key: sum(int(item.get(key, 0)) for item in telemetry)
                for key in ("retrieval_calls", "grader_calls", "finding_calls", "synthesis_calls", "hitl_calls")
            } if telemetry else None,
            "role_retries": evaluator_metrics["runtime"].get("role_retries"),
            "tokens": evaluator_metrics["runtime"].get("tokens"),
            "attempts": sum(retrieval_attempts) if retrieval_attempts else None,
            "checkpoint_resume_latency_seconds": evaluator_metrics["runtime"].get("checkpoint_resume_latency_seconds"),
        },
        "budget_limits": config.budgets.model_dump(mode="json"), "route_counts": dict(routes),
    }


def _computation_gate(predictions: list[dict[str, Any]]) -> bool:
    unsupported_capabilities = {
        "arithmetic", "statistical_computation", "sql", "other_unsupported"
    }
    cases = [
        item
        for item in predictions
        if item.get("expected_capability") in unsupported_capabilities
    ]
    return bool(cases) and all(
        item.get("observed", {}).get("answer_outcome") == "unsupported"
        and bool(item.get("observed", {}).get("task_capabilities"))
        and set(item["observed"]["task_capabilities"])
        == {item["expected_capability"]}
        for item in cases
    )


def _load_stage_report(
    path: Path | None,
    target_stage: TargetStage,
    config: V2Config,
    evaluated_commit: str | None,
) -> StageReportReference:
    if path is None or not path.is_file():
        return StageReportReference(
            target_stage=target_stage,
            path=str(path) if path is not None else None,
            error="stage evaluator artifact not provided",
        )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        envelope = ExternalReportEnvelope.model_validate(payload)
        _validate_external_report(
            payload,
            envelope,
            expected_producer=f"v2_{target_stage.removeprefix('v2_')}_stage_evaluator",
            config=config,
            evaluated_commit=evaluated_commit,
        )
        metrics = payload.get("metrics")
        if not isinstance(metrics, dict):
            raise ValueError("stage report metrics must be an object")
        required_audit_fields = {
            "technical_failure_count",
            "degraded_retrieval_count",
            "invariant_violation_count",
            "schema_invariant_violation_count",
            "provenance_violation_count",
            "citation_violation_count",
            "budget_violation_count",
        }
        if not required_audit_fields <= set(metrics):
            raise ValueError("stage report is missing required audit metrics")
        audit = _audit_counts_from_metrics(metrics)
        audit_clear = not any(audit.model_dump().values())
        runtime_clear = (
            metrics["technical_failure_count"] == 0
            and metrics["degraded_retrieval_count"] == 0
            and metrics["invariant_violation_count"] == 0
        )
        declared_stage = payload.get("target_stage")
        if target_stage == "v2_1":
            stage_ok = declared_stage == "v2_1" and {
                "retrieval_completed_count",
                "grader_completed_count",
            } <= set(metrics)
            complete = (
                runtime_clear
                and audit_clear
            )
        elif target_stage == "v2_2":
            stage_ok = declared_stage == "v2_2" and {
                "recovery_count",
                "finding_count",
            } <= set(metrics)
            complete = (
                runtime_clear
                and audit_clear
            )
        else:
            stage_ok = declared_stage == "v2_3" and {
                "interrupt_count",
                "resume_count",
                "final_execution_status",
            } <= set(metrics)
            cross_ref = payload.get("cross_process_acceptance_ref")
            cross_digest = payload.get("cross_process_acceptance_digest")
            cross_path = Path(cross_ref) if isinstance(cross_ref, str) else None
            cross_link_ok = bool(
                cross_path
                and cross_path.is_file()
                and isinstance(cross_digest, str)
                and _sha256(cross_path) == cross_digest
            )
            complete = (
                int(metrics.get("interrupt_count", 0)) >= 1
                and int(metrics.get("resume_count", 0)) >= 1
                and metrics.get("final_execution_status") == "completed"
                and runtime_clear
                and audit_clear
                and cross_link_ok
            )
        if not stage_ok:
            raise ValueError(f"artifact is not a real {target_stage} stage evaluator report")
        return StageReportReference(
            target_stage=target_stage,
            run_id=envelope.run_id,
            path=str(path),
            digest=_sha256(path),
            available=True,
            evaluation_complete=bool(complete),
            audit_counts=audit,
            cross_process_acceptance_ref=(
                str(cross_path) if target_stage == "v2_3" and cross_path else None
            ),
            cross_process_acceptance_digest=(
                cross_digest if target_stage == "v2_3" else None
            ),
            error=None if complete else "stage evaluator reports incomplete execution",
        )
    except Exception as exc:
        return StageReportReference(
            target_stage=target_stage,
            path=str(path),
            error=_safe_error(exc)["message"],
        )


def _load_evaluator_reports(
    *,
    planning_report: Path | None,
    module4_gold_report: Path | None,
    answer_ragas_report: Path | None,
    config: V2Config,
    evaluated_commit: str | None,
) -> tuple[list[EvaluatorReportReference], dict[str, Any]]:
    planning_ref, planning_metrics = _load_planning_report(
        planning_report, config, evaluated_commit
    )
    module4_ref, module4_metrics = _load_module4_report(
        module4_gold_report, config, evaluated_commit
    )
    answer_ref, answer_metrics = _load_answer_report(
        answer_ragas_report, config, evaluated_commit
    )
    return (
        [planning_ref, module4_ref, answer_ref],
        {
            "planning": planning_metrics,
            "evidence_route": module4_metrics,
            "answer": answer_metrics,
            "runtime": {
                "hitl_waiting_seconds": None,
                "retrieval_latency_seconds": None,
                "role_retries": None,
                "tokens": None,
                "checkpoint_resume_latency_seconds": None,
            },
        },
    )


def _load_stage_runtime_metrics(paths: list[Path | None]) -> dict[str, Any]:
    metrics_by_stage: list[dict[str, Any]] = []
    for path in paths:
        if path is None or not path.is_file():
            continue
        try:
            metrics = json.loads(path.read_text(encoding="utf-8")).get("metrics", {})
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(metrics, dict):
            metrics_by_stage.append(metrics)

    def first_value(*names: str) -> Any:
        return next(
            (
                metrics[name]
                for metrics in metrics_by_stage
                for name in names
                if metrics.get(name) is not None
            ),
            None,
        )

    checkpoint = {
        name: first_value(name)
        for name in (
            "checkpoint_load_latency_seconds",
            "checkpoint_write_latency_seconds",
            "resume_latency_seconds",
        )
    }
    return {
        "hitl_waiting_seconds": first_value("hitl_waiting_seconds"),
        "retrieval_latency_seconds": first_value(
            "retrieval_latency_seconds", "retrieval_total_latency_seconds"
        ),
        "role_retries": first_value("role_retries"),
        "tokens": first_value("tokens", "role_tokens"),
        "checkpoint_resume_latency_seconds": (
            checkpoint if any(value is not None for value in checkpoint.values()) else None
        ),
    }


def _missing_evaluator(
    name: Literal["planning", "module4_grade_route", "answer_ragas"],
    path: Path | None,
    error: str,
) -> EvaluatorReportReference:
    return EvaluatorReportReference(
        evaluator=name,
        path=str(path) if path is not None else None,
        error=error,
    )


def _audit_counts_from_metrics(metrics: dict[str, Any]) -> InvariantCounts:
    return InvariantCounts(
        schema_invariant_violation_count=int(
            metrics.get(
                "schema_invariant_violation_count",
                metrics.get("invariant_violation_count", 0),
            )
        ),
        provenance_violation_count=int(
            metrics.get("provenance_violation_count", 0)
        ),
        citation_violation_count=int(metrics.get("citation_violation_count", 0)),
        budget_violation_count=int(metrics.get("budget_violation_count", 0)),
        retrieval_degraded_queries_count=int(
            metrics.get(
                "retrieval_degraded_queries_count",
                metrics.get("degraded_retrieval_count", 0),
            )
        ),
    )


def _validate_external_report(
    payload: dict[str, Any],
    envelope: ExternalReportEnvelope,
    *,
    expected_producer: str,
    config: V2Config,
    evaluated_commit: str | None,
) -> None:
    if envelope.producer != expected_producer:
        raise ValueError(f"unexpected report producer: {envelope.producer}")
    if evaluated_commit is None or envelope.git_commit != evaluated_commit:
        raise ValueError("report git_commit does not match evaluated commit")
    if envelope.git_dirty:
        raise ValueError("dirty evaluator artifact cannot support a full baseline")
    if envelope.resolved_config != config.resolved_record():
        raise ValueError("report resolved_config does not match baseline config")
    if envelope.artifact_digest != _json_digest(
        {**payload, "artifact_digest": ""}
    ):
        raise ValueError("report artifact_digest mismatch")


def _load_planning_report(
    path: Path | None,
    config: V2Config,
    evaluated_commit: str | None,
) -> tuple[EvaluatorReportReference, dict[str, Any]]:
    empty = {
        "complexity_accuracy": None,
        "capability_accuracy": None,
        "decomposition_requirement_coverage": None,
        "structural_violation_counts": None,
    }
    if path is None or not path.is_file():
        return _missing_evaluator("planning", path, "planning report not provided"), empty
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        envelope = ExternalReportEnvelope.model_validate(payload)
        _validate_external_report(
            payload,
            envelope,
            expected_producer="module2_planning_evaluator",
            config=config,
            evaluated_commit=evaluated_commit,
        )
        dataset = payload["dataset"]
        metrics = payload["metrics"]
        structural = payload["structural_violations"]
        simple_correct = int(metrics["simple_capability_correct"])
        simple_total = int(metrics["simple_capability_total"])
        complex_correct = int(metrics["complex_task_capability_correct"])
        complex_total = int(metrics["complex_task_capability_total"])
        capability_total = simple_total + complex_total
        values = {
            "complexity_accuracy": metrics.get("complexity_accuracy"),
            "capability_accuracy": (
                (simple_correct + complex_correct) / capability_total
                if capability_total
                else None
            ),
            "decomposition_requirement_coverage": metrics.get(
                "micro_pipeline_requirement_coverage"
            ),
            "structural_violation_counts": structural,
        }
        audit = InvariantCounts(
            schema_invariant_violation_count=int(
                structural.get(
                    "schema_invariant_violations",
                    sum(int(value) for value in structural.values()),
                )
            )
        )
        complete = (
            payload.get("evaluation_incomplete") is False
            and dataset.get("qa_sha256") == _digest_if_exists(QA_DATASET)
            and dataset.get("annotation_sha256") == FROZEN_ANNOTATION_DATASET_SHA256
            and all(values[key] is not None for key in values if key != "structural_violation_counts")
        )
        return EvaluatorReportReference(
            evaluator="planning",
            run_id=envelope.run_id,
            path=str(path),
            digest=_sha256(path),
            available=True,
            evaluation_complete=complete,
            audit_counts=audit,
            error=None if complete else "planning evaluation incomplete or dataset identity mismatch",
        ), values
    except Exception as exc:
        return _missing_evaluator("planning", path, _safe_error(exc)["message"]), empty


def _load_module4_report(
    path: Path | None,
    config: V2Config,
    evaluated_commit: str | None,
) -> tuple[EvaluatorReportReference, dict[str, Any]]:
    fields = (
        "relevance_accuracy",
        "answerability_accuracy",
        "ambiguity_accuracy",
        "recoverability_accuracy",
        "failure_reason_accuracy",
        "grade_exact_match_accuracy",
        "route_accuracy",
        "recovery_strategy_accuracy",
    )
    empty = {field: None for field in fields}
    if path is None or not path.is_file():
        return _missing_evaluator(
            "module4_grade_route", path, "Module 4 Gold report not provided"
        ), empty
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        envelope = ExternalReportEnvelope.model_validate(payload)
        _validate_external_report(
            payload,
            envelope,
            expected_producer="module4_gold_replay_evaluator",
            config=config,
            evaluated_commit=evaluated_commit,
        )
        metrics = payload["metrics"]
        values = {field: metrics.get(field) for field in fields}
        if "invariant_violation_count" not in metrics:
            raise ValueError("Module 4 Gold report is missing invariant_violation_count")
        audit = InvariantCounts(
            schema_invariant_violation_count=int(
                metrics["invariant_violation_count"]
            )
        )
        complete = (
            payload.get("dataset", {}).get("sha256") == FROZEN_MODULE4_GOLD_SHA256
            and payload.get("evaluation_incomplete") is False
            and all(value is not None for value in values.values())
            and audit.schema_invariant_violation_count == 0
        )
        return EvaluatorReportReference(
            evaluator="module4_grade_route",
            run_id=envelope.run_id,
            path=str(path),
            digest=_sha256(path),
            available=True,
            evaluation_complete=complete,
            audit_counts=audit,
            error=None if complete else "Module 4 Gold evaluation incomplete or dataset identity mismatch",
        ), values
    except Exception as exc:
        return _missing_evaluator(
            "module4_grade_route", path, _safe_error(exc)["message"]
        ), empty


def _load_answer_report(
    path: Path | None,
    config: V2Config,
    evaluated_commit: str | None,
) -> tuple[EvaluatorReportReference, dict[str, Any]]:
    empty = {
        "supported_subset_ragas": None,
        "outcome_accuracy": None,
        "unsupported_computation_recall": None,
        "abstention_correctness": None,
    }
    if path is None or not path.is_file():
        return _missing_evaluator("answer_ragas", path, "answer RAGAS report not provided"), empty
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        envelope = ExternalReportEnvelope.model_validate(payload)
        _validate_external_report(
            payload,
            envelope,
            expected_producer="v2_answer_ragas_evaluator",
            config=config,
            evaluated_commit=evaluated_commit,
        )
        dataset = payload["dataset"]
        metrics = payload["metrics"]
        required_audit_fields = {
            "schema_invariant_violation_count",
            "provenance_violation_count",
            "citation_violation_count",
            "budget_violation_count",
            "retrieval_degraded_queries_count",
        }
        if not required_audit_fields <= set(metrics):
            raise ValueError("answer report is missing required audit metrics")
        audit = _audit_counts_from_metrics(metrics)
        ragas = metrics.get("supported_subset_ragas")
        complete = (
            dataset.get("qa_sha256") == _digest_if_exists(QA_DATASET)
            and dataset.get("annotation_sha256") == FROZEN_ANNOTATION_DATASET_SHA256
            and payload.get("evaluation_incomplete") is False
            and isinstance(ragas, dict)
            and bool(ragas)
            and all(value is not None for value in ragas.values())
            and metrics.get("outcome_accuracy") is not None
            and metrics.get("unsupported_computation_recall") is not None
            and metrics.get("abstention_correctness") is not None
        )
        return EvaluatorReportReference(
            evaluator="answer_ragas",
            run_id=envelope.run_id,
            path=str(path),
            digest=_sha256(path),
            available=True,
            evaluation_complete=complete,
            audit_counts=audit,
            error=None if complete else "answer/RAGAS evaluation incomplete or dataset mismatch",
        ), {
            "supported_subset_ragas": ragas,
            "outcome_accuracy": metrics.get("outcome_accuracy"),
            "unsupported_computation_recall": metrics.get(
                "unsupported_computation_recall"
            ),
            "abstention_correctness": metrics.get("abstention_correctness"),
        }
    except Exception as exc:
        return _missing_evaluator("answer_ragas", path, _safe_error(exc)["message"]), empty


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


def _check_frozen_dataset_integrity(
    records: list[DatasetRecord],
) -> dict[str, Any]:
    actual = {record.name: record for record in records}
    expected = {
        "retrieval_eval_v2": FROZEN_RETRIEVAL_DATASET_SHA256,
        "v2_qa_annotations": FROZEN_ANNOTATION_DATASET_SHA256,
        "module4_grade_route_gold": FROZEN_MODULE4_GOLD_SHA256,
    }
    checks = {
        name: bool(
            actual.get(name)
            and actual[name].available
            and actual[name].sha256 == digest
        )
        for name, digest in expected.items()
    }
    return {
        "passed": all(checks.values()),
        "evaluated": True,
        "checks": checks,
        "expected_sha256": expected,
        "actual_sha256": {
            name: actual[name].sha256 if name in actual else None for name in expected
        },
    }


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


def _check_cross_process_gate(
    path: Path | None, *, current_git_commit: str | None = None
) -> dict[str, Any]:
    if path is None or not path.is_file():
        return {"passed": False, "evaluated": False, "status": "not_provided" if path is None else "not_found"}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        evidence = CrossProcessEvidence.model_validate(payload)
        digest = evidence.artifact_digest
        digest_ok = digest == _json_digest(
            {key: value for key, value in payload.items() if key != "artifact_digest"}
        )
        names = [step.name for step in evidence.steps]
        negative_names = [item.name for item in evidence.negative_contracts]
        invocations = [*evidence.steps, *evidence.negative_contracts]
        ids_ok = all(
            step.request_id == evidence.request_id
            and step.thread_id == evidence.thread_id
            for step in evidence.steps
        )
        process_evidence_ok = (
            sorted(item.invocation_index for item in invocations)
            == list(range(1, len(invocations) + 1))
            and all(item.command and item.exit_code == 0 for item in invocations)
        )
        negative_ok = negative_names == [
            "invalid_payload",
            "duplicate_resume",
            "stale_resume",
            "expired_checkpoint",
            "lease_recovery",
        ] and all(item.passed for item in evidence.negative_contracts)
        allowed_commits = {MODULE8_FROZEN_IMPLEMENTATION_SHA}
        if current_git_commit is not None:
            allowed_commits.add(current_git_commit)
        commit_ok = evidence.git_commit in allowed_commits
        passed = digest_ok and names == ["start", "status_waiting", "resume", "status_completed"] and ids_ok and evidence.request_id == evidence.thread_id and negative_ok and process_evidence_ok and commit_ok and all(step.passed for step in evidence.steps)
        return {"passed": passed, "evaluated": True, "status": "checked", "path": str(path), "sha256": _sha256(path), "run_id": evidence.run_id, "digest_valid": digest_ok, "git_commit_allowed": commit_ok, "allowed_git_commits": sorted(allowed_commits), "request_id": evidence.request_id, "thread_id": evidence.thread_id, "step_count": len(evidence.steps), "negative_contracts": negative_ok, "process_invocation_evidence": process_evidence_ok}
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
    parser.add_argument(
        "--profile",
        choices=("development_contract", "development_real", "full_baseline"),
    )
    parser.add_argument("--retrieval-report", type=Path, default=DEFAULT_RETRIEVAL_REPORT)
    parser.add_argument("--cross-process-report", type=Path, default=None)
    parser.add_argument("--planning-report", type=Path)
    parser.add_argument("--module4-gold-report", type=Path)
    parser.add_argument("--answer-ragas-report", type=Path)
    parser.add_argument("--stage-v21-report", type=Path)
    parser.add_argument("--stage-v22-report", type=Path)
    parser.add_argument("--stage-v23-report", type=Path)
    args = parser.parse_args()
    report = evaluate_baseline(
        dataset_path=args.dataset,
        output_root=args.output_root,
        mode=args.mode,
        profile=args.profile,
        retrieval_report=args.retrieval_report,
        cross_process_report=args.cross_process_report,
        planning_report=args.planning_report,
        module4_gold_report=args.module4_gold_report,
        answer_ragas_report=args.answer_ragas_report,
        stage_v21_report=args.stage_v21_report,
        stage_v22_report=args.stage_v22_report,
        stage_v23_report=args.stage_v23_report,
    )
    print(json.dumps({"run_id": report["run_id"], "output": str(args.output_root / report["run_id"]), "baseline_status": report["baseline_status"], "failed_gates": report["failed_gates"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
