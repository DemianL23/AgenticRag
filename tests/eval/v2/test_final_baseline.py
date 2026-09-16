from __future__ import annotations

import json
from pathlib import Path

import pytest
from types import SimpleNamespace

from agenticrag.v2.config import V2Config
from agenticrag.v2.schemas import (
    Evidence,
    EvidenceGrade,
    EvidenceOccurrence,
    GradeRecord,
    GroundedFinding,
    QueryRevision,
    RetrievalAttempt,
    RetrievalTask,
    StageRunResult,
    SynthesizedAnswer,
)
import eval.v2.final_baseline as baseline
from eval.v2.final_baseline import (
    ANNOTATION_DATASET,
    FROZEN_ANNOTATION_DATASET_SHA256,
    FROZEN_MODULE4_GOLD_SHA256,
    FROZEN_RETRIEVAL_METRICS,
    MODULE4_GOLD_DATASET,
    MODULE8_FROZEN_IMPLEMENTATION_SHA,
    QA_DATASET,
    RETRIEVAL_DATASET,
    DEFAULT_DATASET,
    _check_cross_process_gate,
    _check_frozen_dataset_integrity,
    _check_retrieval_gate,
    _computation_gate,
    _audit_tasks,
    _json_digest,
    evaluate_baseline,
    load_workflow_scenarios,
    scenario_coverage,
)


def _write_json(path: Path, payload: dict[str, object]) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _mutate_signed_report(path: Path, **metric_updates: object) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["metrics"].update(metric_updates)
    payload["artifact_digest"] = ""
    payload["artifact_digest"] = _json_digest(payload)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _signed_report(
    payload: dict[str, object], *, producer: str, config: V2Config | None = None
) -> dict[str, object]:
    report = {
        **payload,
        "report_schema_version": 1,
        "producer": producer,
        "git_commit": baseline._git_commit() or "0" * 40,
        "git_dirty": False,
        "resolved_config": (config or V2Config()).resolved_record(),
        "artifact_digest": "",
    }
    report["artifact_digest"] = _json_digest(report)
    return report


def _valid_retrieval_report(path: Path) -> Path:
    return _write_json(
        path,
        {
            "dataset": str(RETRIEVAL_DATASET),
            "dataset_size": 47,
            "evaluated_queries": 47,
            "fallback_queries": 0,
            "invalid_score_queries": 0,
            "final_metrics": {
                key: value
                for key, value in FROZEN_RETRIEVAL_METRICS.items()
                if key != "Recall@20"
            },
            "rrf_candidate_metrics": {
                "Recall@20": FROZEN_RETRIEVAL_METRICS["Recall@20"]
            },
        },
    )


def _valid_cross_process_report(path: Path, *, git_commit: str) -> Path:
    request_id = "8db660d1-9b3b-4d68-8dca-73b32bd5374e"
    step_names = (
        "start",
        "status_waiting",
        "resume",
        "status_completed",
    )
    negative_names = (
        "invalid_payload",
        "stale_resume",
        "duplicate_resume",
        "expired_checkpoint",
        "lease_recovery",
    )
    invocation_order = {
        "start": 1,
        "status_waiting": 2,
        "invalid_payload": 3,
        "stale_resume": 4,
        "resume": 5,
        "duplicate_resume": 6,
        "status_completed": 7,
        "expired_checkpoint": 8,
        "lease_recovery": 9,
    }

    def invocation(name: str, index: int) -> dict[str, object]:
        state_digest = "a" * 64
        waiting_state = {
            "pending_hitl_request_id": "HITL_001",
            "execution_status": "waiting_user",
        }
        result: dict[str, object] = {"contract": name}
        if name == "invalid_payload":
            result.update(
                error_code="resume_payload_invalid",
                execution_status="waiting_user",
            )
        if name == "stale_resume":
            result.update(
                error_code="resume_conflict",
                execution_status="waiting_user",
                resumable=True,
                business_state_before=waiting_state,
                business_state_after=dict(waiting_state),
            )
        if name == "duplicate_resume":
            result.update(
                execution_status="completed",
                new_query_revision_count=1,
                hitl_rounds=1,
            )
        if name == "expired_checkpoint":
            result.update(error_code="checkpoint_expired")
        if name == "lease_recovery":
            result.update(execution_status="completed", answer_outcome="complete")
        return {
            "name": name,
            "passed": True,
            "request_id": request_id,
            "thread_id": request_id,
            "invocation_index": index,
            "command": ["python", "-m", "eval.v2.module8_worker", name],
            "exit_code": 0,
            "pid": index,
            "business_state_before_digest": state_digest,
            "business_state_after_digest": state_digest,
            "telemetry": {
                "retrieval_calls": 0,
                "grader_calls": 0,
                "finding_calls": 0,
                "synthesis_calls": 0,
                "hitl_calls": 0,
            },
            "result": result,
        }

    payload: dict[str, object] = {
        "schema_version": 1,
        "producer": "module8_cross_process_acceptance",
        "run_id": "cross-process-run",
        "git_commit": git_commit,
        "request_id": request_id,
        "thread_id": request_id,
        "steps": [
            invocation(name, invocation_order[name]) for name in step_names
        ],
        "negative_contracts": [
            invocation(name, invocation_order[name]) for name in negative_names
        ],
        "artifact_digest": "",
    }
    payload["artifact_digest"] = _json_digest(
        {key: value for key, value in payload.items() if key != "artifact_digest"}
    )
    return _write_json(path, payload)


def _mutate_cross_invocation(
    path: Path,
    name: str,
    *,
    telemetry: dict[str, int] | None = None,
    after_digest: str | None = None,
    request_id: str | None = None,
    thread_id: str | None = None,
    result_updates: dict[str, object] | None = None,
) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    invocation = next(
        item
        for item in [*payload["steps"], *payload["negative_contracts"]]
        if item["name"] == name
    )
    if telemetry is not None:
        invocation["telemetry"].update(telemetry)
    if after_digest is not None:
        invocation["business_state_after_digest"] = after_digest
    if request_id is not None:
        invocation["request_id"] = request_id
    if thread_id is not None:
        invocation["thread_id"] = thread_id
    if result_updates is not None:
        invocation["result"].update(result_updates)
    payload["artifact_digest"] = _json_digest(
        {key: value for key, value in payload.items() if key != "artifact_digest"}
    )
    path.write_text(json.dumps(payload), encoding="utf-8")


def _valid_evaluator_and_stage_reports(root: Path) -> dict[str, Path]:
    annotation_ids = [
        json.loads(line)["finqa_id"]
        for line in ANNOTATION_DATASET.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    planning = _write_json(
        root / "planning.json",
        _signed_report({
            "run_id": "planning-run",
            "dataset": {
                "qa_sha256": baseline._sha256(QA_DATASET),
                "annotation_sha256": FROZEN_ANNOTATION_DATASET_SHA256,
            },
            "metrics": {
                "complexity_accuracy": 0.8,
                "simple_capability_correct": 5,
                "simple_capability_total": 6,
                "complex_task_capability_correct": 10,
                "complex_task_capability_total": 12,
                "micro_pipeline_requirement_coverage": 0.75,
            },
            "structural_violations": {"empty_task_violations": 0},
            "evaluation_incomplete": False,
        }, producer="module2_planning_evaluator"),
    )
    module4 = _write_json(
        root / "module4.json",
        _signed_report({
            "run_id": "module4-run",
            "dataset": {"sha256": FROZEN_MODULE4_GOLD_SHA256},
            "metrics": {
                name: 1.0
                for name in (
                    "relevance_accuracy",
                    "answerability_accuracy",
                    "ambiguity_accuracy",
                    "recoverability_accuracy",
                    "failure_reason_accuracy",
                    "grade_exact_match_accuracy",
                    "route_accuracy",
                    "recovery_strategy_accuracy",
                )
            } | {"invariant_violation_count": 0},
            "evaluation_incomplete": False,
        }, producer="module4_gold_replay_evaluator"),
    )
    answer = _write_json(
        root / "answer.json",
        _signed_report({
            "run_id": "answer-run",
            "dataset": {
                "qa_sha256": baseline._sha256(QA_DATASET),
                "annotation_sha256": FROZEN_ANNOTATION_DATASET_SHA256,
                "sample_count": len(annotation_ids),
            },
            "metrics": {
                "supported_subset_ragas": {
                    "faithfulness": 0.9,
                    "answer_correctness": 0.8,
                },
                "outcome_accuracy": 0.9,
                "unsupported_computation_recall": 1.0,
                "abstention_correctness": 1.0,
                "schema_invariant_violation_count": 0,
                "provenance_violation_count": 0,
                "citation_violation_count": 0,
                "budget_violation_count": 0,
                "retrieval_degraded_queries_count": 0,
            },
            "evaluation_incomplete": False,
        }, producer="v2_answer_ragas_evaluator"),
    )
    v21 = _write_json(
        root / "v21.json",
        _signed_report({
            "run_id": "v21-run",
            "target_stage": "v2_1",
            "metrics": {
                "retrieval_completed_count": 1,
                "grader_completed_count": 1,
                "technical_failure_count": 0,
                "degraded_retrieval_count": 0,
                "invariant_violation_count": 0,
                "schema_invariant_violation_count": 0,
                "provenance_violation_count": 0,
                "citation_violation_count": 0,
                "budget_violation_count": 0,
            },
            "evaluation_incomplete": False,
        }, producer="v2_1_stage_evaluator"),
    )
    v22 = _write_json(
        root / "v22.json",
        _signed_report({
            "run_id": "v22-run",
            "target_stage": "v2_2",
            "metrics": {
                "recovery_count": 1,
                "finding_count": 1,
                "technical_failure_count": 0,
                "invariant_violation_count": 0,
                "degraded_retrieval_count": 0,
                "schema_invariant_violation_count": 0,
                "provenance_violation_count": 0,
                "citation_violation_count": 0,
                "budget_violation_count": 0,
            },
        }, producer="v2_2_stage_evaluator"),
    )
    stage_cross = _valid_cross_process_report(
        root / "stage-cross.json", git_commit=baseline._git_commit() or "0" * 40
    )
    v23 = _write_json(
        root / "v23.json",
        _signed_report({
            "run_id": "v23-run",
            "target_stage": "v2_3",
            "cross_process_acceptance_ref": str(stage_cross),
            "cross_process_acceptance_digest": baseline._sha256(stage_cross),
            "metrics": {
                "interrupt_count": 1,
                "resume_count": 1,
                "final_execution_status": "completed",
                "invariant_violation_count": 0,
                "technical_failure_count": 0,
                "degraded_retrieval_count": 0,
                "schema_invariant_violation_count": 0,
                "provenance_violation_count": 0,
                "citation_violation_count": 0,
                "budget_violation_count": 0,
            },
        }, producer="v2_3_stage_evaluator"),
    )
    return {
        "planning_report": planning,
        "module4_gold_report": module4,
        "answer_ragas_report": answer,
        "stage_v21_report": v21,
        "stage_v22_report": v22,
        "stage_v23_report": v23,
        "cross_process_report": stage_cross,
    }


def _passing_scenario_prediction(
    scenario: object, _config: object
) -> dict[str, object]:
    expected = scenario.expected.model_dump(mode="json")
    return {
        "scenario_id": scenario.scenario_id,
        "mode": scenario.mode,
        "stage": scenario.stage,
        "expected_capability": scenario.capability,
        "passed": True,
        "status": (
            "technical_failure"
            if scenario.expected.technical_failure
            else scenario.expected.execution_status
        ),
        "expected": expected,
        "observed": {
            "answer_outcome": scenario.expected.answer_outcome,
            "task_capabilities": (
                [scenario.capability]
                if scenario.capability != "mixed"
                else ["retrieval_synthesis", "arithmetic"]
            ),
            "route": scenario.expected.route,
            "route_sequence": scenario.expected.route_sequence,
            "retrieval_attempt_count": scenario.expected.retrieval_attempt_count or 0,
            "schema_invariant_violation_count": 0,
            "provenance_violation_count": 0,
            "citation_violation_count": 0,
            "budget_violation_count": 0,
            "degraded_retrieval_count": 0,
        },
        "violations": [],
        "elapsed_seconds": 0.0,
    }


def test_workflow_scenario_dataset_has_explicit_coverage() -> None:
    records = load_workflow_scenarios(DEFAULT_DATASET)
    coverage = scenario_coverage(records)
    assert coverage["scenario_count"] == 30
    assert coverage["stage_counts"] == {"v2_1": 2, "v2_2": 19, "v2_3": 9}
    for tag in (
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
    ):
        assert coverage["tag_counts"][tag] >= 2


def test_workflow_scenario_schema_rejects_invalid_route_strategy_pair(tmp_path: Path) -> None:
    path = tmp_path / "invalid.jsonl"
    record = {
        "scenario_id": "bad",
        "stage": "v2_2",
        "mode": "contract",
        "question": "question",
        "language": "en",
        "complexity": "simple",
        "capability": "retrieval_synthesis",
        "tags": ["supported"],
        "expected": {
            "execution_status": "completed",
            "answer_outcome": "complete",
            "route": "answer",
            "recovery_strategy": "hyde",
        },
    }
    path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="only recover"):
        load_workflow_scenarios(path)


def test_contract_baseline_report_is_immutable_and_contains_digests(tmp_path: Path) -> None:
    retrieval_report = tmp_path / "retrieval.json"
    retrieval_report.write_text(
        json.dumps(
                {
                    "dataset": str(RETRIEVAL_DATASET),
                    "dataset_size": 47,
                    "evaluated_queries": 47,
                    "fallback_queries": 0,
                    "invalid_score_queries": 0,
                    "final_metrics": {key: value for key, value in FROZEN_RETRIEVAL_METRICS.items() if key != "Recall@20"},
                    "rrf_candidate_metrics": {"Recall@20": FROZEN_RETRIEVAL_METRICS["Recall@20"]},
            }
        ),
        encoding="utf-8",
    )
    report = evaluate_baseline(
        dataset_path=DEFAULT_DATASET,
        output_root=tmp_path / "reports",
        run_id="candidate-1",
        mode="contract",
        config=V2Config(),
        retrieval_report=retrieval_report,
    )
    assert report["metrics"]["contract_scenario_count"] == 27
    assert report["metrics"]["technical_failure_count"] == 0
    assert report["hard_gates"]["dataset_schema_and_coverage"] is True
    assert report["hard_gates"]["v1_2_retrieval_regression"] is True
    assert report["hard_gates"]["baseline_scenarios_terminal_contract"] is True
    assert report["hard_gates"]["v2_3_cross_process"] is False
    assert report["evaluation_profile"] == "development_contract"
    assert report["evaluation_incomplete"] is True
    assert report["freeze_eligible"] is False
    assert len(report["digests"]["dataset_sha256"]) == 64
    assert len(report["artifact_digest"]) == 64
    assert {item["name"] for item in report["dataset_records"]} >= {
        "retrieval_eval_v2",
        "qa",
        "v2_qa_annotations",
        "v2_workflow_scenarios",
        "module4_grade_route_gold",
    }
    report_path = tmp_path / "reports" / "candidate-1" / "report.json"
    assert json.loads(report_path.read_text(encoding="utf-8"))["digests"] == report["digests"]
    assert all(item["evaluation_complete"] is False for item in report["stage_reports"])
    assert not (report_path.parent / "stage_v2_1.json").exists()
    with pytest.raises(FileExistsError):
        evaluate_baseline(
            dataset_path=DEFAULT_DATASET,
            output_root=tmp_path / "reports",
            run_id="candidate-1",
            mode="contract",
            config=V2Config(),
            retrieval_report=retrieval_report,
        )


def test_real_mode_is_explicitly_separate_from_contract_mode(tmp_path: Path) -> None:
    records = load_workflow_scenarios(DEFAULT_DATASET)
    assert sum(record.mode == "real" for record in records) == 3
    assert sum(record.mode == "contract" for record in records) == 27


def test_contract_scenarios_execute_frozen_graph_and_recovery_history(tmp_path: Path) -> None:
    report = evaluate_baseline(
        dataset_path=DEFAULT_DATASET,
        output_root=tmp_path / "reports",
        run_id="contract-execution",
        mode="contract",
        config=V2Config(),
    )
    predictions = [
        json.loads(line)
        for line in (tmp_path / "reports" / "contract-execution" / "predictions.jsonl").read_text().splitlines()
    ]
    recovery = next(item for item in predictions if item["scenario_id"] == "contract_direct_success")
    assert recovery["passed"] is True
    assert recovery["observed"]["route_sequence"] == ["recover", "answer"]
    assert recovery["observed"]["recovery_strategies"] == ["direct_rewrite"]
    assert recovery["observed"]["telemetry"]["retrieval_calls"] == 2
    assert report["metrics"]["scenario_pass_count"] == 27


def test_weak_47_query_retrieval_report_fails_regression_gate(tmp_path: Path) -> None:
    weak = tmp_path / "weak.json"
    weak.write_text(
        json.dumps(
            {
                "dataset": str(RETRIEVAL_DATASET),
                "dataset_size": 47,
                "evaluated_queries": 47,
                "fallback_queries": 0,
                "invalid_score_queries": 0,
                "final_metrics": {"Recall@1": 0.0, "Recall@3": 0.0, "Recall@5": 0.0, "MRR@5": 0.0},
                "rrf_candidate_metrics": {"Recall@20": 0.0},
            }
        ),
        encoding="utf-8",
    )
    result = _check_retrieval_gate(weak)
    assert result["evaluated"] is True
    assert result["passed"] is False


def test_arbitrary_cross_process_pass_boolean_cannot_pass(tmp_path: Path) -> None:
    path = tmp_path / "cross.json"
    path.write_text(json.dumps({"passed": True}), encoding="utf-8")
    result = _check_cross_process_gate(path)
    assert result["passed"] is False
    assert result["evaluated"] is False


def test_missing_cross_process_evidence_makes_candidate_incomplete(tmp_path: Path) -> None:
    report = evaluate_baseline(
        dataset_path=DEFAULT_DATASET,
        output_root=tmp_path / "reports",
        run_id="incomplete",
        mode="contract",
        config=V2Config(),
    )
    assert report["evaluation_incomplete"] is True
    assert report["freeze_eligible"] is False
    assert report["baseline_status"] == "not_eligible"
    assert "v2_3_cross_process" in report["failed_gates"]


def test_cross_process_evidence_cannot_substitute_for_v23_stage_report(
    tmp_path: Path,
) -> None:
    cross = _valid_cross_process_report(
        tmp_path / "cross.json", git_commit=baseline._git_commit() or "0" * 40
    )
    report = evaluate_baseline(
        output_root=tmp_path / "out",
        run_id="cross-is-not-stage",
        mode="contract",
        config=V2Config(),
        cross_process_report=cross,
    )
    v23 = next(
        item for item in report["stage_reports"] if item["target_stage"] == "v2_3"
    )
    assert report["hard_gates"]["v2_3_cross_process"] is True
    assert v23["evaluation_complete"] is False
    assert report["evaluation_completeness"]["stage_reports_evaluated"] is False


def test_report_digest_round_trip_and_immutable_output(tmp_path: Path) -> None:
    retrieval = tmp_path / "retrieval.json"
    retrieval.write_text(
        json.dumps(
            {
                "dataset": str(RETRIEVAL_DATASET),
                "dataset_size": 47,
                "evaluated_queries": 47,
                "fallback_queries": 0,
                "invalid_score_queries": 0,
                "final_metrics": {key: value for key, value in FROZEN_RETRIEVAL_METRICS.items() if key != "Recall@20"},
                "rrf_candidate_metrics": {"Recall@20": FROZEN_RETRIEVAL_METRICS["Recall@20"]},
            }
        ),
        encoding="utf-8",
    )
    root = tmp_path / "reports"
    report = evaluate_baseline(dataset_path=DEFAULT_DATASET, output_root=root, run_id="digest", mode="contract", config=V2Config(), retrieval_report=retrieval)
    persisted = json.loads((root / "digest" / "report.json").read_text())
    assert persisted["artifact_digest"] == report["artifact_digest"]
    with pytest.raises(FileExistsError):
        evaluate_baseline(dataset_path=DEFAULT_DATASET, output_root=root, run_id="digest", mode="contract", config=V2Config(), retrieval_report=retrieval)


def test_wrong_computation_answer_fails_computation_gate() -> None:
    predictions = [{"scenario_id": "contract_unsupported_arithmetic", "expected_capability": "arithmetic", "observed": {"answer_outcome": "complete", "task_capabilities": ["arithmetic"]}}]
    assert _computation_gate(predictions) is False


def test_provenance_auditor_counts_wrong_task_finding() -> None:
    task = RetrievalTask(
        id="SQ_001", ordinal=1, query="q", intent="i", capability="retrieval_synthesis",
        grounded_finding=GroundedFinding(task_id="SQ_999", text="bad", evidence_ids=["E1"]),
    )
    evidence = {"E1": Evidence(evidence_id="E1", chunk_id="E1", content="c", doc_id="d", source="s", page=1, occurrences=[])}
    stage = SimpleNamespace(execution_status="completed", answer_outcome="complete", final_answer=None)
    _, counts = _audit_tasks([task], evidence, stage, config=V2Config())
    assert counts.provenance_violation_count == 1


def test_git_dirty_disqualifies_candidate(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(baseline, "_git_dirty", lambda: True)
    report = evaluate_baseline(dataset_path=DEFAULT_DATASET, output_root=tmp_path / "reports", run_id="dirty", mode="contract", config=V2Config())
    assert report["git_dirty"] is True
    assert report["freeze_eligible"] is False
    assert report["baseline_status"] == "not_eligible"


def test_contract_only_is_never_eligible_even_with_valid_cross_evidence(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    commit = "c" * 40
    monkeypatch.setattr(baseline, "_git_commit", lambda: commit)
    monkeypatch.setattr(baseline, "_git_dirty", lambda: False)
    reports = _valid_evaluator_and_stage_reports(tmp_path)
    result = evaluate_baseline(
        output_root=tmp_path / "out",
        run_id="contract-only",
        mode="contract",
        config=V2Config(),
        retrieval_report=_valid_retrieval_report(tmp_path / "retrieval.json"),
        **reports,
    )
    assert result["evaluation_profile"] == "development_contract"
    assert result["evaluation_incomplete"] is True
    assert result["freeze_eligible"] is False


def test_real_only_profile_is_never_eligible(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def fake_real(scenario: object, _config: object) -> dict[str, object]:
        return {
            "scenario_id": scenario.scenario_id,
            "mode": "real",
            "stage": scenario.stage,
            "expected_capability": scenario.capability,
            "passed": True,
            "status": scenario.expected.execution_status,
            "expected": scenario.expected.model_dump(mode="json"),
            "observed": {
                "answer_outcome": scenario.expected.answer_outcome,
                "task_capabilities": [scenario.capability],
            },
            "violations": [],
            "elapsed_seconds": 0.0,
        }

    monkeypatch.setattr(baseline, "_real_prediction", fake_real)
    monkeypatch.setattr(baseline, "_git_dirty", lambda: False)
    result = evaluate_baseline(
        output_root=tmp_path / "out",
        run_id="real-only",
        mode="real",
        config=V2Config(),
    )
    assert result["evaluation_profile"] == "development_real"
    assert result["evaluation_incomplete"] is True
    assert result["freeze_eligible"] is False


def test_full_profile_missing_planning_evaluation_is_incomplete(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(baseline, "_git_dirty", lambda: False)
    monkeypatch.setattr(baseline, "_contract_prediction", _passing_scenario_prediction)
    monkeypatch.setattr(baseline, "_real_prediction", _passing_scenario_prediction)
    reports = _valid_evaluator_and_stage_reports(tmp_path)
    reports["planning_report"] = tmp_path / "missing-planning.json"
    result = evaluate_baseline(
        output_root=tmp_path / "out",
        run_id="missing-planning",
        profile="full_baseline",
        config=V2Config(),
        **reports,
    )
    assert result["metrics"]["planning"]["complexity_accuracy"] is None
    assert result["evaluation_completeness"]["planning_evaluator_evaluated"] is False
    assert result["evaluation_incomplete"] is True


def test_planning_provider_failure_stays_incomplete_without_schema_mislabel(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    commit = "c" * 40
    monkeypatch.setattr(baseline, "_git_commit", lambda: commit)
    monkeypatch.setattr(baseline, "_git_dirty", lambda: False)
    monkeypatch.setattr(baseline, "_contract_prediction", _passing_scenario_prediction)
    monkeypatch.setattr(baseline, "_real_prediction", _passing_scenario_prediction)
    reports = _valid_evaluator_and_stage_reports(tmp_path)
    payload = json.loads(reports["planning_report"].read_text(encoding="utf-8"))
    payload["evaluation_incomplete"] = True
    payload["structural_violations"] = {
        "schema_invariant_violations": 0,
        "technical_planning_failures": 1,
    }
    payload["artifact_digest"] = ""
    payload["artifact_digest"] = _json_digest(payload)
    reports["planning_report"].write_text(json.dumps(payload), encoding="utf-8")

    result = evaluate_baseline(
        output_root=tmp_path / "out",
        run_id="planning-provider-incomplete",
        profile="full_baseline",
        config=V2Config(),
        retrieval_report=_valid_retrieval_report(tmp_path / "retrieval.json"),
        **reports,
    )

    assert result["invariant_counts"]["schema_invariant_violation_count"] == 0
    assert result["hard_gates"]["schema_invariant_zero"] is True
    assert result["evaluation_incomplete"] is True
    assert result["freeze_eligible"] is False


def test_actual_planning_and_module4_artifact_metrics_are_populated(
    tmp_path: Path,
) -> None:
    reports = _valid_evaluator_and_stage_reports(tmp_path)
    result = evaluate_baseline(
        output_root=tmp_path / "out",
        run_id="integrated-metrics",
        mode="contract",
        config=V2Config(),
        **reports,
    )
    assert result["metrics"]["planning"] == {
        "complexity_accuracy": 0.8,
        "capability_accuracy": 15 / 18,
        "decomposition_requirement_coverage": 0.75,
        "structural_violation_counts": {"empty_task_violations": 0},
    }
    assert result["metrics"]["evidence_route"]["route_accuracy"] == 1.0
    assert result["metrics"]["evidence_route"]["grade_exact_match_accuracy"] == 1.0
    assert result["metrics"]["answer"]["supported_subset_ragas"] == {
        "faithfulness": 0.9,
        "answer_correctness": 0.8,
    }


def test_missing_answer_evaluator_is_unavailable_not_zero(tmp_path: Path) -> None:
    result = evaluate_baseline(
        output_root=tmp_path / "out",
        run_id="missing-answer",
        mode="contract",
        config=V2Config(),
    )
    assert result["metrics"]["answer"]["supported_subset_ragas"] is None
    assert result["evaluation_completeness"]["answer_ragas_evaluator_evaluated"] is False
    assert result["evaluation_incomplete"] is True


def test_integrated_answer_ragas_failure_is_sanitized_and_blocks_eligibility(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    async def broken_answer_evaluator(**_kwargs: object) -> dict[str, object]:
        raise RuntimeError("Authorization: Bearer answer-ragas-secret initialization failed")

    monkeypatch.setattr(baseline, "_git_dirty", lambda: False)
    monkeypatch.setattr(baseline, "_contract_prediction", _passing_scenario_prediction)
    monkeypatch.setattr(baseline, "_real_prediction", _passing_scenario_prediction)
    monkeypatch.setattr(
        "eval.v2.answer_baseline.evaluate_v2_answers", broken_answer_evaluator
    )
    reports = _valid_evaluator_and_stage_reports(tmp_path)
    reports["answer_ragas_report"] = None
    result = evaluate_baseline(
        output_root=tmp_path / "out",
        run_id="answer-ragas-init-failure",
        profile="full_baseline",
        config=V2Config(),
        retrieval_report=_valid_retrieval_report(tmp_path / "retrieval.json"),
        **reports,
    )

    assert result["baseline_status"] == "not_eligible"
    assert result["freeze_eligible"] is False
    assert result["evaluation_incomplete"] is True
    assert result["evaluation_completeness"]["answer_ragas_evaluator_evaluated"] is False
    assert result["evaluator_failures"] == [
        {
            "evaluator": "answer_ragas",
            "exception_type": "RuntimeError",
            "summary": "Authorization=[REDACTED] initialization failed",
            "validation_errors": [],
        }
    ]
    serialized = json.dumps(result)
    assert "answer-ragas-secret" not in serialized
    assert "answer RAGAS report not provided" == next(
        item["error"]
        for item in result["evaluator_reports"]
        if item["evaluator"] == "answer_ragas"
    )


def test_terminal_unresolved_tag_rejects_waiting_user(tmp_path: Path) -> None:
    records = [
        json.loads(line)
        for line in DEFAULT_DATASET.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    target = next(item for item in records if item["scenario_id"] == "contract_unresolved_success")
    target["expected"].update(
        execution_status="waiting_user", answer_outcome=None, resumable=True
    )
    path = tmp_path / "invalid-unresolved.jsonl"
    path.write_text("".join(json.dumps(item) + "\n" for item in records))
    with pytest.raises(ValueError, match="terminal_unresolved"):
        load_workflow_scenarios(path)


def test_true_hitl_budget_exhaustion_and_bounded_recovery_are_executed(
    tmp_path: Path,
) -> None:
    evaluate_baseline(
        output_root=tmp_path / "out",
        run_id="history",
        mode="contract",
        config=V2Config(),
    )
    predictions = {
        item["scenario_id"]: item
        for item in map(
            json.loads,
            (tmp_path / "out" / "history" / "predictions.jsonl").read_text().splitlines(),
        )
    }
    unresolved = predictions["contract_unresolved_success"]
    assert unresolved["observed"]["answer_outcome"] == "unresolved"
    assert unresolved["observed"]["hitl_rounds"] == 1
    assert unresolved["observed"]["query_revision_count"] == 2
    for scenario_id in (
        "contract_direct_boundary",
        "contract_step_boundary",
        "contract_hyde_boundary",
    ):
        item = predictions[scenario_id]
        assert item["observed"]["route_sequence"][-2:] == ["recover", "no_knowledge"]
        assert not any(attempt.endswith("_003") for attempt in item["observed"]["retrieval_attempt_ids"])


@pytest.mark.parametrize(
    ("dataset_name", "source_path"),
    [
        ("v2_qa_annotations", ANNOTATION_DATASET),
        ("module4_grade_route_gold", MODULE4_GOLD_DATASET),
    ],
)
def test_modified_frozen_dataset_fails_integrity(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    dataset_name: str,
    source_path: Path,
) -> None:
    modified = tmp_path / source_path.name
    modified.write_bytes(source_path.read_bytes() + b"\n")
    if dataset_name == "v2_qa_annotations":
        monkeypatch.setattr(baseline, "ANNOTATION_DATASET", modified)
    else:
        monkeypatch.setattr(baseline, "MODULE4_GOLD_DATASET", modified)
    records = baseline._all_dataset_records(DEFAULT_DATASET)
    result = _check_frozen_dataset_integrity(records)
    assert result["passed"] is False
    assert result["checks"][dataset_name] is False
    report = evaluate_baseline(
        output_root=tmp_path / "out",
        run_id=f"integrity-{dataset_name}",
        mode="contract",
        config=V2Config(),
    )
    assert report["hard_gates"]["frozen_dataset_integrity"] is False
    assert report["evaluation_incomplete"] is True
    assert report["freeze_eligible"] is False


def test_cross_process_evidence_enforces_git_identity(tmp_path: Path) -> None:
    wrong = _valid_cross_process_report(tmp_path / "wrong.json", git_commit="f" * 40)
    assert _check_cross_process_gate(
        wrong, current_git_commit="c" * 40
    )["passed"] is False
    allowed = _valid_cross_process_report(
        tmp_path / "allowed.json", git_commit=MODULE8_FROZEN_IMPLEMENTATION_SHA
    )
    result = _check_cross_process_gate(allowed, current_git_commit="c" * 40)
    assert result["passed"] is True
    assert result["git_commit_allowed"] is True


@pytest.mark.parametrize(
    ("name", "telemetry"),
    [
        ("duplicate_resume", {"retrieval_calls": 1}),
        ("invalid_payload", {"grader_calls": 1}),
        ("invalid_payload", {"hitl_calls": 1}),
        ("expired_checkpoint", {"retrieval_calls": 1}),
    ],
)
def test_cross_process_negative_contract_rejects_execution_telemetry(
    tmp_path: Path, name: str, telemetry: dict[str, int]
) -> None:
    path = _valid_cross_process_report(
        tmp_path / f"{name}.json", git_commit=MODULE8_FROZEN_IMPLEMENTATION_SHA
    )
    _mutate_cross_invocation(path, name, telemetry=telemetry)

    result = _check_cross_process_gate(path)

    assert result["passed"] is False
    assert result["negative_zero_execution"] is False


@pytest.mark.parametrize(
    "name", [
        "duplicate_resume",
        "invalid_payload",
        "stale_resume",
        "expired_checkpoint",
    ]
)
def test_cross_process_negative_contract_rejects_business_state_mutation(
    tmp_path: Path, name: str
) -> None:
    path = _valid_cross_process_report(
        tmp_path / f"{name}.json", git_commit=MODULE8_FROZEN_IMPLEMENTATION_SHA
    )
    _mutate_cross_invocation(path, name, after_digest="b" * 64)

    result = _check_cross_process_gate(path)

    assert result["passed"] is False
    assert result["negative_zero_business_mutation"] is False


def test_cross_process_stale_resume_requires_pending_waiting_evidence(
    tmp_path: Path,
) -> None:
    path = _valid_cross_process_report(
        tmp_path / "stale-after-completed.json",
        git_commit=MODULE8_FROZEN_IMPLEMENTATION_SHA,
    )
    _mutate_cross_invocation(
        path,
        "stale_resume",
        result_updates={
            "execution_status": "completed",
            "resumable": False,
            "business_state_after": {"pending_hitl_request_id": None},
        },
    )

    result = _check_cross_process_gate(path)

    assert result["passed"] is False
    assert result["stale_pending_identity"] is False


def test_cross_process_primary_negative_contract_requires_main_identity(
    tmp_path: Path,
) -> None:
    path = _valid_cross_process_report(
        tmp_path / "foreign-stale.json",
        git_commit=MODULE8_FROZEN_IMPLEMENTATION_SHA,
    )
    _mutate_cross_invocation(
        path,
        "stale_resume",
        request_id="other-request",
        thread_id="other-request",
    )

    result = _check_cross_process_gate(path)

    assert result["passed"] is False
    assert result["primary_request_negative_identity"] is False


def test_cross_process_valid_expired_and_waiting_stale_contracts_pass(
    tmp_path: Path,
) -> None:
    path = _valid_cross_process_report(
        tmp_path / "valid-negative-contracts.json",
        git_commit=MODULE8_FROZEN_IMPLEMENTATION_SHA,
    )

    result = _check_cross_process_gate(path)

    assert result["passed"] is True
    assert result["negative_zero_execution"] is True
    assert result["negative_zero_business_mutation"] is True
    assert result["stale_pending_identity"] is True


def test_provenance_auditor_rejects_wrong_task_occurrence() -> None:
    attempt = RetrievalAttempt(
        id="ATT_SQ001_QR001_001",
        ordinal=1,
        strategy="original",
        retrieval_query="q",
        evidence_ids=["E1"],
    )
    revision = QueryRevision(
        id="QR_SQ001_001",
        ordinal=1,
        source="original",
        query="q",
        retrieval_attempts=[attempt],
    )
    grade = EvidenceGrade(
        relevance="strong",
        answerability="sufficient",
        ambiguity="none",
        recoverability="none",
        failure_reason="none",
        reason="supported",
        supporting_evidence_ids=["E1"],
    )
    record = GradeRecord(
        id="GR_SQ001_001",
        query_revision_id=revision.id,
        input_attempt_ids=[attempt.id],
        input_evidence_ids=["E1"],
        grade=grade,
    )
    task = RetrievalTask(
        id="SQ_001",
        ordinal=1,
        query="q",
        intent="i",
        capability="retrieval_synthesis",
        query_revisions=[revision],
        grade_records=[record],
        grounded_finding=GroundedFinding(
            task_id="SQ_001", text="fact", evidence_ids=["E1"]
        ),
    )
    evidence = {
        "E1": Evidence(
            evidence_id="E1",
            chunk_id="E1",
            content="fact",
            doc_id="d",
            source="s",
            page=1,
            occurrences=[
                EvidenceOccurrence(
                    task_id="SQ_999",
                    query_revision_id=revision.id,
                    retrieval_attempt_id=attempt.id,
                    strategy="original",
                    final_rank=1,
                )
            ],
        )
    }
    stage = SimpleNamespace(execution_status="completed", answer_outcome="complete", final_answer=None)
    _, counts = _audit_tasks([task], evidence, stage, config=V2Config())
    assert counts.provenance_violation_count == 1


def test_only_full_baseline_profile_can_become_eligible(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    commit = "c" * 40
    monkeypatch.setattr(baseline, "_git_commit", lambda: commit)
    monkeypatch.setattr(baseline, "_git_dirty", lambda: False)

    monkeypatch.setattr(baseline, "_contract_prediction", _passing_scenario_prediction)
    monkeypatch.setattr(baseline, "_real_prediction", _passing_scenario_prediction)
    reports = _valid_evaluator_and_stage_reports(tmp_path)
    result = evaluate_baseline(
        output_root=tmp_path / "out",
        run_id="full",
        profile="full_baseline",
        config=V2Config(),
        retrieval_report=_valid_retrieval_report(tmp_path / "retrieval.json"),
        **reports,
    )
    assert result["evaluation_completeness"]["complete"] is True
    assert result["evaluation_incomplete"] is False
    assert result["freeze_eligible"] is True
    assert result["baseline_status"] == "eligible"


def test_real_nontechnical_terminal_mismatch_blocks_freeze(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    commit = "c" * 40
    monkeypatch.setattr(baseline, "_git_commit", lambda: commit)
    monkeypatch.setattr(baseline, "_git_dirty", lambda: False)
    monkeypatch.setattr(baseline, "_contract_prediction", _passing_scenario_prediction)

    def mismatched_real(scenario: object, config: object) -> dict[str, object]:
        result = _passing_scenario_prediction(scenario, config)
        if scenario.scenario_id == "real_v22_complex_zh":
            result["passed"] = False
            result["status"] = "completed"
            result["observed"]["answer_outcome"] = "partial"
            result["violations"] = ["answer_outcome mismatch"]
        return result

    monkeypatch.setattr(baseline, "_real_prediction", mismatched_real)
    reports = _valid_evaluator_and_stage_reports(tmp_path)
    result = evaluate_baseline(
        output_root=tmp_path / "out",
        run_id="real-terminal-mismatch",
        profile="full_baseline",
        config=V2Config(),
        retrieval_report=_valid_retrieval_report(tmp_path / "retrieval.json"),
        **reports,
    )
    assert result["metrics"]["technical_failure_count"] == 0
    assert result["hard_gates"][
        "ordinary_baseline_zero_unexpected_technical_failures"
    ] is True
    assert result["hard_gates"][
        "required_real_model_scenarios_terminal_contract"
    ] is False
    assert result["freeze_eligible"] is False


def test_answer_computation_recall_is_a_hard_gate(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    commit = "c" * 40
    monkeypatch.setattr(baseline, "_git_commit", lambda: commit)
    monkeypatch.setattr(baseline, "_git_dirty", lambda: False)
    monkeypatch.setattr(baseline, "_contract_prediction", _passing_scenario_prediction)
    monkeypatch.setattr(baseline, "_real_prediction", _passing_scenario_prediction)
    reports = _valid_evaluator_and_stage_reports(tmp_path)
    _mutate_signed_report(
        reports["answer_ragas_report"], unsupported_computation_recall=0.5
    )
    result = evaluate_baseline(
        output_root=tmp_path / "out",
        run_id="computation-recall",
        profile="full_baseline",
        config=V2Config(),
        retrieval_report=_valid_retrieval_report(tmp_path / "retrieval.json"),
        **reports,
    )
    assert result["hard_gates"]["computation_capability_safety"] is False
    assert result["freeze_eligible"] is False


@pytest.mark.parametrize(
    ("metric", "gate"),
    [
        ("retrieval_degraded_queries_count", "retrieval_degraded_queries_zero"),
        ("citation_violation_count", "citation_violation_zero"),
        ("provenance_violation_count", "provenance_violation_zero"),
    ],
)
def test_answer_audit_counts_feed_global_hard_gates(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    metric: str,
    gate: str,
) -> None:
    commit = "c" * 40
    monkeypatch.setattr(baseline, "_git_commit", lambda: commit)
    monkeypatch.setattr(baseline, "_git_dirty", lambda: False)
    monkeypatch.setattr(baseline, "_contract_prediction", _passing_scenario_prediction)
    monkeypatch.setattr(baseline, "_real_prediction", _passing_scenario_prediction)
    reports = _valid_evaluator_and_stage_reports(tmp_path)
    _mutate_signed_report(reports["answer_ragas_report"], **{metric: 1})
    result = evaluate_baseline(
        output_root=tmp_path / "out",
        run_id=f"answer-audit-{metric}",
        profile="full_baseline",
        config=V2Config(),
        retrieval_report=_valid_retrieval_report(tmp_path / "retrieval.json"),
        **reports,
    )
    assert result["hard_gates"][gate] is False
    assert result["freeze_eligible"] is False


def test_module4_invariant_violation_feeds_schema_gate(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(baseline, "_git_dirty", lambda: False)
    reports = _valid_evaluator_and_stage_reports(tmp_path)
    _mutate_signed_report(reports["module4_gold_report"], invariant_violation_count=1)
    result = evaluate_baseline(
        output_root=tmp_path / "out",
        run_id="module4-invariant",
        mode="contract",
        config=V2Config(),
        **reports,
    )
    assert result["invariant_counts"]["schema_invariant_violation_count"] >= 1
    assert result["hard_gates"]["schema_invariant_zero"] is False


@pytest.mark.parametrize("stage_key", ["stage_v22_report", "stage_v23_report"])
def test_stage_degraded_retrieval_feeds_global_gate(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, stage_key: str
) -> None:
    monkeypatch.setattr(baseline, "_git_dirty", lambda: False)
    reports = _valid_evaluator_and_stage_reports(tmp_path)
    _mutate_signed_report(reports[stage_key], degraded_retrieval_count=1)
    result = evaluate_baseline(
        output_root=tmp_path / "out",
        run_id=f"stage-degraded-{stage_key}",
        mode="contract",
        config=V2Config(),
        **reports,
    )
    assert result["invariant_counts"]["retrieval_degraded_queries_count"] >= 1
    assert result["hard_gates"]["retrieval_degraded_queries_zero"] is False


def test_v23_final_state_provenance_violation_feeds_global_gate(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(baseline, "_git_dirty", lambda: False)
    reports = _valid_evaluator_and_stage_reports(tmp_path)
    _mutate_signed_report(
        reports["stage_v23_report"], provenance_violation_count=1
    )

    result = evaluate_baseline(
        output_root=tmp_path / "out",
        run_id="v23-provenance",
        mode="contract",
        config=V2Config(),
        **reports,
    )

    assert result["invariant_counts"]["provenance_violation_count"] >= 1
    assert result["hard_gates"]["provenance_violation_zero"] is False
