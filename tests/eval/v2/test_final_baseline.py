from __future__ import annotations

import json
from pathlib import Path

import pytest
from types import SimpleNamespace

from agenticrag.v2.config import V2Config
from agenticrag.v2.schemas import Evidence, EvidenceOccurrence, GroundedFinding, RetrievalTask, StageRunResult, SynthesizedAnswer
import eval.v2.final_baseline as baseline
from eval.v2.final_baseline import (
    FROZEN_RETRIEVAL_METRICS,
    RETRIEVAL_DATASET,
    DEFAULT_DATASET,
    _check_cross_process_gate,
    _check_retrieval_gate,
    _computation_gate,
    _audit_tasks,
    evaluate_baseline,
    load_workflow_scenarios,
    scenario_coverage,
)


def test_workflow_scenario_dataset_has_explicit_coverage() -> None:
    records = load_workflow_scenarios(DEFAULT_DATASET)
    coverage = scenario_coverage(records)
    assert coverage["scenario_count"] == 29
    assert coverage["stage_counts"] == {"v2_1": 2, "v2_2": 17, "v2_3": 10}
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
    assert report["metrics"]["contract_scenario_count"] == 26
    assert report["metrics"]["technical_failure_count"] == 0
    assert report["hard_gates"]["dataset_schema_and_coverage"] is True
    assert report["hard_gates"]["v1_2_retrieval_regression"] is True
    assert report["hard_gates"]["baseline_scenarios_terminal_contract"] is True
    assert report["hard_gates"]["v2_3_cross_process"] is False
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
    assert sum(record.mode == "contract" for record in records) == 26


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
    assert report["metrics"]["scenario_pass_count"] == 26


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
    predictions = [{"scenario_id": "contract_unsupported_arithmetic", "observed": {"answer_outcome": "complete"}}]
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
