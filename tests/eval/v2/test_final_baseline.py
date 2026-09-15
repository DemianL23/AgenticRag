from __future__ import annotations

import json
from pathlib import Path

import pytest

from agenticrag.v2.config import V2Config
from eval.v2.final_baseline import (
    DEFAULT_DATASET,
    evaluate_baseline,
    load_workflow_scenarios,
    scenario_coverage,
)


def test_workflow_scenario_dataset_has_explicit_coverage() -> None:
    records = load_workflow_scenarios(DEFAULT_DATASET)
    coverage = scenario_coverage(records)
    assert coverage["scenario_count"] == 29
    assert coverage["stage_counts"] == {"v2_1": 3, "v2_2": 16, "v2_3": 10}
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
                "dataset_size": 47,
                "evaluated_queries": 47,
                "fallback_queries": 0,
                "invalid_score_queries": 0,
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
    assert len(report["digests"]["artifact_manifest_sha256"]) == 64
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
