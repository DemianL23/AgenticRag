from __future__ import annotations

from pathlib import Path

from agenticrag.v2.config import V2Config, V2PersistenceConfig
from eval.v2.final_baseline import _check_cross_process_gate
from eval.v2.module8 import (
    CrossProcessEvidence,
    evaluate_module8_cross_process_acceptance,
)


def test_cross_process_producer_emits_separate_valid_artifacts(
    tmp_path: Path,
) -> None:
    config = V2Config(
        persistence=V2PersistenceConfig(
            sqlite_path=str(tmp_path / "acceptance.sqlite3")
        )
    )
    result = evaluate_module8_cross_process_acceptance(
        config=config,
        run_id="cross-process",
        output_root=tmp_path / "reports",
        sqlite_path=tmp_path / "acceptance.sqlite3",
    )
    cross_path = Path(result["cross_process_path"])
    stage_path = Path(result["stage_report_path"])
    evidence = CrossProcessEvidence.model_validate_json(
        cross_path.read_text(encoding="utf-8")
    )

    assert result["all_passed"] is True
    assert cross_path != stage_path
    assert cross_path.is_file() and stage_path.is_file()
    assert [step.name for step in evidence.steps] == [
        "start",
        "status_waiting",
        "resume",
        "status_completed",
    ]
    assert all(step.command and step.exit_code == 0 for step in evidence.steps)
    assert all(item.passed for item in evidence.negative_contracts)
    assert _check_cross_process_gate(
        cross_path, current_git_commit=evidence.git_commit
    )["passed"] is True

    stage = result["stage_report"]
    assert stage["producer"] == "v2_3_stage_evaluator"
    assert stage["metrics"]["resume_count"] >= 1
    assert stage["metrics"]["accepted_resume_count"] >= 1
    assert stage["metrics"]["final_execution_status"] == "completed"
    assert stage["cross_process_acceptance_ref"] == str(cross_path)
    assert stage["cross_process_acceptance_digest"]
