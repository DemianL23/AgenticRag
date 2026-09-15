from __future__ import annotations

from pathlib import Path

import pytest

from agenticrag.v2.config import V2Config, V2PersistenceConfig
import eval.v2.final_baseline as baseline
import eval.v2.module8 as module8
from eval.v2.final_baseline import _check_cross_process_gate
from eval.v2.module8 import (
    CrossProcessEvidence,
    evaluate_module8_cross_process_acceptance,
)


def test_cross_process_producer_emits_separate_valid_artifacts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    commit = "c" * 40
    monkeypatch.setattr(module8, "_git_commit", lambda: commit)
    monkeypatch.setattr(module8, "_git_dirty", lambda: False)
    config = V2Config(
        persistence=V2PersistenceConfig(
            sqlite_path=str(tmp_path / "baseline.sqlite3")
        )
    )
    result = evaluate_module8_cross_process_acceptance(
        config=config,
        run_id="cross-process",
        output_root=tmp_path / "reports",
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
    zero_side_effects = {
        item.name: item
        for item in evidence.negative_contracts
        if item.name in {"invalid_payload", "duplicate_resume", "stale_resume"}
    }
    assert set(zero_side_effects) == {
        "invalid_payload",
        "duplicate_resume",
        "stale_resume",
    }
    assert all(
        not any(item.telemetry.model_dump().values())
        and item.business_state_before_digest
        == item.business_state_after_digest
        for item in zero_side_effects.values()
    )
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
    assert stage["resolved_config"] == config.resolved_record()
    assert stage["runtime_overrides"]["persistence"]["sqlite_path"] == str(
        tmp_path / "reports" / "cross-process" / "acceptance.sqlite3"
    )
    assert (
        stage["runtime_overrides"]["persistence"]["sqlite_path"]
        != config.persistence.sqlite_path
    )
    completed = next(step for step in evidence.steps if step.name == "status_completed")
    assert completed.result["audit_source"] == "durable_checkpoint_state"
    assert stage["metrics"]["provenance_violation_count"] == completed.result[
        "audit_counts"
    ]["provenance_violation_count"]
    reference = baseline._load_stage_report(
        stage_path,
        "v2_3",
        config,
        commit,
    )
    assert reference.available is True
    assert reference.evaluation_complete is True
