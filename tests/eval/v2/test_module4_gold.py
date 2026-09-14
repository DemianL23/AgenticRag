from __future__ import annotations

import json
from pathlib import Path

import pytest

from agenticrag.v2.grading import EvidenceGradingError
from agenticrag.v2.schemas import EvidenceGrade
from eval.v2.module4_gold import (
    GoldExpected,
    Module4GoldFixture,
    Module4GoldJudge,
    Module4GoldJudgeConfig,
    author_gold_dataset,
    build_gold_judge_prompt,
    derive_gold_route,
    extract_gold_inputs,
    load_gold_dataset,
    replay_module4_gold,
    sha256,
    validate_gold_fixture,
)


def _source_report() -> dict[str, object]:
    predictions = []
    for index in range(14):
        qa_id = f"qa-{index:03d}"
        task_id = "SQ_001"
        predictions.append(
            {
                "qa_id": qa_id,
                "observed": {
                    "tasks": [
                        {
                            "id": task_id,
                            "ordinal": 1,
                            "query": f"query {index}",
                            "intent": "extract the fact",
                            "capability": "retrieval_synthesis",
                            "query_revisions": [
                                {
                                    "id": "QR_SQ001_001",
                                    "query": f"query {index}",
                                    "source": "original",
                                }
                            ],
                            "grade_records": [{"grade": {"leak": "never project"}}],
                            "routing_decisions": [{"route": "answer"}],
                        }
                    ],
                    "retrieval_results": {
                        task_id: {
                            "evidence": [
                                {
                                    "evidence_id": f"e-{index}",
                                    "content": f"fact {index}",
                                    "doc_id": "doc-1",
                                    "source": "source.pdf",
                                    "page": 1,
                                }
                            ]
                        }
                    },
                },
            }
        )
    return {
        "run_id": "source-run",
        "git_commit": "source-commit",
        "predictions": predictions,
    }


def _expected_answer() -> GoldExpected:
    return GoldExpected(
        relevance="strong",
        answerability="sufficient",
        ambiguity="none",
        recoverability="none",
        failure_reason="none",
        supporting_evidence_ids=["e1"],
        route="answer",
    )


def _fixture(gold_id: str, expected: GoldExpected | None = None) -> Module4GoldFixture:
    fixture = Module4GoldFixture(
        gold_id=gold_id,
        qa_id=f"qa-{gold_id}",
        source_run_id="source-run",
        source_git_commit="source-commit",
        source_task_id="SQ_001",
        source_task_ordinal=1,
        task={"query": "question", "intent": "extract fact"},
        query_revision={"query": "question", "source": "original"},
        evidence=[
            {
                "evidence_id": "e1",
                "content": "fact",
                "doc_id": "doc-1",
                "source": "source.pdf",
                "page": 1,
            }
        ],
        expected=expected or _expected_answer(),
    )
    validate_gold_fixture(fixture)
    return fixture


class _FakeGoldJudge:
    def __init__(self, grade: EvidenceGrade) -> None:
        self.config = Module4GoldJudgeConfig()
        self.grade_value = grade

    def grade(self, fixture_input: dict[str, object]) -> tuple[EvidenceGrade, int]:
        evidence = fixture_input["evidence"]
        assert isinstance(evidence, list)
        grade = self.grade_value.model_copy(
            update={"supporting_evidence_ids": [evidence[0]["evidence_id"]]}
        )
        return grade, 1


class _FakeReplayGrader:
    def __init__(self, grades: list[EvidenceGrade | Exception]) -> None:
        self.grades = list(grades)
        self.calls = 0

    def grade(self, **kwargs: object) -> tuple[EvidenceGrade, int]:
        self.calls += 1
        result = self.grades.pop(0)
        if isinstance(result, Exception):
            raise result
        return result, 1


def test_extract_gold_inputs_uses_explicit_whitelist_without_prediction_leakage() -> None:
    inputs = extract_gold_inputs(_source_report())

    assert len(inputs) == 14
    assert set(inputs[0]) == {
        "gold_id",
        "qa_id",
        "source_run_id",
        "source_git_commit",
        "source_task_id",
        "source_task_ordinal",
        "task",
        "query_revision",
        "evidence",
    }
    serialized = json.dumps(inputs, ensure_ascii=False)
    assert "grade_records" not in serialized
    assert "routing_decisions" not in serialized
    assert "leak" not in serialized
    assert "answer" not in serialized


def test_authoring_uses_independent_judge_and_derives_route_from_policy(tmp_path: Path) -> None:
    source_path = tmp_path / "source-report.json"
    output_path = tmp_path / "gold.jsonl"
    source_path.write_text(json.dumps(_source_report()), encoding="utf-8")
    result = author_gold_dataset(
        source_path,
        output_path=output_path,
        judge=_FakeGoldJudge(_expected_answer().as_grade()),
    )

    fixtures = load_gold_dataset(output_path)
    assert result["fixture_count"] == 14
    assert result["judge_config"]["model"] == "qwen-plus"
    assert all(fixture.expected.route == "answer" for fixture in fixtures)
    assert all("grade_records" not in fixture.model_dump_json() for fixture in fixtures)


def test_authoring_can_write_missing_output(tmp_path: Path) -> None:
    source_path = tmp_path / "source-report.json"
    source_path.write_text(json.dumps(_source_report()), encoding="utf-8")
    output_path = tmp_path / "gold.jsonl"

    result = author_gold_dataset(
        source_path,
        output_path=output_path,
        judge=_FakeGoldJudge(_expected_answer().as_grade()),
    )

    assert Path(result["dataset_path"]) == output_path
    assert output_path.exists()


def test_authoring_refuses_existing_output_without_force(tmp_path: Path) -> None:
    source_path = tmp_path / "source-report.json"
    output_path = tmp_path / "gold.jsonl"
    source_path.write_text(json.dumps(_source_report()), encoding="utf-8")
    output_path.write_text("sentinel\n", encoding="utf-8")

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        author_gold_dataset(
            source_path,
            output_path=output_path,
            judge=_FakeGoldJudge(_expected_answer().as_grade()),
        )
    assert output_path.read_text(encoding="utf-8") == "sentinel\n"


def test_authoring_force_allows_existing_output(tmp_path: Path) -> None:
    source_path = tmp_path / "source-report.json"
    output_path = tmp_path / "gold.jsonl"
    source_path.write_text(json.dumps(_source_report()), encoding="utf-8")
    output_path.write_text("sentinel\n", encoding="utf-8")

    author_gold_dataset(
        source_path,
        output_path=output_path,
        judge=_FakeGoldJudge(_expected_answer().as_grade()),
        force=True,
    )

    assert output_path.read_text(encoding="utf-8") != "sentinel\n"
    assert len(load_gold_dataset(output_path)) == 14


def test_authoring_default_uses_unique_candidate_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_path = tmp_path / "source-report.json"
    source_path.write_text(json.dumps(_source_report()), encoding="utf-8")
    candidate_root = tmp_path / "candidates"
    monkeypatch.setattr("eval.v2.module4_gold.DEFAULT_CANDIDATE_ROOT", candidate_root)

    result = author_gold_dataset(
        source_path,
        judge=_FakeGoldJudge(_expected_answer().as_grade()),
    )

    candidate_path = Path(result["dataset_path"])
    assert candidate_path.parent.parent == candidate_root
    assert candidate_path.name == "candidate.jsonl"
    assert candidate_path.exists()


def test_gold_judge_prompt_has_only_frozen_inputs_and_no_production_prediction() -> None:
    fixture = _fixture("M4G_001")
    prompt = build_gold_judge_prompt(
        {
            "task": fixture.task.model_dump(mode="json"),
            "query_revision": fixture.query_revision.model_dump(mode="json"),
            "evidence": [item.model_dump(mode="json") for item in fixture.evidence],
        }
    )

    assert "production Grader output" in prompt
    assert "route" in prompt
    assert "grade_records" not in prompt
    assert "routing_decisions" not in prompt
    assert "fact" in prompt


def test_gold_judge_prompt_defines_partial_vs_none_generically() -> None:
    fixture = _fixture("M4G_partial_none")
    prompt = build_gold_judge_prompt(
        {
            "task": fixture.task.model_dump(mode="json"),
            "query_revision": fixture.query_revision.model_dump(mode="json"),
            "evidence": [item.model_dump(mode="json") for item in fixture.evidence],
        }
    )

    assert "at least one explicitly requested target fact" in prompt
    assert "none means the Evidence establishes none" in prompt
    assert "Adjacent, proxy, or merely correlated metrics" in prompt
    assert "Netflix gross margin history by year" not in prompt


def test_gold_fixture_route_must_match_frozen_policy() -> None:
    fixture = _fixture("M4G_001")
    grade = fixture.expected.as_grade()
    assert derive_gold_route(fixture, grade) == ("answer", None)


def test_gold_fixture_recover_route_uses_frozen_strategy_mapping() -> None:
    expected = GoldExpected(
        relevance="strong",
        answerability="partial",
        ambiguity="none",
        recoverability="likely",
        failure_reason="insufficient_coverage",
        missing_information=["missing fact"],
        supporting_evidence_ids=["e1"],
        route="recover",
        recovery_strategy="direct_rewrite",
    )
    fixture = _fixture("M4G_recover", expected)
    assert derive_gold_route(fixture, expected.as_grade()) == (
        "recover",
        "direct_rewrite",
    )


def test_gold_fixture_rejects_route_not_derived_from_policy() -> None:
    fixture = _fixture("M4G_001")
    invalid = fixture.model_copy(
        update={"expected": fixture.expected.model_copy(update={"route": "no_knowledge"})}
    )
    with pytest.raises(ValueError, match="route/strategy"):
        validate_gold_fixture(invalid)


def test_replay_evaluator_does_not_call_planner_or_retrieval_and_scores_fields(tmp_path: Path) -> None:
    first = _fixture("M4G_001")
    second_expected = GoldExpected(
        relevance="weak",
        answerability="none",
        ambiguity="none",
        recoverability="none",
        failure_reason="none",
        route="no_knowledge",
    )
    second = _fixture("M4G_002", second_expected)
    gold_path = tmp_path / "gold.jsonl"
    gold_path.write_text(
        "".join(json.dumps(item.model_dump(mode="json")) + "\n" for item in (first, second)),
        encoding="utf-8",
    )
    third = _fixture(
        "M4G_003",
        _expected_answer().model_copy(update={"supporting_evidence_ids": ["e1"]}),
    )
    mismatched_grade = first.expected.as_grade().model_copy(update={"relevance": "weak"})
    grader = _FakeReplayGrader(
        [first.expected.as_grade(), second.expected.as_grade(), mismatched_grade]
    )
    gold_path.write_text(
        "".join(
            json.dumps(item.model_dump(mode="json")) + "\n"
            for item in (first, second, third)
        ),
        encoding="utf-8",
    )

    report = replay_module4_gold(
        gold_path, grader=grader, output_root=tmp_path / "reports"
    )

    assert grader.calls == 3
    assert report["metrics"]["task_count"] == 3
    assert report["metrics"]["relevance_accuracy"] == 2 / 3
    assert report["metrics"]["answerability_accuracy"] == 1.0
    assert report["metrics"]["grade_exact_match_accuracy"] == 2 / 3
    assert report["metrics"]["route_accuracy"] == 2 / 3
    assert report["metrics"]["technical_failure_count"] == 0
    assert report["metrics"]["invariant_violation_count"] == 0


def test_replay_technical_failure_is_separate_from_route_accuracy(tmp_path: Path) -> None:
    first = _fixture("M4G_001")
    second = _fixture("M4G_002")
    gold_path = tmp_path / "gold.jsonl"
    gold_path.write_text(
        "".join(json.dumps(item.model_dump(mode="json")) + "\n" for item in (first, second)),
        encoding="utf-8",
    )
    error = EvidenceGradingError(attempts=2, cause=ValueError("invalid grade"))
    report = replay_module4_gold(
        gold_path,
        grader=_FakeReplayGrader([first.expected.as_grade(), error]),
        output_root=tmp_path / "reports",
    )

    assert report["metrics"]["successful_task_count"] == 1
    assert report["metrics"]["technical_failure_count"] == 1
    assert report["metrics"]["route_accuracy"] == 1.0


def test_frozen_real_gold_dataset_has_expected_shape_and_digest() -> None:
    path = Path("eval/datasets/v2_module4_grade_route_gold.jsonl")
    fixtures = load_gold_dataset(path)

    assert len(fixtures) == 14
    assert len({fixture.qa_id for fixture in fixtures}) == 8
    assert not any(fixture.needs_review for fixture in fixtures)
    assert sha256(path) == "b2204ec505d76c3fd57dd060378ea0e9e5946482e4987b6b0b6c6a58ff1e594c"
    adjudicated = next(
        fixture for fixture in fixtures if fixture.gold_id == "M4G_indEN_00260_012"
    )
    assert adjudicated.expected.answerability == "none"
    assert adjudicated.expected.supporting_evidence_ids == []
    assert adjudicated.expected.route == "recover"
    assert adjudicated.expected.recovery_strategy == "direct_rewrite"
    assert adjudicated.needs_review is False
    assert any("adjacent" in note.lower() for note in adjudicated.review_notes)
