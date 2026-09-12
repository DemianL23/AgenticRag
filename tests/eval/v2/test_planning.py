from __future__ import annotations

import json
from pathlib import Path

import pytest

from agenticrag.v2.config import DecisionModelConfig, V2Config
from agenticrag.v2.planning import PlanningResult
from agenticrag.v2.schemas import ComplexityDecision, DecompositionResult, TaskDraft
from eval.v2.planning import (
    CoverageJudgeResult,
    PlanningAnnotation,
    PlanningJudgeConfig,
    PlanningCoverageJudge,
    RequiredInformationUnit,
    evaluate_planning,
    load_planning_samples,
    validate_coverage_result,
)


class FakeStructuredModel:
    def __init__(self, output: object) -> None:
        self.output = output
        self.calls = 0
        self.prompts: list[str] = []

    def with_structured_output(self, schema: object) -> "FakeStructuredModel":
        return self

    def invoke(self, prompt: str) -> object:
        self.calls += 1
        self.prompts.append(prompt)
        return self.output


class FakePlanner:
    def __init__(self, result: PlanningResult) -> None:
        self.result = result

    def plan(self, question: str) -> PlanningResult:
        return self.result


def _write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    path.write_text("".join(json.dumps(record) + "\n" for record in records))


def test_formal_sidecar_joins_qa_by_finqa_id() -> None:
    samples, digests = load_planning_samples()
    assert len(samples) == 19
    assert {sample.finqa_id for sample in samples} == {
        sample.annotation.finqa_id for sample in samples
    }
    assert len(digests["annotation_sha256"]) == 64


def test_planning_judge_is_eval_only_and_not_a_runtime_decision_role() -> None:
    config = V2Config()
    judge_config = PlanningJudgeConfig(model="independent-judge")

    assert not hasattr(config.decision_models, "planning_judge")
    with pytest.raises(ValueError):
        DecisionModelConfig(role="planning_judge")
    assert judge_config.model == "independent-judge"
    assert judge_config.temperature == 0


def test_sidecar_join_fails_on_missing_id(tmp_path: Path) -> None:
    qa = tmp_path / "qa.jsonl"
    annotations = tmp_path / "annotations.jsonl"
    _write_jsonl(qa, [{"finqa_id": "a", "question": "问题"}])
    _write_jsonl(
        annotations,
        [
            {
                "finqa_id": "b",
                "complexity": "simple",
                "capability": "retrieval_synthesis",
                "required_information_units": [
                    {"description": "事实", "expected_capability": None}
                ],
                "expected_outcome": "complete",
            }
        ],
    )
    with pytest.raises(ValueError, match="一一对应"):
        load_planning_samples(qa, annotations)


def test_coverage_judge_requires_every_unit_and_valid_match() -> None:
    result = CoverageJudgeResult(
        units=[
            {"unit_index": 0, "covered": True, "matched_predicted_task_index": 0, "reason": "匹配"},
            {"unit_index": 1, "covered": False, "matched_predicted_task_index": None, "reason": "缺失"},
        ]
    )
    validate_coverage_result(result, unit_count=2, task_count=1)
    with pytest.raises(ValueError):
        validate_coverage_result(result, unit_count=1, task_count=1)


def test_planning_eval_records_metrics_and_independent_judge(tmp_path: Path) -> None:
    qa = tmp_path / "qa.jsonl"
    annotations = tmp_path / "annotations.jsonl"
    _write_jsonl(
        qa,
        [
            {"finqa_id": "simple", "question": "保险风险包括哪些？"},
            {"finqa_id": "complex", "question": "查询 A 和 B"},
        ],
    )
    _write_jsonl(
        annotations,
        [
            {
                "finqa_id": "simple",
                "complexity": "simple",
                "capability": "retrieval_synthesis",
                "required_information_units": [
                    {"description": "风险类型", "expected_capability": None}
                ],
                "expected_outcome": "complete",
            },
            {
                "finqa_id": "complex",
                "complexity": "complex",
                "capability": None,
                "required_information_units": [
                    {"description": "A 的事实", "expected_capability": "retrieval_synthesis"},
                    {"description": "B 的事实", "expected_capability": "retrieval_synthesis"},
                ],
                "expected_outcome": "complete",
            },
        ],
    )
    config = V2Config()
    judge_config = PlanningJudgeConfig()
    simple = PlanningResult.from_router(
        question="保险风险包括哪些？",
        decision=ComplexityDecision(
            complexity="simple", capability="retrieval_synthesis", reason="simple"
        ),
        router_attempts=1,
    )
    complex_result = PlanningResult(
        question="查询 A 和 B",
        normalized_question="查询 A 和 B",
        complexity_decision=ComplexityDecision(
            complexity="complex", capability=None, reason="complex"
        ),
        decomposition=DecompositionResult(
            tasks=[
                TaskDraft(query="A 的事实", intent="查询 A", capability="retrieval_synthesis"),
                TaskDraft(query="B 的事实", intent="查询 B", capability="retrieval_synthesis"),
            ],
            decomposition_complete=True,
        ),
        router_attempts=1,
        decomposer_attempts=1,
    )
    planner = FakePlanner(simple)

    class TwoPlanPlanner(FakePlanner):
        def plan(self, question: str) -> PlanningResult:
            return simple if "保险" in question else complex_result

    judge = PlanningCoverageJudge(
        judge_config,
        model=FakeStructuredModel(
            {
                "units": [
                    {"unit_index": 0, "covered": True, "matched_predicted_task_index": 0, "reason": "A"},
                    {"unit_index": 1, "covered": True, "matched_predicted_task_index": 1, "reason": "B"},
                ]
            }
        ),
    )
    report = evaluate_planning(
        qa,
        annotations,
        config=config,
        planner=TwoPlanPlanner(simple),
        judge=judge,
        run_id="test-run",
    )

    assert report["dataset"]["sample_count"] == 2
    assert report["metrics"]["complexity_accuracy"] == 1.0
    assert report["metrics"]["simple_capability_accuracy"] == 1.0
    assert report["metrics"]["macro_pipeline_requirement_coverage"] == 1.0
    assert report["metrics"]["micro_pipeline_requirement_coverage"] == 1.0
    assert report["metrics"]["macro_decomposer_requirement_coverage_on_correct_route"] == 1.0
    assert report["metrics"]["micro_decomposer_requirement_coverage_on_correct_route"] == 1.0
    assert report["metrics"]["complex_task_capability_accuracy"] == 1.0
    assert report["structural_violations"]["schema_invariant_violations"] == 0
    assert judge._model.calls == 1


def test_router_mismatch_is_quality_error_not_structural_violation(tmp_path: Path) -> None:
    qa = tmp_path / "qa.jsonl"
    annotations = tmp_path / "annotations.jsonl"
    _write_jsonl(qa, [{"finqa_id": "complex", "question": "查询 A 和 B"}])
    _write_jsonl(
        annotations,
        [
            {
                "finqa_id": "complex",
                "complexity": "complex",
                "capability": None,
                "required_information_units": [
                    {"description": "A 的事实", "expected_capability": "retrieval_synthesis"},
                    {"description": "B 的事实", "expected_capability": "retrieval_synthesis"},
                ],
                "expected_outcome": "complete",
            }
        ],
    )
    config = V2Config()
    planner = FakePlanner(
        PlanningResult.from_router(
            question="查询 A 和 B",
            decision=ComplexityDecision(
                complexity="simple", capability="retrieval_synthesis", reason="one task"
            ),
            router_attempts=1,
        )
    )
    judge = PlanningCoverageJudge(
        PlanningJudgeConfig(),
        model=FakeStructuredModel(
            {
                "units": [
                    {"unit_index": 0, "covered": False, "matched_predicted_task_index": None, "reason": "无 task"},
                    {"unit_index": 1, "covered": False, "matched_predicted_task_index": None, "reason": "无 task"},
                ]
            }
        ),
    )
    report = evaluate_planning(
        qa,
        annotations,
        config=config,
        planner=planner,
        judge=judge,
        run_id="mismatch-run",
    )

    assert report["structural_violations"]["schema_invariant_violations"] == 0
    assert report["metrics"]["macro_pipeline_requirement_coverage"] == 0.0
    assert report["metrics"]["micro_pipeline_requirement_coverage"] == 0.0
    assert report["metrics"]["macro_decomposer_requirement_coverage_on_correct_route"] is None
    assert report["metrics"]["micro_decomposer_requirement_coverage_on_correct_route"] is None
    assert report["metrics"]["unmatched_gold_units_count"] == 2
