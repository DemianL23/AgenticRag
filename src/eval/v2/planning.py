"""Module 2 planning dataset, semantic judge, and evaluation report."""

from __future__ import annotations

import hashlib
import json
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from pydantic import Field, model_validator

from agenticrag.v2.config import V2Config
from agenticrag.v2.planning import PlanningError, PlanningService, _invoke_structured
from agenticrag.v2.schemas import TaskDraft, V2Model
from agenticrag.v2.types import Complexity, GlobalAnswerOutcome, TaskCapability

DEFAULT_QA_PATH = Path("qa.jsonl")
DEFAULT_ANNOTATION_PATH = Path("eval/datasets/v2_qa_annotations.jsonl")
DEFAULT_OUTPUT_ROOT = Path("artifacts/eval/v2/module2_planning")


class RequiredInformationUnit(V2Model):
    description: str = Field(min_length=1)
    expected_capability: TaskCapability | None

    @model_validator(mode="after")
    def validate_description(self) -> "RequiredInformationUnit":
        if not self.description.strip():
            raise ValueError("required_information_unit.description 不能为空")
        return self


class PlanningAnnotation(V2Model):
    finqa_id: str = Field(min_length=1)
    complexity: Complexity
    capability: TaskCapability | None
    required_information_units: list[RequiredInformationUnit] = Field(min_length=1)
    expected_outcome: GlobalAnswerOutcome

    @model_validator(mode="after")
    def validate_annotation_contract(self) -> "PlanningAnnotation":
        if not self.finqa_id.strip():
            raise ValueError("finqa_id 不能为空")
        if self.complexity == "simple":
            if self.capability is None:
                raise ValueError("simple annotation 必须有 top-level capability")
            if any(unit.expected_capability is not None for unit in self.required_information_units):
                raise ValueError("simple annotation 的 unit capability 必须为 null")
        else:
            if self.capability is not None:
                raise ValueError("complex annotation 的 top-level capability 必须为 null")
            if any(unit.expected_capability is None for unit in self.required_information_units):
                raise ValueError("complex annotation 的 unit 必须有 expected_capability")
        return self


@dataclass(frozen=True, slots=True)
class PlanningSample:
    finqa_id: str
    question: str
    annotation: PlanningAnnotation
    qa_record: dict[str, Any]


class CoverageUnitResult(V2Model):
    unit_index: int = Field(ge=0)
    covered: bool
    matched_predicted_task_index: int | None = Field(default=None, ge=0)
    reason: str = Field(min_length=1)


class CoverageJudgeResult(V2Model):
    units: list[CoverageUnitResult] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_unique_unit_indices(self) -> "CoverageJudgeResult":
        indices = [unit.unit_index for unit in self.units]
        if len(indices) != len(set(indices)):
            raise ValueError("Coverage Judge unit_index 必须唯一")
        return self


class PlanningCoverageJudge:
    """Independent structured judge for requirement-to-task semantic coverage."""

    def __init__(self, config: V2Config | None = None, *, model: Any | None = None) -> None:
        self.config = config or V2Config.from_env()
        self._model = model

    def judge(
        self,
        *,
        question: str,
        requirements: list[RequiredInformationUnit],
        predicted_tasks: list[TaskDraft],
    ) -> tuple[CoverageJudgeResult, int]:
        model = self._model
        if model is None:
            from agenticrag.v2.planning import create_decision_chat_model

            model = create_decision_chat_model(self.config.decision_models.planning_judge)
        return _invoke_structured(
            model,
            CoverageJudgeResult,
            _build_judge_prompt(question, requirements, predicted_tasks),
            role="planning_judge",
            retry_policy=self.config.decision_models.planning_judge.retry_policy,
            post_validate=lambda result: validate_coverage_result(
                result, len(requirements), len(predicted_tasks)
            ),
        )


def load_planning_samples(
    qa_path: Path = DEFAULT_QA_PATH,
    annotation_path: Path = DEFAULT_ANNOTATION_PATH,
) -> tuple[list[PlanningSample], dict[str, str]]:
    qa_records = _load_jsonl(qa_path, "qa")
    annotation_records = _load_jsonl(annotation_path, "v2 annotation")
    qa_by_id = _index_records(qa_records, qa_path)
    annotations = [PlanningAnnotation.model_validate(record) for record in annotation_records]
    annotation_by_id = _index_annotations(annotations, annotation_path)
    if set(qa_by_id) != set(annotation_by_id):
        raise ValueError("qa.jsonl 与 v2_qa_annotations.jsonl 的 finqa_id 必须完全一一对应")

    samples: list[PlanningSample] = []
    for finqa_id, qa_record in qa_by_id.items():
        question = qa_record.get("question")
        if not isinstance(question, str) or not question.strip():
            raise ValueError(f"QA {finqa_id} 缺少合法 question")
        samples.append(
            PlanningSample(
                finqa_id=finqa_id,
                question=question,
                annotation=annotation_by_id[finqa_id],
                qa_record=qa_record,
            )
        )
    return samples, {
        "qa_sha256": _sha256(qa_path),
        "annotation_sha256": _sha256(annotation_path),
    }


def evaluate_planning(
    qa_path: Path = DEFAULT_QA_PATH,
    annotation_path: Path = DEFAULT_ANNOTATION_PATH,
    *,
    config: V2Config | None = None,
    planner: PlanningService | None = None,
    judge: PlanningCoverageJudge | None = None,
    run_id: str | None = None,
) -> dict[str, Any]:
    """Run one auditable Module 2 planning evaluation."""
    config = config or V2Config.from_env()
    samples, dataset_digests = load_planning_samples(qa_path, annotation_path)
    planner = planner or PlanningService(config)
    if judge is not None and judge.config != config:
        raise ValueError("Planning Judge 必须使用同一 resolved V2Config 的独立 role 配置")

    started = time.perf_counter()
    structural = {
        "empty_task_violations": 0,
        "duplicate_task_violations": 0,
        "task_limit_violations": 0,
        "schema_invariant_violations": 0,
        "simple_decomposer_calls": 0,
        "complex_missing_decomposition": 0,
        "complex_null_capability_contract_violations": 0,
    }
    per_sample: list[dict[str, Any]] = []
    complexity_correct = 0
    simple_capability_correct = 0
    simple_total = 0
    complex_capability_correct = 0
    complex_capability_total = 0
    unmatched_gold_units = 0
    coverage_values: list[float] = []
    covered_units_total = 0
    requirement_units_total = 0
    evaluation_incomplete = False

    for sample in samples:
        annotation = sample.annotation
        sample_started = time.perf_counter()
        item: dict[str, Any] = {
            "finqa_id": sample.finqa_id,
            "gold": annotation.model_dump(mode="json"),
            "router": {
                "gold": annotation.complexity,
                "predicted": None,
                "correct": False,
            },
            "decomposition": None,
            "coverage_judge": None,
            "errors": [],
            "latency_seconds": {},
        }
        try:
            result = planner.plan(sample.question)
        except PlanningError as exc:
            structural["schema_invariant_violations"] += 1
            evaluation_incomplete = True
            item["errors"].append(exc.execution_error.model_dump(mode="json"))
            item["latency_seconds"]["planning_total"] = time.perf_counter() - sample_started
            per_sample.append(item)
            continue

        decision = result.complexity_decision
        item["router"] = {
            "gold": annotation.complexity,
            "predicted": decision.complexity,
            "predicted_capability": decision.capability,
            "correct": decision.complexity == annotation.complexity,
            "attempts": result.router_attempts,
        }
        if item["router"]["correct"]:
            complexity_correct += 1

        predicted_complex = decision.complexity == "complex"
        decomposition = result.decomposition

        if annotation.complexity == "simple":
            simple_total += 1
            if decision.capability == annotation.capability:
                simple_capability_correct += 1
        if predicted_complex:
            if decision.capability is not None:
                structural["complex_null_capability_contract_violations"] += 1
                structural["schema_invariant_violations"] += 1
            if decomposition is None:
                structural["complex_missing_decomposition"] += 1
                structural["schema_invariant_violations"] += 1
            else:
                item["decomposition"] = {
                    "result": decomposition.model_dump(mode="json"),
                    "attempts": result.decomposer_attempts,
                }
                _record_decomposition_structure(
                    decomposition, config.budgets.max_subqueries, structural
                )
        else:
            if decision.capability is None:
                structural["schema_invariant_violations"] += 1
            if result.decomposer_attempts:
                structural["simple_decomposer_calls"] += 1
                structural["schema_invariant_violations"] += 1

        if annotation.complexity == "complex":
            tasks = decomposition.tasks if predicted_complex and decomposition else []
            if judge is None:
                evaluation_incomplete = True
                item["errors"].append(
                    {
                        "code": "planning_judge_not_configured",
                        "message": "semantic coverage skipped because no independent Judge was configured",
                    }
                )
            else:
                judge_started = time.perf_counter()
                try:
                    coverage, judge_attempts = judge.judge(
                        question=sample.question,
                        requirements=annotation.required_information_units,
                        predicted_tasks=tasks,
                    )
                    item["coverage_judge"] = {
                        "result": coverage.model_dump(mode="json"),
                        "attempts": judge_attempts,
                    }
                    item["latency_seconds"]["planning_judge"] = time.perf_counter() - judge_started
                    covered = sum(unit.covered for unit in coverage.units)
                    total = len(annotation.required_information_units)
                    coverage_values.append(covered / total)
                    covered_units_total += covered
                    requirement_units_total += total
                    for unit_result in coverage.units:
                        if not unit_result.covered:
                            unmatched_gold_units += 1
                            continue
                        if unit_result.matched_predicted_task_index is None:
                            continue
                        predicted_index = unit_result.matched_predicted_task_index
                        if predicted_index >= len(tasks):
                            continue
                        expected_capability = annotation.required_information_units[
                            unit_result.unit_index
                        ].expected_capability
                        if expected_capability is not None:
                            complex_capability_total += 1
                            if tasks[predicted_index].capability == expected_capability:
                                complex_capability_correct += 1
                except PlanningError as exc:
                    evaluation_incomplete = True
                    item["errors"].append(exc.execution_error.model_dump(mode="json"))
                    item["latency_seconds"]["planning_judge"] = time.perf_counter() - judge_started
        item["latency_seconds"]["planning_total"] = time.perf_counter() - sample_started
        per_sample.append(item)

    report = {
        "run_id": run_id or str(uuid4()),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "git_dirty": _git_dirty(),
        "dataset": {
            "qa_path": str(qa_path),
            "annotation_path": str(annotation_path),
            "qa_sha256": dataset_digests["qa_sha256"],
            "annotation_sha256": dataset_digests["annotation_sha256"],
            "sample_count": len(samples),
        },
        "resolved_config": config.resolved_record(),
        "metrics": {
            "complexity_accuracy": complexity_correct / len(samples) if samples else None,
            "complexity_correct": complexity_correct,
            "complexity_total": len(samples),
            "simple_capability_accuracy": simple_capability_correct / simple_total
            if simple_total
            else None,
            "simple_capability_correct": simple_capability_correct,
            "simple_capability_total": simple_total,
            "macro_decomposition_requirement_coverage": sum(coverage_values) / len(coverage_values)
            if coverage_values
            else None,
            "micro_decomposition_requirement_coverage": covered_units_total / requirement_units_total
            if requirement_units_total
            else None,
            "complex_task_capability_accuracy": complex_capability_correct / complex_capability_total
            if complex_capability_total
            else None,
            "unmatched_gold_units_count": unmatched_gold_units,
        },
        "structural_violations": structural,
        "evaluation_incomplete": evaluation_incomplete,
        "per_sample": per_sample,
        "elapsed_seconds": time.perf_counter() - started,
    }
    return report


def save_report(report: dict[str, Any], output: Path) -> None:
    if output.exists():
        raise FileExistsError(f"拒绝覆盖既有 Module 2 report：{output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")


def validate_coverage_result(
    result: CoverageJudgeResult, unit_count: int, task_count: int
) -> None:
    expected_indices = set(range(unit_count))
    actual_indices = {unit.unit_index for unit in result.units}
    if actual_indices != expected_indices:
        raise ValueError("Coverage Judge 必须为每一个 gold unit 返回一次结果")
    for unit in result.units:
        if unit.covered and unit.matched_predicted_task_index is None:
            raise ValueError("covered=true 必须有 matched_predicted_task_index")
        if not unit.covered and unit.matched_predicted_task_index is not None:
            raise ValueError("covered=false 的 matched_predicted_task_index 必须为 null")
        if unit.matched_predicted_task_index is not None and unit.matched_predicted_task_index >= task_count:
            raise ValueError("matched_predicted_task_index 超出 predicted tasks 范围")


def _record_decomposition_structure(
    decomposition: Any, max_subqueries: int, structural: dict[str, int]
) -> None:
    tasks = decomposition.tasks
    if any(not task.query.strip() or not task.intent.strip() for task in tasks):
        structural["empty_task_violations"] += 1
        structural["schema_invariant_violations"] += 1
    normalized = [" ".join(task.query.split()).casefold() for task in tasks]
    if len(normalized) != len(set(normalized)):
        structural["duplicate_task_violations"] += 1
        structural["schema_invariant_violations"] += 1
    if decomposition.decomposition_complete and not 2 <= len(tasks) <= max_subqueries:
        structural["task_limit_violations"] += 1
        structural["schema_invariant_violations"] += 1


def _build_judge_prompt(
    question: str,
    requirements: list[RequiredInformationUnit],
    predicted_tasks: list[TaskDraft],
) -> str:
    requirement_text = "\n".join(
        f"{index}. {unit.description}" for index, unit in enumerate(requirements)
    )
    task_text = "\n".join(
        f"{index}. query={task.query}; intent={task.intent}; capability={task.capability}"
        for index, task in enumerate(predicted_tasks)
    )
    return (
        "你是独立的 Module 2 Planning Judge。不要判断答案是否正确，也不要使用任何 gold answer。\n"
        "只判断每个 gold required information unit 是否被 predicted task 在语义上覆盖。\n"
        "covered=true 时必须给出对应 predicted task index，covered=false 时 index 必须为 null。\n"
        "必须为每个 unit 返回一次结果，并给出简短理由。\n\n"
        f"原始问题：\n{question}\n\n"
        f"Gold required information units：\n{requirement_text}\n\n"
        f"Predicted tasks：\n{task_text}"
    )


def _load_jsonl(path: Path, label: str) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"{label} dataset 不存在：{path}")
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{label} dataset 第 {line_number} 行 JSON 非法") from exc
        if not isinstance(value, dict):
            raise ValueError(f"{label} dataset 第 {line_number} 行必须是 JSON object")
        records.append(value)
    if not records:
        raise ValueError(f"{label} dataset 不能为空：{path}")
    return records


def _index_records(records: list[dict[str, Any]], path: Path) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for record in records:
        finqa_id = record.get("finqa_id")
        if not isinstance(finqa_id, str) or not finqa_id.strip():
            raise ValueError(f"{path} 存在缺失或非法 finqa_id")
        if finqa_id in indexed:
            raise ValueError(f"{path} 存在重复 finqa_id：{finqa_id}")
        indexed[finqa_id] = record
    return indexed


def _index_annotations(
    records: list[PlanningAnnotation], path: Path
) -> dict[str, PlanningAnnotation]:
    indexed: dict[str, PlanningAnnotation] = {}
    for record in records:
        if record.finqa_id in indexed:
            raise ValueError(f"{path} 存在重复 finqa_id：{record.finqa_id}")
        indexed[record.finqa_id] = record
    return indexed


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_commit() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _git_dirty() -> bool | None:
    result = subprocess.run(
        ["git", "status", "--porcelain"], capture_output=True, text=True, check=False
    )
    return result.stdout.strip() != "" if result.returncode == 0 else None


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="运行 V2 Module 2 Planning Evaluation")
    parser.add_argument("--qa", type=Path, default=DEFAULT_QA_PATH)
    parser.add_argument("--annotations", type=Path, default=DEFAULT_ANNOTATION_PATH)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    report = evaluate_planning(args.qa, args.annotations, judge=PlanningCoverageJudge())
    output = args.output or DEFAULT_OUTPUT_ROOT / report["run_id"] / "report.json"
    save_report(report, output)
    print(json.dumps({"output": str(output), "metrics": report["metrics"]}, ensure_ascii=False))
