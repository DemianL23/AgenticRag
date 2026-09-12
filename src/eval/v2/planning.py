"""Module 2 planning dataset, semantic judge, and evaluation report."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from pydantic import Field, model_validator

from agenticrag.v2.config import ModelRetryPolicy, V2Config
from agenticrag.v2.planning import PlanningError, PlanningService, _invoke_structured
from agenticrag.v2.schemas import TaskDraft, V2Model
from agenticrag.v2.types import Complexity, GlobalAnswerOutcome, TaskCapability

DEFAULT_QA_PATH = Path("qa.jsonl")
DEFAULT_ANNOTATION_PATH = Path("eval/datasets/v2_qa_annotations.jsonl")
DEFAULT_OUTPUT_ROOT = Path("artifacts/eval/v2/module2_planning")


class PlanningJudgeConfig(V2Model):
    """Eval-only config; this is intentionally not a V2 runtime role."""

    provider: str = "openai_compatible"
    model: str = "qwen-plus"
    model_revision: str | None = None
    endpoint_identifier: str = "default"
    temperature: float = Field(default=0.0, ge=0.0, le=0.0)
    timeout_seconds: float = Field(default=120.0, gt=0.0)
    max_tokens: int = Field(default=1024, gt=0)
    thinking: bool = False
    retry_policy: ModelRetryPolicy = Field(default_factory=ModelRetryPolicy)

    @classmethod
    def from_env(cls) -> "PlanningJudgeConfig":
        try:
            from dotenv import load_dotenv

            load_dotenv(override=False)
        except ImportError:  # pragma: no cover - project dependency
            pass
        return cls(
            provider=os.getenv("V2_EVAL_PLANNING_JUDGE_PROVIDER", "openai_compatible"),
            model=os.getenv("V2_EVAL_PLANNING_JUDGE_MODEL", "qwen-plus"),
            model_revision=_read_optional("V2_EVAL_PLANNING_JUDGE_MODEL_REVISION"),
            endpoint_identifier=os.getenv(
                "V2_EVAL_PLANNING_JUDGE_ENDPOINT_IDENTIFIER", "eval-planning-judge"
            ),
            temperature=_read_float("V2_EVAL_PLANNING_JUDGE_TEMPERATURE", 0.0),
            timeout_seconds=_read_float(
                "V2_EVAL_PLANNING_JUDGE_TIMEOUT_SECONDS", 120.0
            ),
            max_tokens=_read_int("V2_EVAL_PLANNING_JUDGE_MAX_TOKENS", 1024),
            thinking=_read_bool("V2_EVAL_PLANNING_JUDGE_THINKING", False),
            retry_policy=ModelRetryPolicy(
                max_attempts=_read_int("V2_EVAL_PLANNING_JUDGE_MAX_ATTEMPTS", 2),
                retry_timeout=_read_bool(
                    "V2_EVAL_PLANNING_JUDGE_RETRY_TIMEOUT", True
                ),
                retry_transient_provider_error=_read_bool(
                    "V2_EVAL_PLANNING_JUDGE_RETRY_TRANSIENT", True
                ),
                retry_rate_limit=_read_bool(
                    "V2_EVAL_PLANNING_JUDGE_RETRY_RATE_LIMIT", True
                ),
                retry_structured_output=_read_bool(
                    "V2_EVAL_PLANNING_JUDGE_RETRY_STRUCTURED", True
                ),
            ),
        )

    def resolved_record(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


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
    matched_predicted_task_indices: list[int]
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

    def __init__(
        self,
        config: PlanningJudgeConfig | None = None,
        *,
        model: Any | None = None,
    ) -> None:
        self.config = config or PlanningJudgeConfig.from_env()
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
            model = create_planning_judge_model(self.config)
        return _invoke_structured(
            model,
            CoverageJudgeResult,
            _build_judge_prompt(question, requirements, predicted_tasks),
            role="planning_judge",
            retry_policy=self.config.retry_policy,
            post_validate=lambda result: validate_coverage_result(
                result, len(requirements), len(predicted_tasks)
            ),
        )


def create_planning_judge_model(config: PlanningJudgeConfig) -> Any:
    """Create the eval-only Judge model without adding a runtime role."""
    from agenticrag.generation.config import GenerationConfig

    try:
        from langchain_openai import ChatOpenAI
    except ImportError as exc:  # pragma: no cover - exercised only without extra
        raise RuntimeError(
            "Planning Judge 需要可选依赖，请执行：uv sync --extra generation"
        ) from exc

    generation_defaults = GenerationConfig.from_env()
    base_url = os.getenv(
        "V2_EVAL_PLANNING_JUDGE_BASE_URL", generation_defaults.base_url
    )
    api_key = os.getenv("V2_EVAL_PLANNING_JUDGE_API_KEY") or generation_defaults.api_key
    return ChatOpenAI(
        model=config.model,
        api_key=api_key,
        base_url=base_url,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
        timeout=config.timeout_seconds,
        max_retries=0,
        extra_body={"enable_thinking": config.thinking},
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
    pipeline_coverage_values: list[float] = []
    pipeline_covered_units = 0
    pipeline_requirement_units = 0
    pipeline_unmatched_units = 0
    decomposer_coverage_values: list[float] = []
    decomposer_covered_units = 0
    decomposer_requirement_units = 0
    decomposer_unmatched_units = 0
    matched_gold_units_with_expected_capability = 0
    complex_capability_unmatched_units = 0
    unmatched_predicted_tasks_count = 0
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
            total = len(annotation.required_information_units)
            pipeline_requirement_units += total
            if not predicted_complex:
                pipeline_coverage_values.append(0.0)
                pipeline_unmatched_units += total
                complex_capability_unmatched_units += total
                item["coverage_judge"] = {
                    "status": "not_run",
                    "reason": "router_predicted_simple",
                }
            elif decomposition is None:
                pipeline_coverage_values.append(0.0)
                decomposer_coverage_values.append(0.0)
                decomposer_requirement_units += total
                pipeline_unmatched_units += total
                decomposer_unmatched_units += total
                complex_capability_unmatched_units += total
                item["coverage_judge"] = {
                    "status": "not_run",
                    "reason": "decomposition_missing",
                }
            elif judge is None:
                evaluation_incomplete = True
                item["errors"].append(
                    {
                        "code": "planning_judge_not_configured",
                        "message": "semantic coverage skipped because no independent Judge was configured",
                    }
                )
            else:
                tasks = decomposition.tasks
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
                    coverage_value = covered / total
                    pipeline_coverage_values.append(coverage_value)
                    decomposer_coverage_values.append(coverage_value)
                    pipeline_covered_units += covered
                    pipeline_unmatched_units += total - covered
                    decomposer_covered_units += covered
                    decomposer_requirement_units += total
                    decomposer_unmatched_units += total - covered
                    matched_task_indices = {
                        index
                        for unit_result in coverage.units
                        if unit_result.covered
                        for index in unit_result.matched_predicted_task_indices
                    }
                    unmatched_predicted_tasks_count += len(tasks) - len(
                        matched_task_indices
                    )
                    for unit_result in coverage.units:
                        if not unit_result.covered:
                            complex_capability_unmatched_units += 1
                            continue
                        predicted_indices = unit_result.matched_predicted_task_indices
                        if any(index >= len(tasks) for index in predicted_indices):
                            complex_capability_unmatched_units += 1
                            continue
                        expected_capability = annotation.required_information_units[
                            unit_result.unit_index
                        ].expected_capability
                        if expected_capability is not None:
                            matched_gold_units_with_expected_capability += 1
                            if all(
                                tasks[index].capability == expected_capability
                                for index in predicted_indices
                            ):
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
        "model_configs": {
            "router": config.decision_models.router.model_dump(mode="json"),
            "decomposer": config.decision_models.decomposer.model_dump(mode="json"),
            "planning_judge": judge.config.resolved_record() if judge is not None else None,
        },
        "metrics": {
            "complexity_accuracy": complexity_correct / len(samples) if samples else None,
            "complexity_correct": complexity_correct,
            "complexity_total": len(samples),
            "simple_capability_accuracy": simple_capability_correct / simple_total
            if simple_total
            else None,
            "simple_capability_correct": simple_capability_correct,
            "simple_capability_total": simple_total,
            "macro_pipeline_requirement_coverage": sum(pipeline_coverage_values)
            / len(pipeline_coverage_values)
            if pipeline_coverage_values
            else None,
            "micro_pipeline_requirement_coverage": pipeline_covered_units
            / pipeline_requirement_units
            if pipeline_requirement_units
            else None,
            "pipeline_coverage_covered_units": pipeline_covered_units,
            "pipeline_coverage_total_units": pipeline_requirement_units,
            "pipeline_unmatched_units": pipeline_unmatched_units,
            "macro_decomposer_requirement_coverage_on_correct_route": sum(
                decomposer_coverage_values
            )
            / len(decomposer_coverage_values)
            if decomposer_coverage_values
            else None,
            "micro_decomposer_requirement_coverage_on_correct_route": decomposer_covered_units
            / decomposer_requirement_units
            if decomposer_requirement_units
            else None,
            "decomposer_coverage_covered_units": decomposer_covered_units,
            "decomposer_coverage_total_units": decomposer_requirement_units,
            "decomposer_unmatched_units": decomposer_unmatched_units,
            "complex_task_capability_accuracy": complex_capability_correct
            / matched_gold_units_with_expected_capability
            if matched_gold_units_with_expected_capability
            else None,
            "complex_task_capability_correct": complex_capability_correct,
            "matched_gold_units_with_expected_capability": matched_gold_units_with_expected_capability,
            "complex_task_capability_total": matched_gold_units_with_expected_capability,
            "unmatched_gold_units_count": complex_capability_unmatched_units,
            "unmatched_predicted_tasks_count": unmatched_predicted_tasks_count,
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
        matched_indices = unit.matched_predicted_task_indices
        if unit.covered and not matched_indices:
            raise ValueError("covered=true 必须有 matched_predicted_task_indices")
        if not unit.covered and matched_indices:
            raise ValueError(
                "covered=false 的 matched_predicted_task_indices 必须为空"
            )
        if len(matched_indices) != len(set(matched_indices)):
            raise ValueError("matched_predicted_task_indices 不得重复")
        if any(index < 0 or index >= task_count for index in matched_indices):
            raise ValueError(
                "matched_predicted_task_indices 超出 predicted tasks 范围"
            )


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
        "只判断每个 gold required information unit 是否被一个或多个 predicted tasks 在语义上完整覆盖。\n"
        "一个 gold unit 可以由多个 predicted tasks 联合覆盖；不要强制 gold unit 与 predicted task 一一对应。\n"
        "Decomposition 可以采用不同但语义等价的分组方式，covered 判断依据 matched tasks 的语义并集；不要因为分组方式不同就判未覆盖。\n"
        "只有 matched tasks 的语义并集完整覆盖 gold unit 时才能 covered=true，不能把部分覆盖判为 covered=true。\n"
        "covered=true 时 matched_predicted_task_indices 必须为非空且包含一个或多个合法 index；covered=false 时必须为 []。\n"
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


def _read_optional(name: str) -> str | None:
    value = os.getenv(name)
    return value.strip() if value and value.strip() else None


def _read_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"{name} 必须是整数") from exc


def _read_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(f"{name} 必须是数字") from exc


def _read_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"{name} 必须是 true/false")


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
