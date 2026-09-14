"""V2.1 real-document stage evaluation.

The manifest is an observation/evaluation contract.  Its expected fields are
never passed to the production Planner or Evidence Grader.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from agenticrag.v2.config import V2Config
from agenticrag.v2.module4 import Module4Service, materialize_retrieval_tasks, unsupported_routing_decision
from agenticrag.v2.planning import PlanningResult
from agenticrag.v2.schemas import ComplexityDecision, DecompositionResult, TaskDraft

DEFAULT_MANIFEST = Path("eval/datasets/v2_module4_real_eval.jsonl")
DEFAULT_OUTPUT_ROOT = Path("artifacts/eval/v2/module4_v2_1")


def load_manifest(path: Path = DEFAULT_MANIFEST) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Module 4 manifest 不存在：{path}")
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"manifest 第 {line_number} 行 JSON 非法") from exc
        if not isinstance(record, dict):
            raise ValueError(f"manifest 第 {line_number} 行必须是 object")
        _validate_manifest_record(record, line_number)
        records.append(record)
    if len(records) != 19:
        raise ValueError(f"Module 4 manifest 必须有 19 条，实际 {len(records)} 条")
    ids = [record["qa_id"] for record in records]
    if len(ids) != len(set(ids)):
        raise ValueError("Module 4 manifest qa_id 必须唯一")
    return records


def _validate_manifest_record(record: dict[str, Any], line_number: int) -> None:
    qa_id = record.get("qa_id")
    question = record.get("question")
    expected = record.get("expected")
    retrieval = record.get("retrieval")
    if not isinstance(qa_id, str) or not qa_id.strip():
        raise ValueError(f"manifest 第 {line_number} 行缺少合法 qa_id")
    if not isinstance(question, str) or not question.strip():
        raise ValueError(f"manifest 第 {line_number} 行缺少合法 question")
    if not isinstance(expected, dict) or not isinstance(retrieval, dict):
        raise ValueError(f"manifest 第 {line_number} 行 expected/retrieval 必须是 object")
    route = expected.get("route")
    expected_to_run = retrieval.get("expected_to_run")
    if route == "unsupported":
        if expected_to_run is not False or record.get("evaluation_incomplete") is not False:
            raise ValueError(f"manifest 第 {line_number} 行 unsupported contract 非法")
    elif expected_to_run is True:
        if record.get("evaluation_incomplete") is not True:
            raise ValueError(f"manifest 第 {line_number} 行 supported contract 非法")
    else:
        raise ValueError(f"manifest 第 {line_number} 行 expected route/expected_to_run 不受支持")


def evaluate_module4(
    manifest_path: Path = DEFAULT_MANIFEST,
    *,
    config: V2Config | None = None,
    service: Module4Service | None = None,
    run_id: str | None = None,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
) -> dict[str, Any]:
    config = config or V2Config.from_env()
    records = load_manifest(manifest_path)
    run_id = run_id or str(uuid4())
    report_dir = output_root / run_id
    if report_dir.exists():
        raise FileExistsError(f"拒绝覆盖 Module 4 report：{report_dir}")

    started = time.perf_counter()
    predictions: list[dict[str, Any]] = []
    service = service
    unsupported_total = unsupported_correct = 0
    retriever_calls = grader_calls = 0
    technical_failures = degraded_count = 0
    evaluation_incomplete = 0
    invariant_violations = 0
    supported_observed = 0

    def attach_invariants(item: dict[str, Any]) -> None:
        nonlocal invariant_violations
        violations = check_module4_invariants(item.get("observed"))
        item["invariant_violations"] = violations
        invariant_violations += len(violations)

    for record in records:
        expected = record["expected"]
        expected_route = expected.get("route")
        item: dict[str, Any] = {
            "qa_id": record["qa_id"],
            "question": record["question"],
            "evaluation_incomplete": bool(record.get("evaluation_incomplete")),
            "expected": {
                "complexity": expected.get("complexity"),
                "capability": expected.get("capability"),
                "route": expected_route,
                "outcome": expected.get("outcome"),
            },
            "observed": None,
            "errors": [],
        }
        if item["evaluation_incomplete"]:
            evaluation_incomplete += 1

        if expected_route == "unsupported":
            unsupported_total += 1
            tasks = _deterministic_unsupported_tasks(record)
            decisions = [unsupported_routing_decision(task) for task in tasks]
            tasks = [
                task.model_copy(
                    update={
                        "execution_status": "completed",
                        "answer_outcome": "unsupported",
                        "terminal_reason": f"unsupported_capability:{task.capability}",
                        "routing_decisions": [decision],
                    }
                )
                for task, decision in zip(tasks, decisions, strict=True)
            ]
            correct = all(decision.route == expected_route for decision in decisions)
            unsupported_correct += int(correct)
            item["observed"] = {
                "mode": "deterministic_capability_gold",
                "tasks": [task.model_dump(mode="json") for task in tasks],
                "task_order": [task.id for task in tasks],
                "routing_decisions": [decision.model_dump(mode="json") for decision in decisions],
                "retriever_called": False,
                "grader_called": False,
                "correct": correct,
            }
            attach_invariants(item)
            predictions.append(item)
            continue

        supported_observed += 1
        if service is None:
            try:
                service = Module4Service(config)
            except Exception as exc:
                technical_failures += 1
                item["errors"].append(_error("service_initialization_failed", exc))
                item["observed"] = {"mode": "runtime", "status": "technical_failure"}
                attach_invariants(item)
                predictions.append(item)
                continue
        try:
            result = service.run(record["question"])
            task_values = list(result.tasks)
            retriever_calls += sum(
                1 for task in task_values if task.query_revisions
            )
            grader_calls += sum(1 for task in task_values if task.grade_records)
            degraded_count += sum(
                int(retrieval.retrieval_degraded)
                for retrieval in result.retrieval_results.values()
            )
            if result.stage_result.execution_status == "failed":
                technical_failures += 1
            item["observed"] = {
                "mode": "runtime",
                "status": result.stage_result.execution_status,
                "stage_result": result.stage_result.model_dump(mode="json"),
                "planning_result": result.planning_result.model_dump(mode="json")
                if result.planning_result is not None
                else None,
                "tasks": [task.model_dump(mode="json") for task in task_values],
                "task_order": [task.id for task in task_values],
                "evidence": [item.model_dump(mode="json") for item in result.evidence.values()],
                "retrieval_results": {
                    task_id: retrieval.model_dump(mode="json")
                    for task_id, retrieval in result.retrieval_results.items()
                },
            }
        except Exception as exc:
            technical_failures += 1
            item["errors"].append(_error("module4_runtime_failed", exc))
            item["observed"] = {"mode": "runtime", "status": "technical_failure"}
        attach_invariants(item)
        predictions.append(item)

    report = {
        "run_id": run_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "git_dirty": _git_dirty(),
        "dataset": {
            "manifest_path": str(manifest_path),
            "manifest_sha256": _sha256(manifest_path),
            "sample_count": len(records),
        },
        "resolved_config": config.resolved_record(),
        "model_configs": {
            "router": config.decision_models.router.model_dump(mode="json"),
            "decomposer": config.decision_models.decomposer.model_dump(mode="json"),
            "grader": config.decision_models.grader.model_dump(mode="json"),
        },
        "metrics": {
            "complete_eval_count": len(records) - evaluation_incomplete,
            "evaluation_incomplete_count": evaluation_incomplete,
            "unsupported_route_accuracy": unsupported_correct / unsupported_total
            if unsupported_total else None,
            "unsupported_correct": unsupported_correct,
            "unsupported_total": unsupported_total,
            "retrieval_completed_count": sum(
                int(
                    bool(task.get("query_revisions"))
                    and task.get("id") in (item.get("observed") or {}).get("retrieval_results", {})
                )
                for item in predictions
                for task in (item.get("observed") or {}).get("tasks", [])
            ),
            "grader_completed_count": grader_calls,
            "route_accuracy": None,
            "grade_accuracy": None,
            "retriever_calls": retriever_calls,
            "grader_calls": grader_calls,
            "technical_failure_count": technical_failures,
            "degraded_retrieval_count": degraded_count,
            "invariant_violation_count": invariant_violations,
            "supported_runtime_sample_count": supported_observed,
            "observed_route_distribution": dict(
                Counter(
                    decision["route"]
                    for item in predictions
                    for task in (item.get("observed") or {}).get("tasks", [])
                    for decision in task.get("routing_decisions", [])
                )
            ),
        },
        "evaluation_incomplete": evaluation_incomplete > 0,
        "invariant_evaluation": "evaluated",
        "predictions": predictions,
        "elapsed_seconds": time.perf_counter() - started,
    }
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    with (report_dir / "predictions.jsonl").open("w") as handle:
        for item in predictions:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    return report


def _deterministic_unsupported_tasks(record: dict[str, Any]):
    expected = record["expected"]
    mapping = expected["task_mapping"]
    if expected["complexity"] == "simple":
        tasks = [TaskDraft(query=record["question"], intent="capability policy fixture", capability=expected["capability"])]
    else:
        tasks = [
            TaskDraft(query=unit["description"], intent=unit["description"], capability=unit["expected_capability"])
            for unit in mapping["required_information_units"]
        ]
    planning = PlanningResult(
        question=record["question"],
        normalized_question=record["question"],
        complexity_decision=ComplexityDecision(
            complexity=expected["complexity"],
            capability=expected.get("capability"),
            reason="deterministic manifest capability-policy fixture",
        ),
        simple_task=tasks[0] if expected["complexity"] == "simple" else None,
        decomposition=DecompositionResult(tasks=tasks, decomposition_complete=True)
        if expected["complexity"] == "complex"
        else None,
        router_attempts=1,
        decomposer_attempts=1 if expected["complexity"] == "complex" else 0,
    )
    return materialize_retrieval_tasks(planning)


def check_module4_invariants(observed: dict[str, Any] | None) -> list[str]:
    """Check observable Module 4 contracts without treating quality as a violation."""
    if not observed or observed.get("status") == "technical_failure":
        return []
    violations: list[str] = []
    tasks = observed.get("tasks", [])
    if not isinstance(tasks, list):
        return ["tasks_not_list"]
    task_order = observed.get("task_order")
    task_ids = [task.get("id") for task in tasks]
    if task_order is not None and task_order != task_ids:
        violations.append("task_order_not_stable")
    retrieval_results = observed.get("retrieval_results", {})
    if not isinstance(retrieval_results, dict):
        violations.append("retrieval_results_not_map")
        retrieval_results = {}

    for task in tasks:
        task_id = task.get("id")
        capability = task.get("capability")
        status = task.get("execution_status")
        outcome = task.get("answer_outcome")
        grades = task.get("grade_records", [])
        routes = task.get("routing_decisions", [])
        if status == "failed" and outcome is not None:
            violations.append(f"{task_id}:failed_task_has_outcome")
        if status == "failed" and not task.get("error"):
            violations.append(f"{task_id}:failed_task_missing_error")
        if capability != "retrieval_synthesis":
            if retrieval_results.get(task_id) is not None:
                violations.append(f"{task_id}:unsupported_has_retrieval_result")
            if grades:
                violations.append(f"{task_id}:unsupported_has_grade")
            if not routes or routes[-1].get("route") != "unsupported":
                violations.append(f"{task_id}:unsupported_route_missing")
            elif routes[-1].get("grade_record_id") is not None:
                violations.append(f"{task_id}:unsupported_route_has_grade")
            if status != "completed" or outcome != "unsupported":
                violations.append(f"{task_id}:unsupported_status_contract")
            if task.get("query_revisions"):
                violations.append(f"{task_id}:unsupported_has_query_revision")
            continue

        retrieval = retrieval_results.get(task_id)
        if status != "failed" and task.get("query_revisions") and retrieval is None:
            violations.append(f"{task_id}:retrieval_result_missing")
        if status == "failed":
            continue
        if retrieval is not None:
            if not grades:
                violations.append(f"{task_id}:successful_retrieval_missing_grade")
            else:
                record = grades[-1]
                input_ids = record.get("input_evidence_ids", [])
                grade = record.get("grade", {})
                supporting = grade.get("supporting_evidence_ids", [])
                evidence_ids = {
                    item.get("evidence_id")
                    for item in retrieval.get("evidence", [])
                }
                if len(input_ids) != len(set(input_ids)):
                    violations.append(f"{task_id}:grade_input_ids_duplicate")
                if set(input_ids) != evidence_ids:
                    violations.append(f"{task_id}:grade_input_not_final_evidence")
                if not set(supporting) <= set(input_ids):
                    violations.append(f"{task_id}:grade_supporting_id_outside_input")
            if not routes:
                violations.append(f"{task_id}:successful_grade_missing_route")
            else:
                route = routes[-1]
                if route.get("route") == "unsupported":
                    violations.append(f"{task_id}:supported_task_unsupported_route")
                if grades and route.get("grade_record_id") != grades[-1].get("id"):
                    violations.append(f"{task_id}:route_grade_record_mismatch")
                if route.get("route") == "recover" and not route.get("recovery_strategy"):
                    violations.append(f"{task_id}:recover_route_missing_strategy")
                if route.get("route") != "recover" and route.get("recovery_strategy") is not None:
                    violations.append(f"{task_id}:non_recover_route_has_strategy")

    stage_result = observed.get("stage_result")
    if isinstance(stage_result, dict):
        if any(task.get("execution_status") == "failed" for task in tasks):
            if stage_result.get("execution_status") != "failed":
                violations.append("stage_failed_task_status_mismatch")
    return violations


def _error(code: str, exc: Exception) -> dict[str, Any]:
    return {"code": code, "message": str(exc), "exception_type": type(exc).__name__}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_commit() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return None


def _git_dirty() -> bool | None:
    try:
        return bool(subprocess.run(["git", "status", "--short"], capture_output=True, text=True, check=True).stdout.strip())
    except Exception:
        return None


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Run the V2.1 Module 4 real-document eval")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args()
    report = evaluate_module4(args.manifest, output_root=args.output_root)
    print(json.dumps({"run_id": report["run_id"], "output": str(args.output_root / report["run_id"])}, ensure_ascii=False))


if __name__ == "__main__":
    main()
