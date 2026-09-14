"""V2.1 explicit LangGraph workflow.

This graph stops after proposing a route.  It intentionally has no recovery,
answer generation, HITL, or persistence behavior.
"""

from __future__ import annotations

from typing import Literal

from langgraph.graph import END, START, StateGraph

from .config import V2Config
from .grading import EvidenceGrader, EvidenceGradingError
from .ids import new_request_id, validate_request_id
from .module4 import (
    make_grade_record,
    materialize_retrieval_tasks,
    route_for,
    unsupported_routing_decision,
)
from .planning import PlanningError, PlanningResult, PlanningService
from .policies import detect_response_language, normalize_query
from .retrieval import RetrievalFanoutService
from .schemas import ExecutionError, RetrievalTask, StageRunResult
from .state import V2State


def initial_v2_1_state(
    question: str,
    *,
    request_id: str | None = None,
    response_language: str | None = None,
) -> V2State:
    normalized = normalize_query(question)
    resolved_request_id = request_id or new_request_id()
    validate_request_id(resolved_request_id)
    language = response_language or detect_response_language(normalized)
    if language not in {"zh", "en"}:
        raise ValueError("response_language 必须是 zh 或 en")
    return {
        "request_id": resolved_request_id,
        "target_stage": "v2_1",
        "original_question": question,
        "normalized_query": normalized,
        "response_language": language,
        "complexity_decision": None,
        "planning_result": None,
        "task_order": [],
        "tasks": {},
        "retrieval_results": {},
        "evidence": {},
        "pending_hitl_request": None,
        "hitl_rounds": 0,
        "execution_status": "running",
        "answer_outcome": None,
        "final_answer": None,
        "error": None,
        "state_schema_version": "v2.1",
        "stage_result": None,
    }


def build_graph_v2_1(
    *,
    config: V2Config | None = None,
    planner: PlanningService | None = None,
    retrieval: RetrievalFanoutService | None = None,
    grader: EvidenceGrader | None = None,
):
    config = config or V2Config.from_env()
    planner = planner or PlanningService(config)
    if retrieval is None:
        from .retrieval import RetrievalFanoutService, V12RetrievalAdapter

        retrieval = RetrievalFanoutService(V12RetrievalAdapter(), config)
    grader = grader or EvidenceGrader(config)

    workflow = StateGraph(V2State)
    workflow.add_node("initialize", lambda state: {})
    workflow.add_node("plan", lambda state: _plan_node(state, planner))
    workflow.add_node("materialize_tasks", lambda state: _materialize_node(state, config))
    workflow.add_node("retrieve", lambda state: _retrieve_node(state, retrieval))
    workflow.add_node("grade", lambda state: _grade_node(state, grader))
    workflow.add_node("route", lambda state: _route_node(state, config))
    workflow.add_node("stage_finalize", _finalize_node)
    workflow.add_edge(START, "initialize")
    workflow.add_edge("initialize", "plan")
    workflow.add_conditional_edges(
        "plan", _continue_after_plan, {"materialize_tasks": "materialize_tasks", "stage_finalize": "stage_finalize"}
    )
    workflow.add_edge("materialize_tasks", "retrieve")
    workflow.add_conditional_edges(
        "retrieve", _continue_after_retrieve, {"grade": "grade", "stage_finalize": "stage_finalize"}
    )
    workflow.add_conditional_edges(
        "grade", _continue_after_grade, {"route": "route", "stage_finalize": "stage_finalize"}
    )
    workflow.add_edge("route", "stage_finalize")
    workflow.add_edge("stage_finalize", END)
    return workflow.compile()


def _plan_node(state: V2State, planner: PlanningService) -> dict[str, object]:
    if state["execution_status"] == "failed":
        return {}
    try:
        result = planner.plan(state["original_question"])
    except PlanningError as exc:
        return {"execution_status": "failed", "error": exc.execution_error}
    except Exception as exc:
        return {
            "execution_status": "failed",
            "error": ExecutionError(
                code="planning_failed",
                message="Module 2 planning technical failure",
                stage="module4_planning",
                details={"exception_type": type(exc).__name__},
            ),
        }
    return {
        "planning_result": result,
        "complexity_decision": result.complexity_decision,
        "normalized_query": result.normalized_question,
    }


def _materialize_node(state: V2State, config: V2Config) -> dict[str, object]:
    planning = state.get("planning_result")
    if planning is None:
        return {
            "execution_status": "failed",
            "error": ExecutionError(
                code="planning_missing",
                message="PlanningResult missing before task materialization",
                stage="module4_materialization",
            ),
        }
    try:
        tasks = materialize_retrieval_tasks(
            planning, max_subqueries=config.budgets.max_subqueries
        )
    except Exception as exc:
        return {
            "execution_status": "failed",
            "error": ExecutionError(
                code="materialization_failed",
                message="PlanningResult cannot be materialized",
                stage="module4_materialization",
                details={"exception_type": type(exc).__name__},
            ),
        }
    return {"tasks": {task.id: task for task in tasks}, "task_order": [task.id for task in tasks]}


def _retrieve_node(state: V2State, retrieval: RetrievalFanoutService) -> dict[str, object]:
    tasks = list(state.get("tasks", {}).values())
    if not tasks:
        return {}
    try:
        result = retrieval.retrieve_tasks(tasks)
    except Exception as exc:
        return {
            "execution_status": "failed",
            "error": ExecutionError(
                code="retrieval_failed",
                message="Module 3 retrieval fan-out failed",
                stage="module4_retrieval",
                details={"exception_type": type(exc).__name__},
            ),
        }
    updates: dict[str, RetrievalTask] = {task.id: task for task in result.tasks}
    failed = next((task for task in result.tasks if task.execution_status == "failed"), None)
    output: dict[str, object] = {
        "tasks": updates,
        "retrieval_results": result.retrieval_results,
        "evidence": result.evidence,
    }
    if failed is not None:
        output["execution_status"] = "failed"
        output["error"] = failed.error
    return output


def _grade_node(state: V2State, grader: EvidenceGrader) -> dict[str, object]:
    task_updates: dict[str, RetrievalTask] = {}
    for task_id, task in sorted(state.get("tasks", {}).items(), key=lambda item: (item[1].ordinal, item[0])):
        if task.execution_status == "failed" or task.answer_outcome == "unsupported":
            continue
        retrieval_result = state.get("retrieval_results", {}).get(task_id)
        if retrieval_result is None:
            continue
        try:
            revision = task.query_revisions[-1]
            grade, _attempts = grader.grade(
                task=task,
                revision=revision,
                evidence=list(retrieval_result.evidence),
            )
            record = make_grade_record(
                task=task, revision=revision, evidence=list(retrieval_result.evidence), grade=grade
            )
            task_updates[task_id] = task.model_copy(update={"grade_records": [*task.grade_records, record]})
        except EvidenceGradingError as exc:
            task_updates[task_id] = task.model_copy(
                update={
                    "execution_status": "failed",
                    "answer_outcome": None,
                    "error": exc.execution_error,
                }
            )
            return {
                "tasks": task_updates,
                "execution_status": "failed",
                "error": exc.execution_error,
            }
        except Exception as exc:
            error = ExecutionError(
                code="grader_failed",
                message="Evidence Grader technical failure",
                stage="module4_grading",
                details={"task_id": task_id, "exception_type": type(exc).__name__},
            )
            task_updates[task_id] = task.model_copy(update={"execution_status": "failed", "error": error})
            return {"tasks": task_updates, "execution_status": "failed", "error": error}
    return {"tasks": task_updates}


def _route_node(state: V2State, config: V2Config) -> dict[str, object]:
    task_updates: dict[str, RetrievalTask] = {}
    try:
        for task_id, task in sorted(state.get("tasks", {}).items(), key=lambda item: (item[1].ordinal, item[0])):
            if task.execution_status == "failed":
                continue
            if task.answer_outcome == "unsupported":
                decision = unsupported_routing_decision(task)
            else:
                if not task.grade_records:
                    continue
                revision = task.query_revisions[-1]
                record = task.grade_records[-1]
                decision = route_for(task=task, revision=revision, grade_record=record, config=config)
            task_updates[task_id] = task.model_copy(
                update={"routing_decisions": [*task.routing_decisions, decision]}
            )
    except Exception as exc:
        error = ExecutionError(
            code="routing_failed",
            message="Deterministic routing policy failed",
            stage="module4_routing",
            details={"exception_type": type(exc).__name__},
        )
        return {"tasks": task_updates, "execution_status": "failed", "error": error}
    return {"tasks": task_updates}


def _continue_after_plan(state: V2State) -> Literal["materialize_tasks", "stage_finalize"]:
    return "stage_finalize" if state["execution_status"] == "failed" else "materialize_tasks"


def _continue_after_retrieve(state: V2State) -> Literal["grade", "stage_finalize"]:
    return "stage_finalize" if state["execution_status"] == "failed" else "grade"


def _continue_after_grade(state: V2State) -> Literal["route", "stage_finalize"]:
    return "stage_finalize" if state["execution_status"] == "failed" else "route"


def _finalize_node(state: V2State) -> dict[str, object]:
    status = "failed" if state["execution_status"] == "failed" else "completed"
    error = state.get("error")
    try:
        stage = StageRunResult(
            request_id=state["request_id"],
            target_stage="v2_1",
            execution_status=status,
            answer_outcome=None,
            final_answer=None,
            resumable=False,
            error=error,
        )
    except Exception as exc:
        execution_error = ExecutionError(
            code="stage_finalize_failed",
            message="V2.1 StageRunResult validation failed",
            stage="module4_finalize",
            details={"exception_type": type(exc).__name__},
        )
        stage = StageRunResult(
            request_id=state["request_id"],
            target_stage="v2_1",
            execution_status="failed",
            error=execution_error,
        )
        return {"execution_status": "failed", "error": execution_error, "stage_result": stage}
    return {"execution_status": status, "stage_result": stage}
