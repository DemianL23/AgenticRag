"""Explicit non-persistent LangGraph workflows for V2.1 and V2.2."""

from __future__ import annotations

from typing import Literal

from langgraph.graph import END, START, StateGraph

from .config import V2Config
from .answering import (
    FindingGenerator,
    SynthesisGenerator,
    build_answer_limitations,
    build_hitl_request,
    deterministic_terminal_answer,
    validate_synthesized_answer,
)
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
from .recovery import RecoveryExecutionError, RecoveryService
from .retrieval import RetrievalFanoutService
from .schemas import (
    ExecutionError,
    RetrievalTask,
    StageRunResult,
    SynthesizedAnswer,
)
from .state import V2State


def initial_v2_1_state(
    question: str,
    *,
    request_id: str | None = None,
    response_language: str | None = None,
) -> V2State:
    return _initial_state(
        question,
        target_stage="v2_1",
        request_id=request_id,
        response_language=response_language,
    )


def initial_v2_2_state(
    question: str,
    *,
    request_id: str | None = None,
    response_language: str | None = None,
) -> V2State:
    return _initial_state(
        question,
        target_stage="v2_2",
        request_id=request_id,
        response_language=response_language,
    )


def _initial_state(
    question: str,
    *,
    target_stage: Literal["v2_1", "v2_2"],
    request_id: str | None,
    response_language: str | None,
) -> V2State:
    normalized = normalize_query(question)
    resolved_request_id = request_id or new_request_id()
    validate_request_id(resolved_request_id)
    language = response_language or detect_response_language(normalized)
    if language not in {"zh", "en"}:
        raise ValueError("response_language 必须是 zh 或 en")
    return {
        "request_id": resolved_request_id,
        "target_stage": target_stage,
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
        "state_schema_version": target_stage,
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
    output: dict[str, object] = {
        "tasks": updates,
        "retrieval_results": result.retrieval_results,
        "evidence": result.evidence,
    }
    return output


def _grade_node(state: V2State, grader: EvidenceGrader) -> dict[str, object]:
    task_updates: dict[str, RetrievalTask] = {}
    first_error: ExecutionError | None = None
    for task_id, task in sorted(state.get("tasks", {}).items(), key=lambda item: (item[1].ordinal, item[0])):
        if task.execution_status == "failed" or task.answer_outcome == "unsupported":
            continue
        retrieval_result = state.get("retrieval_results", {}).get(task_id)
        if retrieval_result is None:
            error = ExecutionError(
                code="missing_retrieval_result",
                message="healthy retrieval task has no RetrievalResult",
                stage="module4_grading",
                details={"task_id": task_id},
            )
            task_updates[task_id] = task.model_copy(
                update={"execution_status": "failed", "answer_outcome": None, "error": error}
            )
            first_error = first_error or error
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
            first_error = first_error or exc.execution_error
            task_updates[task_id] = task.model_copy(
                update={
                    "execution_status": "failed",
                    "answer_outcome": None,
                    "error": exc.execution_error,
                }
            )
        except Exception as exc:
            error = ExecutionError(
                code="grader_failed",
                message="Evidence Grader technical failure",
                stage="module4_grading",
                details={"task_id": task_id, "exception_type": type(exc).__name__},
            )
            first_error = first_error or error
            task_updates[task_id] = task.model_copy(
                update={
                    "execution_status": "failed",
                    "answer_outcome": None,
                    "error": error,
                }
            )
    output: dict[str, object] = {"tasks": task_updates}
    # This is diagnostic state only.  It deliberately does not short-circuit
    # the route node; finalization evaluates all task outcomes together.
    if first_error is not None:
        output["error"] = first_error
    return output


def _route_node(state: V2State, config: V2Config) -> dict[str, object]:
    task_updates: dict[str, RetrievalTask] = {}
    first_error: ExecutionError | None = None
    for task_id, task in sorted(state.get("tasks", {}).items(), key=lambda item: (item[1].ordinal, item[0])):
        if task.execution_status == "failed":
            continue
        try:
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
                details={"task_id": task_id, "exception_type": type(exc).__name__},
            )
            first_error = first_error or error
            task_updates[task_id] = task.model_copy(
                update={"execution_status": "failed", "answer_outcome": None, "error": error}
            )
    output: dict[str, object] = {"tasks": task_updates}
    if first_error is not None:
        output["error"] = first_error
    return output


def _continue_after_plan(state: V2State) -> Literal["materialize_tasks", "stage_finalize"]:
    return "stage_finalize" if state["execution_status"] == "failed" else "materialize_tasks"


def _continue_after_retrieve(state: V2State) -> Literal["grade", "stage_finalize"]:
    return "stage_finalize" if state["execution_status"] == "failed" else "grade"


def _continue_after_grade(state: V2State) -> Literal["route", "stage_finalize"]:
    return "stage_finalize" if state["execution_status"] == "failed" else "route"


def _finalize_node(state: V2State) -> dict[str, object]:
    failed_tasks = [
        task for task in state.get("tasks", {}).values() if task.execution_status == "failed"
    ]
    status = "failed" if state["execution_status"] == "failed" or failed_tasks else "completed"
    error = state.get("error")
    if error is None and failed_tasks:
        error = failed_tasks[0].error
    if error is None and failed_tasks:
        error = ExecutionError(
            code="task_failed",
            message="one or more required tasks failed",
            stage="module4_finalize",
        )
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


def build_graph_v2_2(
    *,
    config: V2Config | None = None,
    planner: PlanningService | None = None,
    retrieval: RetrievalFanoutService | None = None,
    grader: EvidenceGrader | None = None,
    recovery: RecoveryService | None = None,
    finding: FindingGenerator | None = None,
    synthesis: SynthesisGenerator | None = None,
):
    """Build the non-persistent V2.2 Recovery → Finding → Answer graph."""
    config = config or V2Config.from_env()
    planner = planner or PlanningService(config)
    if retrieval is None:
        from .retrieval import V12RetrievalAdapter

        retrieval = RetrievalFanoutService(V12RetrievalAdapter(), config)
    grader = grader or EvidenceGrader(config)
    recovery = recovery or RecoveryService(config)
    finding = finding or FindingGenerator(config)
    synthesis = synthesis or SynthesisGenerator(config)

    workflow = StateGraph(V2State)
    workflow.add_node("initialize", lambda state: {})
    workflow.add_node("plan", lambda state: _plan_node(state, planner))
    workflow.add_node("materialize_tasks", lambda state: _materialize_node(state, config))
    workflow.add_node("retrieve", lambda state: _retrieve_node(state, retrieval))
    workflow.add_node("grade", lambda state: _grade_node(state, grader))
    workflow.add_node("route", lambda state: _route_node(state, config))
    workflow.add_node("recover", lambda state: _recover_node(state, recovery))
    workflow.add_node("terminalize_routes", lambda state: _terminalize_routes_node(state, config))
    workflow.add_node("generate_findings", lambda state: _finding_node(state, finding))
    workflow.add_node("aggregate_outcomes", _aggregate_node)
    workflow.add_node("simple_answer", lambda state: _simple_answer_node(state))
    workflow.add_node("synthesize", lambda state: _synthesis_node(state, synthesis))
    workflow.add_node("terminal_answer", _terminal_answer_node)
    workflow.add_node("validate_final_answer", _validate_final_answer_node)
    workflow.add_node("stage_finalize", _finalize_v22_node)

    workflow.add_edge(START, "initialize")
    workflow.add_edge("initialize", "plan")
    workflow.add_conditional_edges(
        "plan",
        _continue_after_plan,
        {"materialize_tasks": "materialize_tasks", "stage_finalize": "stage_finalize"},
    )
    workflow.add_edge("materialize_tasks", "retrieve")
    workflow.add_conditional_edges(
        "retrieve",
        _continue_after_retrieve,
        {"grade": "grade", "stage_finalize": "stage_finalize"},
    )
    workflow.add_conditional_edges(
        "grade",
        _continue_after_grade,
        {"route": "route", "stage_finalize": "stage_finalize"},
    )
    workflow.add_conditional_edges(
        "route",
        _continue_after_route_v22,
        {"recover": "recover", "terminalize_routes": "terminalize_routes", "stage_finalize": "stage_finalize"},
    )
    workflow.add_edge("recover", "terminalize_routes")
    workflow.add_conditional_edges(
        "terminalize_routes",
        _continue_after_terminalize_v22,
        {"waiting": "stage_finalize", "findings": "generate_findings"},
    )
    workflow.add_edge("generate_findings", "aggregate_outcomes")
    workflow.add_conditional_edges(
        "aggregate_outcomes",
        _continue_after_aggregate_v22,
        {
            "terminal_answer": "terminal_answer",
            "simple_answer": "simple_answer",
            "synthesize": "synthesize",
            "stage_finalize": "stage_finalize",
        },
    )
    workflow.add_edge("terminal_answer", "validate_final_answer")
    workflow.add_edge("simple_answer", "validate_final_answer")
    workflow.add_edge("synthesize", "validate_final_answer")
    workflow.add_edge("validate_final_answer", "stage_finalize")
    workflow.add_edge("stage_finalize", END)
    return workflow.compile()


def _continue_after_route_v22(
    state: V2State,
) -> Literal["recover", "terminalize_routes", "stage_finalize"]:
    if state["execution_status"] == "failed":
        return "stage_finalize"
    for task in _ordered_tasks(state):
        if task.routing_decisions and task.routing_decisions[-1].route == "recover":
            return "recover"
    return "terminalize_routes"


def _continue_after_terminalize_v22(
    state: V2State,
) -> Literal["waiting", "findings"]:
    return "waiting" if state["execution_status"] == "waiting_user" else "findings"


def _continue_after_aggregate_v22(
    state: V2State,
) -> Literal["terminal_answer", "simple_answer", "synthesize", "stage_finalize"]:
    if state["execution_status"] == "failed":
        return "stage_finalize"
    if state["answer_outcome"] in {"no_knowledge", "unsupported", "unresolved"}:
        return "terminal_answer"
    decision = state.get("complexity_decision")
    if decision is not None and decision.complexity == "simple":
        return "simple_answer"
    return "synthesize"


def _ordered_tasks(state: V2State) -> list[RetrievalTask]:
    return sorted(
        state.get("tasks", {}).values(), key=lambda task: (task.ordinal, task.id)
    )


def _recover_node(state: V2State, recovery: RecoveryService) -> dict[str, object]:
    task_updates: dict[str, RetrievalTask] = {}
    retrieval_updates = dict(state.get("retrieval_results", {}))
    evidence_updates: dict[str, object] = {}
    first_error: ExecutionError | None = None
    for task in _ordered_tasks(state):
        if not task.routing_decisions or task.routing_decisions[-1].route != "recover":
            continue
        try:
            result = recovery.recover(
                task=task,
                evidence_by_id=state.get("evidence", {}),
            )
            task_updates[task.id] = result.task
            retrieval_updates[task.id] = result.retrieval_result
            for item in result.retrieval_result.evidence:
                evidence_updates[item.evidence_id] = item
        except RecoveryExecutionError as exc:
            task_updates[task.id] = exc.failed_task
            first_error = first_error or exc.execution_error
            if exc.retrieval_result is not None:
                retrieval_updates[task.id] = exc.retrieval_result
                for item in exc.retrieval_result.evidence:
                    evidence_updates[item.evidence_id] = item
        except Exception as exc:
            error = ExecutionError(
                code="recovery_failed",
                message="Recovery precondition or execution failed",
                stage="module6_recovery",
                details={"task_id": task.id, "exception_type": type(exc).__name__},
            )
            task_updates[task.id] = task.model_copy(
                update={"execution_status": "failed", "answer_outcome": None, "error": error}
            )
            first_error = first_error or error
    output: dict[str, object] = {
        "tasks": task_updates,
        "retrieval_results": retrieval_updates,
        "evidence": evidence_updates,
    }
    if first_error is not None:
        output["error"] = first_error
    return output


def _terminalize_routes_node(state: V2State, config: V2Config) -> dict[str, object]:
    task_updates: dict[str, RetrievalTask] = {}
    waiting_tasks: dict[str, RetrievalTask] = {}
    first_error: ExecutionError | None = None
    for task in _ordered_tasks(state):
        if task.execution_status == "failed":
            continue
        route = task.routing_decisions[-1].route if task.routing_decisions else None
        if route == "no_knowledge":
            task_updates[task.id] = task.model_copy(
                update={
                    "execution_status": "completed",
                    "answer_outcome": "no_knowledge",
                    "terminal_reason": "deterministic route: no_knowledge",
                }
            )
        elif route == "unsupported":
            task_updates[task.id] = task.model_copy(
                update={
                    "execution_status": "completed",
                    "answer_outcome": "unsupported",
                    "terminal_reason": task.terminal_reason or "unsupported capability",
                }
            )
        elif route in {"clarify", "scope_select"}:
            waiting = task.model_copy(
                update={"execution_status": "waiting_user", "answer_outcome": None}
            )
            task_updates[task.id] = waiting
            waiting_tasks[task.id] = waiting
        elif route == "answer":
            task_updates[task.id] = task
        else:
            error = ExecutionError(
                code="routing_missing",
                message="Task has no terminal routing decision",
                stage="module6_terminalize",
                details={"task_id": task.id},
            )
            failed = task.model_copy(
                update={"execution_status": "failed", "answer_outcome": None, "error": error}
            )
            task_updates[task.id] = failed
            first_error = first_error or error

    output: dict[str, object] = {"tasks": task_updates}
    if waiting_tasks:
        merged_tasks = dict(state.get("tasks", {}))
        merged_tasks.update(task_updates)
        try:
            pending = build_hitl_request(
                request_id=state["request_id"],
                tasks=merged_tasks,
                evidence_by_id=state.get("evidence", {}),
                max_scope_options=config.budgets.max_scope_options,
            )
            output["pending_hitl_request"] = pending
            output["execution_status"] = "waiting_user"
        except Exception as exc:
            error = ExecutionError(
                code="hitl_payload_invalid",
                message="V2.2 waiting payload generation failed",
                stage="module6_terminalize",
                details={"exception_type": type(exc).__name__},
            )
            output["tasks"] = {
                task_id: task.model_copy(
                    update={"execution_status": "failed", "answer_outcome": None, "error": error}
                )
                for task_id, task in task_updates.items()
            }
            output["execution_status"] = "running"
            first_error = first_error or error
    if first_error is not None:
        output["error"] = first_error
    return output


def _finding_node(state: V2State, generator: FindingGenerator) -> dict[str, object]:
    task_updates: dict[str, RetrievalTask] = {}
    first_error: ExecutionError | None = None
    for task in _ordered_tasks(state):
        if task.execution_status == "failed":
            continue
        if not task.routing_decisions or task.routing_decisions[-1].route != "answer":
            continue
        try:
            role = "simple_answer" if state["complexity_decision"].complexity == "simple" else "finding"
            finding, _attempts = generator.generate(
                task=task,
                evidence_by_id=state.get("evidence", {}),
                response_language=state["response_language"],
                role=role,
            )
            task_updates[task.id] = task.model_copy(
                update={
                    "grounded_finding": finding,
                    "execution_status": "completed",
                    "answer_outcome": "complete",
                    "terminal_reason": "grounded finding generated",
                }
            )
        except Exception as exc:
            error = exc.execution_error if hasattr(exc, "execution_error") else ExecutionError(
                code="finding_generation_failed",
                message="GroundedFinding generation technical failure",
                stage="module6_finding",
                details={"task_id": task.id, "exception_type": type(exc).__name__},
            )
            failed = task.model_copy(
                update={"execution_status": "failed", "answer_outcome": None, "error": error}
            )
            task_updates[task.id] = failed
            first_error = first_error or error
    output: dict[str, object] = {"tasks": task_updates}
    if first_error is not None:
        output["error"] = first_error
    return output


def _aggregate_node(state: V2State) -> dict[str, object]:
    from .policies import aggregate_task_outcomes

    tasks = _ordered_tasks(state)
    try:
        status, outcome = aggregate_task_outcomes(tasks)
    except Exception as exc:
        error = ExecutionError(
            code="outcome_aggregation_failed",
            message="Global outcome aggregation failed",
            stage="module6_aggregation",
            details={"exception_type": type(exc).__name__},
        )
        return {"execution_status": "failed", "answer_outcome": None, "error": error}
    error = state.get("error")
    if status == "failed" and error is None:
        error = next((task.error for task in tasks if task.error is not None), None)
    return {"execution_status": status, "answer_outcome": outcome, "error": error}


def _simple_answer_node(state: V2State) -> dict[str, object]:
    findings = [task.grounded_finding for task in _ordered_tasks(state) if task.grounded_finding]
    tasks = _ordered_tasks(state)
    if len(findings) != 1:
        return {"execution_status": "failed", "answer_outcome": None, "error": _module6_error("simple_answer_invalid", "Simple path requires exactly one finding")}
    finding = findings[0]
    answer = SynthesizedAnswer(
        answer=finding.text,
        citation_evidence_ids=finding.evidence_ids,
        limitations=[],
    )
    try:
        validate_synthesized_answer(
            answer,
            tasks=tasks,
            findings=[finding],
            evidence_by_id=state.get("evidence", {}),
            global_outcome="complete",
        )
    except Exception as exc:
        return {"execution_status": "failed", "answer_outcome": None, "final_answer": None, "error": _module6_error("citation_validation_failed", str(exc))}
    return {"final_answer": answer}


def _synthesis_node(state: V2State, generator: SynthesisGenerator) -> dict[str, object]:
    tasks = _ordered_tasks(state)
    findings = [task.grounded_finding for task in tasks if task.grounded_finding]
    limitations = build_answer_limitations(tasks)
    try:
        answer, _attempts = generator.generate(
            original_question=state["original_question"],
            tasks=tasks,
            findings=findings,
            evidence_by_id=state.get("evidence", {}),
            limitations=limitations,
            response_language=state["response_language"],
        )
        return {"final_answer": answer}
    except Exception as exc:
        error = exc.execution_error if hasattr(exc, "execution_error") else _module6_error(
            "synthesis_failed", "Final synthesis technical failure"
        )
        return {"execution_status": "failed", "answer_outcome": None, "final_answer": None, "error": error}


def _terminal_answer_node(state: V2State) -> dict[str, object]:
    outcome = state.get("answer_outcome")
    if outcome not in {"no_knowledge", "unsupported", "unresolved"}:
        return {"execution_status": "failed", "answer_outcome": None, "error": _module6_error("terminal_answer_invalid", "Invalid terminal outcome")}
    answer = SynthesizedAnswer(
        answer=deterministic_terminal_answer(outcome, response_language=state["response_language"]),
        citation_evidence_ids=[],
        limitations=build_answer_limitations(_ordered_tasks(state)),
    )
    return {"final_answer": answer}


def _validate_final_answer_node(state: V2State) -> dict[str, object]:
    answer = state.get("final_answer")
    if answer is None:
        return {"execution_status": "failed", "answer_outcome": None, "error": _module6_error("citation_validation_failed", "Final answer missing")}
    findings = [task.grounded_finding for task in _ordered_tasks(state) if task.grounded_finding]
    try:
        validate_synthesized_answer(
            answer,
            tasks=_ordered_tasks(state),
            findings=findings,
            evidence_by_id=state.get("evidence", {}),
            global_outcome=state["answer_outcome"],
        )
    except Exception as exc:
        return {"execution_status": "failed", "answer_outcome": None, "final_answer": None, "error": _module6_error("citation_validation_failed", str(exc))}
    return {}


def _finalize_v22_node(state: V2State) -> dict[str, object]:
    status = state["execution_status"]
    stage_kwargs: dict[str, object] = {
        "request_id": state["request_id"],
        "target_stage": "v2_2",
        "execution_status": status,
        "answer_outcome": state.get("answer_outcome"),
        "final_answer": state.get("final_answer"),
        "resumable": False,
        "pending_hitl_request": state.get("pending_hitl_request"),
        "error": state.get("error"),
    }
    if status == "waiting_user":
        stage_kwargs.update(answer_outcome=None, final_answer=None, resumable=False)
    if status == "failed":
        stage_kwargs.update(answer_outcome=None, final_answer=None, pending_hitl_request=None)
        if stage_kwargs["error"] is None:
            stage_kwargs["error"] = _module6_error("module6_failed", "V2.2 request failed")
    try:
        stage = StageRunResult(**stage_kwargs)
    except Exception as exc:
        error = _module6_error("stage_finalize_failed", str(exc))
        stage = StageRunResult(
            request_id=state["request_id"],
            target_stage="v2_2",
            execution_status="failed",
            error=error,
        )
        return {"execution_status": "failed", "answer_outcome": None, "final_answer": None, "error": error, "stage_result": stage}
    return {"stage_result": stage}


def _module6_error(code: str, message: str) -> ExecutionError:
    return ExecutionError(code=code, message=message, stage="module6")
