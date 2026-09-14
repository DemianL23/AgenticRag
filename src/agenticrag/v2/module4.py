"""Module 4 orchestration: grade evidence and propose deterministic routes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from .config import V2Config
from .grading import EvidenceGrader, EvidenceGradingError
from .ids import grade_record_id, routing_decision_id, task_id
from .planning import PlanningResult, PlanningService
from .policies import capability_outcome, retrieval_attempt_available, routing_decision
from .retrieval import RetrievalFanoutService, V12RetrievalAdapter
from .schemas import (
    Evidence,
    ExecutionError,
    GradeRecord,
    QueryRevision,
    RetrievalResult,
    RetrievalTask,
    RoutingDecision,
    StageRunResult,
    TaskDraft,
)
from .state import V2State


class MaterializationError(ValueError):
    """The PlanningResult cannot be converted to executable RetrievalTasks."""


@dataclass(frozen=True, slots=True)
class Module4Run:
    """Serializable business output of a V2.1 run."""

    planning_result: PlanningResult | None
    tasks: tuple[RetrievalTask, ...]
    evidence: dict[str, Evidence]
    retrieval_results: dict[str, RetrievalResult]
    stage_result: StageRunResult


def materialize_retrieval_tasks(
    planning: PlanningResult, *, max_subqueries: int = 4
) -> list[RetrievalTask]:
    """Create program-assigned SQ IDs without changing LLM task order."""
    if isinstance(max_subqueries, bool) or not isinstance(max_subqueries, int) or max_subqueries < 1:
        raise ValueError("max_subqueries 必须是正整数")
    decision = planning.complexity_decision
    if decision.complexity == "simple":
        if planning.simple_task is None or planning.decomposition is not None:
            raise MaterializationError("simple PlanningResult contract 无法 materialize")
        return [_task_from_draft(planning.simple_task, 1)]

    decomposition = planning.decomposition
    if decomposition is None:
        raise MaterializationError("complex PlanningResult 缺少 DecompositionResult")
    if not decomposition.decomposition_complete:
        if decomposition.tasks:
            raise MaterializationError("decomposition_limit 不得 materialize TaskDraft")
        return []
    if not 2 <= len(decomposition.tasks) <= max_subqueries:
        raise MaterializationError(
            "complete DecompositionResult 的 task 数量超出 bounded max_subqueries"
        )
    return [_task_from_draft(draft, ordinal) for ordinal, draft in enumerate(decomposition.tasks, 1)]


def _task_from_draft(draft: TaskDraft, ordinal: int) -> RetrievalTask:
    return RetrievalTask(
        id=task_id(ordinal),
        ordinal=ordinal,
        query=draft.query,
        intent=draft.intent,
        capability=draft.capability,
        required=True,
    )


class Module4Service:
    """Non-persistent V2.1 application service backed by the StateGraph."""

    def __init__(
        self,
        config: V2Config | None = None,
        *,
        planner: PlanningService | None = None,
        retrieval: RetrievalFanoutService | None = None,
        grader: EvidenceGrader | None = None,
    ) -> None:
        self.config = config or V2Config.from_env()
        self.planner = planner or PlanningService(self.config)
        self.retrieval = retrieval or RetrievalFanoutService(
            V12RetrievalAdapter(), self.config
        )
        self.grader = grader or EvidenceGrader(self.config)

    def run(
        self,
        question: str,
        *,
        request_id: str | None = None,
        response_language: str | None = None,
    ) -> Module4Run:
        from .graph import build_graph_v2_1, initial_v2_1_state

        graph = build_graph_v2_1(
            config=self.config,
            planner=self.planner,
            retrieval=self.retrieval,
            grader=self.grader,
        )
        initial = initial_v2_1_state(
            question,
            request_id=request_id,
            response_language=response_language,
        )
        state = graph.invoke(initial)
        stage_result = state["stage_result"]
        assert stage_result is not None
        planning = state.get("planning_result")
        return Module4Run(
            planning_result=planning,
            tasks=tuple(state.get("tasks", {}).values()),
            evidence=state.get("evidence", {}),
            retrieval_results=state.get("retrieval_results", {}),
            stage_result=stage_result,
        )


def unsupported_routing_decision(task: RetrievalTask) -> RoutingDecision:
    """Create a deterministic capability route without fake Grade/Evidence."""
    outcome = capability_outcome(task.capability)
    if outcome != "unsupported":
        raise ValueError("unsupported_routing_decision 只接受 unsupported capability")
    return RoutingDecision(
        id=f"ROUTE_{task.id.replace('_', '')}_CAPABILITY_001",
        grade_record_id=None,
        route="unsupported",
        reason=f"capability policy: {task.capability} is unsupported in V2",
    )


def make_grade_record(
    *, task: RetrievalTask, revision: QueryRevision, evidence: Sequence[Evidence], grade: Any
) -> GradeRecord:
    return GradeRecord(
        id=grade_record_id(task.id, revision.id, len(task.grade_records) + 1),
        query_revision_id=revision.id,
        input_attempt_ids=[attempt.id for attempt in revision.retrieval_attempts],
        input_evidence_ids=[item.evidence_id for item in evidence],
        grade=grade,
    )


def route_for(
    *, task: RetrievalTask, revision: QueryRevision, grade_record: GradeRecord, config: V2Config
) -> RoutingDecision:
    decision = routing_decision(
        decision_id=routing_decision_id(task.id, revision.id, len(task.routing_decisions) + 1),
        grade_record_id=grade_record.id,
        capability=task.capability,
        grade=grade_record.grade,
        input_evidence_ids=grade_record.input_evidence_ids,
        retrieval_budget_available=retrieval_attempt_available(revision, config.budgets),
    )
    return decision
