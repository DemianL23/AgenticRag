"""Module 7 in-memory HITL resume contract.

This module owns the application boundary between a pending HITL request and
the affected-task continuation.  It deliberately does not implement a
LangGraph interrupt, a checkpointer, or any persistence mechanism; those are
Module 8 concerns.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .answering import (
    FindingGenerator,
    SynthesisGenerator,
    build_answer_limitations,
    deterministic_terminal_answer,
    validate_synthesized_answer,
)
from .config import V2Config
from .grading import EvidenceGrader, EvidenceGradingError, _safe_exception_message
from .ids import query_revision_id, retrieval_attempt_id
from .module4 import make_grade_record, route_for
from .policies import (
    aggregate_task_outcomes,
    capability_is_supported,
    hitl_round_available,
    query_revision_available,
    validate_resume_request,
)
from .recovery import RecoveryExecutionError, RecoveryService
from .retrieval import V12RetrievalAdapter, V12RetrievalBackend
from .schemas import (
    Evidence,
    ExecutionError,
    GroundedFinding,
    HITLRequest,
    QueryRevision,
    RetrievalAttempt,
    RetrievalResult,
    RetrievalTask,
    ResumeRequest,
    SynthesizedAnswer,
)
from .state import V2State, merge_evidence
from .types import GlobalAnswerOutcome, GlobalExecutionStatus, RetrievalStrategy


class HITLResumeError(ValueError):
    """A deterministic, non-mutating failure at the resume boundary."""

    def __init__(
        self,
        *,
        code: str,
        message: str,
        details: dict[str, object] | None = None,
    ) -> None:
        self.execution_error = ExecutionError(
            code=code,
            message=message,
            stage="module7_hitl_resume",
            retryable=False,
            details=details or {},
        )
        super().__init__(message)


@dataclass(frozen=True, slots=True)
class HITLResumeResult:
    """Serializable result of one accepted in-memory resume."""

    state: V2State
    affected_task_ids: tuple[str, ...]
    resumed_task_ids: tuple[str, ...]
    new_query_revision_ids: tuple[str, ...]


def await_user_input(state: V2State) -> dict[str, object]:
    """Represent the side-effect-free await boundary for a pending request.

    The function intentionally does not call ``interrupt()``.  It is the
    logical boundary that Module 8 can later wrap with a durable await node.
    """

    if state.get("target_stage") != "v2_3":
        raise HITLResumeError(
            code="request_not_resumable",
            message="await_user_input 只接受 target_stage=v2_3 的 state",
        )
    if state.get("execution_status") != "waiting_user":
        raise HITLResumeError(
            code="request_not_resumable",
            message="只有 waiting_user state 才能进入 await_user_input",
        )
    if state.get("pending_hitl_request") is None:
        raise HITLResumeError(
            code="request_not_resumable",
            message="waiting_user state 必须有 pending HITLRequest",
        )
    return {
        "execution_status": "waiting_user",
        "answer_outcome": None,
        "final_answer": None,
    }


class HITLResumeService:
    """Validate and continue only the tasks affected by a HITL response."""

    def __init__(
        self,
        config: V2Config | None = None,
        *,
        backend: V12RetrievalBackend | None = None,
        grader: EvidenceGrader | None = None,
        recovery: RecoveryService | None = None,
        finding: FindingGenerator | None = None,
        synthesis: SynthesisGenerator | None = None,
    ) -> None:
        self.config = config or V2Config.from_env()
        self.backend = backend or V12RetrievalAdapter()
        self.grader = grader or EvidenceGrader(self.config)
        self.recovery = recovery or RecoveryService(
            self.config, backend=self.backend, grader=self.grader
        )
        self.finding = finding or FindingGenerator(self.config)
        self.synthesis = synthesis or SynthesisGenerator(self.config)

    def resume(self, state: V2State, request: ResumeRequest) -> HITLResumeResult:
        """Accept one validated response and return a new logical state.

        All request/state validation happens before constructing any new
        revision or calling retrieval, grading, recovery, or answer models.
        The input mapping and its domain objects are never mutated in place.
        """

        if state.get("target_stage") != "v2_3":
            raise HITLResumeError(
                code="request_not_resumable",
                message="HITL resume 只接受 target_stage=v2_3 的 state",
            )

        pending = state.get("pending_hitl_request")
        if state.get("execution_status") != "waiting_user" or pending is None:
            raise HITLResumeError(
                code="request_not_resumable",
                message="当前 request 没有合法的 pending HITL 状态",
            )
        if pending.request_id != state.get("request_id"):
            raise HITLResumeError(
                code="request_not_resumable",
                message="pending HITLRequest 不属于当前 request",
            )
        try:
            validate_resume_request(request, pending)
            affected = self._resolve_affected_tasks(state, pending)
            if not hitl_round_available(
                state.get("hitl_rounds", 0), self.config.budgets
            ):
                raise HITLResumeError(
                    code="request_not_resumable",
                    message="HITL round budget 已耗尽",
                )
            inputs = self._collect_user_inputs(pending, request, affected)
            revision_available = {
                task.id: query_revision_available(task, self.config.budgets)
                for task in affected
            }
        except HITLResumeError:
            raise
        except Exception as exc:
            raise HITLResumeError(
                code="resume_payload_invalid",
                message="ResumeRequest 未通过 deterministic validation",
                details={
                    "cause_type": type(exc).__name__,
                    "cause_message": _safe_exception_message(exc),
                },
            ) from exc

        tasks = dict(state.get("tasks", {}))
        retrieval_results = dict(state.get("retrieval_results", {}))
        evidence = dict(state.get("evidence", {}))
        resumed_ids: list[str] = []
        revision_ids: list[str] = []

        for task in affected:
            user_input = inputs[task.id]
            if not revision_available[task.id]:
                tasks[task.id] = task.model_copy(
                    update={
                        "execution_status": "completed",
                        "answer_outcome": "unresolved",
                        "terminal_reason": "hitl_budget_exhausted",
                        "error": None,
                    }
                )
                continue

            revision = self._new_revision(task, user_input)
            revision_ids.append(revision.id)
            resumed_ids.append(task.id)
            resumed_task = task.model_copy(
                update={
                    "query_revisions": [*task.query_revisions, revision],
                    "execution_status": "running",
                    "answer_outcome": None,
                    "grounded_finding": None,
                    "error": None,
                }
            )
            resumed_task, retrieval_result, new_evidence = self._retrieve_revision(
                resumed_task, revision
            )
            tasks[task.id] = resumed_task
            if retrieval_result is not None:
                retrieval_results[task.id] = retrieval_result
            if new_evidence:
                try:
                    evidence = merge_evidence(evidence, new_evidence)
                except Exception as exc:
                    error = self._error(
                        "retrieval_failed",
                        "HITL revision Evidence merge failed",
                        task_id=task.id,
                        cause=exc,
                    )
                    tasks[task.id] = self._failed_task(resumed_task, error)
                    continue
            if resumed_task.execution_status == "failed":
                continue
            tasks[task.id] = self._grade_and_route(resumed_task, evidence)

            current = tasks[task.id]
            if current.execution_status == "failed":
                continue
            current, _recovery_result, recovery_evidence, recovery_retrieval = self._execute_route(
                current,
                evidence,
            )
            if recovery_retrieval is not None:
                retrieval_results[task.id] = recovery_retrieval
            if recovery_evidence:
                try:
                    evidence = merge_evidence(evidence, recovery_evidence)
                except Exception as exc:
                    error = self._error(
                        "regrade_failed",
                        "HITL recovery Evidence merge failed",
                        task_id=task.id,
                        cause=exc,
                    )
                    current = self._failed_task(current, error)
            if current.execution_status != "failed":
                current = self._complete_route(
                    current,
                    evidence,
                    simple=state.get("complexity_decision") is not None
                    and state["complexity_decision"].complexity == "simple",
                    response_language=state["response_language"],
                )
            tasks[task.id] = current

        next_state = dict(state)
        next_state.update(
            {
                "tasks": tasks,
                "retrieval_results": retrieval_results,
                "evidence": evidence,
                "pending_hitl_request": None,
                "hitl_rounds": state.get("hitl_rounds", 0) + 1,
                "execution_status": "running",
                "answer_outcome": None,
                "final_answer": None,
                "error": None,
                "stage_result": None,
            }
        )
        finalized = self._finalize(next_state)
        return HITLResumeResult(
            state=finalized,
            affected_task_ids=tuple(task.id for task in affected),
            resumed_task_ids=tuple(resumed_ids),
            new_query_revision_ids=tuple(revision_ids),
        )

    def _resolve_affected_tasks(
        self, state: V2State, pending: HITLRequest
    ) -> list[RetrievalTask]:
        tasks = state.get("tasks", {})
        affected_ids: list[str] = []
        actions: dict[str, str] = {}
        for item in pending.items:
            for task_id in item.affected_task_ids:
                task = tasks.get(task_id)
                if task is None:
                    raise ValueError(f"HITL affected task 不存在：{task_id}")
                if task.execution_status != "waiting_user":
                    raise ValueError(f"HITL affected task 状态非法：{task_id}")
                if not task.routing_decisions:
                    raise ValueError(f"HITL affected task 缺少 routing decision：{task_id}")
                route = task.routing_decisions[-1].route
                if route != item.action:
                    raise ValueError(f"HITL action 与 task route 不一致：{task_id}")
                if not capability_is_supported(task.capability):
                    raise ValueError(f"HITL affected task capability 不可恢复：{task_id}")
                if task.grounded_finding is not None:
                    raise ValueError(f"waiting task 不应已有 GroundedFinding：{task_id}")
                if item.action == "scope_select":
                    valid_evidence = {
                        evidence_id
                        for revision in task.query_revisions
                        for attempt in revision.retrieval_attempts
                        for evidence_id in attempt.evidence_ids
                    }
                    for option in item.scope_options:
                        if not set(option.evidence_ids) <= valid_evidence:
                            raise ValueError(
                                f"ScopeOption 引用了当前 task 之外的 Evidence：{task_id}"
                            )
                previous_action = actions.setdefault(task_id, item.action)
                if previous_action != item.action:
                    raise ValueError(f"同一 task 不能有不同 HITL action：{task_id}")
                if task_id not in affected_ids:
                    affected_ids.append(task_id)
        return sorted(
            (tasks[task_id] for task_id in affected_ids),
            key=lambda task: (task.ordinal, task.id),
        )

    def _collect_user_inputs(
        self,
        pending: HITLRequest,
        request: ResumeRequest,
        affected: list[RetrievalTask],
    ) -> dict[str, dict[str, str]]:
        responses = {response.item_id: response for response in request.responses}
        inputs = {task.id: {} for task in affected}
        for item in pending.items:
            response = responses[item.id]
            if item.action == "clarify":
                assert response.clarify_values is not None
                values = {
                    key: response.clarify_values[key]
                    for key in sorted(response.clarify_values)
                }
                for task_id in item.affected_task_ids:
                    self._merge_user_values(inputs[task_id], values)
            else:
                assert response.selected_option_id is not None
                option = next(
                    option
                    for option in item.scope_options
                    if option.id == response.selected_option_id
                )
                for task_id in item.affected_task_ids:
                    self._merge_user_values(inputs[task_id], {"scope": option.value})
        if any(not values for values in inputs.values()):
            raise ValueError("合法 Resume 必须为每个 affected task 提供用户输入")
        return inputs

    @staticmethod
    def _merge_user_values(target: dict[str, str], values: dict[str, str]) -> None:
        for key, value in values.items():
            existing = target.get(key)
            if existing is not None and existing != value:
                raise ValueError(f"同一 task 的 HITL 输入冲突：{key}")
            target[key] = value

    @staticmethod
    def _new_revision(
        task: RetrievalTask, user_input: dict[str, str]
    ) -> QueryRevision:
        if not task.query_revisions:
            raise ValueError("HITL resume task 必须已有 QueryRevision")
        previous = task.query_revisions[-1]
        query = _merge_query(previous.query, user_input)
        return QueryRevision(
            id=query_revision_id(task.id, previous.ordinal + 1),
            ordinal=previous.ordinal + 1,
            source="hitl",
            query=query,
            user_input=dict(sorted(user_input.items())),
        )

    def _retrieve_revision(
        self, task: RetrievalTask, revision: QueryRevision
    ) -> tuple[RetrievalTask, RetrievalResult | None, dict[str, Evidence]]:
        attempt = RetrievalAttempt(
            id=retrieval_attempt_id(task.id, revision.id, 1),
            ordinal=1,
            strategy="user_clarified",
            retrieval_query=revision.query,
        )
        try:
            result = self.backend.retrieve(
                task_id=task.id,
                query_revision_id=revision.id,
                attempt_id=attempt.id,
                query=revision.query,
                strategy="user_clarified",
            )
            attempt = attempt.model_copy(
                update={
                    "evidence_ids": [item.evidence_id for item in result.evidence],
                    "retrieval_degraded": result.retrieval_degraded,
                    "retrieval_degraded_reason": result.degraded_reason,
                    "latency": result.latency,
                    "trace_ref": result.trace_ref,
                }
            )
            updated_revision = revision.model_copy(update={"retrieval_attempts": [attempt]})
            updated_task = task.model_copy(
                update={"query_revisions": [*task.query_revisions[:-1], updated_revision]}
            )
            return (
                updated_task,
                result,
                {item.evidence_id: item for item in result.evidence},
            )
        except Exception as exc:
            failed_revision = revision.model_copy(update={"retrieval_attempts": [attempt]})
            failed_task = task.model_copy(
                update={
                    "query_revisions": [*task.query_revisions[:-1], failed_revision],
                    "execution_status": "failed",
                    "answer_outcome": None,
                    "error": self._error(
                        "retrieval_failed",
                        "HITL user_clarified retrieval technical failure",
                        task_id=task.id,
                        cause=exc,
                    ),
                }
            )
            return failed_task, None, {}

    def _grade_and_route(
        self,
        task: RetrievalTask,
        evidence: dict[str, Evidence],
    ) -> RetrievalTask:
        revision = task.query_revisions[-1]
        attempt = revision.retrieval_attempts[-1]
        current_evidence = [evidence[item] for item in attempt.evidence_ids]
        try:
            grade, _attempts = self.grader.grade(
                task=task,
                revision=revision,
                evidence=current_evidence,
            )
            record = make_grade_record(
                task=task,
                revision=revision,
                evidence=current_evidence,
                grade=grade,
            )
            graded = task.model_copy(
                update={"grade_records": [*task.grade_records, record]}
            )
            decision = route_for(
                task=graded,
                revision=revision,
                grade_record=record,
                config=self.config,
            )
            return graded.model_copy(
                update={"routing_decisions": [*graded.routing_decisions, decision]}
            )
        except EvidenceGradingError as exc:
            return self._failed_task(task, exc.execution_error)
        except Exception as exc:
            return self._failed_task(
                task,
                self._error(
                    "grader_failed",
                    "HITL revision grading technical failure",
                    task_id=task.id,
                    cause=exc,
                ),
            )

    def _execute_route(
        self,
        task: RetrievalTask,
        evidence: dict[str, Evidence],
    ) -> tuple[RetrievalTask, Any | None, dict[str, Evidence], RetrievalResult | None]:
        route = task.routing_decisions[-1].route
        if route == "recover":
            try:
                result = self.recovery.recover(task=task, evidence_by_id=evidence)
                return (
                    result.task,
                    result,
                    {item.evidence_id: item for item in result.retrieval_result.evidence},
                    result.retrieval_result,
                )
            except RecoveryExecutionError as exc:
                return (
                    exc.failed_task,
                    None,
                    {
                        item.evidence_id: item
                        for item in (exc.retrieval_result.evidence if exc.retrieval_result else [])
                    },
                    exc.retrieval_result,
                )
            except Exception as exc:
                return (
                    self._failed_task(
                        task,
                        self._error(
                            "recovery_failed",
                            "HITL revision recovery technical failure",
                            task_id=task.id,
                            cause=exc,
                        ),
                    ),
                    None,
                    {},
                    None,
                )
        return task, None, {}, None

    def _complete_route(
        self,
        task: RetrievalTask,
        evidence: dict[str, Evidence],
        *,
        simple: bool,
        response_language: str,
    ) -> RetrievalTask:
        route = task.routing_decisions[-1].route
        if route == "answer":
            try:
                finding, _attempts = self.finding.generate(
                    task=task,
                    evidence_by_id=evidence,
                    response_language=response_language,  # type: ignore[arg-type]
                    role="simple_answer" if simple else "finding",
                )
                return task.model_copy(
                    update={
                        "grounded_finding": finding,
                        "execution_status": "completed",
                        "answer_outcome": "complete",
                        "terminal_reason": "grounded finding generated",
                        "error": None,
                    }
                )
            except Exception as exc:
                error = getattr(exc, "execution_error", None) or self._error(
                    "finding_generation_failed",
                    "HITL revision Finding generation technical failure",
                    task_id=task.id,
                    cause=exc,
                )
                return self._failed_task(task, error)
        if route == "no_knowledge":
            return task.model_copy(
                update={
                    "execution_status": "completed",
                    "answer_outcome": "no_knowledge",
                    "terminal_reason": "deterministic route: no_knowledge",
                }
            )
        if route == "unsupported":
            return task.model_copy(
                update={
                    "execution_status": "completed",
                    "answer_outcome": "unsupported",
                    "terminal_reason": "unsupported capability",
                }
            )
        if route in {"clarify", "scope_select"}:
            return task.model_copy(
                update={
                    "execution_status": "completed",
                    "answer_outcome": "unresolved",
                    "terminal_reason": "hitl_budget_exhausted",
                }
            )
        return self._failed_task(
            task,
            self._error(
                "routing_failed",
                "HITL revision produced an unsupported route",
                task_id=task.id,
            ),
        )

    def _finalize(self, state: V2State) -> V2State:
        tasks = sorted(
            state.get("tasks", {}).values(), key=lambda task: (task.ordinal, task.id)
        )
        status, outcome = aggregate_task_outcomes(tasks)
        evidence = state.get("evidence", {})
        final_answer: SynthesizedAnswer | None = None
        error: ExecutionError | None = None
        if status == "failed":
            error = next((task.error for task in tasks if task.error), None)
        elif outcome in {"no_knowledge", "unsupported", "unresolved"}:
            final_answer = SynthesizedAnswer(
                answer=deterministic_terminal_answer(
                    outcome, response_language=state["response_language"]
                ),
                citation_evidence_ids=[],
                limitations=build_answer_limitations(tasks),
            )
        elif state.get("complexity_decision") is not None and state[
            "complexity_decision"
        ].complexity == "simple":
            findings = [task.grounded_finding for task in tasks if task.grounded_finding]
            if len(findings) != 1:
                status, outcome = "failed", None
                error = self._error(
                    "simple_answer_invalid",
                    "Simple resume path requires exactly one Finding",
                )
            else:
                final_answer = SynthesizedAnswer(
                    answer=findings[0].text,
                    citation_evidence_ids=findings[0].evidence_ids,
                    limitations=[],
                )
        else:
            findings = [task.grounded_finding for task in tasks if task.grounded_finding]
            limitations = build_answer_limitations(tasks)
            try:
                final_answer, _attempts = self.synthesis.generate(
                    original_question=state["original_question"],
                    tasks=tasks,
                    findings=findings,
                    evidence_by_id=evidence,
                    limitations=limitations,
                    response_language=state["response_language"],
                )
            except Exception as exc:
                status, outcome, final_answer = "failed", None, None
                error = getattr(exc, "execution_error", None) or self._error(
                    "synthesis_failed",
                    "HITL resume final synthesis technical failure",
                    cause=exc,
                )
        if status != "failed" and final_answer is not None:
            try:
                validate_synthesized_answer(
                    final_answer,
                    tasks=tasks,
                    findings=[task.grounded_finding for task in tasks if task.grounded_finding],
                    evidence_by_id=evidence,
                    global_outcome=outcome,  # type: ignore[arg-type]
                )
            except Exception as exc:
                status, outcome, final_answer = "failed", None, None
                error = self._error(
                    "citation_validation_failed",
                    "HITL resume final answer validation failed",
                    cause=exc,
                )
        stage = self._stage_result(
            state,
            status=status,
            outcome=outcome,
            final_answer=final_answer,
            error=error,
        )
        next_state = dict(state)
        next_state.update(
            {
                "execution_status": status,
                "answer_outcome": outcome,
                "final_answer": final_answer,
                "error": error,
                "stage_result": stage,
            }
        )
        return next_state  # type: ignore[return-value]

    @staticmethod
    def _stage_result(
        state: V2State,
        *,
        status: GlobalExecutionStatus,
        outcome: GlobalAnswerOutcome | None,
        final_answer: SynthesizedAnswer | None,
        error: ExecutionError | None,
    ):
        from .schemas import StageRunResult

        if status == "failed":
            return StageRunResult(
                request_id=state["request_id"],
                target_stage=state["target_stage"],
                execution_status="failed",
                answer_outcome=None,
                final_answer=None,
                resumable=False,
                pending_hitl_request=None,
                error=error
                or ExecutionError(
                    code="module7_failed",
                    message="HITL resume failed",
                    stage="module7_hitl_resume",
                ),
            )
        return StageRunResult(
            request_id=state["request_id"],
            target_stage=state["target_stage"],
            execution_status="completed",
            answer_outcome=outcome,
            final_answer=final_answer,
            resumable=False,
            pending_hitl_request=None,
            error=None,
        )

    @staticmethod
    def _failed_task(task: RetrievalTask, error: ExecutionError) -> RetrievalTask:
        return task.model_copy(
            update={"execution_status": "failed", "answer_outcome": None, "error": error}
        )

    @staticmethod
    def _error(
        code: str,
        message: str,
        *,
        task_id: str | None = None,
        cause: Exception | None = None,
    ) -> ExecutionError:
        details: dict[str, object] = {}
        if task_id is not None:
            details["task_id"] = task_id
        if cause is not None:
            details.update(
                cause_type=type(cause).__name__,
                cause_message=_safe_exception_message(cause),
            )
        return ExecutionError(
            code=code,
            message=message,
            stage="module7_hitl_resume",
            retryable=False,
            details=details,
        )


def _merge_query(query: str, user_input: dict[str, str]) -> str:
    values = "; ".join(f"{key}={user_input[key]}" for key in sorted(user_input))
    return f"{query} [user-provided HITL input: {values}]"
