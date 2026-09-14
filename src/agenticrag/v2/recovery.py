"""Bounded Module 5 recovery execution.

This module turns a deterministic ``recover`` decision into one corrective
retrieval attempt on the current QueryRevision.  It deliberately stops after
re-grade and re-route; answer generation belongs to Module 6.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Literal, Sequence

from pydantic import Field, ValidationError

from .config import V2Config
from .grading import (
    EvidenceGrader,
    EvidenceGradingError,
    _safe_exception_message,
)
from .ids import retrieval_attempt_id
from .module4 import make_grade_record, route_for
from .planning import (
    PlanningError,
    StructuredOutputContractError,
    _invoke_structured,
    create_decision_chat_model,
)
from .policies import RECOVERY_BY_FAILURE_REASON, normalize_query, retrieval_attempt_available
from .retrieval import (
    V12RetrievalAdapter,
    V12RetrievalBackend,
)
from .schemas import (
    Evidence,
    ExecutionError,
    EvidenceGrade,
    GradeRecord,
    QueryRevision,
    RetrievalAttempt,
    RetrievalResult,
    RetrievalTask,
    RoutingDecision,
    V2Model,
)
from .types import RecoveryStrategy
from .state import merge_evidence


class RecoveryArtifactPayload(V2Model):
    """The only structured output the Rewrite Model is allowed to produce."""

    retrieval_query: str = Field(min_length=1)


class RecoveryArtifact(V2Model):
    """A retrieval-only artifact; it is never Evidence."""

    strategy: RecoveryStrategy
    retrieval_query: str = Field(min_length=1)
    is_evidence: Literal[False] = False


class RecoveryGenerationError(RuntimeError):
    """The Rewrite Model could not produce a valid bounded artifact."""

    def __init__(self, *, attempts: int, cause: Exception) -> None:
        self.attempts = attempts
        self.cause = cause
        marker = str(cause).lower()
        if "recovery_duplicate_query" in marker:
            code = "recovery_duplicate_query"
        elif isinstance(cause, (ValidationError, StructuredOutputContractError)) or any(
            token in marker for token in ("structured", "schema", "parse", "json")
        ):
            code = "invalid_recovery_artifact"
        else:
            code = "recovery_model_failed"
        self.execution_error = ExecutionError(
            code=code,
            message=f"Recovery artifact generation failed after {attempts} attempt(s)",
            stage="module5_recovery",
            retryable=False,
            details={
                "role": "rewrite",
                "attempts": str(attempts),
                "cause_type": type(cause).__name__,
                "cause_message": _safe_exception_message(cause),
            },
        )
        super().__init__(self.execution_error.message)


class RecoveryExecutionError(RuntimeError):
    """A technical recovery failure with the task history preserved."""

    def __init__(
        self,
        *,
        task: RetrievalTask,
        execution_error: ExecutionError,
        cause: Exception | None = None,
    ) -> None:
        self.execution_error = execution_error
        self.cause = cause
        self.failed_task = task.model_copy(
            update={
                "execution_status": "failed",
                "answer_outcome": None,
                "error": execution_error,
            }
        )
        super().__init__(execution_error.message)


@dataclass(frozen=True, slots=True)
class RecoveryRun:
    """Serializable business output after Attempt 2 re-grade and re-route."""

    task: RetrievalTask
    artifact: RecoveryArtifact
    retrieval_result: RetrievalResult
    evidence: dict[str, Evidence]
    grade_record: GradeRecord
    routing_decision: RoutingDecision
    rewrite_attempts: int


RECOVERY_SYSTEM_PROMPT = """你是 Agentic RAG V2 的 Recovery Rewrite Model。
Recovery strategy 已由确定性 RoutingPolicy 选定，你不得选择、修改或输出 strategy、route 或 failure_reason。
你只能输出一个 retrieval_query，用于一次有界的 corrective retrieval；不能回答用户问题，不能生成事实结论。

direct_rewrite：保持原任务的实体、时间范围和必需信息需求不变，只改写为更适合检索的表达。
step_back：保留任务主题和实体，降低过度具体的限定，退回到抽象一级但仍保持任务约束。
hyde：生成一个 hypothetical document-style retrieval artifact，用于帮助检索；它不是 Evidence，不能被引用。

禁止增加新的业务目标、删除原任务所需信息、跨任务推理或引用其他任务。
只返回完整 JSON 对象：{"retrieval_query": "..."}。
"""


class RecoveryArtifactGenerator:
    """Generate one strategy-specific retrieval artifact with bounded retry."""

    def __init__(self, config: V2Config | None = None, *, model: Any | None = None) -> None:
        self.config = config or V2Config.from_env()
        self._model = model

    def generate(
        self,
        *,
        task: RetrievalTask,
        revision: QueryRevision,
        grade: EvidenceGrade,
        strategy: RecoveryStrategy,
        evidence: Sequence[Evidence],
    ) -> tuple[RecoveryArtifact, int]:
        if strategy not in RECOVERY_BY_FAILURE_REASON.values():
            raise ValueError(f"不支持的 recovery strategy：{strategy}")
        used_queries = {
            normalize_query(attempt.retrieval_query).casefold()
            for attempt in revision.retrieval_attempts
        }
        prompt = build_recovery_prompt(
            task=task,
            revision=revision,
            grade=grade,
            strategy=strategy,
            evidence=evidence,
        )
        model = self._model or create_decision_chat_model(
            self.config.decision_models.rewrite
        )

        def validate_payload(payload: RecoveryArtifactPayload) -> None:
            if not payload.retrieval_query.strip():
                raise StructuredOutputContractError(
                    "invalid_recovery_artifact: retrieval_query 不能为空"
                )
            normalized = normalize_query(payload.retrieval_query).casefold()
            if normalized in used_queries:
                raise StructuredOutputContractError(
                    "recovery_duplicate_query: retrieval_query 与当前 revision 已执行 query 重复"
                )

        try:
            payload, attempts = _invoke_structured(
                model,
                RecoveryArtifactPayload,
                prompt,
                role="rewrite",
                retry_policy=self.config.decision_models.rewrite.retry_policy,
                post_validate=validate_payload,
                repair_prompt_builder=_build_recovery_repair_prompt,
            )
        except PlanningError as exc:
            raise RecoveryGenerationError(attempts=exc.attempts, cause=exc.cause) from exc

        return (
            RecoveryArtifact(
                strategy=strategy,
                retrieval_query=normalize_query(payload.retrieval_query),
                is_evidence=False,
            ),
            attempts,
        )


def build_recovery_prompt(
    *,
    task: RetrievalTask,
    revision: QueryRevision,
    grade: EvidenceGrade,
    strategy: RecoveryStrategy,
    evidence: Sequence[Evidence],
) -> str:
    """Build a strategy-specific prompt from current-task state only."""
    context = {
        "strategy": strategy,
        "task": {"query": task.query, "intent": task.intent},
        "query_revision": {"query": revision.query, "source": revision.source},
        "grade": {
            "relevance": grade.relevance,
            "answerability": grade.answerability,
            "ambiguity": grade.ambiguity,
            "recoverability": grade.recoverability,
            "failure_reason": grade.failure_reason,
            "missing_information": grade.missing_information,
            "missing_slots": grade.missing_slots,
        },
        "current_final_evidence": [
            {
                "evidence_id": item.evidence_id,
                "content": item.content,
                "doc_id": item.doc_id,
                "source": item.source,
                "page": item.page,
            }
            for item in evidence
        ],
        "executed_retrieval_queries": [
            attempt.retrieval_query for attempt in revision.retrieval_attempts
        ],
    }
    return (
        RECOVERY_SYSTEM_PROMPT
        + "\n本次 strategy 由系统固定为："
        + strategy
        + "。\n输入：\n"
        + json.dumps(context, ensure_ascii=False, indent=2)
    )


def _build_recovery_repair_prompt(original_prompt: str, cause: Exception) -> str:
    return (
        f"{original_prompt}\n\n"
        "上一轮 Recovery artifact 未通过 structured contract。请重新生成完整对象，"
        "不要解释修复过程，不要改变系统指定的 strategy。\n"
        f"Failure type: {type(cause).__name__}\n"
        f"Failure: {_safe_exception_message(cause)}\n"
        "retrieval_query 必须非空，并且不能与 executed_retrieval_queries 中的任何 query 重复。"
    )


class RecoveryService:
    """Execute one bounded recovery attempt and its current-revision re-grade."""

    def __init__(
        self,
        config: V2Config | None = None,
        *,
        generator: RecoveryArtifactGenerator | None = None,
        backend: V12RetrievalBackend | None = None,
        grader: EvidenceGrader | None = None,
    ) -> None:
        self.config = config or V2Config.from_env()
        self.generator = generator or RecoveryArtifactGenerator(self.config)
        self.backend = backend or V12RetrievalAdapter()
        self.grader = grader or EvidenceGrader(self.config)

    def recover(
        self,
        *,
        task: RetrievalTask,
        evidence_by_id: dict[str, Evidence],
    ) -> RecoveryRun:
        revision, attempt, grade_record, routing = self._validate_preconditions(
            task, evidence_by_id
        )
        current_evidence = [evidence_by_id[evidence_id] for evidence_id in attempt.evidence_ids]
        try:
            artifact, rewrite_attempts = self.generator.generate(
                task=task,
                revision=revision,
                grade=grade_record.grade,
                strategy=routing.recovery_strategy,
                evidence=current_evidence,
            )
        except RecoveryGenerationError as exc:
            raise RecoveryExecutionError(
                task=task, execution_error=exc.execution_error, cause=exc
            ) from exc

        attempt2_id = retrieval_attempt_id(task.id, revision.id, 2)
        attempt2 = RetrievalAttempt(
            id=attempt2_id,
            ordinal=2,
            strategy=artifact.strategy,
            retrieval_query=artifact.retrieval_query,
        )
        try:
            retrieval_result = self.backend.retrieve(
                task_id=task.id,
                query_revision_id=revision.id,
                attempt_id=attempt2.id,
                query=artifact.retrieval_query,
                strategy=artifact.strategy,
            )
            attempt2 = attempt2.model_copy(
                update={
                    "evidence_ids": [item.evidence_id for item in retrieval_result.evidence],
                    "retrieval_degraded": retrieval_result.retrieval_degraded,
                    "retrieval_degraded_reason": retrieval_result.degraded_reason,
                    "latency": retrieval_result.latency,
                    "trace_ref": retrieval_result.trace_ref,
                }
            )
        except Exception as exc:
            failed_task = _append_attempt(task, revision, attempt2)
            raise RecoveryExecutionError(
                task=failed_task,
                execution_error=_recovery_error(
                    code="retrieval_failed",
                    message="Recovery Attempt 2 retrieval technical failure",
                    details={"exception_type": type(exc).__name__},
                ),
                cause=exc,
            ) from exc

        updated_task = _append_attempt(task, revision, attempt2)
        try:
            union_evidence = merge_evidence(
                {item.evidence_id: item for item in current_evidence},
                {item.evidence_id: item for item in retrieval_result.evidence},
            )
            if len(union_evidence) > 10:
                raise ValueError("当前 revision Evidence union 不得超过 10 条")
            ordered_ids = list(
                dict.fromkeys([*attempt.evidence_ids, *attempt2.evidence_ids])
            )
            union_list = [union_evidence[evidence_id] for evidence_id in ordered_ids]
            grade, _grader_attempts = self.grader.grade(
                task=updated_task,
                revision=updated_task.query_revisions[-1],
                evidence=union_list,
            )
            new_grade_record = make_grade_record(
                task=updated_task,
                revision=updated_task.query_revisions[-1],
                evidence=union_list,
                grade=grade,
            )
            graded_task = updated_task.model_copy(
                update={"grade_records": [*updated_task.grade_records, new_grade_record]}
            )
            new_routing = route_for(
                task=graded_task,
                revision=graded_task.query_revisions[-1],
                grade_record=new_grade_record,
                config=self.config,
            )
            routed_task = graded_task.model_copy(
                update={
                    "routing_decisions": [*graded_task.routing_decisions, new_routing]
                }
            )
        except EvidenceGradingError as exc:
            raise RecoveryExecutionError(
                task=updated_task,
                execution_error=exc.execution_error,
                cause=exc,
            ) from exc
        except Exception as exc:
            raise RecoveryExecutionError(
                task=updated_task,
                execution_error=_recovery_error(
                    code="regrade_failed",
                    message="Recovery re-grade or re-route technical failure",
                    details={"exception_type": type(exc).__name__},
                ),
                cause=exc,
            ) from exc

        return RecoveryRun(
            task=routed_task,
            artifact=artifact,
            retrieval_result=retrieval_result,
            evidence=union_evidence,
            grade_record=new_grade_record,
            routing_decision=new_routing,
            rewrite_attempts=rewrite_attempts,
        )

    def _validate_preconditions(
        self, task: RetrievalTask, evidence_by_id: dict[str, Evidence]
    ) -> tuple[QueryRevision, RetrievalAttempt, Any, RoutingDecision]:
        if task.capability != "retrieval_synthesis":
            raise ValueError("Recovery 只接受 retrieval_synthesis capability")
        if task.execution_status in {"failed", "completed", "waiting_user"}:
            raise ValueError("Recovery 不接受已终止或 waiting_user task")
        if not task.query_revisions:
            raise ValueError("Recovery task 必须至少有一个 QueryRevision")
        revision = task.query_revisions[-1]
        if not retrieval_attempt_available(revision, self.config.budgets):
            raise ValueError("Recovery retrieval attempt budget 已耗尽")
        if len(revision.retrieval_attempts) != 1:
            raise ValueError("Recovery 当前 revision 必须恰好有一个 RetrievalAttempt")
        if not task.grade_records:
            raise ValueError("Recovery task 必须至少有一个 GradeRecord")
        if not task.routing_decisions:
            raise ValueError("Recovery task 必须至少有一个 RoutingDecision")
        attempt = revision.retrieval_attempts[0]
        grade_record = task.grade_records[-1]
        routing = task.routing_decisions[-1]
        if grade_record.query_revision_id != revision.id:
            raise ValueError("最新 GradeRecord 不属于当前 QueryRevision")
        if grade_record.input_attempt_ids != [attempt.id]:
            raise ValueError("Recovery 初始 GradeRecord 必须只引用 Attempt 1")
        if set(grade_record.input_evidence_ids) != set(attempt.evidence_ids):
            raise ValueError("Recovery 初始 GradeRecord Evidence 必须等于 Attempt 1 Evidence")
        if routing.route != "recover" or routing.recovery_strategy is None:
            raise ValueError("Recovery 只接受最新 route=recover 且有 strategy 的 task")
        if routing.grade_record_id != grade_record.id:
            raise ValueError("最新 RoutingDecision 未引用最新 GradeRecord")
        expected_strategy = RECOVERY_BY_FAILURE_REASON.get(grade_record.grade.failure_reason)
        if (
            grade_record.grade.recoverability != "likely"
            or expected_strategy != routing.recovery_strategy
        ):
            raise ValueError("最新 RoutingDecision strategy 不符合 frozen recovery mapping")
        missing = set(attempt.evidence_ids) - set(evidence_by_id)
        if missing:
            raise ValueError(f"Recovery 当前 Evidence 缺失：{sorted(missing)}")
        return revision, attempt, grade_record, routing


def _append_attempt(
    task: RetrievalTask, revision: QueryRevision, attempt: RetrievalAttempt
) -> RetrievalTask:
    updated_revision = revision.model_copy(
        update={"retrieval_attempts": [*revision.retrieval_attempts, attempt]}
    )
    revisions = [*task.query_revisions[:-1], updated_revision]
    return task.model_copy(update={"query_revisions": revisions})


def _recovery_error(
    *, code: str, message: str, details: dict[str, str] | None = None
) -> ExecutionError:
    return ExecutionError(
        code=code,
        message=message,
        stage="module5_recovery",
        retryable=False,
        details=details or {},
    )
