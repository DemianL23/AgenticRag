"""Pydantic domain contracts for V2.

These models describe workflow data only. They intentionally contain no model,
retriever, database, network client, lock, or other runtime object.
"""

from __future__ import annotations

from math import isfinite
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .ids import validate_request_id
from .types import (
    AnswerLimitationKind,
    Complexity,
    EvidenceFailureReason,
    GlobalAnswerOutcome,
    GlobalExecutionStatus,
    HITLAction,
    RecoveryStrategy,
    ResponseLanguage,
    RetrievalStrategy,
    Route,
    TargetStage,
    TaskAnswerOutcome,
    TaskCapability,
    TaskExecutionStatus,
)


class V2Model(BaseModel):
    """Strict, assignment-validating, JSON-serializable domain model."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)


def _nonempty(value: str) -> str:
    if not value.strip():
        raise ValueError("value 不能为空")
    return value


def _finite_nonnegative(value: float) -> float:
    if not isfinite(value) or value < 0:
        raise ValueError("耗时必须是有限的非负数")
    return value


def _positive_rank(value: int | None) -> int | None:
    if value is not None and (isinstance(value, bool) or value <= 0):
        raise ValueError("rank 必须是正整数")
    return value


class ExecutionError(V2Model):
    code: str = Field(min_length=1)
    message: str = Field(min_length=1)
    stage: str | None = None
    retryable: bool = False
    details: dict[str, str] = Field(default_factory=dict)

    _validate_code = field_validator("code", "message")(_nonempty)


class RetrievalLatency(V2Model):
    query_embedding_seconds: float = 0.0
    dense_search_seconds: float = 0.0
    bm25_search_seconds: float = 0.0
    merge_rrf_seconds: float = 0.0
    candidate_total_seconds: float = 0.0
    rerank_seconds: float = 0.0
    total_seconds: float = 0.0

    _validate_values = field_validator(
        "query_embedding_seconds",
        "dense_search_seconds",
        "bm25_search_seconds",
        "merge_rrf_seconds",
        "candidate_total_seconds",
        "rerank_seconds",
        "total_seconds",
    )(_finite_nonnegative)


class ComplexityDecision(V2Model):
    complexity: Complexity
    capability: TaskCapability | None
    reason: str = Field(min_length=1)

    _validate_reason = field_validator("reason")(_nonempty)

    @model_validator(mode="after")
    def validate_capability_contract(self) -> "ComplexityDecision":
        if self.complexity == "simple" and self.capability is None:
            raise ValueError("simple ComplexityDecision 必须声明 capability")
        if self.complexity == "complex" and self.capability is not None:
            raise ValueError("complex ComplexityDecision 的 capability 必须为 null")
        return self


class TaskDraft(V2Model):
    query: str = Field(min_length=1)
    intent: str = Field(min_length=1)
    capability: TaskCapability

    _validate_text = field_validator("query", "intent")(_nonempty)


class DecompositionResult(V2Model):
    tasks: list[TaskDraft] = Field(default_factory=list)
    decomposition_complete: bool
    failure_reason: str | None = None

    @model_validator(mode="after")
    def validate_decomposition_contract(self) -> "DecompositionResult":
        if self.decomposition_complete:
            if not self.tasks:
                raise ValueError("decomposition_complete=true 时 tasks 不能为空")
            if self.failure_reason is not None:
                raise ValueError("完整 decomposition 不应有 failure_reason")
        else:
            if self.failure_reason != "decomposition_limit":
                raise ValueError(
                    "decomposition_complete=false 时 failure_reason 必须为 decomposition_limit"
                )
            if self.tasks:
                raise ValueError(
                    "decomposition_complete=false 时 tasks 必须为空，禁止保留超限 TaskDraft"
                )
        normalized = [task.query.casefold().strip() for task in self.tasks]
        if len(normalized) != len(set(normalized)):
            raise ValueError("TaskDraft query 不得重复")
        return self


class RetrievalAttempt(V2Model):
    id: str = Field(min_length=1)
    ordinal: int = Field(gt=0)
    strategy: RetrievalStrategy
    retrieval_query: str = Field(min_length=1)
    evidence_ids: list[str] = Field(default_factory=list)
    retrieval_degraded: bool = False
    retrieval_degraded_reason: str | None = None
    latency: RetrievalLatency = Field(default_factory=RetrievalLatency)
    trace_ref: str | None = None

    _validate_text = field_validator("id", "retrieval_query")(_nonempty)

    @field_validator("evidence_ids")
    @classmethod
    def validate_evidence_ids(cls, value: list[str]) -> list[str]:
        if any(not item.strip() for item in value):
            raise ValueError("evidence_ids 不能包含空字符串")
        if len(value) != len(set(value)):
            raise ValueError("evidence_ids 不得重复")
        return value

    @model_validator(mode="after")
    def validate_degraded_reason(self) -> "RetrievalAttempt":
        if self.retrieval_degraded and not self.retrieval_degraded_reason:
            raise ValueError("retrieval_degraded=true 时必须记录 reason")
        if not self.retrieval_degraded and self.retrieval_degraded_reason is not None:
            raise ValueError("正常 retrieval 不应有 degraded reason")
        return self


class QueryRevision(V2Model):
    id: str = Field(min_length=1)
    ordinal: int = Field(gt=0)
    source: str
    query: str = Field(min_length=1)
    user_input: dict[str, str] | None = None
    retrieval_attempts: list[RetrievalAttempt] = Field(default_factory=list)

    _validate_text = field_validator("id", "query")(_nonempty)

    @field_validator("source")
    @classmethod
    def validate_source(cls, value: str) -> str:
        if value not in {"original", "hitl"}:
            raise ValueError("QueryRevision.source 必须是 original 或 hitl")
        return value

    @field_validator("user_input")
    @classmethod
    def validate_user_input(cls, value: dict[str, str] | None) -> dict[str, str] | None:
        if value is not None and any(not key.strip() or not item.strip() for key, item in value.items()):
            raise ValueError("user_input 的 key/value 不能为空")
        return value


class EvidenceOccurrence(V2Model):
    task_id: str = Field(min_length=1)
    query_revision_id: str = Field(min_length=1)
    retrieval_attempt_id: str = Field(min_length=1)
    strategy: RetrievalStrategy
    dense_rank: int | None = None
    bm25_rank: int | None = None
    rrf_rank: int | None = None
    final_rank: int = Field(gt=0)

    _validate_ids = field_validator(
        "task_id", "query_revision_id", "retrieval_attempt_id"
    )(_nonempty)
    _validate_ranks = field_validator("dense_rank", "bm25_rank", "rrf_rank")(_positive_rank)


class Evidence(V2Model):
    evidence_id: str = Field(min_length=1)
    chunk_id: str = Field(min_length=1)
    content: str = Field(min_length=1)
    doc_id: str = Field(min_length=1)
    source: str = Field(min_length=1)
    page: int = Field(gt=0)
    occurrences: list[EvidenceOccurrence] = Field(default_factory=list)

    _validate_text = field_validator(
        "evidence_id", "chunk_id", "content", "doc_id", "source"
    )(_nonempty)

    @model_validator(mode="after")
    def validate_stable_id(self) -> "Evidence":
        if self.evidence_id != self.chunk_id:
            raise ValueError("Evidence.evidence_id 必须等于稳定 chunk_id")
        return self


class EvidenceGrade(V2Model):
    relevance: Literal["none", "weak", "strong"]
    answerability: Literal["none", "partial", "sufficient"]
    ambiguity: Literal["none", "missing_slot", "multiple_candidates"]
    recoverability: Literal["none", "likely"]
    failure_reason: EvidenceFailureReason
    reason: str = Field(min_length=1)
    missing_information: list[str] = Field(default_factory=list)
    missing_slots: list[str] = Field(default_factory=list)
    supporting_evidence_ids: list[str] = Field(default_factory=list)

    _validate_reason = field_validator("reason")(_nonempty)

    @field_validator("missing_information", "missing_slots")
    @classmethod
    def validate_missing_values(cls, value: list[str]) -> list[str]:
        if any(not item.strip() for item in value):
            raise ValueError("missing information/slots 不能包含空字符串")
        return value

    @field_validator("supporting_evidence_ids")
    @classmethod
    def validate_supporting_ids(cls, value: list[str]) -> list[str]:
        if any(not item.strip() for item in value):
            raise ValueError("supporting evidence ID 不能为空")
        if len(value) != len(set(value)):
            raise ValueError("supporting_evidence_ids 不得重复")
        return value

    @model_validator(mode="after")
    def validate_cross_fields(self) -> "EvidenceGrade":
        if self.answerability in {"sufficient", "partial"} and not self.supporting_evidence_ids:
            raise ValueError("partial/sufficient 必须至少有 supporting evidence")
        if self.answerability == "none" and self.supporting_evidence_ids:
            raise ValueError("answerability=none 时 supporting evidence 必须为空")
        if self.relevance == "none" and self.answerability == "sufficient":
            raise ValueError("relevance=none 不能对应 sufficient")
        if self.recoverability == "likely":
            if self.answerability == "sufficient":
                raise ValueError("recoverability=likely 不能对应 sufficient")
            if self.failure_reason == "none":
                raise ValueError("recoverability=likely 必须有 failure_reason")
        if self.failure_reason == "none" and self.recoverability != "none":
            raise ValueError("failure_reason=none 时 recoverability 必须为 none")
        if self.ambiguity == "missing_slot" and not self.missing_slots:
            raise ValueError("ambiguity=missing_slot 时 missing_slots 不能为空")
        return self

    def validate_against_evidence_ids(self, evidence_ids: set[str]) -> None:
        """Validate the supporting ID subset against a concrete Grader input."""
        unknown = set(self.supporting_evidence_ids) - evidence_ids
        if unknown:
            raise ValueError(f"supporting evidence 不属于 Grader 输入：{sorted(unknown)}")


class GradeRecord(V2Model):
    id: str = Field(min_length=1)
    query_revision_id: str = Field(min_length=1)
    input_attempt_ids: list[str] = Field(default_factory=list)
    input_evidence_ids: list[str] = Field(default_factory=list)
    grade: EvidenceGrade

    _validate_ids = field_validator("id", "query_revision_id")(_nonempty)

    @field_validator("input_attempt_ids", "input_evidence_ids")
    @classmethod
    def validate_input_ids(cls, value: list[str]) -> list[str]:
        if any(not item.strip() for item in value):
            raise ValueError("input IDs 不能为空")
        if len(value) != len(set(value)):
            raise ValueError("input IDs 不得重复")
        return value

    @model_validator(mode="after")
    def validate_grade_subset(self) -> "GradeRecord":
        self.grade.validate_against_evidence_ids(set(self.input_evidence_ids))
        return self


class RoutingDecision(V2Model):
    id: str = Field(min_length=1)
    grade_record_id: str | None = None
    route: Route
    recovery_strategy: RecoveryStrategy | None = None
    reason: str = Field(min_length=1)

    _validate_text = field_validator("id", "reason")(_nonempty)

    @model_validator(mode="after")
    def validate_strategy_contract(self) -> "RoutingDecision":
        if self.route == "recover" and self.recovery_strategy is None:
            raise ValueError("recover route 必须声明 recovery_strategy")
        if self.route != "recover" and self.recovery_strategy is not None:
            raise ValueError("只有 recover route 可以声明 recovery_strategy")
        return self


class GroundedFinding(V2Model):
    task_id: str = Field(min_length=1)
    text: str = Field(min_length=1)
    evidence_ids: list[str] = Field(min_length=1, max_length=3)

    _validate_text = field_validator("task_id", "text")(_nonempty)

    @field_validator("evidence_ids")
    @classmethod
    def validate_evidence_ids(cls, value: list[str]) -> list[str]:
        if any(not item.strip() for item in value):
            raise ValueError("Finding evidence ID 不能为空")
        if len(value) != len(set(value)):
            raise ValueError("Finding evidence ID 不得重复")
        return value


class AnswerLimitation(V2Model):
    task_id: str = Field(min_length=1)
    kind: AnswerLimitationKind
    reason: str = Field(min_length=1)

    _validate_text = field_validator("task_id", "reason")(_nonempty)


class SynthesizedAnswer(V2Model):
    answer: str = Field(min_length=1)
    citation_evidence_ids: list[str] = Field(default_factory=list)
    limitations: list[AnswerLimitation] = Field(default_factory=list)

    _validate_answer = field_validator("answer")(_nonempty)

    @field_validator("citation_evidence_ids")
    @classmethod
    def validate_citations(cls, value: list[str]) -> list[str]:
        if any(not item.strip() for item in value):
            raise ValueError("citation evidence ID 不能为空")
        if len(value) != len(set(value)):
            raise ValueError("citation evidence ID 不得重复")
        return value


class RetrievalTask(V2Model):
    id: str = Field(min_length=1)
    ordinal: int = Field(gt=0)
    query: str = Field(min_length=1)
    intent: str = Field(min_length=1)
    capability: TaskCapability
    required: bool = True
    query_revisions: list[QueryRevision] = Field(default_factory=list)
    grade_records: list[GradeRecord] = Field(default_factory=list)
    routing_decisions: list[RoutingDecision] = Field(default_factory=list)
    grounded_finding: GroundedFinding | None = None
    execution_status: TaskExecutionStatus = "pending"
    answer_outcome: TaskAnswerOutcome | None = None
    terminal_reason: str | None = None
    error: ExecutionError | None = None

    _validate_text = field_validator("id", "query", "intent")(_nonempty)

    @field_validator("required")
    @classmethod
    def validate_required(cls, value: bool) -> bool:
        if value is not True:
            raise ValueError("V2 不支持 optional task，required 必须为 true")
        return value

    @model_validator(mode="after")
    def validate_status_outcome(self) -> "RetrievalTask":
        if self.execution_status in {"failed", "waiting_user"} and self.answer_outcome is not None:
            raise ValueError("failed/waiting_user task 的 answer_outcome 必须为 null")
        if self.execution_status == "completed" and self.answer_outcome is None:
            raise ValueError("completed task 必须有 answer_outcome")
        if self.execution_status in {"pending", "running"} and self.answer_outcome is not None:
            raise ValueError("pending/running task 不能有 answer_outcome")
        if self.execution_status == "failed" and self.error is None:
            raise ValueError("failed task 必须记录 technical error")
        return self


class ScopeOption(V2Model):
    id: str = Field(min_length=1)
    label: str = Field(min_length=1)
    value: str = Field(min_length=1)
    description: str = Field(min_length=1)
    evidence_ids: list[str] = Field(default_factory=list)

    _validate_text = field_validator("id", "label", "value", "description")(_nonempty)

    @field_validator("evidence_ids")
    @classmethod
    def validate_evidence_ids(cls, value: list[str]) -> list[str]:
        if any(not item.strip() for item in value):
            raise ValueError("ScopeOption evidence ID 不能为空")
        if len(value) != len(set(value)):
            raise ValueError("ScopeOption evidence ID 不得重复")
        return value


class HITLItem(V2Model):
    id: str = Field(min_length=1)
    action: HITLAction
    affected_task_ids: list[str] = Field(min_length=1)
    question: str = Field(min_length=1)
    missing_slots: list[str] = Field(default_factory=list)
    scope_options: list[ScopeOption] = Field(default_factory=list)

    _validate_text = field_validator("id", "question")(_nonempty)

    @field_validator("affected_task_ids", "missing_slots")
    @classmethod
    def validate_text_lists(cls, value: list[str]) -> list[str]:
        if any(not item.strip() for item in value):
            raise ValueError("HITL ID/slot 不能为空")
        if len(value) != len(set(value)):
            raise ValueError("HITL ID/slot 不得重复")
        return value

    @model_validator(mode="after")
    def validate_action_payload(self) -> "HITLItem":
        if self.action == "clarify":
            if not self.missing_slots:
                raise ValueError("clarify 必须提供 missing_slots")
            if self.scope_options:
                raise ValueError("clarify 不应提供 scope_options")
        else:
            if len(self.scope_options) < 2 or len(self.scope_options) > 5:
                raise ValueError("scope_select 必须有 2 到 5 个 options")
            if self.missing_slots:
                raise ValueError("scope_select 不应提供 missing_slots")
        return self


class HITLRequest(V2Model):
    id: str = Field(min_length=1)
    request_id: str = Field(min_length=1)
    items: list[HITLItem] = Field(min_length=1)

    _validate_text = field_validator("id")(_nonempty)
    _validate_request_id = field_validator("request_id")(validate_request_id)

    @model_validator(mode="after")
    def validate_unique_items(self) -> "HITLRequest":
        ids = [item.id for item in self.items]
        if len(ids) != len(set(ids)):
            raise ValueError("HITL item ID 必须唯一")
        option_ids = [option.id for item in self.items for option in item.scope_options]
        if len(option_ids) != len(set(option_ids)):
            raise ValueError("ScopeOption ID 在 request 内必须唯一")
        return self


class HITLResponse(V2Model):
    item_id: str = Field(min_length=1)
    clarify_values: dict[str, str] | None = None
    selected_option_id: str | None = None

    _validate_item = field_validator("item_id")(_nonempty)

    @model_validator(mode="after")
    def validate_one_action(self) -> "HITLResponse":
        if (self.clarify_values is None) == (self.selected_option_id is None):
            raise ValueError("HITLResponse 必须且只能提交 clarify_values 或 selected_option_id")
        if self.clarify_values is not None and any(
            not key.strip() or not value.strip() for key, value in self.clarify_values.items()
        ):
            raise ValueError("clarify_values 的 key/value 不能为空")
        if self.selected_option_id is not None and not self.selected_option_id.strip():
            raise ValueError("selected_option_id 不能为空")
        return self


class ResumeRequest(V2Model):
    request_id: str = Field(min_length=1)
    hitl_request_id: str = Field(min_length=1)
    responses: list[HITLResponse] = Field(min_length=1)

    _validate_request_id = field_validator("request_id")(validate_request_id)
    _validate_hitl_request_id = field_validator("hitl_request_id")(_nonempty)

    @model_validator(mode="after")
    def validate_unique_responses(self) -> "ResumeRequest":
        ids = [response.item_id for response in self.responses]
        if len(ids) != len(set(ids)):
            raise ValueError("ResumeRequest response item_id 必须唯一")
        return self


class RetrievalResult(V2Model):
    evidence: list[Evidence] = Field(max_length=5)
    retrieval_degraded: bool = False
    degraded_reason: str | None = None
    latency: RetrievalLatency = Field(default_factory=RetrievalLatency)
    trace_ref: str | None = None

    @model_validator(mode="after")
    def validate_degraded_reason(self) -> "RetrievalResult":
        if self.retrieval_degraded and not self.degraded_reason:
            raise ValueError("degraded retrieval 必须记录 reason")
        if not self.retrieval_degraded and self.degraded_reason is not None:
            raise ValueError("正常 retrieval 不应有 degraded_reason")
        return self


class StageRunResult(V2Model):
    request_id: str = Field(min_length=1)
    target_stage: TargetStage
    execution_status: GlobalExecutionStatus
    answer_outcome: GlobalAnswerOutcome | None = None
    final_answer: SynthesizedAnswer | None = None
    resumable: bool = False
    pending_hitl_request: HITLRequest | None = None
    trace_ref: str | None = None
    error: ExecutionError | None = None

    _validate_request = field_validator("request_id")(validate_request_id)

    @model_validator(mode="after")
    def validate_terminal_contract(self) -> "StageRunResult":
        if self.execution_status == "failed":
            if (
                self.answer_outcome is not None
                or self.final_answer is not None
                or self.resumable
                or self.pending_hitl_request is not None
            ):
                raise ValueError(
                    "failed StageRunResult 的 outcome/final/resumable/pending_hitl "
                    "必须为空或 false"
                )
            if self.error is None:
                raise ValueError("failed StageRunResult 必须有 error")
        elif self.execution_status == "waiting_user":
            if self.answer_outcome is not None or self.final_answer is not None:
                raise ValueError("waiting_user StageRunResult 不能有 outcome/final_answer")
            if self.pending_hitl_request is None:
                raise ValueError("waiting_user StageRunResult 必须有 pending_hitl_request")
            expected_resumable = self.target_stage == "v2_3"
            if self.resumable != expected_resumable:
                raise ValueError(
                    "V2.2 waiting_user 必须 resumable=false，V2.3 必须 resumable=true"
                )
        else:
            if self.resumable or self.pending_hitl_request is not None:
                raise ValueError("非 waiting_user StageRunResult 不应有 HITL 状态")
            if self.execution_status == "running" and (
                self.answer_outcome is not None or self.final_answer is not None
            ):
                raise ValueError("running StageRunResult 不能有 outcome/final_answer")
        if self.execution_status == "completed" and self.answer_outcome is None:
            # V2.1 may complete its stage before producing a business outcome.
            if self.target_stage != "v2_1":
                raise ValueError("V2.2/V2.3 completed StageRunResult 必须有 outcome")
        if self.execution_status == "completed" and self.target_stage != "v2_1":
            if self.final_answer is None:
                raise ValueError("V2.2/V2.3 completed StageRunResult 必须有 final_answer")
        return self


def json_record(model: BaseModel) -> dict[str, Any]:
    """Serialize a domain object without pickle or runtime object support."""
    return model.model_dump(mode="json")
