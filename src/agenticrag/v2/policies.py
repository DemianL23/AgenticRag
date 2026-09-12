"""Deterministic V2 policies and cross-object validators."""

from __future__ import annotations

import unicodedata
from typing import Iterable

from .config import V2BudgetConfig
from .schemas import (
    ComplexityDecision,
    DecompositionResult,
    Evidence,
    EvidenceGrade,
    GroundedFinding,
    HITLRequest,
    QueryRevision,
    ResumeRequest,
    RetrievalTask,
    RoutingDecision,
)
from .types import (
    GlobalAnswerOutcome,
    GlobalExecutionStatus,
    TaskAnswerOutcome,
    TaskCapability,
)

SUPPORTED_CAPABILITY: TaskCapability = "retrieval_synthesis"
RECOVERY_BY_FAILURE_REASON = {
    "irrelevant_evidence": "direct_rewrite",
    "insufficient_coverage": "direct_rewrite",
    "query_mismatch": "direct_rewrite",
    "overly_specific": "step_back",
    "terminology_gap": "hyde",
}


def normalize_query(query: str) -> str:
    """Apply only semantics-preserving whitespace and Unicode normalization."""
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query 不能为空")
    return " ".join(unicodedata.normalize("NFC", query).split())


def detect_response_language(text: str) -> str:
    """Detect zh/en from the dominant CJK or Latin alphabet count."""
    normalized = normalize_query(text)
    cjk = sum("\u4e00" <= char <= "\u9fff" for char in normalized)
    latin = sum(char.isascii() and char.isalpha() for char in normalized)
    if cjk == 0 and latin == 0:
        return "zh"
    return "zh" if cjk >= latin else "en"


def capability_is_supported(capability: TaskCapability) -> bool:
    return capability == SUPPORTED_CAPABILITY


def capability_outcome(capability: TaskCapability) -> TaskAnswerOutcome | None:
    """Return the pre-retrieval business outcome for a capability."""
    return None if capability_is_supported(capability) else "unsupported"


def retrieval_attempt_available(
    revision: QueryRevision, budget: V2BudgetConfig
) -> bool:
    _validate_count(len(revision.retrieval_attempts), "retrieval attempt")
    return len(revision.retrieval_attempts) < budget.max_retrieval_attempts_per_revision


def query_revision_available(task: RetrievalTask, budget: V2BudgetConfig) -> bool:
    _validate_count(len(task.query_revisions), "query revision")
    return len(task.query_revisions) < budget.max_query_revisions


def hitl_round_available(hitl_rounds: int, budget: V2BudgetConfig) -> bool:
    _validate_count(hitl_rounds, "HITL round")
    return hitl_rounds < budget.max_hitl_rounds


def _validate_count(count: int, label: str) -> None:
    if isinstance(count, bool) or count < 0:
        raise ValueError(f"{label} count 必须是非负整数")


def validate_decomposition(
    decision: ComplexityDecision,
    result: DecompositionResult,
    budget: V2BudgetConfig,
) -> None:
    if decision.complexity == "simple":
        if decision.capability is None:
            raise ValueError("simple decision 必须声明 capability")
        if result.tasks:
            raise ValueError("simple 请求不应有 Decomposer tasks")
        return
    if decision.capability is not None:
        raise ValueError("complex decision 的 capability 必须为 null")
    if not result.decomposition_complete:
        if result.failure_reason != "decomposition_limit":
            raise ValueError("decomposition limit reason 非法")
        if result.tasks:
            raise ValueError("decomposition limit 结果不得包含 TaskDraft")
        return
    if not 2 <= len(result.tasks) <= budget.max_subqueries:
        raise ValueError("complex tasks 数量必须在 2 到 max_subqueries 之间")
    normalized = [normalize_query(task.query).casefold() for task in result.tasks]
    if len(normalized) != len(set(normalized)):
        raise ValueError("complex task query 不得重复")


def routing_decision(
    *,
    decision_id: str,
    grade_record_id: str | None,
    capability: TaskCapability,
    grade: EvidenceGrade,
    input_evidence_ids: Iterable[str],
    retrieval_budget_available: bool,
    reason_prefix: str = "deterministic policy",
) -> RoutingDecision:
    """Apply the frozen route priority; LLM output cannot override it."""
    if not capability_is_supported(capability):
        raise ValueError(
            "routing_decision 只接受 retrieval_synthesis capability；"
            "unsupported capability 必须由 Capability Policy 终止"
        )
    available = set(input_evidence_ids)
    grade.validate_against_evidence_ids(available)
    supporting_valid = bool(grade.supporting_evidence_ids) and set(
        grade.supporting_evidence_ids
    ) <= available
    if grade.ambiguity == "missing_slot":
        return RoutingDecision(
            id=decision_id,
            grade_record_id=grade_record_id,
            route="clarify",
            reason=f"{reason_prefix}: missing slot",
        )
    if grade.ambiguity == "multiple_candidates":
        return RoutingDecision(
            id=decision_id,
            grade_record_id=grade_record_id,
            route="scope_select",
            reason=f"{reason_prefix}: multiple candidates",
        )
    if (
        grade.relevance == "strong"
        and grade.answerability == "sufficient"
        and supporting_valid
    ):
        return RoutingDecision(
            id=decision_id,
            grade_record_id=grade_record_id,
            route="answer",
            reason=f"{reason_prefix}: sufficient strong evidence",
        )
    if grade.recoverability == "likely" and retrieval_budget_available:
        strategy = RECOVERY_BY_FAILURE_REASON.get(grade.failure_reason)
        if strategy is not None:
            return RoutingDecision(
                id=decision_id,
                grade_record_id=grade_record_id,
                route="recover",
                recovery_strategy=strategy,
                reason=f"{reason_prefix}: recoverable evidence gap",
            )
    return RoutingDecision(
        id=decision_id,
        grade_record_id=grade_record_id,
        route="no_knowledge",
        reason=f"{reason_prefix}: no actionable evidence",
    )


def aggregate_task_outcomes(
    tasks: Iterable[RetrievalTask],
) -> tuple[GlobalExecutionStatus, GlobalAnswerOutcome | None]:
    """Aggregate only task contracts; never infer outcome from generated text."""
    task_list = list(tasks)
    if not task_list:
        raise ValueError("至少需要一个 required task")
    required = [task for task in task_list if task.required]
    valid_findings = [task for task in required if task.grounded_finding is not None]
    if valid_findings:
        if all(task.answer_outcome == "complete" for task in required):
            return "completed", "complete"
        return "completed", "partial"
    if any(task.execution_status == "failed" for task in required):
        return "failed", None
    if any(task.answer_outcome == "unresolved" for task in required):
        return "completed", "unresolved"
    if any(task.answer_outcome == "unsupported" for task in required):
        return "completed", "unsupported"
    return "completed", "no_knowledge"


def validate_finding_provenance(
    finding: GroundedFinding,
    task: RetrievalTask,
    *,
    evidence_by_id: dict[str, Evidence] | None = None,
    max_evidence: int = 3,
) -> None:
    if finding.task_id != task.id:
        raise ValueError("Finding task_id 与 RetrievalTask 不一致")
    if not 1 <= len(finding.evidence_ids) <= max_evidence:
        raise ValueError("Finding evidence 数量超出限制")
    if not task.grade_records:
        raise ValueError("Finding 必须基于至少一条 GradeRecord")
    supporting = set(task.grade_records[-1].grade.supporting_evidence_ids)
    if not set(finding.evidence_ids) <= supporting:
        raise ValueError("Finding 引用了最后一次 Grade 不支持的 Evidence")
    if evidence_by_id is not None and not set(finding.evidence_ids) <= set(evidence_by_id):
        raise ValueError("Finding 引用了不存在的 Evidence")


def validate_hitl_request(
    request: HITLRequest,
    tasks: dict[str, RetrievalTask],
    evidence_by_task: dict[str, set[str]],
    *,
    max_scope_options: int = 5,
) -> None:
    seen_options: set[str] = set()
    for item in request.items:
        unknown_tasks = set(item.affected_task_ids) - set(tasks)
        if unknown_tasks:
            raise ValueError(f"HITL affected task 不存在：{sorted(unknown_tasks)}")
        valid_evidence = set().union(
            *(evidence_by_task.get(task_id, set()) for task_id in item.affected_task_ids)
        )
        for option in item.scope_options:
            if option.id in seen_options:
                raise ValueError("ScopeOption ID 必须唯一")
            seen_options.add(option.id)
            if not set(option.evidence_ids) <= valid_evidence:
                raise ValueError("ScopeOption 引用了相关 task 之外的 Evidence")
        if item.action == "scope_select" and len(item.scope_options) > max_scope_options:
            raise ValueError("ScopeOption 超出配置上限")


def validate_resume_request(
    resume: ResumeRequest,
    pending: HITLRequest,
) -> None:
    if resume.request_id != pending.request_id:
        raise ValueError("Resume request_id 不匹配")
    if resume.hitl_request_id != pending.id:
        raise ValueError("Resume hitl_request_id 不匹配")
    by_id = {item.id: item for item in pending.items}
    if set(by_id) != {response.item_id for response in resume.responses}:
        raise ValueError("Resume responses 必须与 HITL items 一一对应")
    for response in resume.responses:
        item = by_id[response.item_id]
        if item.action == "clarify":
            if response.clarify_values is None:
                raise ValueError("clarify 必须提交 clarify_values")
            if set(item.missing_slots) - set(response.clarify_values):
                raise ValueError("clarify_values 未覆盖全部 missing_slots")
            if response.selected_option_id is not None:
                raise ValueError("clarify 不允许提交 option")
        else:
            valid_ids = {option.id for option in item.scope_options}
            if response.selected_option_id not in valid_ids:
                raise ValueError("scope_select option ID 非法")
            if response.clarify_values is not None:
                raise ValueError("scope_select 不允许提交 clarify_values")
