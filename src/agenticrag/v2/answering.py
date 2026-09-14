"""Grounded Finding, answer synthesis, and final-answer contracts for V2.2."""

from __future__ import annotations

import json
import os
from collections.abc import Iterable, Sequence
from typing import Any

from pydantic import Field, field_validator

from .config import AnswerModelConfig, AnswerRole, V2Config
from .grading import _safe_exception_message
from .ids import hitl_item_id, hitl_request_id, scope_option_id
from .planning import (
    PlanningError,
    StructuredOutputContractError,
    _invoke_structured,
)
from .policies import validate_finding_provenance, validate_hitl_request
from .schemas import (
    AnswerLimitation,
    Evidence,
    EvidenceGrade,
    ExecutionError,
    GroundedFinding,
    HITLItem,
    HITLRequest,
    RetrievalTask,
    ScopeOption,
    SynthesizedAnswer,
    V2Model,
)
from .types import AnswerLimitationKind, GlobalAnswerOutcome, ResponseLanguage


class FindingPayload(V2Model):
    """The only fields the Finding Answer Model may generate."""

    text: str = Field(min_length=1)
    evidence_ids: list[str] = Field(min_length=1, max_length=3)

    @field_validator("text")
    @classmethod
    def validate_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Finding text 不能为空")
        return value.strip()


class SynthesisPayload(V2Model):
    """The only fields the Final Synthesizer may generate."""

    answer: str = Field(min_length=1)
    citation_evidence_ids: list[str] = Field(default_factory=list)

    @field_validator("answer")
    @classmethod
    def validate_answer(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("synthesized answer 不能为空")
        return value.strip()

    @field_validator("citation_evidence_ids")
    @classmethod
    def validate_citation_ids(cls, value: list[str]) -> list[str]:
        if any(not item.strip() for item in value):
            raise ValueError("citation evidence ID 不能为空")
        if len(value) != len(set(value)):
            raise ValueError("citation evidence ID 不得重复")
        return value


class AnswerGenerationError(RuntimeError):
    """Bounded Answer Model failure with safe diagnostics."""

    def __init__(self, *, code: str, role: AnswerRole, attempts: int, cause: Exception) -> None:
        self.code = code
        self.role = role
        self.attempts = attempts
        self.cause = cause
        details = {
            "role": role,
            "attempts": str(attempts),
            "cause_type": type(cause).__name__,
            "cause_message": _safe_exception_message(cause),
        }
        if isinstance(cause, PlanningError):
            details["root_cause_type"] = type(cause.cause).__name__
            details["root_cause_message"] = _safe_exception_message(cause.cause)
        self.execution_error = ExecutionError(
            code=code,
            message=f"{role} Answer Model failed after {attempts} attempt(s)",
            stage="module6_answering",
            retryable=False,
            details=details,
        )
        super().__init__(self.execution_error.message)


class FindingGenerator:
    """Generate and validate one task-scoped GroundedFinding."""

    def __init__(self, config: V2Config | None = None, *, model: Any | None = None) -> None:
        self.config = config or V2Config.from_env()
        self._model = model

    def generate(
        self,
        *,
        task: RetrievalTask,
        evidence_by_id: dict[str, Evidence],
        response_language: ResponseLanguage,
        role: AnswerRole = "finding",
    ) -> tuple[GroundedFinding, int]:
        if not task.routing_decisions or task.routing_decisions[-1].route != "answer":
            raise ValueError("只有 answer route 才能生成 GroundedFinding")
        if not task.grade_records:
            raise ValueError("Finding 必须有最新 GradeRecord")
        grade_record = task.grade_records[-1]
        supporting_ids = list(grade_record.grade.supporting_evidence_ids)
        if not supporting_ids or any(item not in evidence_by_id for item in supporting_ids):
            raise ValueError("Finding 的 supporting Evidence 不完整")
        evidence = [evidence_by_id[item] for item in supporting_ids]
        prompt = build_finding_prompt(
            task=task,
            grade=grade_record.grade,
            evidence=evidence,
            response_language=response_language,
            role=role,
        )
        answer_config = _answer_model_config(self.config, role)
        model = self._model or create_answer_chat_model(answer_config)

        def validate_payload(payload: FindingPayload) -> None:
            finding = GroundedFinding(
                task_id=task.id,
                text=payload.text,
                evidence_ids=payload.evidence_ids,
            )
            validate_finding_provenance(
                finding,
                task,
                evidence_by_id=evidence_by_id,
                max_evidence=self.config.budgets.max_evidence_per_finding,
            )

        try:
            payload, attempts = _invoke_structured(
                model,
                FindingPayload,
                prompt,
                role=role,
                retry_policy=answer_config.retry_policy,
                post_validate=validate_payload,
                repair_prompt_builder=_build_answer_repair_prompt,
            )
        except PlanningError as exc:
            raise AnswerGenerationError(
                code=_answer_error_code(exc.cause),
                role=role,
                attempts=exc.attempts,
                cause=exc,
            ) from exc
        except Exception as exc:
            raise AnswerGenerationError(
                code=_answer_error_code(exc), role=role, attempts=1, cause=exc
            ) from exc
        return (
            GroundedFinding(
                task_id=task.id,
                text=payload.text,
                evidence_ids=payload.evidence_ids,
            ),
            attempts,
        )


class SynthesisGenerator:
    """Generate a final answer from already verified Findings only."""

    def __init__(self, config: V2Config | None = None, *, model: Any | None = None) -> None:
        self.config = config or V2Config.from_env()
        self._model = model

    def generate(
        self,
        *,
        original_question: str,
        tasks: Sequence[RetrievalTask],
        findings: Sequence[GroundedFinding],
        evidence_by_id: dict[str, Evidence],
        limitations: list[AnswerLimitation],
        response_language: ResponseLanguage,
    ) -> tuple[SynthesizedAnswer, int]:
        allowed_ids = sorted({item for finding in findings for item in finding.evidence_ids})
        if not findings or any(item not in evidence_by_id for item in allowed_ids):
            raise ValueError("Synthesizer 必须只接收合法 Finding Evidence")
        prompt = build_synthesis_prompt(
            original_question=original_question,
            tasks=tasks,
            findings=findings,
            evidence_by_id=evidence_by_id,
            allowed_ids=allowed_ids,
            response_language=response_language,
        )
        model = self._model or create_answer_chat_model(self.config.answer_models.synthesis)

        def validate_payload(payload: SynthesisPayload) -> None:
            answer = SynthesizedAnswer(
                answer=payload.answer,
                citation_evidence_ids=payload.citation_evidence_ids,
                limitations=limitations,
            )
            validate_synthesized_answer(
                answer,
                tasks=tasks,
                findings=findings,
                evidence_by_id=evidence_by_id,
                global_outcome="partial" if limitations else "complete",
            )

        try:
            payload, attempts = _invoke_structured(
                model,
                SynthesisPayload,
                prompt,
                role="synthesis",
                retry_policy=self.config.answer_models.synthesis.retry_policy,
                post_validate=validate_payload,
                repair_prompt_builder=_build_answer_repair_prompt,
            )
        except PlanningError as exc:
            raise AnswerGenerationError(
                code=_answer_error_code(exc.cause, default="synthesis_failed"),
                role="synthesis",
                attempts=exc.attempts,
                cause=exc,
            ) from exc
        except Exception as exc:
            raise AnswerGenerationError(
                code=_answer_error_code(exc, default="synthesis_failed"),
                role="synthesis",
                attempts=1,
                cause=exc,
            ) from exc
        return (
            SynthesizedAnswer(
                answer=payload.answer,
                citation_evidence_ids=payload.citation_evidence_ids,
                limitations=limitations,
            ),
            attempts,
        )


def build_finding_prompt(
    *,
    task: RetrievalTask,
    grade: EvidenceGrade,
    evidence: Sequence[Evidence],
    response_language: ResponseLanguage,
    role: AnswerRole,
) -> str:
    payload = {
        "task": {"task_id": task.id, "query": task.query, "intent": task.intent},
        "grade": {
            "relevance": grade.relevance,
            "answerability": grade.answerability,
            "ambiguity": grade.ambiguity,
            "supporting_evidence_ids": grade.supporting_evidence_ids,
        },
        "allowed_evidence": [_evidence_record(item) for item in evidence],
        "response_language": response_language,
    }
    return (
        f"你是 Agentic RAG V2 的 {role} Answer Model。\n"
        "只能根据提供的 Evidence 生成该 task 的有据中间结论；不能使用外部知识。\n"
        "只返回 text 和 evidence_ids。evidence_ids 必须逐字复制 allowed_evidence 中的 evidence_id，"
        "不能生成 route、outcome、status 或新的事实来源。\n"
        "请使用指定 response_language。\n输入：\n"
        + json.dumps(payload, ensure_ascii=False, indent=2)
    )


def build_synthesis_prompt(
    *,
    original_question: str,
    tasks: Sequence[RetrievalTask],
    findings: Sequence[GroundedFinding],
    evidence_by_id: dict[str, Evidence],
    allowed_ids: Sequence[str],
    response_language: ResponseLanguage,
) -> str:
    finding_payload = []
    for finding in findings:
        finding_payload.append(
            {
                "task_id": finding.task_id,
                "text": finding.text,
                "evidence_ids": finding.evidence_ids,
                "evidence": [_evidence_record(evidence_by_id[item]) for item in finding.evidence_ids],
            }
        )
    task_payload = [
        {
            "task_id": task.id,
            "query": task.query,
            "intent": task.intent,
            "answer_outcome": task.answer_outcome,
            "terminal_reason": task.terminal_reason,
            "technical_failure": task.error.code if task.error else None,
        }
        for task in tasks
    ]
    payload = {
        "original_question": original_question,
        "tasks": task_payload,
        "verified_findings": finding_payload,
        "allowed_citation_evidence_ids": list(allowed_ids),
        "response_language": response_language,
    }
    return (
        "你是 Agentic RAG V2 的 synthesis Answer Model。\n"
        "只能根据 verified_findings 及其引用的 Evidence 回答 original_question。"
        "任务 outcome 和 limitations 已由程序确定，不要重新判断；不能加入未被 Findings 支持的新事实。\n"
        "只返回 answer 和 citation_evidence_ids。citation IDs 必须逐字复制 allowed_citation_evidence_ids，"
        "不能输出 route、status、outcome 或 limitations。\n"
        "请使用指定 response_language。\n输入：\n"
        + json.dumps(payload, ensure_ascii=False, indent=2)
    )


def validate_synthesized_answer(
    answer: SynthesizedAnswer,
    *,
    tasks: Sequence[RetrievalTask],
    findings: Sequence[GroundedFinding],
    evidence_by_id: dict[str, Evidence],
    global_outcome: GlobalAnswerOutcome,
) -> None:
    allowed_ids = {item for finding in findings for item in finding.evidence_ids}
    citations = set(answer.citation_evidence_ids)
    if not citations <= allowed_ids:
        raise ValueError("Final citation 引用了未被 GroundedFinding 使用的 Evidence")
    if not citations <= set(evidence_by_id):
        raise ValueError("Final citation 引用了不存在的 Evidence")
    tasks_by_id = {task.id: task for task in tasks}
    for finding in findings:
        task = tasks_by_id.get(finding.task_id)
        if task is None:
            raise ValueError("GroundedFinding 引用了不存在的 Task")
        validate_finding_provenance(
            finding,
            task,
            evidence_by_id=evidence_by_id,
            max_evidence=3,
        )
    expected = build_answer_limitations(tasks)
    if global_outcome == "complete" and answer.limitations:
        raise ValueError("complete answer 不应有 limitations")
    if global_outcome == "partial":
        if _limitation_kinds(answer.limitations) != _limitation_kinds(expected):
            raise ValueError("partial answer 的 limitations 必须覆盖所有未 complete task")
    if global_outcome in {"no_knowledge", "unsupported", "unresolved"}:
        if _limitation_kinds(answer.limitations) != _limitation_kinds(expected):
            raise ValueError("terminal answer 的 limitations 必须覆盖所有未 complete task")


def _limitation_kinds(
    limitations: Sequence[AnswerLimitation],
) -> dict[str, AnswerLimitationKind]:
    return {item.task_id: item.kind for item in limitations}


def build_answer_limitations(tasks: Iterable[RetrievalTask]) -> list[AnswerLimitation]:
    limitations: list[AnswerLimitation] = []
    for task in sorted((item for item in tasks if item.required), key=lambda item: (item.ordinal, item.id)):
        kind: AnswerLimitationKind | None = None
        if task.execution_status == "failed":
            kind = "technical_failure"
        elif task.execution_status == "waiting_user":
            kind = "unresolved"
        elif task.answer_outcome in {"no_knowledge", "unsupported", "unresolved"}:
            kind = task.answer_outcome
        if kind is None:
            continue
        reason = task.terminal_reason or (task.error.message if task.error else kind)
        limitations.append(AnswerLimitation(task_id=task.id, kind=kind, reason=reason[:500]))
    return limitations


def deterministic_terminal_answer(
    outcome: GlobalAnswerOutcome, *, response_language: ResponseLanguage
) -> str:
    messages = {
        "zh": {
            "no_knowledge": "根据当前检索到的资料无法确定。",
            "unsupported": "当前 V2 不支持该任务所需的能力。",
            "unresolved": "需要补充或选择信息后才能继续。",
        },
        "en": {
            "no_knowledge": "The answer cannot be determined from the retrieved evidence.",
            "unsupported": "The requested capability is not supported by V2.",
            "unresolved": "More information or a scope selection is required to continue.",
        },
    }
    if outcome not in {"no_knowledge", "unsupported", "unresolved"}:
        raise ValueError("只有业务 terminal outcome 才能生成 deterministic response")
    return messages[response_language][outcome]


def build_hitl_request(
    *,
    request_id: str,
    tasks: dict[str, RetrievalTask],
    evidence_by_id: dict[str, Evidence],
    max_scope_options: int,
) -> HITLRequest:
    items: list[HITLItem] = []
    option_ordinal = 1
    evidence_by_task: dict[str, set[str]] = {}
    for task in tasks.values():
        evidence_by_task[task.id] = {
            evidence_id
            for revision in task.query_revisions
            for attempt in revision.retrieval_attempts
            for evidence_id in attempt.evidence_ids
            if evidence_id in evidence_by_id
        }
    waiting_tasks = sorted(
        (task for task in tasks.values() if task.routing_decisions and task.routing_decisions[-1].route in {"clarify", "scope_select"}),
        key=lambda item: (item.ordinal, item.id),
    )
    for item_ordinal, task in enumerate(waiting_tasks, start=1):
        route = task.routing_decisions[-1].route
        grade = task.grade_records[-1].grade
        if route == "clarify":
            item = HITLItem(
                id=hitl_item_id(item_ordinal),
                action="clarify",
                affected_task_ids=[task.id],
                question=f"请补充：{', '.join(grade.missing_slots)}",
                missing_slots=grade.missing_slots,
                scope_options=[],
            )
        else:
            candidate_ids = sorted(evidence_by_task.get(task.id, set()))[:max_scope_options]
            option_count = max(2, min(max_scope_options, len(candidate_ids)))
            options: list[ScopeOption] = []
            for index in range(option_count):
                candidate_id = candidate_ids[index] if index < len(candidate_ids) else None
                options.append(
                    ScopeOption(
                        id=scope_option_id(option_ordinal),
                        label=f"候选范围 {index + 1}",
                        value=candidate_id or f"candidate_{index + 1}",
                        description=(
                            f"基于 Evidence {candidate_id} 的候选范围"
                            if candidate_id
                            else "候选范围需要用户确认"
                        ),
                        evidence_ids=[candidate_id] if candidate_id else [],
                    )
                )
                option_ordinal += 1
            item = HITLItem(
                id=hitl_item_id(item_ordinal),
                action="scope_select",
                affected_task_ids=[task.id],
                question="请选择需要回答的范围。",
                missing_slots=[],
                scope_options=options,
            )
        items.append(item)
    request = HITLRequest(id=hitl_request_id(1), request_id=request_id, items=items)
    validate_hitl_request(request, tasks, evidence_by_task, max_scope_options=max_scope_options)
    return request


def _evidence_record(item: Evidence) -> dict[str, object]:
    return {
        "evidence_id": item.evidence_id,
        "content": item.content,
        "doc_id": item.doc_id,
        "source": item.source,
        "page": item.page,
    }


def _build_answer_repair_prompt(original_prompt: str, cause: Exception) -> str:
    return (
        f"{original_prompt}\n\n"
        "上一轮结构化输出未通过 contract。请返回完整对象，不要输出 patch、route、outcome 或 status。\n"
        f"Failure type: {type(cause).__name__}\n"
        f"Failure: {_safe_exception_message(cause)}\n"
        "只能逐字复制输入中允许的 Evidence ID，不能删除 citation 错误后继续。"
    )


def _answer_error_code(cause: Exception, default: str = "finding_generation_failed") -> str:
    marker = str(cause).lower()
    if any(token in marker for token in ("evidence", "citation", "provenance")):
        return "citation_validation_failed"
    return default


def _answer_model_config(config: V2Config, role: AnswerRole) -> AnswerModelConfig:
    return {
        "simple_answer": config.answer_models.simple_answer,
        "finding": config.answer_models.finding,
        "synthesis": config.answer_models.synthesis,
    }[role]


def create_answer_chat_model(config: AnswerModelConfig) -> Any:
    """Create an OpenAI-compatible Answer Model with role-specific settings."""
    from agenticrag.generation.config import GenerationConfig

    try:
        from langchain_openai import ChatOpenAI
    except ImportError as exc:  # pragma: no cover - optional runtime dependency
        raise RuntimeError("Answer Model 需要安装 generation optional extra") from exc
    defaults = GenerationConfig.from_env()
    role_prefix = f"V2_{config.role.upper()}"
    base_url = os.getenv(f"{role_prefix}_BASE_URL") or os.getenv("V2_ANSWER_BASE_URL") or defaults.base_url
    api_key = os.getenv("V2_ANSWER_API_KEY") or defaults.api_key
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
