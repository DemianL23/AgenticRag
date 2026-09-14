"""Module 4 Evidence Grader runtime.

The grader describes the evidence it sees.  It never chooses a route; route
selection remains the deterministic policy in :mod:`agenticrag.v2.policies`.
"""

from __future__ import annotations

import json
import re
from typing import Any

from .config import V2Config
from .planning import PlanningError, _invoke_structured, create_decision_chat_model
from .schemas import Evidence, EvidenceGrade, ExecutionError, QueryRevision, RetrievalTask


EVIDENCE_GRADER_SYSTEM_PROMPT = """你是 Agentic RAG V2 的 Evidence Grader。
你只评估给定 RetrievalTask 的当前证据状态，不决定下一步 route，也不输出 recovery strategy。

只能根据输入的 task query、task intent、当前 QueryRevision 和当前 revision 的 V1.2 Final Top-5 Evidence 判断：
- relevance: none / weak / strong
- answerability: none / partial / sufficient
- ambiguity: none / missing_slot / multiple_candidates
- recoverability: none / likely
- failure_reason: none / irrelevant_evidence / insufficient_coverage / query_mismatch / overly_specific / terminology_gap
- reason、missing_information、missing_slots、supporting_evidence_ids

严格规则：
- supporting_evidence_ids 只能引用本次输入的 Evidence ID，不得编造 ID。
- allowed_supporting_evidence_ids 是唯一合法 ID 列表；supporting_evidence_ids 必须从中逐字复制 exact ID。
  不允许改写、缩写、重建或猜测 Evidence ID。请复制 supporting_evidence_ids exactly from
  allowed_supporting_evidence_ids，不要自行构造 Evidence ID。 Copy supporting_evidence_ids exactly from
  allowed_supporting_evidence_ids. Never construct an Evidence ID yourself.
- answerability=sufficient 或 partial 时至少引用一条 supporting evidence。
- answerability=none 时 supporting_evidence_ids 必须为空。
- relevance=none 时不能为 sufficient。
- recoverability=likely 时 answerability 不能为 sufficient，且 failure_reason 不能为 none。
- failure_reason=none 时 recoverability 必须为 none。
- ambiguity=missing_slot 时 missing_slots 必须非空。

missing_slot 与 missing_information 的语义边界：
- missing_slot 是 query-side ambiguity，只能表示 TASK QUERY 本身没有提供一个回答所必需的参数值。
  判断只看 query 是否明确提供了该参数；Evidence 中碰巧出现候选值，也不能替代用户指定参数。
  典型参数包括未指定 year、region、company/entity 或无法从上下文推断的必要时间范围。
- multiple_candidates 表示 TASK QUERY 已经提供了名称、指代、概念或范围，但该表达本身可以解析成
  多个明确、有限、合理且可枚举的候选，应让用户在候选中选择。
- 不要把当前 Evidence 中缺失的事实当成 missing_slot。
- 如果 task query 已经完整明确，但 Evidence 没有包含全部所需事实，必须使用
  ambiguity=none、missing_slots=[]，并把缺失事实写入 missing_information。
- 只有 query 本身无法解释时才请求用户澄清；绝不能把 evidence deficiency 变成 clarify。
- missing_slot 后续对应 clarify；multiple_candidates 后续对应 scope_select。
- 如果补充检索有合理机会找到缺失事实，通常使用 recoverability=likely，并选择合适的
  failure_reason，例如 insufficient_coverage、query_mismatch、irrelevant_evidence、
  overly_specific 或 terminology_gap。

最小对比示例（只表达判定原则，不是待回答的真实样本）：
 A. Task：“该年度的研发费用是多少？” Evidence：“2022研发费用……；2023研发费用……”
   query 没有明确提供 year；Evidence 中有多个年份也不能替代用户指定，因此 ambiguity=missing_slot、
   missing_slots=["year"]。
 B. Task：“2023年的研发费用是多少？” Evidence：“只找到2022年研发费用。”
   query 已完整但证据不足，因此 ambiguity=none、missing_slots=[]、
   missing_information=["2023研发费用"]、recoverability=likely、
   failure_reason=insufficient_coverage；不能使用 missing_slot。
 C. Task：“公司的利润是多少？”上下文明确存在营业利润、净利润、归母净利润三个
    有限且合理的候选。此时 ambiguity=multiple_candidates、missing_slots=[]，不能使用
    missing_slot。
 D. Task：“A公司过去5年的员工流失率是多少？” Evidence：“只有2024年员工流失率为8%。”
    query 已完整，但证据缺少其他年份，因此 ambiguity=none、missing_slots=[]、
    missing_information=["过去其他年份的员工流失率"]、recoverability=likely、
    failure_reason=insufficient_coverage；不能使用 missing_slot。

不要使用任何 gold answer、expected label、评测标签或其他 task 的证据。只返回 EvidenceGrade 结构化结果。"""


class EvidenceGradingError(RuntimeError):
    """Technical failure while producing or validating an EvidenceGrade."""

    def __init__(self, *, attempts: int, cause: Exception) -> None:
        self.attempts = attempts
        self.cause = cause
        code = (
            cause.execution_error.code
            if isinstance(cause, PlanningError)
            else "grader_failed"
        )
        self.execution_error = ExecutionError(
            code=code,
            message=f"Evidence Grader structured output failed after {attempts} attempt(s)",
            stage="module4_grading",
            retryable=False,
            details=_failure_details(cause, attempts),
        )
        super().__init__(self.execution_error.message)


class EvidenceGrader:
    """Call the configured production ``grader`` Decision Model."""

    def __init__(self, config: V2Config | None = None, *, model: Any | None = None) -> None:
        self.config = config or V2Config.from_env()
        self._model = model

    def grade(
        self,
        *,
        task: RetrievalTask,
        revision: QueryRevision,
        evidence: list[Evidence],
    ) -> tuple[EvidenceGrade, int]:
        """Grade one task against its current revision's allowed Evidence set.

        The initial grade accepts exactly one attempt's Final Top-5.  A recovery
        re-grade accepts the exact union of the two attempts in the current
        revision, still without admitting evidence from another revision/task.
        """
        expected_evidence_ids = _expected_revision_evidence_ids(revision)
        max_evidence = 5 if len(revision.retrieval_attempts) == 1 else 10
        if len(evidence) > max_evidence:
            raise ValueError(
                "Evidence Grader 输入不得超过当前 revision 的 evidence 上限"
            )
        evidence_ids_in_input = [item.evidence_id for item in evidence]
        if len(evidence_ids_in_input) != len(set(evidence_ids_in_input)):
            raise ValueError("Evidence Grader 输入 Evidence ID 不得重复")
        if set(evidence_ids_in_input) != expected_evidence_ids:
            raise ValueError(
                "Evidence Grader 输入必须严格等于当前 revision 的 Final Top-5 "
                "或两次 Attempt 的 Evidence union"
            )
        evidence_ids = set(evidence_ids_in_input)
        prompt = build_evidence_grader_prompt(task=task, revision=revision, evidence=evidence)
        model = self._model or create_decision_chat_model(self.config.decision_models.grader)
        try:
            return _invoke_structured(
                model,
                EvidenceGrade,
                prompt,
                role="grader",
                retry_policy=self.config.decision_models.grader.retry_policy,
                post_validate=lambda value: value.validate_against_evidence_ids(evidence_ids),
                repair_prompt_builder=_build_grader_repair_prompt,
            )
        except PlanningError as exc:
            raise EvidenceGradingError(
                attempts=exc.attempts,
                cause=exc,
            ) from exc
        except Exception as exc:
            raise EvidenceGradingError(attempts=1, cause=exc) from exc


def build_evidence_grader_prompt(
    *, task: RetrievalTask, revision: QueryRevision, evidence: list[Evidence]
) -> str:
    """Build a bounded prompt with no diagnostics or evaluation labels."""
    evidence_ids_in_input = [item.evidence_id for item in evidence]
    evidence_payload = [
        {
            "evidence_id": item.evidence_id,
            "content": item.content,
            "doc_id": item.doc_id,
            "source": item.source,
            "page": item.page,
        }
        for item in evidence
    ]
    context = {
        "task": {"query": task.query, "intent": task.intent},
        "query_revision": {
            "id": revision.id,
            "ordinal": revision.ordinal,
            "source": revision.source,
            "query": revision.query,
        },
        "final_top5_evidence": evidence_payload,
        "allowed_supporting_evidence_ids": evidence_ids_in_input,
    }
    return f"{EVIDENCE_GRADER_SYSTEM_PROMPT}\n\n输入：\n{json.dumps(context, ensure_ascii=False, indent=2)}"


def _build_grader_repair_prompt(original_prompt: str, cause: Exception) -> str:
    """Append bounded, sanitized contract feedback for the one repair attempt."""
    return (
        f"{original_prompt}\n\n"
        "上一轮输出违反了 EvidenceGrade structured contract。请重新生成完整的 EvidenceGrade 对象，"
        "不要输出 patch、route 或 recovery strategy。保持对 task 与 evidence 的事实判断不变。\n"
        f"Failure type: {type(cause).__name__}\n"
        f"Failure: {_safe_exception_message(cause)}\n"
        "修复要求：supporting_evidence_ids 只能逐字复制输入中的 "
        "allowed_supporting_evidence_ids；answerability=none 时必须返回 []。"
    )


_MAX_DIAGNOSTIC_MESSAGE_LENGTH = 2000


def _failure_details(cause: Exception, attempts: int) -> dict[str, str]:
    """Return bounded diagnostics without serializing provider/runtime objects."""
    details = {
        "role": "grader",
        "attempts": str(attempts),
        "cause_type": type(cause).__name__,
        "cause_message": _safe_exception_message(cause),
    }
    if isinstance(cause, PlanningError):
        root_cause = cause.cause
        details.update(
            {
                "root_cause_type": type(root_cause).__name__,
                "root_cause_message": _safe_exception_message(root_cause),
            }
        )
    return details


def _safe_exception_message(cause: Exception) -> str:
    """Truncate and redact common credential forms in exception messages."""
    message = str(cause)[:_MAX_DIAGNOSTIC_MESSAGE_LENGTH]
    message = re.sub(
        r"(?i)(authorization\s*[:=]\s*(?:bearer\s+)?)[^\s,;}\]]+",
        r"\1<redacted>",
        message,
    )
    message = re.sub(
        r"(?i)(api[_-]?key\s*[:=]\s*)[^\s,;}\]]+",
        r"\1<redacted>",
        message,
    )
    message = re.sub(r"(?i)\bbearer\s+[A-Za-z0-9._~+/=-]+", "Bearer <redacted>", message)
    message = re.sub(r"(?i)\bsk-[A-Za-z0-9._-]+", "<redacted>", message)
    return message


def _expected_revision_evidence_ids(revision: QueryRevision) -> set[str]:
    """Return the only Evidence IDs legal for this revision's current grade."""
    attempts = revision.retrieval_attempts
    if not attempts:
        return set()
    if len(attempts) > 2:
        raise ValueError("Evidence Grader 不支持超过两次 RetrievalAttempt")
    if [attempt.ordinal for attempt in attempts] != list(range(1, len(attempts) + 1)):
        raise ValueError("RetrievalAttempt ordinal 必须从 1 连续到当前 attempt")
    attempt_ids = [attempt.id for attempt in attempts]
    if len(attempt_ids) != len(set(attempt_ids)):
        raise ValueError("RetrievalAttempt ID 不得重复")
    if any(len(attempt.evidence_ids) > 5 for attempt in attempts):
        raise ValueError("每次 RetrievalAttempt 最多只能有 5 个 Final Evidence")
    return {
        evidence_id
        for attempt in attempts
        for evidence_id in attempt.evidence_ids
    }
