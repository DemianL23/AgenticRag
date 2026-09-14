"""Module 4 Evidence Grader runtime.

The grader describes the evidence it sees.  It never chooses a route; route
selection remains the deterministic policy in :mod:`agenticrag.v2.policies`.
"""

from __future__ import annotations

import json
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
- answerability=sufficient 或 partial 时至少引用一条 supporting evidence。
- answerability=none 时 supporting_evidence_ids 必须为空。
- relevance=none 时不能为 sufficient。
- recoverability=likely 时 answerability 不能为 sufficient，且 failure_reason 不能为 none。
- failure_reason=none 时 recoverability 必须为 none。
- ambiguity=missing_slot 时 missing_slots 必须非空。

missing_slot 与 missing_information 的语义边界：
- missing_slot 是 query-side ambiguity，只能表示 TASK QUERY 本身缺少一个必须由用户补充的参数，
  并且仅凭当前 query 无法形成有限、明确、可枚举的候选集合。典型参数包括未指定
  year、region、company/entity 或无法从上下文推断的必要时间范围。
- multiple_candidates 表示 TASK QUERY 本身已有多个明确、有限、合理且可枚举的解释，
  应让用户在候选中选择，而不是请求用户补充一个无法界定的参数。
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
   query 没有指定年度，因此 ambiguity=missing_slot、missing_slots=["year"]。
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
            details={"attempts": str(attempts), "exception_type": type(cause).__name__},
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
        """Grade exactly one task against exactly its current Final Top-5."""
        if len(evidence) > 5:
            raise ValueError("Evidence Grader 输入不得超过当前 Final Top-5")
        evidence_ids_in_input = [item.evidence_id for item in evidence]
        if len(evidence_ids_in_input) != len(set(evidence_ids_in_input)):
            raise ValueError("Evidence Grader 输入 Evidence ID 不得重复")
        if revision.retrieval_attempts:
            current_attempt_ids = set(revision.retrieval_attempts[-1].evidence_ids)
            if set(evidence_ids_in_input) != current_attempt_ids:
                raise ValueError("Evidence Grader 只能接收当前 revision 的 Final Top-5")
        elif evidence:
            raise ValueError("没有 RetrievalAttempt 时不能提供 Evidence")
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
    }
    return f"{EVIDENCE_GRADER_SYSTEM_PROMPT}\n\n输入：\n{json.dumps(context, ensure_ascii=False, indent=2)}"
