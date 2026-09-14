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
