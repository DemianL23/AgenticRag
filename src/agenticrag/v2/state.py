"""Serializable LangGraph state contract for V2.

The graph itself is intentionally introduced in a later module. This module
only defines the state shape and reducers that later graph builders will use.
"""

from __future__ import annotations

import operator
from typing import Annotated, TypedDict

from .schemas import (
    ComplexityDecision,
    Evidence,
    ExecutionError,
    HITLRequest,
    RetrievalTask,
    StageRunResult,
    SynthesizedAnswer,
)
from .types import GlobalAnswerOutcome, GlobalExecutionStatus, ResponseLanguage, TargetStage


def merge_tasks(
    current: dict[str, object], update: dict[str, object]
) -> dict[str, object]:
    """Merge task fan-out updates in stable SQ ordinal order."""
    merged = dict(current)
    merged.update(update)
    return dict(sorted(merged.items(), key=lambda item: _task_sort_key(item[0])))


def merge_dicts(
    current: dict[str, object], update: dict[str, object]
) -> dict[str, object]:
    """Backward-compatible name for the task-only reducer."""
    return merge_tasks(current, update)


def merge_evidence(
    current: dict[str, "Evidence"], update: dict[str, "Evidence"]
) -> dict[str, "Evidence"]:
    """Deduplicate Evidence while appending every occurrence history entry."""
    merged = dict(current)
    for evidence_id, incoming in update.items():
        if evidence_id != incoming.evidence_id or evidence_id != incoming.chunk_id:
            raise ValueError("Evidence map key 必须等于 evidence_id 和 chunk_id")
        existing = merged.get(evidence_id)
        if existing is None:
            merged[evidence_id] = incoming
            continue
        stable_fields = ("content", "doc_id", "source", "page")
        if any(getattr(existing, field) != getattr(incoming, field) for field in stable_fields):
            raise ValueError(f"Evidence {evidence_id} 的稳定内容或 metadata 不一致")
        merged[evidence_id] = existing.model_copy(
            update={"occurrences": [*existing.occurrences, *incoming.occurrences]}
        )
    return merged


def _task_sort_key(task_id: str) -> tuple[int, str]:
    if not task_id.startswith("SQ_") or not task_id[3:].isdigit():
        raise ValueError(f"非法 RetrievalTask ID：{task_id}")
    return int(task_id[3:]), task_id


class V2State(TypedDict):
    request_id: str
    target_stage: TargetStage
    original_question: str
    normalized_query: str
    response_language: ResponseLanguage

    complexity_decision: ComplexityDecision | None
    task_order: list[str]
    tasks: Annotated[dict[str, RetrievalTask], merge_tasks]
    evidence: Annotated[dict[str, Evidence], merge_evidence]

    pending_hitl_request: HITLRequest | None
    hitl_rounds: int

    execution_status: GlobalExecutionStatus
    answer_outcome: GlobalAnswerOutcome | None
    final_answer: SynthesizedAnswer | None
    error: ExecutionError | None

    state_schema_version: str
    stage_result: StageRunResult | None


class V2HistoryState(TypedDict):
    """Append-only history fields used by later graph nodes."""

    query_revisions: Annotated[list[dict[str, object]], operator.add]
    retrieval_attempts: Annotated[list[dict[str, object]], operator.add]
    grade_records: Annotated[list[dict[str, object]], operator.add]
    routing_decisions: Annotated[list[dict[str, object]], operator.add]
