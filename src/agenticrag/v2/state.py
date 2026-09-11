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


def merge_dicts(
    current: dict[str, object], update: dict[str, object]
) -> dict[str, object]:
    """Reducer for deterministic key-based fan-out merges."""
    merged = dict(current)
    merged.update(update)
    return merged


class V2State(TypedDict):
    request_id: str
    target_stage: TargetStage
    original_question: str
    normalized_query: str
    response_language: ResponseLanguage

    complexity_decision: ComplexityDecision | None
    task_order: list[str]
    tasks: Annotated[dict[str, RetrievalTask], merge_dicts]
    evidence: Annotated[dict[str, Evidence], merge_dicts]

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
