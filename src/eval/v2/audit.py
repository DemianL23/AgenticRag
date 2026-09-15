"""Shared eval-only auditors for V2 task history and answer provenance."""

from __future__ import annotations

from typing import Any

from pydantic import Field

from agenticrag.v2.config import V2Config
from agenticrag.v2.schemas import V2Model


class AuditCounts(V2Model):
    schema_invariant_violation_count: int = Field(default=0, ge=0)
    provenance_violation_count: int = Field(default=0, ge=0)
    citation_violation_count: int = Field(default=0, ge=0)
    budget_violation_count: int = Field(default=0, ge=0)
    retrieval_degraded_queries_count: int = Field(default=0, ge=0)

    def __add__(self, other: "AuditCounts") -> "AuditCounts":
        return AuditCounts(
            **{
                field: getattr(self, field) + getattr(other, field)
                for field in type(self).model_fields
            }
        )


def audit_v2_result(
    tasks: list[Any],
    evidence: dict[str, Any],
    stage: Any,
    *,
    config: V2Config,
    state: dict[str, Any] | None = None,
) -> AuditCounts:
    """Audit frozen V2 schema, budget, provenance, and citation contracts."""

    schema = provenance = citation = budget = degraded = 0
    finding_ids: set[str] = set()
    evidence_ids = set(evidence)
    if len(tasks) > config.budgets.max_subqueries:
        budget += 1
    for task in tasks:
        execution_status = getattr(task, "execution_status", None)
        answer_outcome = getattr(task, "answer_outcome", None)
        revisions = list(getattr(task, "query_revisions", []))
        grades = list(getattr(task, "grade_records", []))
        finding = getattr(task, "grounded_finding", None)
        task_id = getattr(task, "id", None)
        if execution_status == "failed" and answer_outcome is not None:
            schema += 1
        if len(revisions) > config.budgets.max_query_revisions or any(
            len(revision.retrieval_attempts)
            > config.budgets.max_retrieval_attempts_per_revision
            for revision in revisions
        ):
            budget += 1
        if any(
            revision.ordinal > config.budgets.max_query_revisions
            or any(
                attempt.ordinal
                > config.budgets.max_retrieval_attempts_per_revision
                for attempt in revision.retrieval_attempts
            )
            for revision in revisions
        ):
            budget += 1
        for revision in revisions:
            degraded += sum(
                int(attempt.retrieval_degraded)
                for attempt in revision.retrieval_attempts
            )
        if finding is None:
            continue
        finding_ids.update(finding.evidence_ids)
        support = set(grades[-1].grade.supporting_evidence_ids) if grades else set()
        latest_revision = revisions[-1].id if revisions else None
        latest_grade_revision = grades[-1].query_revision_id if grades else None
        latest_grade = grades[-1] if grades else None
        current_attempts = revisions[-1].retrieval_attempts if revisions else []
        current_attempt_ids = {attempt.id for attempt in current_attempts}
        attempt_evidence = {
            attempt.id: set(attempt.evidence_ids) for attempt in current_attempts
        }
        grade_input_valid = bool(
            latest_grade
            and set(latest_grade.input_attempt_ids) <= current_attempt_ids
            and set(finding.evidence_ids) <= set(latest_grade.input_evidence_ids)
        )
        occurrence_valid = all(
            any(
                occurrence.task_id == task_id
                and occurrence.query_revision_id == latest_revision
                and occurrence.retrieval_attempt_id in current_attempt_ids
                and evidence_id
                in attempt_evidence.get(occurrence.retrieval_attempt_id, set())
                for occurrence in evidence[evidence_id].occurrences
            )
            and evidence[evidence_id].evidence_id == evidence[evidence_id].chunk_id
            for evidence_id in finding.evidence_ids
            if evidence_id in evidence
        )
        if (
            finding.task_id != task_id
            or not finding.evidence_ids
            or len(finding.evidence_ids) > config.budgets.max_evidence_per_finding
            or not set(finding.evidence_ids) <= evidence_ids
            or not set(finding.evidence_ids) <= support
            or latest_revision != latest_grade_revision
            or not grade_input_valid
            or not occurrence_valid
        ):
            provenance += 1
    if state is not None and state.get("hitl_rounds", 0) > config.budgets.max_hitl_rounds:
        budget += 1
    pending = getattr(stage, "pending_hitl_request", None)
    if pending is not None and any(
        len(item.scope_options) > config.budgets.max_scope_options
        for item in pending.items
    ):
        budget += 1
    final = getattr(stage, "final_answer", None)
    if final is not None and not set(final.citation_evidence_ids) <= finding_ids:
        citation += 1
    return AuditCounts(
        schema_invariant_violation_count=schema,
        provenance_violation_count=provenance,
        citation_violation_count=citation,
        budget_violation_count=budget,
        retrieval_degraded_queries_count=degraded,
    )
