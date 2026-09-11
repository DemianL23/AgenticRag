from __future__ import annotations

from uuid import UUID

import pytest

from agenticrag.v2.config import V2BudgetConfig, V2Config
from agenticrag.v2.ids import (
    grade_record_id,
    new_request_id,
    query_revision_id,
    retrieval_attempt_id,
    task_id,
)
from agenticrag.v2.policies import (
    aggregate_task_outcomes,
    detect_response_language,
    normalize_query,
    routing_decision,
    validate_decomposition,
    validate_finding_provenance,
    validate_hitl_request,
    validate_resume_request,
)
from agenticrag.v2.schemas import (
    ComplexityDecision,
    DecompositionResult,
    EvidenceGrade,
    ExecutionError,
    GradeRecord,
    GroundedFinding,
    HITLItem,
    HITLRequest,
    HITLResponse,
    QueryRevision,
    RetrievalTask,
    ResumeRequest,
    ScopeOption,
    StageRunResult,
    SynthesizedAnswer,
    TaskDraft,
    json_record,
)
from agenticrag.v2.serialization import serialize_state


def _task(
    *,
    identifier: str = "SQ_001",
    status: str = "pending",
    outcome: str | None = None,
    finding: GroundedFinding | None = None,
    capability: str = "retrieval_synthesis",
    error: ExecutionError | None = None,
) -> RetrievalTask:
    return RetrievalTask(
        id=identifier,
        ordinal=1,
        query="query",
        intent="intent",
        capability=capability,
        execution_status=status,
        answer_outcome=outcome,
        grounded_finding=finding,
        error=error,
    )


def _grade(**overrides: object) -> EvidenceGrade:
    values: dict[str, object] = {
        "relevance": "strong",
        "answerability": "sufficient",
        "ambiguity": "none",
        "recoverability": "none",
        "failure_reason": "none",
        "reason": "supported",
        "supporting_evidence_ids": ["chunk-1"],
    }
    values.update(overrides)
    return EvidenceGrade(**values)


def test_budget_and_role_config_are_explicit() -> None:
    config = V2Config()

    assert config.budgets == V2BudgetConfig()
    assert config.budgets.max_subqueries == 4
    assert config.budgets.max_concurrent_subqueries == 1
    assert config.decision_models.router.role == "router"
    assert config.decision_models.grader.role == "grader"
    assert config.answer_models.synthesis.role == "synthesis"
    assert config.decision_models.router is not config.decision_models.grader


def test_stable_ids_are_parent_and_ordinal_based() -> None:
    request = new_request_id()

    assert UUID(request).version == 4
    assert task_id(1) == "SQ_001"
    assert query_revision_id("SQ_001", 1) == "QR_SQ001_001"
    assert retrieval_attempt_id("SQ_001", "QR_SQ001_001", 1) == "ATT_SQ001_QR001_001"
    assert grade_record_id("SQ_001", "QR_SQ001_001", 1) == "GR_SQ001_QR001_001"
    with pytest.raises(ValueError):
        task_id(0)


def test_query_normalization_and_language_detection_are_deterministic() -> None:
    assert normalize_query("  公司\n\t  业务  ") == "公司 业务"
    assert detect_response_language("公司 2023 收入") == "zh"
    assert detect_response_language("What is revenue?") == "en"
    assert detect_response_language("12345") == "zh"


def test_complexity_and_decomposition_contracts() -> None:
    budget = V2BudgetConfig()
    simple = ComplexityDecision(
        complexity="simple", capability="retrieval_synthesis", reason="fact lookup"
    )
    valid = DecompositionResult(
        tasks=[
            TaskDraft(query="revenue", intent="revenue", capability="retrieval_synthesis"),
            TaskDraft(query="profit", intent="profit", capability="retrieval_synthesis"),
        ],
        decomposition_complete=True,
    )
    validate_decomposition(
        ComplexityDecision(complexity="complex", capability=None, reason="multi-part"),
        valid,
        budget,
    )
    with pytest.raises(ValueError):
        ComplexityDecision(complexity="simple", capability=None, reason="missing")
    with pytest.raises(ValueError):
        validate_decomposition(simple, valid, budget)
    with pytest.raises(ValueError):
        DecompositionResult(
            tasks=[
                TaskDraft(query="same", intent="a", capability="retrieval_synthesis"),
                TaskDraft(query="same", intent="b", capability="retrieval_synthesis"),
            ],
            decomposition_complete=True,
        )


def test_task_status_and_evidence_grade_invariants() -> None:
    with pytest.raises(ValueError):
        _task(status="failed", error=None)
    with pytest.raises(ValueError):
        _grade(answerability="none")
    with pytest.raises(ValueError):
        _grade(recoverability="likely")
    with pytest.raises(ValueError):
        _grade(ambiguity="missing_slot")
    with pytest.raises(ValueError):
        _grade(relevance="none")

    finding = GroundedFinding(task_id="SQ_001", text="known", evidence_ids=["chunk-1"])
    task = RetrievalTask(
        **{
            **_task(
                status="completed",
                outcome="complete",
                finding=finding,
            ).model_dump(),
            "grade_records": [
                GradeRecord(
                    id="GR_SQ001_QR001_001",
                    query_revision_id="QR_SQ001_001",
                    input_evidence_ids=["chunk-1"],
                    grade=_grade(),
                )
            ],
        }
    )
    validate_finding_provenance(finding, task)


def test_grade_record_rejects_supporting_id_outside_grader_input() -> None:
    from agenticrag.v2.schemas import GradeRecord

    with pytest.raises(ValueError):
        GradeRecord(
            id="GR_SQ001_QR001_001",
            query_revision_id="QR_SQ001_001",
            input_evidence_ids=["chunk-2"],
            grade=_grade(),
        )


def test_routing_policy_obeys_frozen_priority() -> None:
    unsupported = routing_decision(
        decision_id="ROUTE_001",
        grade_record_id=None,
        capability="arithmetic",
        grade=_grade(),
        input_evidence_ids={"chunk-1"},
        retrieval_budget_available=True,
    )
    assert unsupported.route == "unsupported"

    clarify = routing_decision(
        decision_id="ROUTE_002",
        grade_record_id=None,
        capability="retrieval_synthesis",
        grade=_grade(
            ambiguity="missing_slot",
            answerability="partial",
            supporting_evidence_ids=["chunk-1"],
            missing_slots=["year"],
        ),
        input_evidence_ids={"chunk-1"},
        retrieval_budget_available=True,
    )
    assert clarify.route == "clarify"

    answer = routing_decision(
        decision_id="ROUTE_003",
        grade_record_id=None,
        capability="retrieval_synthesis",
        grade=_grade(),
        input_evidence_ids={"chunk-1"},
        retrieval_budget_available=True,
    )
    assert answer.route == "answer"

    recovery = routing_decision(
        decision_id="ROUTE_004",
        grade_record_id=None,
        capability="retrieval_synthesis",
        grade=_grade(
            relevance="weak",
            answerability="partial",
            recoverability="likely",
            failure_reason="terminology_gap",
        ),
        input_evidence_ids={"chunk-1"},
        retrieval_budget_available=True,
    )
    assert recovery.route == "recover"
    assert recovery.recovery_strategy == "hyde"


def test_global_outcome_aggregation_separates_failure_from_business_outcome() -> None:
    finding = GroundedFinding(task_id="SQ_001", text="known", evidence_ids=["chunk-1"])
    assert aggregate_task_outcomes(
        [_task(status="completed", outcome="complete", finding=finding)]
    ) == ("completed", "complete")
    assert aggregate_task_outcomes(
        [
            _task(status="completed", outcome="complete", finding=finding),
            _task(identifier="SQ_002", status="completed", outcome="unsupported"),
        ]
    ) == ("completed", "partial")
    assert aggregate_task_outcomes(
        [_task(status="failed", error=ExecutionError(code="timeout", message="timeout"))]
    ) == ("failed", None)
    assert aggregate_task_outcomes([_task(status="completed", outcome="unresolved")]) == (
        "completed",
        "unresolved",
    )
    assert aggregate_task_outcomes([_task(status="completed", outcome="unsupported")]) == (
        "completed",
        "unsupported",
    )
    assert aggregate_task_outcomes([_task(status="completed", outcome="no_knowledge")]) == (
        "completed",
        "no_knowledge",
    )


def test_hitl_and_resume_validation_are_deterministic() -> None:
    option_a = ScopeOption(
        id="OPT_001",
        label="2019",
        value="2019",
        description="2019 年",
        evidence_ids=["chunk-1"],
    )
    option_b = option_a.model_copy(update={"id": "OPT_002", "value": "2020"})
    pending = HITLRequest(
        id="HITL_001",
        request_id="request-1",
        items=[
            HITLItem(
                id="ITEM_001",
                action="scope_select",
                affected_task_ids=["SQ_001"],
                question="请选择年份",
                scope_options=[option_a, option_b],
            )
        ],
    )
    validate_hitl_request(
        pending,
        {"SQ_001": _task()},
        {"SQ_001": {"chunk-1"}},
    )
    resume = ResumeRequest(
        request_id="request-1",
        hitl_request_id="HITL_001",
        responses=[HITLResponse(item_id="ITEM_001", selected_option_id="OPT_001")],
    )
    validate_resume_request(resume, pending)
    with pytest.raises(ValueError):
        validate_resume_request(
            resume.model_copy(update={"request_id": "wrong"}), pending
        )


def test_state_serialization_rejects_runtime_objects() -> None:
    task = _task()
    serialized = serialize_state({"task": task, "value": "safe"})
    assert serialized["task"]["id"] == "SQ_001"
    assert json_record(task)["id"] == "SQ_001"
    with pytest.raises(TypeError, match="runtime object"):
        serialize_state({"client": object()})


def test_stage_result_contract_separates_stage_completion_and_hitl() -> None:
    request_id = new_request_id()
    assert StageRunResult(
        request_id=request_id,
        target_stage="v2_1",
        execution_status="completed",
    ).answer_outcome is None

    answer = SynthesizedAnswer(answer="已回答")
    assert StageRunResult(
        request_id=request_id,
        target_stage="v2_2",
        execution_status="completed",
        answer_outcome="complete",
        final_answer=answer,
    ).final_answer == answer
    with pytest.raises(ValueError):
        StageRunResult(
            request_id=request_id,
            target_stage="v2_2",
            execution_status="completed",
            answer_outcome="complete",
        )
    with pytest.raises(ValueError):
        StageRunResult(
            request_id=request_id,
            target_stage="v2_2",
            execution_status="failed",
            error=ExecutionError(code="timeout", message="timeout"),
            answer_outcome="no_knowledge",
        )
