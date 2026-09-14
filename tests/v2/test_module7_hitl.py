from __future__ import annotations

import pytest

from agenticrag.v2.config import V2Config
from agenticrag.v2.hitl import HITLResumeError, HITLResumeService, await_user_input
from agenticrag.v2.ids import grade_record_id, query_revision_id, routing_decision_id
from agenticrag.v2.recovery import RecoveryArtifactGenerator, RecoveryService
from agenticrag.v2.schemas import (
    ComplexityDecision,
    Evidence,
    EvidenceGrade,
    EvidenceOccurrence,
    GroundedFinding,
    GradeRecord,
    HITLItem,
    HITLRequest,
    QueryRevision,
    RetrievalAttempt,
    RetrievalLatency,
    RetrievalResult,
    RetrievalTask,
    ResumeRequest,
    RoutingDecision,
    ScopeOption,
    SynthesizedAnswer,
)
from agenticrag.v2.serialization import serialize_state
from agenticrag.v2.state import V2State
from agenticrag.v2.graph import initial_v2_2_state, initial_v2_3_state


class FakeBackend:
    def __init__(self, evidence_ids: list[str] | list[list[str]], *, fail: bool = False) -> None:
        self.evidence_ids = evidence_ids
        self.fail = fail
        self.calls: list[dict[str, object]] = []

    def retrieve(self, **kwargs: object) -> RetrievalResult:
        self.calls.append(kwargs)
        if self.fail:
            raise RuntimeError("resume retrieval unavailable")
        index = len(self.calls) - 1
        ids = self.evidence_ids[index] if self.evidence_ids and isinstance(self.evidence_ids[0], list) else self.evidence_ids
        task_id = str(kwargs["task_id"])
        revision_id = str(kwargs["query_revision_id"])
        attempt_id = str(kwargs["attempt_id"])
        strategy = str(kwargs["strategy"])
        return RetrievalResult(
            evidence=[
                _evidence(
                    evidence_id,
                    task_id=task_id,
                    revision_id=revision_id,
                    attempt_id=attempt_id,
                    strategy=strategy,
                )
                for evidence_id in ids  # type: ignore[union-attr]
            ],
            latency=RetrievalLatency(total_seconds=0.01),
            trace_ref=attempt_id,
        )


class SequenceGrader:
    def __init__(self, grades: list[EvidenceGrade | Exception]) -> None:
        self.grades = list(grades)
        self.calls: list[dict[str, object]] = []

    def grade(self, **kwargs: object) -> tuple[EvidenceGrade, int]:
        self.calls.append(kwargs)
        result = self.grades.pop(0)
        if isinstance(result, Exception):
            raise result
        return result, 1


class FakeFinding:
    def __init__(self, *, fail: bool = False) -> None:
        self.calls: list[dict[str, object]] = []
        self.fail = fail

    def generate(self, **kwargs: object) -> tuple[GroundedFinding, int]:
        self.calls.append(kwargs)
        if self.fail:
            raise RuntimeError("finding unavailable")
        task = kwargs["task"]
        evidence_by_id = kwargs["evidence_by_id"]
        support = task.grade_records[-1].grade.supporting_evidence_ids
        assert support
        assert support[0] in evidence_by_id
        return GroundedFinding(task_id=task.id, text="verified finding", evidence_ids=[support[0]]), 1


class FakeSynthesis:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def generate(self, **kwargs: object) -> tuple[SynthesizedAnswer, int]:
        self.calls.append(kwargs)
        findings = kwargs["findings"]
        ids = [evidence_id for finding in findings for evidence_id in finding.evidence_ids]
        return SynthesizedAnswer(
            answer="synthesized",
            citation_evidence_ids=ids,
            limitations=kwargs["limitations"],
        ), 1


class SequenceStructuredModel:
    def __init__(self, outputs: list[object]) -> None:
        self.outputs = list(outputs)
        self.calls = 0

    def with_structured_output(self, _schema: object) -> "SequenceStructuredModel":
        return self

    def invoke(self, _prompt: str) -> object:
        self.calls += 1
        output = self.outputs.pop(0)
        if isinstance(output, Exception):
            raise output
        return output


def _evidence(
    evidence_id: str,
    *,
    task_id: str = "SQ_001",
    revision_id: str = "QR_SQ001_001",
    attempt_id: str = "ATT_SQ001_QR001_001",
    strategy: str = "original",
) -> Evidence:
    return Evidence(
        evidence_id=evidence_id,
        chunk_id=evidence_id,
        content=f"fact {evidence_id}",
        doc_id="doc-1",
        source="source.pdf",
        page=1,
        occurrences=[
            EvidenceOccurrence(
                task_id=task_id,
                query_revision_id=revision_id,
                retrieval_attempt_id=attempt_id,
                strategy=strategy,
                final_rank=1,
            )
        ],
    )


def _grade(
    *,
    answerability: str = "none",
    ambiguity: str = "missing_slot",
    recoverability: str = "none",
    failure_reason: str = "none",
    supporting: list[str] | None = None,
    missing_slots: list[str] | None = None,
) -> EvidenceGrade:
    return EvidenceGrade(
        relevance="strong" if answerability == "sufficient" else "none",
        answerability=answerability,  # type: ignore[arg-type]
        ambiguity=ambiguity,  # type: ignore[arg-type]
        recoverability=recoverability,  # type: ignore[arg-type]
        failure_reason=failure_reason,  # type: ignore[arg-type]
        reason="test grade",
        missing_slots=missing_slots or (["year"] if ambiguity == "missing_slot" else []),
        missing_information=["missing fact"] if recoverability == "likely" else [],
        supporting_evidence_ids=supporting or [],
    )


def _waiting_task(
    *,
    request_id: str,
    action: str = "clarify",
    task_id: str = "SQ_001",
    evidence_id: str = "old",
    revisions: int = 1,
) -> tuple[RetrievalTask, HITLRequest, dict[str, Evidence]]:
    revisions_list: list[QueryRevision] = []
    grades: list[GradeRecord] = []
    decisions: list[RoutingDecision] = []
    evidence = {evidence_id: _evidence(evidence_id, task_id=task_id)}
    for ordinal in range(1, revisions + 1):
        revision_id = query_revision_id(task_id, ordinal)
        attempt_id = f"ATT_SQ001_QR{ordinal:03d}_001"
        revision = QueryRevision(
            id=revision_id,
            ordinal=ordinal,
            source="original" if ordinal == 1 else "hitl",
            query="original query" if ordinal == 1 else "previous clarified query",
            user_input=None if ordinal == 1 else {"year": "2018"},
            retrieval_attempts=[
                RetrievalAttempt(
                    id=attempt_id,
                    ordinal=1,
                    strategy="original" if ordinal == 1 else "user_clarified",
                    retrieval_query="original query" if ordinal == 1 else "previous clarified query",
                    evidence_ids=[evidence_id],
                )
            ],
        )
        grade = _grade() if action == "clarify" else _grade(ambiguity="multiple_candidates")
        record = GradeRecord(
            id=grade_record_id(task_id, revision_id, ordinal),
            query_revision_id=revision_id,
            input_attempt_ids=[attempt_id],
            input_evidence_ids=[evidence_id],
            grade=grade,
        )
        decision = RoutingDecision(
            id=routing_decision_id(task_id, revision_id, ordinal),
            grade_record_id=record.id,
            route=action,  # type: ignore[arg-type]
            reason="waiting for user",
        )
        revisions_list.append(revision)
        grades.append(record)
        decisions.append(decision)
    item = (
        HITLItem(
            id="ITEM_001",
            action="clarify",
            affected_task_ids=[task_id],
            question="Which year?",
            missing_slots=["year"],
        )
        if action == "clarify"
        else HITLItem(
            id="ITEM_001",
            action="scope_select",
            affected_task_ids=[task_id],
            question="Select scope",
            scope_options=[
                ScopeOption(id="OPT_001", label="A", value="scope A", description="A"),
                ScopeOption(id="OPT_002", label="B", value="scope B", description="B"),
            ],
        )
    )
    pending = HITLRequest(id="HITL_001", request_id=request_id, items=[item])
    task = RetrievalTask(
        id=task_id,
        ordinal=int(task_id.split("_")[1]),
        query="original query",
        intent="answer the query",
        capability="retrieval_synthesis",
        query_revisions=revisions_list,
        grade_records=grades,
        routing_decisions=decisions,
        execution_status="waiting_user",
    )
    return task, pending, evidence


def _state(
    task: RetrievalTask,
    pending: HITLRequest,
    evidence: dict[str, Evidence],
    *,
    complexity: str = "simple",
    hitl_rounds: int = 0,
    target_stage: str = "v2_3",
) -> V2State:
    initializer = initial_v2_3_state if target_stage == "v2_3" else initial_v2_2_state
    state = initializer("original question", request_id=pending.request_id)
    state.update(
        {
            "complexity_decision": ComplexityDecision(
                complexity=complexity, capability="retrieval_synthesis" if complexity == "simple" else None, reason="test"
            ),
            "tasks": {task.id: task},
            "task_order": [task.id],
            "evidence": evidence,
            "pending_hitl_request": pending,
            "hitl_rounds": hitl_rounds,
            "execution_status": "waiting_user",
        }
    )
    return state


class CountingRecovery:
    def __init__(self) -> None:
        self.calls = 0

    def recover(self, **kwargs: object) -> object:
        self.calls += 1
        raise AssertionError("recovery must not be called for a non-resumable stage")


def _resume(state: V2State, *, response: object, **service_kwargs: object):
    pending = state["pending_hitl_request"]
    assert pending is not None
    request = ResumeRequest(
        request_id=state["request_id"],
        hitl_request_id=pending.id,
        responses=[response],
    )
    return HITLResumeService(V2Config(), **service_kwargs).resume(state, request)


def test_await_user_input_is_a_side_effect_free_boundary() -> None:
    request_id = "00000000-0000-4000-8000-000000000001"
    task, pending, evidence = _waiting_task(request_id=request_id)
    state = _state(task, pending, evidence)
    before = serialize_state(state)
    assert await_user_input(state) == {
        "execution_status": "waiting_user",
        "answer_outcome": None,
        "final_answer": None,
    }
    assert serialize_state(state) == before


def test_v22_waiting_state_rejects_resume_without_mutation_or_execution() -> None:
    request_id = "00000000-0000-4000-8000-000000000002"
    task, pending, evidence = _waiting_task(request_id=request_id)
    state = _state(task, pending, evidence, target_stage="v2_2")
    request = ResumeRequest(
        request_id=request_id,
        hitl_request_id=pending.id,
        responses=[{"item_id": "ITEM_001", "clarify_values": {"year": "2019"}}],
    )
    backend = FakeBackend(["new"])
    grader = SequenceGrader([])
    finding = FakeFinding()
    synthesis = FakeSynthesis()
    recovery = CountingRecovery()
    before = serialize_state(state)

    with pytest.raises(HITLResumeError) as exc_info:
        HITLResumeService(
            V2Config(),
            backend=backend,
            grader=grader,
            finding=finding,
            synthesis=synthesis,
            recovery=recovery,  # type: ignore[arg-type]
        ).resume(state, request)

    assert exc_info.value.execution_error.code == "request_not_resumable"
    assert serialize_state(state) == before
    assert state["hitl_rounds"] == 0
    assert len(state["tasks"]["SQ_001"].query_revisions) == 1
    assert backend.calls == []
    assert grader.calls == []
    assert finding.calls == []
    assert synthesis.calls == []
    assert recovery.calls == 0


def test_await_user_input_rejects_v22_waiting_state() -> None:
    request_id = "00000000-0000-4000-8000-000000000006"
    task, pending, evidence = _waiting_task(request_id=request_id)
    state = _state(task, pending, evidence, target_stage="v2_2")
    before = serialize_state(state)

    with pytest.raises(HITLResumeError) as exc_info:
        await_user_input(state)

    assert exc_info.value.execution_error.code == "request_not_resumable"
    assert serialize_state(state) == before


@pytest.mark.parametrize(
    "mutator",
    [
        lambda request: request.model_copy(update={"request_id": "00000000-0000-4000-8000-000000000002"}),
        lambda request: request.model_copy(update={"hitl_request_id": "HITL_002"}),
        lambda request: request.model_copy(update={"responses": []}),
        lambda request: request.model_copy(
            update={"responses": [request.responses[0], request.responses[0].model_copy(update={"item_id": "ITEM_002"})]}
        ),
    ],
)
def test_invalid_resume_is_zero_mutation_and_zero_execution(mutator) -> None:
    request_id = "00000000-0000-4000-8000-000000000003"
    task, pending, evidence = _waiting_task(request_id=request_id)
    state = _state(task, pending, evidence)
    valid = ResumeRequest(
        request_id=request_id,
        hitl_request_id=pending.id,
        responses=[{"item_id": "ITEM_001", "clarify_values": {"year": "2019"}}],
    )
    invalid = mutator(valid)
    backend = FakeBackend(["new"])
    grader = SequenceGrader([])
    before = serialize_state(state)
    with pytest.raises(HITLResumeError) as exc_info:
        HITLResumeService(V2Config(), backend=backend, grader=grader).resume(state, invalid)
    assert exc_info.value.execution_error.code in {"resume_payload_invalid", "request_not_resumable"}
    assert serialize_state(state) == before
    assert backend.calls == []
    assert grader.calls == []


def test_clarify_resume_creates_qr2_and_user_clarified_attempt_without_extra_llm() -> None:
    request_id = "00000000-0000-4000-8000-000000000004"
    task, pending, evidence = _waiting_task(request_id=request_id)
    state = _state(task, pending, evidence)
    backend = FakeBackend(["new"])
    grader = SequenceGrader([_grade(answerability="sufficient", ambiguity="none", supporting=["new"])])
    finding = FakeFinding()
    result = _resume(
        state,
        response={"item_id": "ITEM_001", "clarify_values": {"year": "2019"}},
        backend=backend,
        grader=grader,
        finding=finding,
    )
    resumed = result.state["tasks"]["SQ_001"]
    revision = resumed.query_revisions[-1]
    attempt = revision.retrieval_attempts[0]
    assert revision.id == "QR_SQ001_002"
    assert revision.source == "hitl"
    assert revision.user_input == {"year": "2019"}
    assert "year=2019" in revision.query
    assert attempt.id == "ATT_SQ001_QR002_001"
    assert attempt.strategy == "user_clarified"
    assert backend.calls[0]["query"] == revision.query
    assert backend.calls[0]["query_revision_id"] == revision.id
    assert grader.calls[0]["evidence"] and [item.evidence_id for item in grader.calls[0]["evidence"]] == ["new"]
    assert finding.calls[0]["role"] == "simple_answer"
    assert result.state["hitl_rounds"] == 1
    assert result.state["pending_hitl_request"] is None
    assert result.state["stage_result"].answer_outcome == "complete"


def test_scope_resume_uses_semantic_option_value_not_option_or_evidence_id() -> None:
    request_id = "00000000-0000-4000-8000-000000000005"
    task, pending, evidence = _waiting_task(request_id=request_id, action="scope_select")
    pending = pending.model_copy(
        update={
            "items": [
                pending.items[0].model_copy(
                    update={
                        "scope_options": [
                            ScopeOption(id="OPT_001", label="A", value="service A", description="A", evidence_ids=["old"]),
                            ScopeOption(id="OPT_002", label="B", value="service B", description="B", evidence_ids=["old"]),
                        ]
                    }
                )
            ]
        }
    )
    state = _state(task, pending, evidence)
    backend = FakeBackend(["new"])
    grader = SequenceGrader([_grade(answerability="sufficient", ambiguity="none", supporting=["new"])])
    result = _resume(
        state,
        response={"item_id": "ITEM_001", "selected_option_id": "OPT_002"},
        backend=backend,
        grader=grader,
        finding=FakeFinding(),
    )
    revision = result.state["tasks"]["SQ_001"].query_revisions[-1]
    assert revision.user_input == {"scope": "service B"}
    assert "service B" in revision.query
    assert "OPT_002" not in revision.query
    assert "old" not in revision.query


def test_affected_task_isolation_preserves_unaffected_tasks_and_findings() -> None:
    request_id = "00000000-0000-4000-8000-000000000006"
    affected, pending, evidence = _waiting_task(request_id=request_id)
    complete_tasks = []
    for task_id, evidence_id in (("SQ_002", "done-2"), ("SQ_003", "done-3")):
        complete, _unused, complete_evidence = _waiting_task(
            request_id=request_id, task_id=task_id, evidence_id=evidence_id
        )
        grade = _grade(answerability="sufficient", ambiguity="none", supporting=[evidence_id])
        record = complete.grade_records[0].model_copy(update={"grade": grade})
        decision = complete.routing_decisions[0].model_copy(update={"route": "answer"})
        finding = GroundedFinding(task_id=task_id, text=f"finding {task_id}", evidence_ids=[evidence_id])
        complete = complete.model_copy(
            update={
                "grade_records": [record],
                "routing_decisions": [decision],
                "grounded_finding": finding,
                "execution_status": "completed",
                "answer_outcome": "complete",
            }
        )
        complete_tasks.append(complete)
        evidence.update(complete_evidence)
    state = _state(affected, pending, evidence, complexity="complex")
    state["tasks"].update({task.id: task for task in complete_tasks})
    state["task_order"] = ["SQ_001", "SQ_002", "SQ_003"]
    before = {task.id: serialize_state({"task": task}) for task in complete_tasks}
    result = _resume(
        state,
        response={"item_id": "ITEM_001", "clarify_values": {"year": "2019"}},
        backend=FakeBackend(["new-1"]),
        grader=SequenceGrader([_grade(answerability="sufficient", ambiguity="none", supporting=["new-1"])]),
        finding=FakeFinding(),
        synthesis=FakeSynthesis(),
    )
    for task in complete_tasks:
        assert serialize_state({"task": result.state["tasks"][task.id]}) == before[task.id]
    assert result.state["tasks"]["SQ_001"].query_revisions[-1].id == "QR_SQ001_002"
    assert result.state["stage_result"].answer_outcome == "complete"


def test_qr2_initial_grade_excludes_qr1_evidence_and_resume_recovery_is_same_revision() -> None:
    request_id = "00000000-0000-4000-8000-000000000007"
    task, pending, evidence = _waiting_task(request_id=request_id)
    backend = FakeBackend([["qr2"], ["qr2", "att2"]])
    grader = SequenceGrader([
        _grade(
            answerability="partial",
            ambiguity="none",
            recoverability="likely",
            failure_reason="insufficient_coverage",
            supporting=["qr2"],
        ),
        _grade(
            answerability="sufficient",
            ambiguity="none",
            supporting=["qr2", "att2"],
        ),
    ])
    rewrite = SequenceStructuredModel([{"retrieval_query": "targeted clarified query"}])
    recovery = RecoveryService(
        V2Config(),
        generator=RecoveryArtifactGenerator(V2Config(), model=rewrite),
        backend=backend,
        grader=grader,
    )
    result = _resume(
        _state(task, pending, evidence),
        response={"item_id": "ITEM_001", "clarify_values": {"year": "2019"}},
        backend=backend,
        grader=grader,
        recovery=recovery,
        finding=FakeFinding(),
    )
    resumed = result.state["tasks"]["SQ_001"]
    revision = resumed.query_revisions[-1]
    assert revision.id == "QR_SQ001_002"
    assert [attempt.strategy for attempt in revision.retrieval_attempts] == ["user_clarified", "direct_rewrite"]
    assert len(resumed.query_revisions) == 2
    assert [item.evidence_id for item in grader.calls[0]["evidence"]] == ["qr2"]
    assert [item.evidence_id for item in grader.calls[1]["evidence"]] == ["qr2", "att2"]
    assert "old" not in [item.evidence_id for item in grader.calls[0]["evidence"]]
    assert result.state["stage_result"].answer_outcome == "complete"
    assert rewrite.calls == 1
    assert len(result.state["evidence"]["qr2"].occurrences) == 2
    assert {item.retrieval_attempt_id for item in result.state["evidence"]["qr2"].occurrences} == {
        "ATT_SQ001_QR002_001",
        "ATT_SQ001_QR002_002",
    }


def test_hitl_budget_exhausted_after_resume_routes_again_to_unresolved_without_second_request() -> None:
    request_id = "00000000-0000-4000-8000-000000000008"
    task, pending, evidence = _waiting_task(request_id=request_id, revisions=2)
    backend = FakeBackend(["never"])
    result = _resume(
        _state(task, pending, evidence),
        response={"item_id": "ITEM_001", "clarify_values": {"year": "2019"}},
        backend=backend,
        grader=SequenceGrader([]),
    )
    resumed = result.state["tasks"]["SQ_001"]
    assert result.new_query_revision_ids == ()
    assert len(resumed.query_revisions) == 2
    assert resumed.execution_status == "completed"
    assert resumed.answer_outcome == "unresolved"
    assert resumed.terminal_reason == "hitl_budget_exhausted"
    assert backend.calls == []
    assert result.state["pending_hitl_request"] is None


def test_scope_route_after_the_only_hitl_round_becomes_unresolved() -> None:
    request_id = "00000000-0000-4000-8000-000000000011"
    task, pending, evidence = _waiting_task(request_id=request_id, action="scope_select")
    result = _resume(
        _state(task, pending, evidence),
        response={"item_id": "ITEM_001", "selected_option_id": "OPT_002"},
        backend=FakeBackend(["new"]),
        grader=SequenceGrader([_grade(ambiguity="multiple_candidates")]),
    )
    resumed = result.state["tasks"]["SQ_001"]
    assert len(resumed.query_revisions) == 2
    assert resumed.answer_outcome == "unresolved"
    assert resumed.terminal_reason == "hitl_budget_exhausted"
    assert result.state["pending_hitl_request"] is None


@pytest.mark.parametrize("failure_kind", ["retrieval", "grader", "finding"])
def test_resume_task_technical_failures_are_isolated(failure_kind: str) -> None:
    request_id = "00000000-0000-4000-8000-000000000009"
    task, pending, evidence = _waiting_task(request_id=request_id)
    backend = FakeBackend(["new"], fail=failure_kind == "retrieval")
    grader = SequenceGrader(
        [RuntimeError("grader down")] if failure_kind == "grader" else [_grade(answerability="sufficient", ambiguity="none", supporting=["new"])]
    )
    finding = FakeFinding(fail=failure_kind == "finding")
    result = _resume(
        _state(task, pending, evidence),
        response={"item_id": "ITEM_001", "clarify_values": {"year": "2019"}},
        backend=backend,
        grader=grader,
        finding=finding,
    )
    resumed = result.state["tasks"]["SQ_001"]
    assert resumed.execution_status == "failed"
    assert resumed.answer_outcome is None
    assert resumed.error is not None
    assert result.state["stage_result"].execution_status == "failed"


def test_validate_resume_strictly_rejects_extra_clarification_slots() -> None:
    request_id = "00000000-0000-4000-8000-000000000010"
    task, pending, evidence = _waiting_task(request_id=request_id)
    state = _state(task, pending, evidence)
    request = ResumeRequest(
        request_id=request_id,
        hitl_request_id=pending.id,
        responses=[
            {"item_id": "ITEM_001", "clarify_values": {"year": "2019", "region": "north"}}
        ],
    )
    with pytest.raises(HITLResumeError) as exc_info:
        HITLResumeService(V2Config()).resume(state, request)
    assert exc_info.value.execution_error.code == "resume_payload_invalid"


def test_scope_resume_rejects_option_with_unknown_evidence_without_execution() -> None:
    request_id = "00000000-0000-4000-8000-000000000012"
    task, pending, evidence = _waiting_task(request_id=request_id, action="scope_select")
    pending = pending.model_copy(
        update={
            "items": [
                pending.items[0].model_copy(
                    update={
                        "scope_options": [
                            ScopeOption(id="OPT_001", label="A", value="A", description="A", evidence_ids=["missing"]),
                            ScopeOption(id="OPT_002", label="B", value="B", description="B", evidence_ids=[]),
                        ]
                    }
                )
            ]
        }
    )
    state = _state(task, pending, evidence)
    backend = FakeBackend(["never"])
    before = serialize_state(state)
    with pytest.raises(HITLResumeError) as exc_info:
        _resume(
            state,
            response={"item_id": "ITEM_001", "selected_option_id": "OPT_002"},
            backend=backend,
            grader=SequenceGrader([]),
        )
    assert exc_info.value.execution_error.code == "resume_payload_invalid"
    assert backend.calls == []
    assert serialize_state(state) == before


def test_resume_failure_keeps_unaffected_finding_and_aggregates_partial() -> None:
    request_id = "00000000-0000-4000-8000-000000000013"
    affected, pending, evidence = _waiting_task(request_id=request_id)
    unaffected, _unused, unaffected_evidence = _waiting_task(
        request_id=request_id, task_id="SQ_002", evidence_id="done"
    )
    complete_grade = _grade(answerability="sufficient", ambiguity="none", supporting=["done"])
    unaffected = unaffected.model_copy(
        update={
            "grade_records": [unaffected.grade_records[0].model_copy(update={"grade": complete_grade})],
            "routing_decisions": [unaffected.routing_decisions[0].model_copy(update={"route": "answer"})],
            "grounded_finding": GroundedFinding(
                task_id="SQ_002", text="existing finding", evidence_ids=["done"]
            ),
            "execution_status": "completed",
            "answer_outcome": "complete",
        }
    )
    evidence.update(unaffected_evidence)
    state = _state(affected, pending, evidence, complexity="complex")
    state["tasks"][unaffected.id] = unaffected
    state["task_order"] = ["SQ_001", "SQ_002"]
    before = serialize_state({"task": unaffected})
    result = _resume(
        state,
        response={"item_id": "ITEM_001", "clarify_values": {"year": "2019"}},
        backend=FakeBackend(["never"], fail=True),
        grader=SequenceGrader([]),
        synthesis=FakeSynthesis(),
    )
    assert result.state["tasks"]["SQ_001"].execution_status == "failed"
    assert result.state["tasks"]["SQ_001"].answer_outcome is None
    assert serialize_state({"task": result.state["tasks"]["SQ_002"]}) == before
    assert result.state["tasks"]["SQ_002"].grounded_finding is not None
    assert result.state["stage_result"].execution_status == "completed"
    assert result.state["stage_result"].answer_outcome == "partial"
