from __future__ import annotations

import pytest

from agenticrag.v2.answering import (
    FindingGenerator,
    SynthesisGenerator,
    build_finding_prompt,
    build_synthesis_prompt,
)
from agenticrag.v2.config import V2Config
from agenticrag.v2.module6 import Module6Service
from agenticrag.v2.planning import PlanningResult
from agenticrag.v2.recovery import RecoveryArtifactGenerator, RecoveryService
from agenticrag.v2.retrieval import RetrievalFanoutService
from agenticrag.v2.schemas import (
    ComplexityDecision,
    DecompositionResult,
    Evidence,
    EvidenceGrade,
    EvidenceOccurrence,
    QueryRevision,
    RetrievalAttempt,
    RetrievalResult,
    RetrievalTask,
    RetrievalLatency,
    TaskDraft,
)


class FakePlanner:
    def __init__(self, result: PlanningResult) -> None:
        self.result = result

    def plan(self, question: str) -> PlanningResult:
        return self.result


class FakeBackend:
    def __init__(self, *, fail_queries: set[str] | None = None) -> None:
        self.calls: list[dict[str, object]] = []
        self.fail_queries = fail_queries or set()

    def retrieve(self, **kwargs: object) -> RetrievalResult:
        self.calls.append(kwargs)
        query = str(kwargs["query"])
        if query in self.fail_queries:
            raise RuntimeError("retrieval unavailable")
        evidence_id = "evidence-" + query.replace(" ", "-")
        return RetrievalResult(
            evidence=[_evidence(evidence_id, kwargs)],
            latency=RetrievalLatency(total_seconds=0.01),
            trace_ref=str(kwargs["attempt_id"]),
        )


class StaticGrader:
    def __init__(self, grades: list[EvidenceGrade | Exception]) -> None:
        self.grades = list(grades)
        self.calls: list[dict[str, object]] = []

    def grade(self, **kwargs: object) -> tuple[EvidenceGrade, int]:
        self.calls.append(kwargs)
        if not self.grades:
            raise AssertionError("unexpected grader call")
        result = self.grades.pop(0)
        if isinstance(result, Exception):
            raise result
        return result, 1


class SequenceStructuredModel:
    def __init__(self, outputs: list[object]) -> None:
        self.outputs = list(outputs)
        self.calls = 0
        self.prompts: list[str] = []

    def with_structured_output(self, schema: object) -> "SequenceStructuredModel":
        return self

    def invoke(self, prompt: str) -> object:
        self.calls += 1
        self.prompts.append(prompt)
        output = self.outputs.pop(0)
        if isinstance(output, Exception):
            raise output
        return output


def _evidence(evidence_id: str, context: dict[str, object]) -> Evidence:
    return Evidence(
        evidence_id=evidence_id,
        chunk_id=evidence_id,
        content=f"content for {evidence_id}",
        doc_id="doc-1",
        source="source.pdf",
        page=1,
        occurrences=[
            EvidenceOccurrence(
                task_id=str(context["task_id"]),
                query_revision_id=str(context["query_revision_id"]),
                retrieval_attempt_id=str(context["attempt_id"]),
                strategy=str(context["strategy"]),
                final_rank=1,
            )
        ],
    )


def _grade(
    evidence_id: str = "evidence-question",
    *,
    answerability: str = "sufficient",
    relevance: str = "strong",
    ambiguity: str = "none",
    recoverability: str = "none",
    failure_reason: str = "none",
    supporting: list[str] | None = None,
    missing_slots: list[str] | None = None,
) -> EvidenceGrade:
    return EvidenceGrade(
        relevance=relevance,
        answerability=answerability,
        ambiguity=ambiguity,
        recoverability=recoverability,
        failure_reason=failure_reason,
        reason="test grade",
        missing_slots=missing_slots or [],
        missing_information=["missing fact"] if answerability != "sufficient" else [],
        supporting_evidence_ids=(supporting if supporting is not None else [evidence_id]),
    )


def _simple_plan(capability: str = "retrieval_synthesis") -> PlanningResult:
    return PlanningResult.from_router(
        question="question",
        decision=ComplexityDecision(
            complexity="simple", capability=capability, reason="one task"
        ),
        router_attempts=1,
    )


def _complex_plan(*queries: str) -> PlanningResult:
    drafts = [
        TaskDraft(query=query, intent=f"intent {query}", capability="retrieval_synthesis")
        for query in queries
    ]
    return PlanningResult(
        question="complex question",
        normalized_question="complex question",
        complexity_decision=ComplexityDecision(
            complexity="complex", capability=None, reason="independent facts"
        ),
        decomposition=DecompositionResult(tasks=drafts, decomposition_complete=True),
        router_attempts=1,
        decomposer_attempts=1,
    )


def _run_service(
    planner: PlanningResult,
    grader: StaticGrader,
    finding_model: SequenceStructuredModel,
    *,
    synthesis_model: SequenceStructuredModel | None = None,
    backend: FakeBackend | None = None,
    recovery: RecoveryService | None = None,
) -> tuple[Module6Service, object]:
    config = V2Config()
    backend = backend or FakeBackend()
    service = Module6Service(
        config,
        planner=FakePlanner(planner),
        retrieval=RetrievalFanoutService(backend, config),
        grader=grader,
        recovery=recovery,
        finding=FindingGenerator(config, model=finding_model),
        synthesis=SynthesisGenerator(config, model=synthesis_model)
        if synthesis_model is not None
        else None,
    )
    return service, service.run(planner.question)


def test_simple_answer_uses_one_simple_model_call_and_no_synthesis() -> None:
    grader = StaticGrader([_grade("evidence-question")])
    finding_model = SequenceStructuredModel([
        {"text": "grounded simple answer", "evidence_ids": ["evidence-question"]}
    ])
    service, result = _run_service(_simple_plan(), grader, finding_model)

    assert result.stage_result.target_stage == "v2_2"
    assert result.stage_result.execution_status == "completed"
    assert result.stage_result.answer_outcome == "complete"
    assert result.stage_result.final_answer is not None
    assert result.stage_result.final_answer.answer == "grounded simple answer"
    assert result.stage_result.final_answer.citation_evidence_ids == ["evidence-question"]
    assert result.stage_result.final_answer.limitations == []
    assert finding_model.calls == 1
    assert service.synthesis._model is None


def test_simple_no_knowledge_and_unsupported_are_deterministic_without_answer_models() -> None:
    no_knowledge_grader = StaticGrader([
        _grade(
            answerability="none",
            relevance="none",
            supporting=[],
        )
    ])
    no_knowledge_model = SequenceStructuredModel([])
    _, no_knowledge = _run_service(_simple_plan(), no_knowledge_grader, no_knowledge_model)
    assert no_knowledge.stage_result.answer_outcome == "no_knowledge"
    assert no_knowledge.stage_result.final_answer is not None
    assert no_knowledge_model.calls == 0

    unsupported_model = SequenceStructuredModel([])
    _, unsupported = _run_service(
        _simple_plan("arithmetic"), StaticGrader([]), unsupported_model
    )
    assert unsupported.stage_result.answer_outcome == "unsupported"
    assert unsupported.stage_result.final_answer is not None
    assert unsupported_model.calls == 0


def test_complex_findings_and_synthesis_are_scoped_and_complete() -> None:
    grader = StaticGrader([_grade("evidence-A"), _grade("evidence-B")])
    finding_model = SequenceStructuredModel([
        {"text": "finding A", "evidence_ids": ["evidence-A"]},
        {"text": "finding B", "evidence_ids": ["evidence-B"]},
    ])
    synthesis_model = SequenceStructuredModel([
        {"answer": "combined answer", "citation_evidence_ids": ["evidence-A", "evidence-B"]}
    ])
    service, result = _run_service(
        _complex_plan("A", "B"),
        grader,
        finding_model,
        synthesis_model=synthesis_model,
    )

    assert result.stage_result.answer_outcome == "complete"
    assert result.stage_result.final_answer is not None
    assert result.stage_result.final_answer.limitations == []
    assert finding_model.calls == 2
    assert synthesis_model.calls == 1
    assert "candidate_pool" not in synthesis_model.prompts[0]
    assert "RRF" not in synthesis_model.prompts[0]
    assert "evidence-A" in synthesis_model.prompts[0]
    assert "evidence-B" in synthesis_model.prompts[0]
    assert all(task.grounded_finding is not None for task in result.tasks)


def test_complex_partial_synthesizes_and_lists_uncompleted_task_limitation() -> None:
    grader = StaticGrader([
        _grade("evidence-A"),
        _grade(answerability="none", relevance="none", supporting=[]),
    ])
    finding_model = SequenceStructuredModel([
        {"text": "finding A", "evidence_ids": ["evidence-A"]}
    ])
    synthesis_model = SequenceStructuredModel([
        {"answer": "partial answer", "citation_evidence_ids": ["evidence-A"]}
    ])
    _, result = _run_service(
        _complex_plan("A", "B"),
        grader,
        finding_model,
        synthesis_model=synthesis_model,
    )

    assert result.stage_result.answer_outcome == "partial"
    assert result.stage_result.final_answer is not None
    assert result.stage_result.final_answer.limitations[0].task_id == "SQ_002"
    assert result.stage_result.final_answer.limitations[0].kind == "no_knowledge"
    assert synthesis_model.calls == 1


def test_retrieval_failure_does_not_short_circuit_other_tasks() -> None:
    backend = FakeBackend(fail_queries={"B"})
    grader = StaticGrader([_grade("evidence-A"), _grade("evidence-C")])
    finding_model = SequenceStructuredModel([
        {"text": "finding A", "evidence_ids": ["evidence-A"]},
        {"text": "finding C", "evidence_ids": ["evidence-C"]},
    ])
    synthesis_model = SequenceStructuredModel([
        {"answer": "partial answer", "citation_evidence_ids": ["evidence-A", "evidence-C"]}
    ])

    _, result = _run_service(
        _complex_plan("A", "B", "C"),
        grader,
        finding_model,
        synthesis_model=synthesis_model,
        backend=backend,
    )

    tasks = {task.id: task for task in result.tasks}
    assert tasks["SQ_002"].execution_status == "failed"
    assert tasks["SQ_002"].answer_outcome is None
    assert tasks["SQ_001"].grounded_finding is not None
    assert tasks["SQ_003"].grounded_finding is not None
    assert result.stage_result.answer_outcome == "partial"
    assert synthesis_model.calls == 1


def test_grader_failure_does_not_short_circuit_other_tasks() -> None:
    grader = StaticGrader([
        _grade("evidence-A"),
        RuntimeError("grader provider unavailable"),
        _grade("evidence-C"),
    ])
    finding_model = SequenceStructuredModel([
        {"text": "finding A", "evidence_ids": ["evidence-A"]},
        {"text": "finding C", "evidence_ids": ["evidence-C"]},
    ])
    synthesis_model = SequenceStructuredModel([
        {"answer": "partial answer", "citation_evidence_ids": ["evidence-A", "evidence-C"]}
    ])

    _, result = _run_service(
        _complex_plan("A", "B", "C"),
        grader,
        finding_model,
        synthesis_model=synthesis_model,
    )

    tasks = {task.id: task for task in result.tasks}
    assert tasks["SQ_002"].execution_status == "failed"
    assert tasks["SQ_001"].grounded_finding is not None
    assert tasks["SQ_003"].grounded_finding is not None
    assert result.stage_result.answer_outcome == "partial"
    assert synthesis_model.calls == 1


def test_finding_provenance_failure_is_task_failure_and_does_not_keep_finding() -> None:
    grader = StaticGrader([_grade("evidence-question")])
    finding_model = SequenceStructuredModel([
        {"text": "invalid", "evidence_ids": ["wrong"]},
        {"text": "still invalid", "evidence_ids": ["wrong"]},
    ])
    _, result = _run_service(_simple_plan(), grader, finding_model)

    task = result.tasks[0]
    assert task.execution_status == "failed"
    assert task.answer_outcome is None
    assert task.grounded_finding is None
    assert task.error is not None
    assert task.error.code == "citation_validation_failed"
    assert result.stage_result.execution_status == "failed"
    assert result.stage_result.final_answer is None


def test_final_synthesis_invalid_citation_fails_request_without_text_fallback() -> None:
    grader = StaticGrader([_grade("evidence-A"), _grade("evidence-B")])
    finding_model = SequenceStructuredModel([
        {"text": "finding A", "evidence_ids": ["evidence-A"]},
        {"text": "finding B", "evidence_ids": ["evidence-B"]},
    ])
    synthesis_model = SequenceStructuredModel([
        {"answer": "bad", "citation_evidence_ids": ["unknown"]},
        {"answer": "bad again", "citation_evidence_ids": ["unknown"]},
    ])
    _, result = _run_service(
        _complex_plan("A", "B"),
        grader,
        finding_model,
        synthesis_model=synthesis_model,
    )

    assert result.stage_result.execution_status == "failed"
    assert result.stage_result.answer_outcome is None
    assert result.stage_result.final_answer is None
    assert all(task.grounded_finding is not None for task in result.tasks)
    assert synthesis_model.calls == 2


def test_clarify_and_scope_select_return_non_resumable_waiting_result() -> None:
    clarify_grade = _grade(
        answerability="none",
        supporting=[],
        ambiguity="missing_slot",
        missing_slots=["year"],
    )
    _, clarify = _run_service(
        _simple_plan(), StaticGrader([clarify_grade]), SequenceStructuredModel([])
    )
    assert clarify.stage_result.execution_status == "waiting_user"
    assert clarify.stage_result.answer_outcome is None
    assert clarify.stage_result.final_answer is None
    assert clarify.stage_result.resumable is False
    assert clarify.stage_result.pending_hitl_request is not None
    assert clarify.stage_result.pending_hitl_request.items[0].action == "clarify"

    scope_grade = _grade(
        answerability="none",
        supporting=[],
        ambiguity="multiple_candidates",
    )
    _, scope = _run_service(
        _simple_plan(), StaticGrader([scope_grade]), SequenceStructuredModel([])
    )
    assert scope.stage_result.execution_status == "waiting_user"
    assert scope.stage_result.pending_hitl_request is not None
    assert scope.stage_result.pending_hitl_request.items[0].action == "scope_select"
    assert len(scope.stage_result.pending_hitl_request.items[0].scope_options) >= 2


def test_recovery_integration_creates_attempt_two_then_finding() -> None:
    config = V2Config()
    backend = FakeBackend()
    initial_and_regrade = StaticGrader([
        _grade(
            "evidence-question",
            answerability="partial",
            recoverability="likely",
            failure_reason="insufficient_coverage",
        ),
        _grade("evidence-corrective-query"),
    ])
    rewrite_model = SequenceStructuredModel([{"retrieval_query": "corrective query"}])
    recovery = RecoveryService(
        config,
        generator=RecoveryArtifactGenerator(config, model=rewrite_model),
        backend=backend,
        grader=initial_and_regrade,
    )
    finding_model = SequenceStructuredModel([
        {"text": "recovered answer", "evidence_ids": ["evidence-corrective-query"]}
    ])
    service = Module6Service(
        config,
        planner=FakePlanner(_simple_plan()),
        retrieval=RetrievalFanoutService(backend, config),
        grader=initial_and_regrade,
        recovery=recovery,
        finding=FindingGenerator(config, model=finding_model),
    )
    result = service.run("question")
    task = result.tasks[0]

    assert result.stage_result.answer_outcome == "complete"
    assert [attempt.ordinal for attempt in task.query_revisions[-1].retrieval_attempts] == [1, 2]
    assert len(task.query_revisions) == 1
    assert len(task.grade_records) == 2
    assert len(task.routing_decisions) == 2
    assert task.routing_decisions[0].route == "recover"
    assert task.routing_decisions[1].route == "answer"
    assert [call["attempt_id"] for call in backend.calls] == [
        "ATT_SQ001_QR001_001",
        "ATT_SQ001_QR001_002",
    ]


def test_finding_and_synthesis_prompts_only_expose_allowed_evidence() -> None:
    task = RetrievalTask(
        id="SQ_001",
        ordinal=1,
        query="question",
        intent="answer question",
        capability="retrieval_synthesis",
        query_revisions=[
            QueryRevision(
                id="QR_SQ001_001",
                ordinal=1,
                source="original",
                query="question",
                retrieval_attempts=[
                    RetrievalAttempt(
                        id="ATT_SQ001_QR001_001",
                        ordinal=1,
                        strategy="original",
                        retrieval_query="question",
                        evidence_ids=["e1", "e2"],
                    )
                ],
            )
        ],
    )
    grade = _grade("e1", supporting=["e1"])
    evidence = {"e1": _evidence("e1", {"task_id": "SQ_001", "query_revision_id": "QR_SQ001_001", "attempt_id": "ATT_SQ001_QR001_001", "strategy": "original"}), "e2": _evidence("e2", {"task_id": "SQ_001", "query_revision_id": "QR_SQ001_001", "attempt_id": "ATT_SQ001_QR001_001", "strategy": "original"})}
    prompt = build_finding_prompt(
        task=task,
        grade=grade,
        evidence=[evidence["e1"]],
        response_language="en",
        role="finding",
    )
    assert "e1" in prompt
    assert "e2" not in prompt
    assert "Gold" not in prompt
    assert "expected" not in prompt

    finding = task.model_copy(update={"grounded_finding": None})
    synthesis_prompt = build_synthesis_prompt(
        original_question="question",
        tasks=[finding],
        findings=[],
        evidence_by_id=evidence,
        allowed_ids=[],
        response_language="en",
    )
    assert "candidate_pool" not in synthesis_prompt
    assert "Gold" not in synthesis_prompt


def test_answer_model_roles_resolve_from_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("V2_SIMPLE_ANSWER_MODEL", "simple-answer-test")
    monkeypatch.setenv("V2_FINDING_MODEL", "finding-test")
    monkeypatch.setenv("V2_SYNTHESIS_MODEL", "synthesis-test")

    config = V2Config.from_env()

    assert config.answer_models.simple_answer.model == "simple-answer-test"
    assert config.answer_models.finding.model == "finding-test"
    assert config.answer_models.synthesis.model == "synthesis-test"
