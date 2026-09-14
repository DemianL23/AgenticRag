from __future__ import annotations

import pytest

from agenticrag.v2.config import V2Config
from agenticrag.v2.grading import EvidenceGrader, EvidenceGradingError
from agenticrag.v2.ids import grade_record_id, routing_decision_id
from agenticrag.v2.module4 import make_grade_record
from agenticrag.v2.recovery import (
    RecoveryArtifactGenerator,
    RecoveryExecutionError,
    RecoveryService,
)
from agenticrag.v2.schemas import (
    Evidence,
    EvidenceGrade,
    EvidenceOccurrence,
    GradeRecord,
    QueryRevision,
    RetrievalAttempt,
    RetrievalLatency,
    RetrievalResult,
    RetrievalTask,
    RoutingDecision,
)


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


class FakeBackend:
    def __init__(
        self, evidence_ids: list[str], *, fail: bool = False, degraded: bool = False
    ) -> None:
        self.evidence_ids = evidence_ids
        self.fail = fail
        self.degraded = degraded
        self.calls: list[dict[str, object]] = []

    def retrieve(self, **kwargs: object) -> RetrievalResult:
        self.calls.append(kwargs)
        if self.fail:
            raise RuntimeError("attempt 2 backend unavailable")
        attempt_id = str(kwargs["attempt_id"])
        revision_id = str(kwargs["query_revision_id"])
        task_id = str(kwargs["task_id"])
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
                for evidence_id in self.evidence_ids
            ],
            retrieval_degraded=self.degraded,
            degraded_reason="reranker_fallback" if self.degraded else None,
            latency=RetrievalLatency(total_seconds=0.25),
            trace_ref=attempt_id,
        )


class SequenceGrader:
    def __init__(self, grades: list[EvidenceGrade]) -> None:
        self.grades = list(grades)
        self.calls: list[list[str]] = []

    def grade(self, *, evidence: list[Evidence], **kwargs: object) -> tuple[EvidenceGrade, int]:
        self.calls.append([item.evidence_id for item in evidence])
        return self.grades.pop(0), 1


class FailingGrader:
    def grade(self, **kwargs: object) -> tuple[EvidenceGrade, int]:
        raise EvidenceGradingError(attempts=2, cause=ValueError("invalid re-grade"))


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


def _grade(*, answerability: str = "partial", **overrides: object) -> EvidenceGrade:
    value: dict[str, object] = {
        "relevance": "weak",
        "answerability": answerability,
        "ambiguity": "none",
        "recoverability": "likely",
        "failure_reason": "insufficient_coverage",
        "reason": "evidence needs a bounded recovery",
        "missing_information": ["missing fact"],
        "missing_slots": [],
        "supporting_evidence_ids": ["A"],
    }
    value.update(overrides)
    return EvidenceGrade.model_validate(value)


def _recovery_task(
    *,
    attempt_count: int = 1,
    strategy: str = "direct_rewrite",
    query: str = "original query",
) -> tuple[RetrievalTask, dict[str, Evidence]]:
    first_ids = ["A", "B", "C", "D", "E"]
    attempt1 = RetrievalAttempt(
        id="ATT_SQ001_QR001_001",
        ordinal=1,
        strategy="original",
        retrieval_query=query,
        evidence_ids=first_ids,
    )
    attempts = [attempt1]
    if attempt_count == 2:
        attempts.append(
            RetrievalAttempt(
                id="ATT_SQ001_QR001_002",
                ordinal=2,
                strategy=strategy,
                retrieval_query="already recovered query",
                evidence_ids=["C", "D", "F", "G", "H"],
            )
        )
    revision = QueryRevision(
        id="QR_SQ001_001",
        ordinal=1,
        source="original",
        query=query,
        retrieval_attempts=attempts,
    )
    failure_reason = {
        "step_back": "overly_specific",
        "hyde": "terminology_gap",
    }.get(strategy, "insufficient_coverage")
    grade = _grade(
        failure_reason=failure_reason,
        supporting_evidence_ids=["A"],
    )
    record = GradeRecord(
        id=grade_record_id("SQ_001", revision.id, 1),
        query_revision_id=revision.id,
        input_attempt_ids=[attempt.id for attempt in attempts],
        input_evidence_ids=first_ids if attempt_count == 1 else first_ids + ["F", "G", "H"],
        grade=grade,
    )
    decision = RoutingDecision(
        id=routing_decision_id("SQ_001", revision.id, 1),
        grade_record_id=record.id,
        route="recover",
        recovery_strategy=strategy,
        reason="deterministic policy: recoverable evidence gap",
    )
    task = RetrievalTask(
        id="SQ_001",
        ordinal=1,
        query=query,
        intent="retrieve the requested fact",
        capability="retrieval_synthesis",
        query_revisions=[revision],
        grade_records=[record],
        routing_decisions=[decision],
        execution_status="running",
    )
    evidence = {evidence_id: _evidence(evidence_id) for evidence_id in first_ids}
    return task, evidence


def test_direct_rewrite_uses_fixed_strategy_and_only_returns_query() -> None:
    task, evidence = _recovery_task()[0:2]
    model = SequenceStructuredModel([{"retrieval_query": "targeted missing fact query"}])
    artifact, attempts = RecoveryArtifactGenerator(
        V2Config(), model=model
    ).generate(
        task=task,
        revision=task.query_revisions[-1],
        grade=task.grade_records[-1].grade,
        strategy="direct_rewrite",
        evidence=list(evidence.values()),
    )

    assert artifact.strategy == "direct_rewrite"
    assert artifact.retrieval_query == "targeted missing fact query"
    assert artifact.is_evidence is False
    assert attempts == 1
    assert model.calls == 1
    assert "missing fact" in model.prompts[0]
    assert "route" in model.prompts[0]
    assert "candidate_pool" not in model.prompts[0]


@pytest.mark.parametrize("strategy", ["step_back", "hyde"])
def test_strategy_specific_artifacts_are_not_evidence(strategy: str) -> None:
    task, evidence = _recovery_task()
    model = SequenceStructuredModel([{"retrieval_query": f"{strategy} retrieval artifact"}])
    artifact, _attempts = RecoveryArtifactGenerator(V2Config(), model=model).generate(
        task=task,
        revision=task.query_revisions[-1],
        grade=task.grade_records[-1].grade,
        strategy=strategy,  # type: ignore[arg-type]
        evidence=list(evidence.values()),
    )

    assert artifact.strategy == strategy
    assert artifact.is_evidence is False
    assert strategy in model.prompts[0]
    if strategy == "hyde":
        assert "hypothetical" in model.prompts[0]


def test_recovery_executes_attempt_two_unions_evidence_and_reroutes_to_answer() -> None:
    task, evidence = _recovery_task()
    backend = FakeBackend(["C", "D", "F", "G", "H"], degraded=True)
    grader = SequenceGrader([
        _grade(
            relevance="strong",
            answerability="sufficient",
            recoverability="none",
            failure_reason="none",
            missing_information=[],
            supporting_evidence_ids=["A", "F"],
            reason="union contains sufficient evidence",
        )
    ])
    service = RecoveryService(
        V2Config(),
        generator=RecoveryArtifactGenerator(
            V2Config(), model=SequenceStructuredModel([{"retrieval_query": "corrective query"}])
        ),
        backend=backend,
        grader=grader,
    )

    result = service.recover(task=task, evidence_by_id=evidence)

    assert result.task.query_revisions[-1].id == "QR_SQ001_001"
    assert [a.ordinal for a in result.task.query_revisions[-1].retrieval_attempts] == [1, 2]
    attempt2 = result.task.query_revisions[-1].retrieval_attempts[-1]
    assert attempt2.id == "ATT_SQ001_QR001_002"
    assert attempt2.strategy == "direct_rewrite"
    assert attempt2.retrieval_query == "corrective query"
    assert attempt2.retrieval_degraded is True
    assert attempt2.retrieval_degraded_reason == "reranker_fallback"
    assert attempt2.latency.total_seconds == 0.25
    assert backend.calls[0]["query_revision_id"] == "QR_SQ001_001"
    assert backend.calls[0]["attempt_id"] == attempt2.id
    assert attempt2.trace_ref == attempt2.id
    assert set(result.evidence) == {"A", "B", "C", "D", "E", "F", "G", "H"}
    assert len(result.evidence["C"].occurrences) == 2
    assert [record.input_attempt_ids for record in result.task.grade_records] == [
        ["ATT_SQ001_QR001_001"],
        ["ATT_SQ001_QR001_001", "ATT_SQ001_QR001_002"],
    ]
    assert len(result.task.routing_decisions) == 2
    assert result.routing_decision.route == "answer"
    assert len(grader.calls) == 1
    assert grader.calls[0] == ["A", "B", "C", "D", "E", "F", "G", "H"]


def test_recovery_after_second_attempt_cannot_recover_again() -> None:
    task, evidence = _recovery_task(attempt_count=2)
    rewrite_model = SequenceStructuredModel([{"retrieval_query": "must not run"}])
    backend = FakeBackend(["X"])
    service = RecoveryService(
        V2Config(),
        generator=RecoveryArtifactGenerator(V2Config(), model=rewrite_model),
        backend=backend,
    )

    with pytest.raises(ValueError, match="budget"):
        service.recover(task=task, evidence_by_id=evidence)
    assert rewrite_model.calls == 0
    assert backend.calls == []
    assert len(task.query_revisions[-1].retrieval_attempts) == 2


def test_regrade_still_recoverable_is_terminated_by_exhausted_budget() -> None:
    task, evidence = _recovery_task()
    result = RecoveryService(
        V2Config(),
        generator=RecoveryArtifactGenerator(
            V2Config(), model=SequenceStructuredModel([{"retrieval_query": "corrective query"}])
        ),
        backend=FakeBackend(["F"]),
        grader=SequenceGrader([_grade()]),
    ).recover(task=task, evidence_by_id=evidence)

    assert result.routing_decision.route == "no_knowledge"
    assert result.routing_decision.recovery_strategy is None
    assert len(result.task.query_revisions[-1].retrieval_attempts) == 2


def test_regrade_failure_marks_task_failed_but_preserves_history() -> None:
    task, evidence = _recovery_task()
    with pytest.raises(RecoveryExecutionError) as raised:
        RecoveryService(
            V2Config(),
            generator=RecoveryArtifactGenerator(
                V2Config(), model=SequenceStructuredModel([{"retrieval_query": "corrective query"}])
            ),
            backend=FakeBackend(["F"]),
            grader=FailingGrader(),
        ).recover(task=task, evidence_by_id=evidence)

    failed = raised.value.failed_task
    assert raised.value.execution_error.code == "grader_failed"
    assert failed.execution_status == "failed"
    assert failed.answer_outcome is None
    assert len(failed.query_revisions[-1].retrieval_attempts) == 2
    assert len(failed.grade_records) == 1
    assert len(failed.routing_decisions) == 1


def test_duplicate_recovery_query_uses_one_bounded_repair_and_fails() -> None:
    task, evidence = _recovery_task()
    model = SequenceStructuredModel(
        [{"retrieval_query": " ORIGINAL   QUERY "}, {"retrieval_query": "original query"}]
    )
    generator = RecoveryArtifactGenerator(V2Config(), model=model)

    with pytest.raises(RecoveryExecutionError) as raised:
        RecoveryService(
            V2Config(),
            generator=generator,
            backend=FakeBackend(["X"]),
        ).recover(task=task, evidence_by_id=evidence)

    assert model.calls == 2
    assert raised.value.execution_error.code == "recovery_duplicate_query"
    assert raised.value.failed_task.query_revisions[-1].retrieval_attempts == task.query_revisions[-1].retrieval_attempts


def test_invalid_empty_artifact_is_bounded_and_becomes_technical_failure() -> None:
    task, evidence = _recovery_task()
    model = SequenceStructuredModel(
        [{"retrieval_query": "   "}, {"retrieval_query": "   "}]
    )

    with pytest.raises(RecoveryExecutionError) as raised:
        RecoveryService(
            V2Config(),
            generator=RecoveryArtifactGenerator(V2Config(), model=model),
            backend=FakeBackend(["X"]),
        ).recover(task=task, evidence_by_id=evidence)

    assert model.calls == 2
    assert raised.value.execution_error.code == "invalid_recovery_artifact"
    assert raised.value.failed_task.answer_outcome is None


def test_recovery_backend_failure_preserves_attempt_one_history() -> None:
    task, evidence = _recovery_task()
    model = SequenceStructuredModel([{"retrieval_query": "corrective query"}])
    with pytest.raises(RecoveryExecutionError) as raised:
        RecoveryService(
            V2Config(),
            generator=RecoveryArtifactGenerator(V2Config(), model=model),
            backend=FakeBackend([], fail=True),
        ).recover(task=task, evidence_by_id=evidence)

    failed = raised.value.failed_task
    assert raised.value.execution_error.code == "retrieval_failed"
    assert failed.answer_outcome is None
    assert [attempt.ordinal for attempt in failed.query_revisions[-1].retrieval_attempts] == [1, 2]
    assert len(failed.grade_records) == 1
    assert len(failed.routing_decisions) == 1


def test_hyde_artifact_is_only_attempt_query_and_never_synthetic_evidence() -> None:
    task, evidence = _recovery_task(strategy="hyde")
    backend = FakeBackend(["real-evidence"])
    grader = SequenceGrader([_grade(answerability="none", supporting_evidence_ids=[])])
    result = RecoveryService(
        V2Config(),
        generator=RecoveryArtifactGenerator(
            V2Config(), model=SequenceStructuredModel([{"retrieval_query": "hypothetical document"}])
        ),
        backend=backend,
        grader=grader,
    ).recover(task=task, evidence_by_id=evidence)

    assert result.artifact.is_evidence is False
    assert result.task.query_revisions[-1].retrieval_attempts[-1].retrieval_query == "hypothetical document"
    assert "hypothetical document" not in result.evidence
    assert "hypothetical document" not in result.grade_record.grade.supporting_evidence_ids


def test_grader_accepts_exact_two_attempt_union_and_rejects_external_evidence() -> None:
    task, _initial_evidence = _recovery_task()
    revision = task.query_revisions[-1].model_copy(
        update={
            "retrieval_attempts": [
                task.query_revisions[-1].retrieval_attempts[0],
                RetrievalAttempt(
                    id="ATT_SQ001_QR001_002",
                    ordinal=2,
                    strategy="direct_rewrite",
                    retrieval_query="corrective query",
                    evidence_ids=["F", "G", "H"],
                ),
            ]
        }
    )
    evidence = [_evidence(item) for item in ["A", "B", "C", "D", "E", "F", "G", "H"]]
    model = SequenceStructuredModel([
        {
            "relevance": "strong",
            "answerability": "sufficient",
            "ambiguity": "none",
            "recoverability": "none",
            "failure_reason": "none",
            "reason": "sufficient",
            "missing_information": [],
            "missing_slots": [],
            "supporting_evidence_ids": ["A"],
        }
    ])
    grade, attempts = EvidenceGrader(V2Config(), model=model).grade(
        task=task,
        revision=revision,
        evidence=evidence,
    )
    assert grade.answerability == "sufficient"
    assert attempts == 1

    with pytest.raises(ValueError, match="严格等于"):
        EvidenceGrader(V2Config(), model=model).grade(
            task=task,
            revision=revision,
            evidence=evidence[:-1] + [_evidence("external")],
        )


def test_grader_initial_attempt_rejects_extra_evidence() -> None:
    task, evidence = _recovery_task()
    revision = task.query_revisions[-1]
    with pytest.raises(ValueError, match="evidence 上限"):
        EvidenceGrader(V2Config(), model=SequenceStructuredModel([])).grade(
            task=task,
            revision=revision,
            evidence=[*evidence.values(), _evidence("external")],
        )
