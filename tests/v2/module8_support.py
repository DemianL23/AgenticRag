from __future__ import annotations

from agenticrag.v2.answering import HITLContentGenerator
from agenticrag.v2.config import V2Config
from agenticrag.v2.graph import initial_v2_3_state
from agenticrag.v2.hitl import HITLResumeService
from agenticrag.v2.planning import PlanningResult
from agenticrag.v2.retrieval import RetrievalFanoutService
from agenticrag.v2.schemas import (
    ComplexityDecision,
    Evidence,
    EvidenceGrade,
    EvidenceOccurrence,
    GroundedFinding,
    RetrievalLatency,
    RetrievalResult,
)


class FakePlanner:
    def __init__(self, result: PlanningResult) -> None:
        self.result = result

    def plan(self, question: str) -> PlanningResult:
        return self.result


class FakeBackend:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def retrieve(self, **kwargs: object) -> RetrievalResult:
        self.calls.append(kwargs)
        strategy = str(kwargs["strategy"])
        evidence_id = "e-original" if strategy == "original" else "e-clarified"
        return RetrievalResult(
            evidence=[
                Evidence(
                    evidence_id=evidence_id,
                    chunk_id=evidence_id,
                    content=f"fact {evidence_id}",
                    doc_id="doc-1",
                    source="fixture.pdf",
                    page=1,
                    occurrences=[
                        EvidenceOccurrence(
                            task_id=str(kwargs["task_id"]),
                            query_revision_id=str(kwargs["query_revision_id"]),
                            retrieval_attempt_id=str(kwargs["attempt_id"]),
                            strategy=str(kwargs["strategy"]),
                            final_rank=1,
                        )
                    ],
                )
            ],
            latency=RetrievalLatency(total_seconds=0.01),
            trace_ref=str(kwargs["attempt_id"]),
            diagnostics={"candidate_pool": ["must stay out of checkpoint"]},
        )


class SequenceGrader:
    def __init__(self, *, initial_missing: bool = True) -> None:
        self.calls: list[dict[str, object]] = []
        self.initial_missing = initial_missing

    def grade(self, **kwargs: object) -> tuple[EvidenceGrade, int]:
        self.calls.append(kwargs)
        evidence = kwargs["evidence"]
        ids = [item.evidence_id for item in evidence]
        if self.initial_missing and len(self.calls) == 1:
            return (
                EvidenceGrade(
                    relevance="none",
                    answerability="none",
                    ambiguity="missing_slot",
                    recoverability="none",
                    failure_reason="none",
                    reason="year is missing",
                    missing_slots=["year"],
                ),
                1,
            )
        return (
            EvidenceGrade(
                relevance="strong",
                answerability="sufficient",
                ambiguity="none",
                recoverability="none",
                failure_reason="none",
                reason="sufficient",
                supporting_evidence_ids=ids[:1],
            ),
            1,
        )


class FakeFinding:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def generate(self, **kwargs: object) -> tuple[GroundedFinding, int]:
        self.calls.append(kwargs)
        task = kwargs["task"]
        support = task.grade_records[-1].grade.supporting_evidence_ids
        return GroundedFinding(task_id=task.id, text="grounded answer", evidence_ids=support[:1]), 1


class FakeStructuredModel:
    def __init__(self, outputs: list[object]) -> None:
        self.outputs = list(outputs)
        self.calls = 0

    def with_structured_output(self, _schema: object) -> "FakeStructuredModel":
        return self

    def invoke(self, _prompt: str) -> object:
        self.calls += 1
        output = self.outputs.pop(0)
        if isinstance(output, Exception):
            raise output
        return output


def simple_plan() -> PlanningResult:
    return PlanningResult.from_router(
        question="Which year?",
        decision=ComplexityDecision(
            complexity="simple",
            capability="retrieval_synthesis",
            reason="one fact",
        ),
        router_attempts=1,
    )


def make_runtime(config: V2Config, *, resume_mode: bool = False):
    from agenticrag.v2.durable import DurableV23Service

    backend = FakeBackend()
    grader = SequenceGrader(initial_missing=not resume_mode)
    finding = FakeFinding()
    hitl_model = FakeStructuredModel([{"question": "Which year?"}])
    hitl = HITLContentGenerator(config, model=hitl_model)
    resume = HITLResumeService(
        config,
        backend=backend,
        grader=grader,
        finding=finding,
    )
    service = DurableV23Service(
        config,
        planner=FakePlanner(simple_plan()),
        retrieval=RetrievalFanoutService(backend, config),
        grader=grader,
        finding=finding,
        hitl=hitl,
        resume_service=resume,
    )
    return service, backend, grader, finding, hitl_model
