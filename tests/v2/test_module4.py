from __future__ import annotations

import time

import pytest

from agenticrag.v2.config import V2Config
from agenticrag.v2.grading import EvidenceGrader
from agenticrag.v2.graph import build_graph_v2_1
from agenticrag.v2.module4 import (
    Module4Service,
    materialize_retrieval_tasks,
    unsupported_routing_decision,
)
from agenticrag.v2.planning import PlanningResult
from agenticrag.v2.retrieval import RetrievalFanoutService
from agenticrag.v2.schemas import (
    ComplexityDecision,
    DecompositionResult,
    Evidence,
    EvidenceGrade,
    EvidenceOccurrence,
    RetrievalResult,
    RetrievalTask,
    TaskDraft,
)


class FakePlanner:
    def __init__(self, result: PlanningResult) -> None:
        self.result = result
        self.calls = 0

    def plan(self, question: str) -> PlanningResult:
        self.calls += 1
        return self.result


class FakeBackend:
    def __init__(
        self,
        *,
        fail: bool = False,
        fail_queries: set[str] | None = None,
        delay_queries: set[str] | None = None,
    ) -> None:
        self.fail = fail
        self.fail_queries = fail_queries or set()
        self.delay_queries = delay_queries or set()
        self.calls: list[str] = []

    def retrieve(self, **kwargs: object) -> RetrievalResult:
        query = str(kwargs["query"])
        self.calls.append(query)
        if query in self.delay_queries:
            time.sleep(0.02)
        if self.fail or query in self.fail_queries:
            raise RuntimeError("retriever unavailable")
        evidence_id = "shared" if query in {"A", "B"} else f"{query}-evidence"
        content = "shared fact" if evidence_id == "shared" else f"fact for {query}"
        return RetrievalResult(
            evidence=[
                Evidence(
                    evidence_id=evidence_id,
                    chunk_id=evidence_id,
                    content=content,
                    doc_id="doc-1",
                    source="source.pdf",
                    page=1,
                    occurrences=[
                        EvidenceOccurrence(
                            task_id=str(kwargs["task_id"]),
                            query_revision_id=str(kwargs["query_revision_id"]),
                            retrieval_attempt_id=str(kwargs["attempt_id"]),
                            strategy="original",
                            final_rank=1,
                        )
                    ],
                )
            ]
        )


class FakeStructuredModel:
    def __init__(self, output: object, *, fail: bool = False) -> None:
        self.output = output
        self.fail = fail
        self.calls = 0
        self.prompts: list[str] = []

    def with_structured_output(self, schema: object) -> "FakeStructuredModel":
        return self

    def invoke(self, prompt: str) -> object:
        self.calls += 1
        self.prompts.append(prompt)
        if self.fail:
            return {"relevance": "invalid"}
        return self.output


class QueryAwareGraderModel(FakeStructuredModel):
    def __init__(self, *, fail_queries: set[str] | None = None) -> None:
        super().__init__({})
        self.fail_queries = fail_queries or set()

    def invoke(self, prompt: str) -> object:
        self.calls += 1
        self.prompts.append(prompt)
        if "shared fact" in prompt:
            if self.fail_queries and "QR_SQ002" in prompt:
                return {"relevance": "invalid"}
            return _grade(supporting_evidence_ids=["shared"])
        for query in ("A", "B", "C"):
            if f"fact for {query}" in prompt:
                return _grade(supporting_evidence_ids=[f"{query}-evidence"])
        return _grade()


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


def _complex_plan_with_capabilities(
    *task_specs: tuple[str, str],
) -> PlanningResult:
    drafts = [
        TaskDraft(query=query, intent=f"intent {query}", capability=capability)
        for query, capability in task_specs
    ]
    return PlanningResult(
        question="complex question",
        normalized_question="complex question",
        complexity_decision=ComplexityDecision(
            complexity="complex", capability=None, reason="mixed independent tasks"
        ),
        decomposition=DecompositionResult(tasks=drafts, decomposition_complete=True),
        router_attempts=1,
        decomposer_attempts=1,
    )


def _grade(**overrides: object) -> dict[str, object]:
    value: dict[str, object] = {
        "relevance": "strong",
        "answerability": "sufficient",
        "ambiguity": "none",
        "recoverability": "none",
        "failure_reason": "none",
        "reason": "evidence is sufficient",
        "supporting_evidence_ids": ["question-evidence"],
    }
    value.update(overrides)
    return value


def test_materialization_assigns_stable_ids_and_does_not_materialize_limit() -> None:
    simple = materialize_retrieval_tasks(_simple_plan())
    assert [task.id for task in simple] == ["SQ_001"]
    complex_tasks = materialize_retrieval_tasks(_complex_plan("A", "B", "C"))
    assert [task.id for task in complex_tasks] == ["SQ_001", "SQ_002", "SQ_003"]

    limit_plan = _complex_plan("A", "B").model_copy(
        update={
            "decomposition": DecompositionResult(
                tasks=[], decomposition_complete=False, failure_reason="decomposition_limit"
            )
        }
    )
    assert materialize_retrieval_tasks(limit_plan) == []


def test_materialization_rejects_complete_over_limit_instead_of_truncating() -> None:
    over_limit = _complex_plan("A", "B", "C", "D", "E")
    with pytest.raises(ValueError, match="bounded|max_subqueries"):
        materialize_retrieval_tasks(over_limit, max_subqueries=4)


def test_unsupported_skips_retrieval_and_grader_and_records_route() -> None:
    backend = FakeBackend()
    grader_model = FakeStructuredModel(_grade())
    config = V2Config()
    result = Module4Service(
        config,
        planner=FakePlanner(_simple_plan("arithmetic")),
        retrieval=RetrievalFanoutService(backend, config),
        grader=EvidenceGrader(config, model=grader_model),
    ).run("calculation")

    task = result.tasks[0]
    assert backend.calls == []
    assert grader_model.calls == 0
    assert task.answer_outcome == "unsupported"
    assert task.routing_decisions[0].route == "unsupported"
    assert result.stage_result.execution_status == "completed"


def test_grader_prompt_is_scoped_to_current_task_and_final_evidence() -> None:
    model = FakeStructuredModel(_grade(supporting_evidence_ids=["question-evidence"]))
    config = V2Config()
    backend = FakeBackend()
    result = Module4Service(
        config,
        planner=FakePlanner(_simple_plan()),
        retrieval=RetrievalFanoutService(backend, config),
        grader=EvidenceGrader(config, model=model),
    ).run("question")
    prompt = model.prompts[0]
    assert "fact for question" in prompt
    assert "final_top5_evidence" in prompt
    assert "candidate_pool" not in prompt
    assert "rrf_top20" not in prompt
    assert "expected.route" not in prompt
    assert result.tasks[0].grade_records[0].input_evidence_ids == ["question-evidence"]


def test_grader_supporting_ids_outside_input_are_technical_failure() -> None:
    result = Module4Service(
        V2Config(),
        planner=FakePlanner(_simple_plan()),
        retrieval=RetrievalFanoutService(FakeBackend(), V2Config()),
        grader=EvidenceGrader(
            V2Config(), model=FakeStructuredModel(_grade(supporting_evidence_ids=["not-input"]))
        ),
    ).run("question")
    assert result.stage_result.execution_status == "failed"
    assert result.stage_result.answer_outcome is None
    assert result.tasks[0].error is not None


@pytest.mark.parametrize(
    ("grade", "route", "strategy"),
    [
        (_grade(), "answer", None),
        (_grade(ambiguity="missing_slot", answerability="none", supporting_evidence_ids=[], missing_slots=["year"]), "clarify", None),
        (_grade(ambiguity="multiple_candidates", answerability="none", supporting_evidence_ids=[]), "scope_select", None),
        (_grade(answerability="partial", recoverability="likely", failure_reason="irrelevant_evidence", supporting_evidence_ids=["question-evidence"]), "recover", "direct_rewrite"),
        (_grade(answerability="none", relevance="none", supporting_evidence_ids=[]), "no_knowledge", None),
    ],
)
def test_deterministic_routes_are_recorded_without_executing_recovery(
    grade: dict[str, object], route: str, strategy: str | None
) -> None:
    model = FakeStructuredModel(grade)
    result = Module4Service(
        V2Config(),
        planner=FakePlanner(_simple_plan()),
        retrieval=RetrievalFanoutService(FakeBackend(), V2Config()),
        grader=EvidenceGrader(V2Config(), model=model),
    ).run("question")
    decision = result.tasks[0].routing_decisions[0]
    assert decision.route == route
    assert decision.recovery_strategy == strategy
    assert model.calls == 1


def test_retrieval_failure_is_failed_with_null_outcome() -> None:
    result = Module4Service(
        V2Config(),
        planner=FakePlanner(_simple_plan()),
        retrieval=RetrievalFanoutService(FakeBackend(fail=True), V2Config()),
        grader=EvidenceGrader(V2Config(), model=FakeStructuredModel(_grade())),
    ).run("question")
    assert result.stage_result.execution_status == "failed"
    assert result.stage_result.answer_outcome is None
    assert result.tasks[0].execution_status == "failed"
    assert result.tasks[0].answer_outcome is None


def test_partial_retrieval_failure_does_not_short_circuit_healthy_tasks() -> None:
    config = V2Config()
    result = Module4Service(
        config,
        planner=FakePlanner(_complex_plan("A", "B", "C")),
        retrieval=RetrievalFanoutService(FakeBackend(fail_queries={"B"}), config),
        grader=EvidenceGrader(config, model=QueryAwareGraderModel()),
    ).run("complex question")

    assert result.stage_result.execution_status == "failed"
    assert result.tasks[1].execution_status == "failed"
    assert result.tasks[1].answer_outcome is None
    assert result.tasks[0].grade_records
    assert result.tasks[2].grade_records
    assert result.tasks[0].routing_decisions
    assert result.tasks[2].routing_decisions
    assert result.tasks[0].query_revisions
    assert result.tasks[2].query_revisions
    assert set(result.retrieval_results) == {"SQ_001", "SQ_003"}


def test_partial_grader_failure_does_not_short_circuit_healthy_tasks() -> None:
    config = V2Config()
    result = Module4Service(
        config,
        planner=FakePlanner(_complex_plan("A", "B", "C")),
        retrieval=RetrievalFanoutService(FakeBackend(), config),
        grader=EvidenceGrader(config, model=QueryAwareGraderModel(fail_queries={"B"})),
    ).run("complex question")

    assert result.stage_result.execution_status == "failed"
    assert result.tasks[1].execution_status == "failed"
    assert result.tasks[1].answer_outcome is None
    assert result.tasks[0].routing_decisions
    assert result.tasks[2].routing_decisions
    assert set(result.retrieval_results) == {"SQ_001", "SQ_002", "SQ_003"}
    assert set(result.evidence) == {"shared", "C-evidence"}


def test_mixed_supported_unsupported_and_failed_tasks_are_all_preserved() -> None:
    config = V2Config()
    backend = FakeBackend(fail_queries={"C"})
    grader_model = QueryAwareGraderModel()
    result = Module4Service(
        config,
        planner=FakePlanner(
            _complex_plan_with_capabilities(
                ("A", "retrieval_synthesis"),
                ("B", "arithmetic"),
                ("C", "retrieval_synthesis"),
            )
        ),
        retrieval=RetrievalFanoutService(backend, config),
        grader=EvidenceGrader(config, model=grader_model),
    ).run("mixed question")

    assert backend.calls == ["A", "C"]
    assert grader_model.calls == 1
    assert result.tasks[0].grade_records and result.tasks[0].routing_decisions
    assert result.tasks[1].answer_outcome == "unsupported"
    assert result.tasks[1].routing_decisions[0].route == "unsupported"
    assert result.tasks[2].execution_status == "failed"
    assert result.tasks[2].answer_outcome is None
    assert result.stage_result.execution_status == "failed"


def test_grader_failure_is_failed_with_null_outcome() -> None:
    result = Module4Service(
        V2Config(),
        planner=FakePlanner(_simple_plan()),
        retrieval=RetrievalFanoutService(FakeBackend(), V2Config()),
        grader=EvidenceGrader(V2Config(), model=FakeStructuredModel({}, fail=True)),
    ).run("question")
    assert result.stage_result.execution_status == "failed"
    assert result.stage_result.answer_outcome is None
    assert result.tasks[0].error is not None
    assert result.tasks[0].error.code == "structured_output_invalid"


def test_complex_graph_is_stable_and_deduplicates_occurrences() -> None:
    backend = FakeBackend(delay_queries={"A"})
    grade_model = FakeStructuredModel(_grade(supporting_evidence_ids=["shared"]))
    config = V2Config()
    result = Module4Service(
        config,
        planner=FakePlanner(_complex_plan("A", "B")),
        retrieval=RetrievalFanoutService(backend, config),
        grader=EvidenceGrader(config, model=grade_model),
    ).run("complex question")
    assert [task.id for task in result.tasks] == ["SQ_001", "SQ_002"]
    assert list(result.evidence) == ["shared"]
    assert len(result.evidence["shared"].occurrences) == 2
    assert [task.routing_decisions[0].route for task in result.tasks] == ["answer", "answer"]
    assert result.stage_result.execution_status == "completed"


def test_graph_builder_can_run_explicit_simple_path() -> None:
    planner = FakePlanner(_simple_plan("arithmetic"))
    graph = build_graph_v2_1(
        config=V2Config(),
        planner=planner,
        retrieval=RetrievalFanoutService(FakeBackend(), V2Config()),
        grader=EvidenceGrader(V2Config(), model=FakeStructuredModel(_grade())),
    )
    state = graph.invoke(
        {
            "request_id": "1e7a9c2d-4f8d-4c18-9e66-8b1de3b7b216",
            "target_stage": "v2_1",
            "original_question": "question",
            "normalized_query": "question",
            "response_language": "en",
            "complexity_decision": None,
            "planning_result": None,
            "task_order": [],
            "tasks": {},
            "retrieval_results": {},
            "evidence": {},
            "pending_hitl_request": None,
            "hitl_rounds": 0,
            "execution_status": "running",
            "answer_outcome": None,
            "final_answer": None,
            "error": None,
            "state_schema_version": "v2.1",
            "stage_result": None,
        }
    )
    assert state["stage_result"].execution_status == "completed"
    assert state["tasks"]["SQ_001"].answer_outcome == "unsupported"
