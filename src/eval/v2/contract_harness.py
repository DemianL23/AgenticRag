"""Deterministic contract harness for the final V2 baseline.

This module is evaluation-only.  Fixtures select injected collaborators; they
never add scenario branches to ``src/agenticrag/v2``.  The real V2 graph and
application services still own planning, retrieval, grading, routing,
recovery, HITL, finding, and synthesis semantics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Any

from agenticrag.v2.answering import FindingGenerator, HITLContentGenerator, SynthesisGenerator
from agenticrag.v2.config import V2Config
from agenticrag.v2.grading import EvidenceGradingError
from agenticrag.v2.hitl import HITLResumeService
from agenticrag.v2.module4 import Module4Service
from agenticrag.v2.module6 import Module6Service
from agenticrag.v2.planning import PlanningResult
from agenticrag.v2.recovery import RecoveryArtifact, RecoveryService
from agenticrag.v2.retrieval import RetrievalFanoutService
from agenticrag.v2.schemas import (
    ComplexityDecision,
    DecompositionResult,
    Evidence,
    EvidenceGrade,
    EvidenceOccurrence,
    ExecutionError,
    GroundedFinding,
    RetrievalResult,
    RetrievalTask,
    RetrievalLatency,
    SynthesizedAnswer,
    StageRunResult,
    ResumeRequest,
    TaskDraft,
)
from agenticrag.v2.types import RecoveryStrategy, Route, TaskCapability


@dataclass(slots=True)
class HarnessTelemetry:
    backend_calls: list[dict[str, Any]] = field(default_factory=list)
    grader_calls: list[dict[str, Any]] = field(default_factory=list)
    finding_calls: list[str] = field(default_factory=list)
    synthesis_calls: int = 0
    hitl_calls: int = 0


class DeterministicPlanner:
    def __init__(self, *, complexity: str, capability: str) -> None:
        self.complexity = complexity
        self.capability = capability

    def plan(self, question: str) -> PlanningResult:
        if self.complexity == "simple":
            decision = ComplexityDecision(complexity="simple", capability=self.capability, reason="deterministic fixture")
            return PlanningResult.from_router(question=question, decision=decision, router_attempts=1)
        decision = ComplexityDecision(complexity="complex", capability=None, reason="deterministic fixture")
        if self.capability == "mixed":
            capabilities = ["retrieval_synthesis", "arithmetic"]
        elif self.capability == "retrieval_synthesis":
            capabilities = ["retrieval_synthesis", "retrieval_synthesis"]
        else:
            capabilities = [self.capability, self.capability]
        decomposition = DecompositionResult(
            decomposition_complete=True,
            tasks=[
                TaskDraft(query=f"{question} part {index}", intent="answer the requested fact", capability=cap)
                for index, cap in enumerate(capabilities, 1)
            ],
        )
        return PlanningResult.from_router(question=question, decision=decision, router_attempts=1).model_copy(
            update={"decomposition": decomposition, "decomposer_attempts": 1}
        )


class DeterministicBackend:
    def __init__(self, telemetry: HarnessTelemetry, *, fault: str) -> None:
        self.telemetry = telemetry
        self.fault = fault

    def retrieve(self, **kwargs: Any) -> RetrievalResult:
        self.telemetry.backend_calls.append(dict(kwargs))
        if self.fault == "retrieval":
            raise RuntimeError("deterministic retrieval fault")
        if self.fault == "timeout":
            raise TimeoutError("deterministic retrieval timeout")
        evidence_id = f"E_{kwargs['task_id']}_{len(self.telemetry.backend_calls):03d}"
        evidence = Evidence(
            evidence_id=evidence_id,
            chunk_id=evidence_id,
            content=f"deterministic evidence for {kwargs['query']}",
            doc_id="contract-doc",
            source="contract-fixture",
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
        return RetrievalResult(
            evidence=[evidence],
            latency=RetrievalLatency(total_seconds=0.001),
            trace_ref=str(kwargs["attempt_id"]),
        )


class DeterministicGrader:
    def __init__(
        self,
        telemetry: HarnessTelemetry,
        *,
        route_sequence: list[Route],
        strategy: RecoveryStrategy | None,
        fault: str,
    ) -> None:
        self.telemetry = telemetry
        self.route_sequence = route_sequence
        self.strategy = strategy
        self.fault = fault
        self._calls: dict[str, int] = {}

    def grade(self, **kwargs: Any) -> tuple[EvidenceGrade, int]:
        task: RetrievalTask = kwargs["task"]
        evidence = kwargs["evidence"]
        self.telemetry.grader_calls.append(dict(kwargs))
        if self.fault == "provider":
            raise RuntimeError("deterministic grader provider fault")
        if self.fault == "schema":
            raise EvidenceGradingError(
                attempts=2,
                cause=ValueError("deterministic structured output fault"),
            )
        call_number = self._calls.get(task.id, 0) + 1
        self._calls[task.id] = call_number
        route = self.route_sequence[min(call_number - 1, len(self.route_sequence) - 1)]
        if route == "recover" or (
            call_number > 1
            and self.route_sequence
            and self.route_sequence[0] == "recover"
            and route == "no_knowledge"
        ):
            failure_reason = {
                "direct_rewrite": "insufficient_coverage",
                "step_back": "overly_specific",
                "hyde": "terminology_gap",
            }[self.strategy or "direct_rewrite"]
            return EvidenceGrade(
                relevance="weak",
                answerability="partial",
                ambiguity="none",
                recoverability="likely",
                failure_reason=failure_reason,
                reason="deterministic recovery precondition",
                missing_information=["requested fact"],
                supporting_evidence_ids=[item.evidence_id for item in evidence],
            ), 1
        if route == "clarify":
            return EvidenceGrade(
                relevance="weak", answerability="none", ambiguity="missing_slot",
                recoverability="none", failure_reason="none", reason="missing year",
                missing_slots=["year"],
            ), 1
        if route == "scope_select":
            return EvidenceGrade(
                relevance="weak", answerability="none", ambiguity="multiple_candidates",
                recoverability="none", failure_reason="none", reason="multiple scopes",
            ), 1
        if route == "no_knowledge":
            return EvidenceGrade(
                relevance="none", answerability="none", ambiguity="none",
                recoverability="none", failure_reason="none", reason="no supported fact",
            ), 1
        return EvidenceGrade(
            relevance="strong", answerability="sufficient", ambiguity="none",
            recoverability="none", failure_reason="none", reason="deterministic supported fact",
            supporting_evidence_ids=[item.evidence_id for item in evidence],
        ), 1


class DeterministicRewrite:
    def __init__(self, strategy: RecoveryStrategy) -> None:
        self.strategy = strategy
        self.calls = 0

    def generate(self, **kwargs: Any) -> tuple[RecoveryArtifact, int]:
        self.calls += 1
        task: RetrievalTask = kwargs["task"]
        return RecoveryArtifact(
            strategy=kwargs["strategy"],
            retrieval_query=f"{task.query} corrective retrieval",
            is_evidence=False,
        ), 1


class DeterministicFinding:
    def __init__(self, telemetry: HarnessTelemetry) -> None:
        self.telemetry = telemetry

    def generate(self, *, task: RetrievalTask, evidence_by_id: dict[str, Evidence], response_language: str, role: str = "finding") -> tuple[GroundedFinding, int]:
        self.telemetry.finding_calls.append(task.id)
        ids = list(task.grade_records[-1].grade.supporting_evidence_ids)
        return GroundedFinding(task_id=task.id, text=f"supported finding for {task.query}", evidence_ids=ids[:3]), 1


class DeterministicSynthesis:
    def __init__(self, telemetry: HarnessTelemetry) -> None:
        self.telemetry = telemetry

    def generate(self, **kwargs: Any) -> tuple[SynthesizedAnswer, int]:
        self.telemetry.synthesis_calls += 1
        findings = kwargs.get("findings", [])
        citations = [evidence_id for finding in findings for evidence_id in finding.evidence_ids]
        return SynthesizedAnswer(answer="deterministic synthesized answer", citation_evidence_ids=list(dict.fromkeys(citations)), limitations=kwargs.get("limitations", [])), 1


class DeterministicHitlModel:
    def __init__(self, telemetry: HarnessTelemetry, *, scope: bool) -> None:
        self.telemetry = telemetry
        self.scope = scope

    def with_structured_output(self, _schema: object) -> "DeterministicHitlModel":
        return self

    def invoke(self, _prompt: str) -> object:
        self.telemetry.hitl_calls += 1
        if self.scope:
            evidence_id = next(iter(re.findall(r"E_SQ_\d{3}_\d{3}", _prompt)), "E_SQ_001_001")
            return {
                "options": [
                    {
                        "label": "Scope A",
                        "value": "scope A",
                        "description": "deterministic scope A",
                        "evidence_ids": [evidence_id],
                    },
                    {
                        "label": "Scope B",
                        "value": "scope B",
                        "description": "deterministic scope B",
                        "evidence_ids": [evidence_id],
                    },
                ]
            }
        return {"question": "Which reporting year should be used?"}


@dataclass(frozen=True, slots=True)
class HarnessResult:
    result: Any
    telemetry: HarnessTelemetry
    backend: DeterministicBackend


def run_contract_scenario(scenario: Any, config: V2Config) -> HarnessResult:
    """Execute a contract fixture through the frozen application graph."""
    route = scenario.fixture.route_sequence[0] if scenario.fixture.route_sequence else scenario.expected.route
    strategy = scenario.fixture.recovery_strategies[0] if scenario.fixture.recovery_strategies else scenario.expected.recovery_strategy
    capability = scenario.capability if scenario.capability != "mixed" else "mixed"
    telemetry = HarnessTelemetry()
    backend = DeterministicBackend(telemetry, fault=scenario.fault)
    planner = DeterministicPlanner(complexity=scenario.complexity, capability=capability)
    route_sequence = list(scenario.fixture.route_sequence) or [route or "answer"]
    grader = DeterministicGrader(
        telemetry,
        route_sequence=route_sequence,
        strategy=strategy,
        fault=scenario.fault,
    )
    retrieval = RetrievalFanoutService(backend, config)
    if scenario.stage == "v2_1":
        service = Module4Service(config, planner=planner, retrieval=retrieval, grader=grader)
        return HarnessResult(service.run(scenario.question, response_language=_language(scenario)), telemetry, backend)
    hitl = None
    if route in {"clarify", "scope_select"}:
        hitl = HITLContentGenerator(config, model=DeterministicHitlModel(telemetry, scope=route == "scope_select"))
    recovery = RecoveryService(
        config,
        generator=DeterministicRewrite(strategy or "direct_rewrite"),
        backend=backend,
        grader=grader,
    )
    if scenario.stage == "v2_3":
        from langgraph.checkpoint.memory import MemorySaver
        from langgraph.types import Command

        from agenticrag.v2.graph import build_graph_v2_3, initial_v2_3_state

        finding = DeterministicFinding(telemetry)
        synthesis = DeterministicSynthesis(telemetry)
        resume_service = HITLResumeService(
            config,
            backend=backend,
            grader=grader,
            recovery=recovery,
            finding=finding,
            synthesis=synthesis,
        )
        graph = build_graph_v2_3(
            checkpointer=MemorySaver(),
            config=config,
            planner=planner,
            retrieval=retrieval,
            grader=grader,
            recovery=recovery,
            finding=finding,
            synthesis=synthesis,
            hitl=hitl,
            resume_service=resume_service,
        )
        initial = initial_v2_3_state(
            scenario.question,
            response_language=_language(scenario),
        )
        graph_config = {"configurable": {"thread_id": initial["request_id"]}}
        graph.invoke(initial, config=graph_config)
        if scenario.fixture.resume_request is not None:
            interrupted_state = graph.get_state(graph_config).values
            pending = interrupted_state["pending_hitl_request"]
            responses = []
            for item in pending.items:
                if item.action == "clarify":
                    responses.append(
                        {
                            "item_id": item.id,
                            "clarify_values": {
                                slot: "2019" for slot in item.missing_slots
                            },
                        }
                    )
                else:
                    responses.append(
                        {
                            "item_id": item.id,
                            "selected_option_id": item.scope_options[0].id,
                        }
                    )
            request = ResumeRequest(
                request_id=interrupted_state["request_id"],
                hitl_request_id=pending.id,
                responses=responses,
            )
            graph.invoke(Command(resume=request.model_dump(mode="json")), config=graph_config)
        state = graph.get_state(graph_config).values
        stage = StageRunResult(
            request_id=state["request_id"],
            target_stage="v2_3",
            execution_status=state["execution_status"],
            answer_outcome=state["answer_outcome"],
            final_answer=state["final_answer"],
            resumable=state["execution_status"] == "waiting_user",
            pending_hitl_request=state["pending_hitl_request"],
            error=state["error"],
        )
        result = type("HarnessGraphResult", (), {})()
        result.tasks = tuple(state["tasks"].values())
        result.evidence = state["evidence"]
        result.stage_result = stage
        result.state = state
        return HarnessResult(result, telemetry, backend)
    service = Module6Service(
        config,
        planner=planner,
        retrieval=retrieval,
        grader=grader,
        recovery=recovery,
        finding=DeterministicFinding(telemetry),
        synthesis=DeterministicSynthesis(telemetry),
        hitl=hitl,
    )
    return HarnessResult(service.run(scenario.question, response_language=_language(scenario)), telemetry, backend)


def _language(scenario: Any) -> str:
    return scenario.language if scenario.language in {"zh", "en"} else "en"
