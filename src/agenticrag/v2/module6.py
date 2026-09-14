"""V2.2 application boundary for Recovery, Findings, and final answers."""

from __future__ import annotations

from dataclasses import dataclass

from .answering import FindingGenerator, SynthesisGenerator
from .config import V2Config
from .grading import EvidenceGrader
from .planning import PlanningResult, PlanningService
from .recovery import RecoveryService
from .retrieval import RetrievalFanoutService, V12RetrievalAdapter
from .schemas import Evidence, RetrievalResult, RetrievalTask, StageRunResult


@dataclass(frozen=True, slots=True)
class Module6Run:
    """Serializable output of a non-persistent V2.2 run."""

    planning_result: PlanningResult | None
    tasks: tuple[RetrievalTask, ...]
    evidence: dict[str, Evidence]
    retrieval_results: dict[str, RetrievalResult]
    stage_result: StageRunResult


class Module6Service:
    """Run the explicit V2.2 graph without persistence or resume."""

    def __init__(
        self,
        config: V2Config | None = None,
        *,
        planner: PlanningService | None = None,
        retrieval: RetrievalFanoutService | None = None,
        grader: EvidenceGrader | None = None,
        recovery: RecoveryService | None = None,
        finding: FindingGenerator | None = None,
        synthesis: SynthesisGenerator | None = None,
    ) -> None:
        self.config = config or V2Config.from_env()
        self.planner = planner or PlanningService(self.config)
        self.retrieval = retrieval or RetrievalFanoutService(
            V12RetrievalAdapter(), self.config
        )
        self.grader = grader or EvidenceGrader(self.config)
        self.recovery = recovery or RecoveryService(self.config)
        self.finding = finding or FindingGenerator(self.config)
        self.synthesis = synthesis or SynthesisGenerator(self.config)

    def run(
        self,
        question: str,
        *,
        request_id: str | None = None,
        response_language: str | None = None,
    ) -> Module6Run:
        from .graph import build_graph_v2_2, initial_v2_2_state

        graph = build_graph_v2_2(
            config=self.config,
            planner=self.planner,
            retrieval=self.retrieval,
            grader=self.grader,
            recovery=self.recovery,
            finding=self.finding,
            synthesis=self.synthesis,
        )
        initial = initial_v2_2_state(
            question,
            request_id=request_id,
            response_language=response_language,
        )
        state = graph.invoke(initial)
        stage_result = state["stage_result"]
        assert stage_result is not None
        return Module6Run(
            planning_result=state.get("planning_result"),
            tasks=tuple(
                sorted(
                    state.get("tasks", {}).values(),
                    key=lambda task: (task.ordinal, task.id),
                )
            ),
            evidence=state.get("evidence", {}),
            retrieval_results=state.get("retrieval_results", {}),
            stage_result=stage_result,
        )
