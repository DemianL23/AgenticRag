from __future__ import annotations

from dataclasses import dataclass

from agenticrag.v2.schemas import StageRunResult, SynthesizedAnswer
from eval.v2.module6 import evaluate_module6


@dataclass
class StubRun:
    stage_result: StageRunResult
    tasks: tuple = ()
    evidence: dict = None
    retrieval_results: dict = None

    def __post_init__(self) -> None:
        self.evidence = self.evidence or {}
        self.retrieval_results = self.retrieval_results or {}


class StubService:
    def run(self, question: str, *, response_language: str | None = None) -> StubRun:
        return StubRun(
            stage_result=StageRunResult(
                request_id="1e7a9c2d-4f8d-4c18-9e66-8b1de3b7b216",
                target_stage="v2_2",
                execution_status="completed",
                answer_outcome="complete",
                final_answer=SynthesizedAnswer(answer="answer"),
            )
        )


def test_module6_stage_report_is_unique_and_records_v22_metrics(tmp_path) -> None:
    report = evaluate_module6(
        "question",
        service=StubService(),
        output_root=tmp_path,
        run_id="run-1",
    )
    assert report["target_stage"] == "v2_2"
    assert report["metrics"]["global_answer_outcome"] == "complete"
    assert (tmp_path / "run-1" / "report.json").exists()
    assert (tmp_path / "run-1" / "predictions.jsonl").exists()

    try:
        evaluate_module6(
            "question",
            service=StubService(),
            output_root=tmp_path,
            run_id="run-1",
        )
    except FileExistsError:
        pass
    else:
        raise AssertionError("stage report must not overwrite an existing run")
