from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

from agenticrag.v2.config import V2Config
from agenticrag.v2.schemas import Evidence, StageRunResult, SynthesizedAnswer
from eval.v2.answer_baseline import evaluate_v2_answers


class _Service:
    def run(self, question: str):
        if question == "supported":
            final = SynthesizedAnswer(
                answer="supported answer", citation_evidence_ids=["E1"], limitations=[]
            )
            return SimpleNamespace(
                stage_result=StageRunResult(
                    request_id="c3936bed-3744-441a-8739-d31580780f88",
                    target_stage="v2_2",
                    execution_status="completed",
                    answer_outcome="complete",
                    final_answer=final,
                ),
                tasks=[SimpleNamespace(capability="retrieval_synthesis")],
                evidence={
                    "E1": Evidence(
                        evidence_id="E1",
                        chunk_id="E1",
                        content="supported context",
                        doc_id="d",
                        source="s",
                        page=1,
                    )
                },
            )
        return SimpleNamespace(
            stage_result=StageRunResult(
                request_id="b7888c5d-823a-44fd-8794-c37507cc3b13",
                target_stage="v2_2",
                execution_status="completed",
                answer_outcome="unsupported",
                final_answer=SynthesizedAnswer(
                    answer="unsupported", citation_evidence_ids=[], limitations=[]
                ),
            ),
            tasks=[SimpleNamespace(capability="arithmetic")],
            evidence={},
        )


class _Evaluator:
    metric_names = ("faithfulness",)

    async def evaluate(self, **_kwargs):
        return SimpleNamespace(
            scores={"faithfulness": 1.0}, reasons={}, errors={}
        )


def test_v2_answer_evaluator_scores_supported_ragas_and_outcomes(
    tmp_path: Path,
) -> None:
    qa = tmp_path / "qa.jsonl"
    annotations = tmp_path / "annotations.jsonl"
    qa.write_text(
        json.dumps({"finqa_id": "a", "question": "supported", "gold": "answer"})
        + "\n"
        + json.dumps({"finqa_id": "b", "question": "compute", "gold": "1"})
        + "\n"
    )
    annotations.write_text(
        json.dumps(
            {
                "finqa_id": "a",
                "complexity": "simple",
                "capability": "retrieval_synthesis",
                "required_information_units": [
                    {"description": "fact", "expected_capability": None}
                ],
                "expected_outcome": "complete",
            }
        )
        + "\n"
        + json.dumps(
            {
                "finqa_id": "b",
                "complexity": "simple",
                "capability": "arithmetic",
                "required_information_units": [
                    {"description": "calculation", "expected_capability": None}
                ],
                "expected_outcome": "unsupported",
            }
        )
        + "\n"
    )

    report = asyncio.run(
        evaluate_v2_answers(
            qa_path=qa,
            annotation_path=annotations,
            config=V2Config(),
            service=_Service(),
            evaluator=_Evaluator(),
        )
    )

    assert report["producer"] == "v2_answer_ragas_evaluator"
    assert report["metrics"]["supported_subset_ragas"]["faithfulness"] == 1.0
    assert report["metrics"]["outcome_accuracy"] == 1.0
    assert report["metrics"]["unsupported_computation_recall"] == 1.0
    assert report["metrics"]["abstention_correctness"] == 1.0
    assert report["evaluation_incomplete"] is False
