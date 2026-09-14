from __future__ import annotations

import hashlib
from pathlib import Path

from agenticrag.v2.config import V2Config
from agenticrag.v2.grading import EvidenceGrader
from agenticrag.v2.module4 import Module4Service
from agenticrag.v2.planning import PlanningResult
from agenticrag.v2.retrieval import RetrievalFanoutService
from agenticrag.v2.schemas import ComplexityDecision, Evidence, EvidenceOccurrence, RetrievalResult
from eval.v2.module4 import check_module4_invariants, evaluate_module4, load_manifest


class _SimplePlanner:
    def plan(self, question: str) -> PlanningResult:
        return PlanningResult.from_router(
            question=question,
            decision=ComplexityDecision(
                complexity="simple", capability="retrieval_synthesis", reason="test"
            ),
            router_attempts=1,
        )


class _SuccessfulBackend:
    def retrieve(self, **kwargs: object) -> RetrievalResult:
        task_id = str(kwargs["task_id"])
        evidence_id = f"e-{task_id}"
        return RetrievalResult(
            evidence=[
                Evidence(
                    evidence_id=evidence_id,
                    chunk_id=evidence_id,
                    content="test evidence",
                    doc_id="doc",
                    source="source",
                    page=1,
                    occurrences=[
                        EvidenceOccurrence(
                            task_id=task_id,
                            query_revision_id=str(kwargs["query_revision_id"]),
                            retrieval_attempt_id=str(kwargs["attempt_id"]),
                            strategy="original",
                            final_rank=1,
                        )
                    ],
                )
            ]
        )


class _InvalidGraderModel:
    def with_structured_output(self, schema: object) -> "_InvalidGraderModel":
        return self

    def invoke(self, prompt: str) -> object:
        return {"relevance": "invalid"}


def test_module4_manifest_is_current_19_sample_contract() -> None:
    path = Path("eval/datasets/v2_module4_real_eval.jsonl")
    records = load_manifest(path)
    assert len(records) == 19
    assert sum(record["expected"]["route"] == "unsupported" for record in records) == 11
    assert sum(record["retrieval"]["expected_to_run"] is True for record in records) == 8
    assert sum(record["evaluation_incomplete"] is True for record in records) == 8


def test_module4_manifest_digest_is_recordable_without_modifying_source() -> None:
    path = Path("eval/datasets/v2_module4_real_eval.jsonl")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert len(digest) == 64


def test_eval_counts_successful_retrieval_even_when_grader_fails(tmp_path: Path) -> None:
    config = V2Config()
    service = Module4Service(
        config,
        planner=_SimplePlanner(),
        retrieval=RetrievalFanoutService(_SuccessfulBackend(), config),
        grader=EvidenceGrader(config, model=_InvalidGraderModel()),
    )
    report = evaluate_module4(
        output_root=tmp_path,
        service=service,
        run_id="retrieval-count-test",
    )

    assert report["metrics"]["retrieval_completed_count"] == 8
    assert report["metrics"]["technical_failure_count"] == 8
    assert report["metrics"]["invariant_violation_count"] == 0


def test_module4_invariant_checker_reports_contract_violations() -> None:
    violations = check_module4_invariants(
        {
            "tasks": [
                {
                    "id": "SQ_001",
                    "capability": "arithmetic",
                    "execution_status": "completed",
                    "answer_outcome": "unsupported",
                    "query_revisions": [],
                    "grade_records": [{"id": "GR_001"}],
                    "routing_decisions": [{"route": "unsupported", "grade_record_id": "GR_001"}],
                }
            ],
            "task_order": ["SQ_001"],
            "retrieval_results": {},
        }
    )

    assert "SQ_001:unsupported_has_grade" in violations
    assert "SQ_001:unsupported_route_has_grade" in violations
