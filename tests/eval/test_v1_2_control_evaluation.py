import asyncio
import json
from dataclasses import dataclass, replace
from pathlib import Path

from agenticrag.generation.schemas import GeneratedAnswer
from agenticrag.retrieval.reranking_retriever import RerankingSearchTrace
from agenticrag.retrieval.schemas import HybridRetrievedChunk, RerankedChunk
from eval.ragas.evaluator import RagasMetricResult
from eval.ragas import runner as ragas_runner
from eval.ragas.runner import evaluate_end_to_end


def _candidate() -> HybridRetrievedChunk:
    return HybridRetrievedChunk(
        content="control evidence",
        score=0.01,
        doc_id="doc_001",
        source="report.pdf",
        page=2,
        chunk_id="doc_001:p0002:c000",
        dense_rank=1,
        bm25_rank=2,
        rrf_score=0.02,
        rrf_rank=1,
    )


def _trace() -> RerankingSearchTrace:
    candidate = _candidate()
    result = RerankedChunk(
        content=candidate.content,
        score=2.5,
        doc_id=candidate.doc_id,
        source=candidate.source,
        page=candidate.page,
        chunk_id=candidate.chunk_id,
        dense_rank=candidate.dense_rank,
        bm25_rank=candidate.bm25_rank,
        rrf_score=candidate.rrf_score,
        rrf_rank=candidate.rrf_rank,
        rerank_score=2.5,
        final_rank=1,
    )
    return RerankingSearchTrace(
        results=(result,),
        candidate_pool=(candidate,),
        rrf_top20=(candidate,),
        fallback_used=False,
        fallback_reason=None,
        invalid_scores=False,
        candidate_seconds=0.1,
        model_load_seconds=0.2,
        rerank_seconds=0.3,
        total_seconds=0.6,
    )


@dataclass
class FakeControlService:
    trace: RerankingSearchTrace

    def answer_with_trace(self, query: str, *, k: int = 5) -> object:
        return type(
            "ControlTrace",
            (),
            {
                "answer": GeneratedAnswer(answer="control answer", citations=()),
                "retrieved_chunks": self.trace.results,
                "retrieval_trace": self.trace,
            },
        )()

    def retrieval_record(self) -> dict[str, object]:
        return {"model_name": "BAAI/bge-reranker-v2-m3", "resolved_revision": "pin"}


class FakeEvaluator:
    metric_names = ("faithfulness",)

    async def evaluate(self, **kwargs: object) -> RagasMetricResult:
        return RagasMetricResult(scores={"faithfulness": 1.0}, reasons={}, errors={})


class RaisingEvaluator:
    metric_names = ("faithfulness",)

    async def evaluate(self, **kwargs: object) -> RagasMetricResult:
        raise RuntimeError("judge unavailable")


class PartiallyScoredEvaluator:
    metric_names = ("faithfulness", "answer_correctness")

    async def evaluate(self, **kwargs: object) -> RagasMetricResult:
        return RagasMetricResult(
            scores={"faithfulness": 1.0, "answer_correctness": None},
            reasons={"faithfulness": "supported"},
            errors={"answer_correctness": "judge metric failed"},
        )


def test_control_report_contains_v12_trace_and_aggregate_diagnostics(
    tmp_path: Path,
) -> None:
    dataset = tmp_path / "qa.jsonl"
    dataset.write_text(
        json.dumps({"id": "q1", "question": "问题", "gold": "答案"}) + "\n",
        encoding="utf-8",
    )

    report = asyncio.run(
        evaluate_end_to_end(
            dataset,
            FakeControlService(_trace()),  # type: ignore[arg-type]
            FakeEvaluator(),
            top_k=5,
            limit=None,
            generation_model="qwen-plus",
            embedding_model="qwen-embedding",
            evaluator_model="qwen-max",
            evaluator_embedding_model="qwen-embedding",
            ragas_version="0.4.3",
            report_schema_version=2,
            report_name="v1_2_end_to_end_control",
            run_id="run-1",
            git_commit="commit-1",
            resolved_config={"retrieval_pipeline": {"final_top_k": 5}},
        )
    )

    sample_trace = report.samples[0].retrieval_trace
    assert report.report_name == "v1_2_end_to_end_control"
    assert report.run_id == "run-1"
    assert report.git_commit == "commit-1"
    assert report.retrieval_record == {
        "model_name": "BAAI/bge-reranker-v2-m3",
        "resolved_revision": "pin",
    }
    assert sample_trace is not None
    assert sample_trace["results"][0]["final_rank"] == 1
    assert sample_trace["results"][0]["rerank_score"] == 2.5
    assert report.retrieval_summary == {
        "traced_queries": 1,
        "retrieval_degraded_queries": 0,
        "fallback_reason_counts": {},
        "total_candidate_seconds": 0.1,
        "total_model_load_seconds": 0.2,
        "total_rerank_seconds": 0.3,
        "total_retrieval_seconds": 0.6,
        "mean_candidate_seconds": 0.1,
        "mean_model_load_seconds": 0.2,
        "mean_rerank_seconds": 0.3,
        "mean_retrieval_seconds": 0.6,
    }


def test_evaluator_failure_keeps_generated_answer_and_retrieval_trace(
    tmp_path: Path,
) -> None:
    dataset = tmp_path / "qa.jsonl"
    dataset.write_text(
        json.dumps({"id": "q1", "question": "问题", "gold": "答案"}) + "\n",
        encoding="utf-8",
    )

    report = asyncio.run(
        evaluate_end_to_end(
            dataset,
            FakeControlService(_trace()),  # type: ignore[arg-type]
            RaisingEvaluator(),
            top_k=5,
            limit=None,
            generation_model="qwen-plus",
            embedding_model="qwen-embedding",
            evaluator_model="qwen-max",
            evaluator_embedding_model="qwen-embedding",
            ragas_version="0.4.3",
        )
    )

    sample = report.samples[0]
    assert sample.generated_answer == "control answer"
    assert sample.retrieved_chunk_ids == ("doc_001:p0002:c000",)
    assert sample.retrieved_contexts == ("control evidence",)
    assert sample.retrieval_trace is not None
    assert sample.evaluation_error == {"evaluator": "RuntimeError: judge unavailable"}
    assert sample.metrics["faithfulness"] is None


def test_control_report_counts_reranker_fallbacks(tmp_path: Path) -> None:
    dataset = tmp_path / "qa.jsonl"
    dataset.write_text(
        json.dumps({"id": "q1", "question": "问题", "gold": "答案"}) + "\n",
        encoding="utf-8",
    )

    report = asyncio.run(
        evaluate_end_to_end(
            dataset,
            FakeControlService(
                replace(
                    _trace(),
                    fallback_used=True,
                    fallback_reason="inference_error",
                )
            ),  # type: ignore[arg-type]
            FakeEvaluator(),
            top_k=5,
            limit=None,
            generation_model="qwen-plus",
            embedding_model="qwen-embedding",
            evaluator_model="qwen-max",
            evaluator_embedding_model="qwen-embedding",
            ragas_version="0.4.3",
        )
    )

    assert report.retrieval_summary is not None
    assert report.retrieval_summary["retrieval_degraded_queries"] == 1
    assert report.retrieval_summary["fallback_reason_counts"] == {
        "inference_error": 1
    }


def test_numeric_failure_keeps_ragas_scores_and_merges_errors(
    tmp_path: Path,
    monkeypatch,
) -> None:
    dataset = tmp_path / "qa.jsonl"
    dataset.write_text(
        json.dumps({"id": "q1", "question": "问题", "gold": "答案"}) + "\n",
        encoding="utf-8",
    )

    def raise_numeric_error(*args: object, **kwargs: object) -> float | None:
        raise RuntimeError("numeric parser failed")

    monkeypatch.setattr(ragas_runner, "numeric_correctness", raise_numeric_error)
    report = asyncio.run(
        evaluate_end_to_end(
            dataset,
            FakeControlService(_trace()),  # type: ignore[arg-type]
            PartiallyScoredEvaluator(),
            top_k=5,
            limit=None,
            generation_model="qwen-plus",
            embedding_model="qwen-embedding",
            evaluator_model="qwen-max",
            evaluator_embedding_model="qwen-embedding",
            ragas_version="0.4.3",
        )
    )

    sample = report.samples[0]
    assert sample.generated_answer == "control answer"
    assert sample.retrieved_contexts == ("control evidence",)
    assert sample.retrieval_trace is not None
    assert sample.metrics == {
        "faithfulness": 1.0,
        "answer_correctness": None,
        "numeric_correctness": None,
    }
    assert sample.metric_reasons == {"faithfulness": "supported"}
    assert sample.evaluation_error == {
        "answer_correctness": "judge metric failed",
        "numeric_correctness": "RuntimeError: numeric parser failed",
    }
