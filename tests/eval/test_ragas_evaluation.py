import asyncio
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from agenticrag.generation.schemas import GeneratedAnswer
from agenticrag.rag.service import RagAnswerTrace
from agenticrag.retrieval.schemas import RetrievedChunk
from eval.ragas.config import RagasEvaluatorConfig
from eval.ragas.dataset import gold_to_reference, load_qa_dataset
from eval.ragas.evaluator import (
    MetricDefinition,
    RagasEvaluator,
    RagasMetricResult,
)
from eval.ragas.report import RagasReport, SampleReport, write_report
from eval.ragas.runner import evaluate_end_to_end, retrieved_contexts_from_chunks


def _chunk(content: str = "检索到的正文") -> RetrievedChunk:
    return RetrievedChunk(
        content=content,
        score=0.2,
        doc_id="doc_000",
        source="corpus/report.pdf",
        page=4,
        chunk_id="doc_000:p0004:c000",
    )


def test_qa_loader_maps_question_and_supported_gold_types(tmp_path: Path) -> None:
    dataset = tmp_path / "qa.jsonl"
    records = [
        {"finqa_id": "a", "question": "问题一", "gold": ["答案甲", "答案乙"]},
        {"uid": "b", "question": "问题二", "gold": 0.0502},
        {"question": "问题三", "gold": "  答案三  "},
    ]
    dataset.write_text(
        "\n".join(json.dumps(record, ensure_ascii=False) for record in records),
        encoding="utf-8",
    )

    samples = load_qa_dataset(dataset)

    assert [sample.sample_id for sample in samples] == ["a", "b", "line_3"]
    assert samples[0].question == "问题一"
    assert samples[0].reference == "- 答案甲\n- 答案乙"
    assert samples[0].task_type is None
    assert samples[1].reference == "0.0502"
    assert samples[2].reference == "答案三"


def test_qa_loader_limit_and_missing_gold_validation(tmp_path: Path) -> None:
    dataset = tmp_path / "qa.jsonl"
    dataset.write_text(
        '{"question":"有效问题","gold":"答案"}\n'
        '{"question":"坏问题"}\n',
        encoding="utf-8",
    )

    assert len(load_qa_dataset(dataset, limit=1)) == 1
    with pytest.raises(ValueError, match="缺少 gold"):
        load_qa_dataset(dataset)


def test_gold_to_reference_rejects_unsupported_values() -> None:
    with pytest.raises(TypeError, match="不受支持"):
        gold_to_reference(None)


def test_retrieved_chunks_become_exact_ragas_contexts() -> None:
    chunks = [_chunk("第一段"), _chunk("第二段")]

    assert retrieved_contexts_from_chunks(chunks) == ["第一段", "第二段"]


class FakeMetricResult:
    def __init__(self, value: float, reason: str | None = None) -> None:
        self.value = value
        self.reason = reason


class RecordingMetric:
    def __init__(self, *, value: float = 0.8, error: Exception | None = None) -> None:
        self.value = value
        self.error = error
        self.calls: list[dict[str, Any]] = []

    async def ascore(self, **kwargs: Any) -> FakeMetricResult:
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        return FakeMetricResult(self.value, "judge reason")


def test_ragas_call_layer_passes_only_metric_inputs_and_isolates_errors() -> None:
    good = RecordingMetric(value=0.75)
    bad = RecordingMetric(error=RuntimeError("judge unavailable"))
    evaluator = RagasEvaluator(
        [
            MetricDefinition("faithfulness", good, ("response", "retrieved_contexts")),
            MetricDefinition("answer_correctness", bad, ("response", "reference")),
        ]
    )

    result = asyncio.run(
        evaluator.evaluate(
            user_input="问题",
            reference="标准答案",
            response="生成答案",
            retrieved_contexts=["正文"],
        )
    )

    assert good.calls == [{"response": "生成答案", "retrieved_contexts": ["正文"]}]
    assert result.scores == {"faithfulness": 0.75, "answer_correctness": None}
    assert result.reasons == {"faithfulness": "judge reason"}
    assert "RuntimeError" in result.errors["answer_correctness"]


def test_evaluator_config_is_independent_from_generation_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dotenv = tmp_path / ".env"
    dotenv.write_text("", encoding="utf-8")
    monkeypatch.setenv("GENERATION_MODEL", "qwen-max")
    monkeypatch.setenv("DASHSCOPE_API_KEY", "shared-test-key")
    monkeypatch.setenv("RAGAS_EVALUATOR_MODEL", "qwen-plus")
    monkeypatch.setenv("RAGAS_EVALUATOR_BASE_URL", "https://judge.example/v1")

    config = RagasEvaluatorConfig.from_env(dotenv)

    assert config.model == "qwen-plus"
    assert config.model != "qwen-max"
    assert config.api_key == "shared-test-key"
    assert config.base_url == "https://judge.example/v1"
    assert config.max_tokens == 4096
    assert config.to_record()["api_key"] is None
    assert config.to_record()["api_key_configured"] is True


def test_evaluator_config_requires_explicit_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dotenv = tmp_path / ".env"
    dotenv.write_text(
        "RAGAS_EVALUATOR_BASE_URL=https://judge.example/v1\n",
        encoding="utf-8",
    )
    monkeypatch.delenv("RAGAS_EVALUATOR_MODEL", raising=False)
    monkeypatch.delenv("RAGAS_EVALUATOR_BASE_URL", raising=False)

    with pytest.raises(ValueError, match="RAGAS_EVALUATOR_MODEL 未配置"):
        RagasEvaluatorConfig.from_env(dotenv)


@dataclass
class FakeService:
    calls: list[tuple[str, int]]

    def answer_with_trace(self, query: str, *, k: int = 5) -> RagAnswerTrace:
        self.calls.append((query, k))
        return RagAnswerTrace(
            query=query,
            answer=GeneratedAnswer(answer="真实生成答案", citations=()),
            retrieved_chunks=(_chunk(),),
        )


class FakeEvaluator:
    metric_names = ("faithfulness", "answer_correctness")

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def evaluate(self, **kwargs: Any) -> RagasMetricResult:
        self.calls.append(kwargs)
        return RagasMetricResult(
            scores={"faithfulness": 1.0, "answer_correctness": 0.5},
            reasons={},
            errors={},
        )


def test_end_to_end_runner_materializes_real_response_and_contexts(
    tmp_path: Path,
) -> None:
    dataset = tmp_path / "qa.jsonl"
    dataset.write_text(
        '{"finqa_id":"qa_1","question":"问题","gold":["标准答案"]}\n',
        encoding="utf-8",
    )
    service = FakeService(calls=[])
    evaluator = FakeEvaluator()

    report = asyncio.run(
        evaluate_end_to_end(
            dataset,
            service,
            evaluator,
            top_k=5,
            limit=1,
            generation_model="qwen-plus",
            embedding_model="Qwen/Qwen3-Embedding-0.6B",
            evaluator_model="qwen-max",
            evaluator_embedding_model="Qwen/Qwen3-Embedding-0.6B",
            ragas_version="0.4.3",
        )
    )

    assert service.calls == [("问题", 5)]
    assert evaluator.calls[0] == {
        "user_input": "问题",
        "reference": "标准答案",
        "response": "真实生成答案",
        "retrieved_contexts": ["检索到的正文"],
    }
    assert report.aggregate_metrics == {
        "faithfulness": 1.0,
        "answer_correctness": 0.5,
        "numeric_correctness": None,
    }
    assert report.samples[0].retrieved_chunk_ids == ("doc_000:p0004:c000",)


@dataclass
class NumericFakeService:
    def answer_with_trace(self, query: str, *, k: int = 5) -> RagAnswerTrace:
        return RagAnswerTrace(
            query=query,
            answer=GeneratedAnswer(answer="现金比率为61.16%", citations=()),
            retrieved_chunks=(_chunk(),),
        )


def test_end_to_end_runner_adds_numeric_correctness_to_sample_and_aggregate(
    tmp_path: Path,
) -> None:
    dataset = tmp_path / "qa.jsonl"
    dataset.write_text(
        '{"finqa_id":"qa_1","question":"问题","gold":0.6116,"task_type":"Implicit_Reasoning"}\n',
        encoding="utf-8",
    )

    report = asyncio.run(
        evaluate_end_to_end(
            dataset,
            NumericFakeService(),
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

    assert report.samples[0].metrics["numeric_correctness"] == 1.0
    assert report.aggregate_metrics["numeric_correctness"] == 1.0
    assert report.aggregate_metric_counts["numeric_correctness"] == 1
    assert "numeric_correctness" in report.metrics


def test_report_serialization_contains_required_audit_fields(tmp_path: Path) -> None:
    sample = SampleReport(
        sample_id="qa_1",
        question="问题",
        reference="标准答案",
        generated_answer="生成答案",
        retrieved_chunk_ids=("chunk_1",),
        retrieved_contexts=("正文",),
        metrics={"faithfulness": 1.0},
        metric_reasons={},
        evaluation_error=None,
    )
    report = RagasReport(
        schema_version=1,
        created_at="2026-09-08T00:00:00+00:00",
        dataset_path="qa.jsonl",
        dataset_size=1,
        top_k=5,
        generation_model="qwen-plus",
        embedding_model="qwen-embedding",
        evaluator_model="qwen-max",
        evaluator_embedding_model="qwen-embedding",
        ragas_version="0.4.3",
        metrics=("faithfulness",),
        context_entity_recall={"enabled": False, "rationale": "test"},
        aggregate_metrics={"faithfulness": 1.0},
        aggregate_metric_counts={"faithfulness": 1},
        successful_samples=1,
        failed_samples=0,
        samples=(sample,),
    )
    output = tmp_path / "ragas_report.json"

    write_report(report, output)
    saved = json.loads(output.read_text(encoding="utf-8"))

    assert saved["ragas_version"] == "0.4.3"
    assert saved["samples"][0]["user_input"] == "问题"
    assert saved["samples"][0]["response"] == "生成答案"
    assert saved["samples"][0]["retrieved_chunk_ids"] == ["chunk_1"]
    assert saved["samples"][0]["evaluation_error"] is None
