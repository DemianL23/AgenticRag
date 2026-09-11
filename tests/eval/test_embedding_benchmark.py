from __future__ import annotations

from agenticrag.rag.integrations.embeddings import EmbeddingConfig
from eval.embedding_benchmark import benchmark_query_embedding


def test_embedding_benchmark_separates_cold_and_warm_calls(monkeypatch) -> None:
    calls: list[str] = []

    class FakeEmbeddings:
        def embed_query(self, query: str) -> list[float]:
            calls.append(query)
            return [0.1, 0.2]

    monkeypatch.setattr(
        "eval.embedding_benchmark.create_embeddings",
        lambda _: FakeEmbeddings(),
    )

    report = benchmark_query_embedding(
        config=EmbeddingConfig(),
        query="测试问题",
        warm_iterations=3,
    )

    assert calls == ["测试问题"] * 4
    assert report["embedding_model"] == "Qwen/Qwen3-Embedding-0.6B"
    assert report["warm_iterations"] == 3
    assert report["timing_seconds"]["warm_query_embedding_seconds"]["count"] == 3
    assert report["warm_latency_excludes_model_load"] is True
