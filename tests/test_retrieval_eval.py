import json
from dataclasses import dataclass
from pathlib import Path

import pytest

from eval.retrieval_metrics import reciprocal_rank
from eval.retrieval_runner import evaluate_retrieval


@dataclass(frozen=True)
class FakeResult:
    chunk_id: str


class FakeRetriever:
    def __init__(self, results: list[str]) -> None:
        self.results = results
        self.calls: list[tuple[str, int]] = []

    def search(self, query: str, k: int = 5) -> list[FakeResult]:
        self.calls.append((query, k))
        return [FakeResult(chunk_id=chunk_id) for chunk_id in self.results[:k]]


def _write_dataset(path: Path, relevant_chunk_id: str = "gold") -> None:
    path.write_text(
        json.dumps(
            {
                "id": "qa_001",
                "query": "测试问题",
                "relevant_chunk_ids": [relevant_chunk_id],
                "metadata": {},
            }
        )
        + "\n",
        encoding="utf-8",
    )


def test_evaluate_retrieval_counts_a_top_k_hit(tmp_path: Path) -> None:
    dataset = tmp_path / "retrieval_eval.jsonl"
    _write_dataset(dataset)
    retriever = FakeRetriever(["gold", "other"])

    report = evaluate_retrieval(dataset, retriever, k=5)

    assert report == {
        "dataset_size": 1,
        "k": 5,
        "metrics": {
            "Recall@1": 1.0,
            "Recall@3": 1.0,
            "Recall@5": 1.0,
            "MRR": 1.0,
        },
        "score_summary": None,
        "queries": [
            {
                "id": "qa_001",
                "query": "测试问题",
                "relevant_chunk_ids": ["gold"],
                "retrieved": [
                    {"rank": 1, "chunk_id": "gold"},
                    {"rank": 2, "chunk_id": "other"},
                ],
                "metrics": {
                    "Recall@1": 1.0,
                    "Recall@3": 1.0,
                    "Recall@5": 1.0,
                    "MRR": 1.0,
                },
                "metadata": {},
                "revision": {},
            }
        ],
    }
    assert retriever.calls == [("测试问题", 5)]


def test_evaluate_retrieval_reports_zero_for_a_miss(tmp_path: Path) -> None:
    dataset = tmp_path / "retrieval_eval.jsonl"
    _write_dataset(dataset)
    retriever = FakeRetriever(["other", "another"])

    report = evaluate_retrieval(dataset, retriever, k=5)

    assert report["metrics"] == {
        "Recall@1": 0.0,
        "Recall@3": 0.0,
        "Recall@5": 0.0,
        "MRR": 0.0,
    }


def test_recall_at_k_is_fraction_for_multiple_relevant_chunks(tmp_path: Path) -> None:
    dataset = tmp_path / "retrieval_eval.jsonl"
    dataset.write_text(
        json.dumps(
            {
                "id": "qa_002",
                "query": "多证据问题",
                "relevant_chunk_ids": ["gold_a", "gold_b"],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    retriever = FakeRetriever(["gold_a", "other", "gold_b"])

    report = evaluate_retrieval(dataset, retriever, k=3)

    assert report["metrics"] == {
        "Recall@1": 0.5,
        "Recall@3": 1.0,
        "MRR": 1.0,
    }


def test_reciprocal_rank_uses_first_relevant_rank() -> None:
    assert reciprocal_rank(
        ["chunk_a", "chunk_b", "gold", "gold"],
        ["gold"],
        k=5,
    ) == pytest.approx(1 / 3)
