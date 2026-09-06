"""Run the offline Dense Retriever benchmark and write its report."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any, Protocol

from agenticrag.integrations.milvus import MilvusConfig
from agenticrag.retrieval.milvus_retriever import MilvusRetriever

from .retrieval_metrics import (
    mean_reciprocal_rank,
    recall_at_k,
    reciprocal_rank,
)


class SearchResult(Protocol):
    """Minimum result shape required by the evaluation runner."""

    chunk_id: str


class Retriever(Protocol):
    """Minimum retriever shape required by the evaluation runner."""

    def search(self, query: str, k: int = 5) -> list[SearchResult]:
        ...


def evaluate_retrieval(
    dataset_path: Path,
    retriever: Retriever,
    k: int = 5,
) -> dict[str, Any]:
    """Evaluate a retriever against a JSONL dataset.

    The default ``k=5`` produces Recall@1, Recall@3, Recall@5, and MRR.
    When a smaller ``k`` is requested, only cutoffs available in the returned
    ranking are reported, plus Recall@k.
    """
    _validate_k(k)
    samples = _load_dataset(Path(dataset_path))
    if not samples:
        raise ValueError(f"评测集不能为空：{dataset_path}")

    recall_totals: dict[int, float] = {
        cutoff: 0.0
        for cutoff in sorted({cutoff for cutoff in (1, 3, 5) if cutoff <= k} | {k})
    }
    reciprocal_ranks: list[float] = []
    query_reports: list[dict[str, Any]] = []
    scores: list[float] = []

    for sample in samples:
        results = retriever.search(sample["query"], k=k)
        retrieved_ids = [_result_chunk_id(result) for result in results]
        relevant_ids = sample["relevant_chunk_ids"]

        for cutoff in recall_totals:
            recall_totals[cutoff] += recall_at_k(
                retrieved_ids,
                relevant_ids,
                k=cutoff,
            )
        query_metrics = {
            f"Recall@{cutoff}": recall_at_k(
                retrieved_ids,
                relevant_ids,
                k=cutoff,
            )
            for cutoff in recall_totals
        }
        query_metrics["MRR"] = reciprocal_rank(retrieved_ids, relevant_ids, k=k)
        reciprocal_ranks.append(query_metrics["MRR"])

        ranked_results = []
        for rank, result in enumerate(results, start=1):
            ranked_result = {
                "rank": rank,
                "chunk_id": _result_chunk_id(result),
            }
            for field_name in ("score", "doc_id", "source", "page"):
                value = getattr(result, field_name, None)
                if value is not None:
                    ranked_result[field_name] = value
                    if field_name == "score":
                        scores.append(float(value))
            ranked_results.append(ranked_result)

        query_reports.append(
            {
                "id": sample["id"],
                "query": sample["query"],
                "relevant_chunk_ids": relevant_ids,
                "retrieved": ranked_results,
                "metrics": query_metrics,
                "metadata": sample.get("metadata", {}),
                "revision": sample.get("revision", {}),
            }
        )

    total = len(samples)
    metrics = {
        f"Recall@{cutoff}": value / total
        for cutoff, value in recall_totals.items()
    }
    metrics["MRR"] = mean_reciprocal_rank(reciprocal_ranks)
    return {
        "dataset_size": total,
        "k": k,
        "metrics": metrics,
        "score_summary": _score_summary(scores),
        "queries": query_reports,
    }


def main() -> None:
    args = _parse_args()
    base_config = MilvusConfig.from_env()
    milvus_config = MilvusConfig(
        uri=args.uri or base_config.uri,
        collection_name=args.collection_name or base_config.collection_name,
    )
    report = evaluate_retrieval(
        args.dataset,
        MilvusRetriever(milvus_config=milvus_config),
        k=args.k,
    )
    _write_json(report, args.output)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"报告已写入：{args.output}")


def _load_dataset(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"评测集不存在：{path}")

    samples: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, raw_line in enumerate(file, start=1):
            if not raw_line.strip():
                continue
            try:
                record = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"评测集第 {line_number} 行不是合法 JSON：{path}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"评测集第 {line_number} 行必须是 JSON 对象：{path}")
            samples.append(_validate_sample(record, line_number, path))
    return samples


def _validate_sample(record: dict[str, Any], line_number: int, path: Path) -> dict[str, Any]:
    sample_id = record.get("id", f"line_{line_number}")
    query = record.get("query")
    relevant_ids = record.get("relevant_chunk_ids")
    if not isinstance(sample_id, str) or not sample_id.strip():
        raise ValueError(f"评测集第 {line_number} 行缺少合法 id：{path}")
    if not isinstance(query, str) or not query.strip():
        raise ValueError(f"评测集第 {line_number} 行缺少合法 query：{path}")
    if (
        not isinstance(relevant_ids, list)
        or not relevant_ids
        or any(not isinstance(value, str) or not value.strip() for value in relevant_ids)
    ):
        raise ValueError(
            f"评测集第 {line_number} 行缺少合法 relevant_chunk_ids：{path}"
        )
    return {
        "id": sample_id.strip(),
        "query": query.strip(),
        "relevant_chunk_ids": [value.strip() for value in relevant_ids],
        "metadata": record.get("metadata", {}),
        "revision": record.get("revision", {}),
    }


def _result_chunk_id(result: SearchResult) -> str:
    chunk_id = getattr(result, "chunk_id", None)
    if not isinstance(chunk_id, str) or not chunk_id.strip():
        raise ValueError("Retriever 返回结果缺少合法 chunk_id")
    return chunk_id.strip()


def _write_json(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.tmp")
    temporary_path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(path)


def _validate_k(k: int) -> None:
    if isinstance(k, bool) or not isinstance(k, int) or k <= 0:
        raise ValueError("k 必须是正整数")


def _score_summary(scores: list[float]) -> dict[str, float | int] | None:
    if not scores:
        return None
    return {
        "count": len(scores),
        "min": min(scores),
        "max": max(scores),
        "mean": statistics.fmean(scores),
        "median": statistics.median(scores),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="评测 Dense Retriever 的 Recall@K 和 MRR。")
    parser.add_argument("--dataset", type=Path, required=True, help="retrieval 评测集 JSONL")
    parser.add_argument("--k", type=int, default=5, help="检索 Top-K，默认 5")
    parser.add_argument("--output", type=Path, default=Path("artifacts/eval/retrieval_report.json"))
    parser.add_argument("--uri", help="Milvus 地址，默认读取 MILVUS_URI")
    parser.add_argument(
        "--collection-name",
        help="Milvus collection 名称，默认读取 MILVUS_COLLECTION",
    )
    return parser.parse_args()
