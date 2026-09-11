"""Run the offline Dense or BM25 Retriever benchmark and write its report."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any, Protocol

from agenticrag.rag.integrations.milvus import MilvusConfig
from agenticrag.rag.integrations.milvus_bm25 import BM25MilvusConfig
from agenticrag.retrieval.bm25_retriever import BM25Retriever, DEFAULT_BM25_TOP_K
from agenticrag.retrieval.hybrid_retriever import HybridRetriever, DEFAULT_HYBRID_TOP_K
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

    def search(self, query: str, k: int = 20) -> list[SearchResult]:
        ...


def evaluate_retrieval(
    dataset_path: Path,
    retriever: Retriever,
    k: int = 5,
    *,
    mrr_key: str = "MRR",
) -> dict[str, Any]:
    """Evaluate a retriever against a JSONL dataset.

    The default ``k=5`` produces Recall@1, Recall@3, Recall@5, and MRR.
    When a smaller ``k`` is requested, only cutoffs available in the returned
    ranking are reported, plus Recall@k.
    """
    _validate_k(k)
    samples = load_retrieval_dataset(Path(dataset_path))
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
        query_metrics[mrr_key] = reciprocal_rank(retrieved_ids, relevant_ids, k=k)
        reciprocal_ranks.append(query_metrics[mrr_key])

        ranked_results = []
        for rank, result in enumerate(results, start=1):
            ranked_result = {
                "rank": rank,
                "chunk_id": _result_chunk_id(result),
            }
            for field_name in (
                "score",
                "doc_id",
                "source",
                "page",
                "dense_rank",
                "bm25_rank",
                "rrf_score",
            ):
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
    metrics[mrr_key] = mean_reciprocal_rank(reciprocal_ranks)
    return {
        "dataset_size": total,
        "k": k,
        "metrics": metrics,
        "score_summary": _score_summary(scores),
        "queries": query_reports,
    }


def main() -> None:
    args = _parse_args()
    if args.retriever == "reranker":
        from agenticrag.reranking.factory import create_reranker
        from agenticrag.retrieval.reranking_retriever import (
            DEFAULT_RERANK_FINAL_TOP_K,
            RerankingRetriever,
        )

        from .reranking_runner import evaluate_reranking

        dense_base = MilvusConfig.from_env()
        bm25_base = BM25MilvusConfig.from_env()
        dense_config = MilvusConfig(
            uri=args.uri or dense_base.uri,
            collection_name=args.dense_collection or dense_base.collection_name,
        )
        bm25_config = BM25MilvusConfig(
            uri=args.uri or bm25_base.uri,
            collection_name=args.bm25_collection or bm25_base.collection_name,
        )
        k = args.k if args.k is not None else DEFAULT_RERANK_FINAL_TOP_K
        output = args.output or Path(
            "artifacts/eval/retrieval_reranker_v1_2_a_report.json"
        )
        if output.exists() and not args.overwrite:
            raise FileExistsError(
                f"V1.2-A 评测报告已存在：{output}；如需覆盖请显式传入 --overwrite"
            )
        retriever = RerankingRetriever(
            hybrid_retriever=HybridRetriever(
                dense_retriever=MilvusRetriever(milvus_config=dense_config),
                bm25_retriever=BM25Retriever(milvus_config=bm25_config),
            ),
            reranker=create_reranker(),
        )
        report = evaluate_reranking(
            args.dataset,
            retriever,
            final_k=k,
            dense_baseline_path=args.dense_baseline_report,
            hybrid_baseline_path=args.hybrid_baseline_report,
        )
        report["collections"] = {
            "dense": dense_config.collection_name,
            "bm25": bm25_config.collection_name,
        }
    elif args.retriever == "hybrid":
        dense_base = MilvusConfig.from_env()
        bm25_base = BM25MilvusConfig.from_env()
        dense_config = MilvusConfig(
            uri=args.uri or dense_base.uri,
            collection_name=args.dense_collection or dense_base.collection_name,
        )
        bm25_config = BM25MilvusConfig(
            uri=args.uri or bm25_base.uri,
            collection_name=args.bm25_collection or bm25_base.collection_name,
        )
        k = args.k if args.k is not None else DEFAULT_HYBRID_TOP_K
        output = args.output or Path("artifacts/eval/retrieval_hybrid_v1_1_report.json")
        if output.exists() and not args.overwrite:
            raise FileExistsError(
                f"Hybrid 评测报告已存在：{output}；如需覆盖请显式传入 --overwrite"
            )
        retriever = HybridRetriever(
            dense_retriever=MilvusRetriever(milvus_config=dense_config),
            bm25_retriever=BM25Retriever(milvus_config=bm25_config),
        )
        report = evaluate_retrieval(
            args.dataset,
            retriever,
            k=k,
            mrr_key=f"MRR@{k}",
        )
        report["retriever"] = "hybrid_rrf"
        report["collections"] = {
            "dense": dense_config.collection_name,
            "bm25": bm25_config.collection_name,
        }
    elif args.retriever == "bm25":
        base_config = BM25MilvusConfig.from_env()
        milvus_config = BM25MilvusConfig(
            uri=args.uri or base_config.uri,
            collection_name=args.collection_name or base_config.collection_name,
        )
        k = args.k if args.k is not None else DEFAULT_BM25_TOP_K
        output = args.output or Path("artifacts/eval/retrieval_bm25_v1_1_report.json")
        if output.exists() and not args.overwrite:
            raise FileExistsError(
                f"BM25 评测报告已存在：{output}；如需覆盖请显式传入 --overwrite"
            )
        report = evaluate_retrieval(
            args.dataset,
            BM25Retriever(milvus_config=milvus_config),
            k=k,
            mrr_key=f"MRR@{k}",
        )
        report["retriever"] = "bm25"
        report["collection_name"] = milvus_config.collection_name
    else:
        base_config = MilvusConfig.from_env()
        milvus_config = MilvusConfig(
            uri=args.uri or base_config.uri,
            collection_name=args.collection_name or base_config.collection_name,
        )
        k = args.k if args.k is not None else 5
        output = args.output or Path("artifacts/eval/retrieval_report.json")
        report = evaluate_retrieval(
            args.dataset,
            MilvusRetriever(milvus_config=milvus_config),
            k=k,
        )
        report["retriever"] = "dense"
        report["collection_name"] = milvus_config.collection_name
    _write_json(report, output)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"报告已写入：{output}")


def load_retrieval_dataset(path: Path) -> list[dict[str, Any]]:
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
    parser = argparse.ArgumentParser(
        description="评测 Dense、BM25、Hybrid 或 Reranker Retrieval 的 Recall@K 和 MRR。"
    )
    parser.add_argument("--dataset", type=Path, required=True, help="retrieval 评测集 JSONL")
    parser.add_argument(
        "--retriever",
        choices=("dense", "bm25", "hybrid", "reranker"),
        default="dense",
        help="检索器类型；reranker 默认最终 Top-5",
    )
    parser.add_argument(
        "--k",
        type=int,
        help="结果 Top-K；不传时 Dense/Reranker=5、BM25/Hybrid=20",
    )
    parser.add_argument("--output", type=Path, help="报告路径；不传时按检索器选择默认路径")
    parser.add_argument("--overwrite", action="store_true", help="允许覆盖已有实验报告")
    parser.add_argument("--uri", help="Milvus 地址，默认读取 MILVUS_URI")
    parser.add_argument(
        "--collection-name",
        help="Milvus collection 名称，默认读取 MILVUS_COLLECTION",
    )
    parser.add_argument(
        "--dense-collection",
        help="Hybrid 的 Dense collection，默认读取 MILVUS_COLLECTION",
    )
    parser.add_argument(
        "--bm25-collection",
        help="Hybrid 的 BM25 collection，默认读取 BM25_MILVUS_COLLECTION",
    )
    parser.add_argument(
        "--dense-baseline-report",
        type=Path,
        default=Path("artifacts/eval/retrieval_report.json"),
        help="V1.2 对比使用的 V0 Dense 报告",
    )
    parser.add_argument(
        "--hybrid-baseline-report",
        type=Path,
        default=Path("artifacts/eval/retrieval_hybrid_v1_1_report.json"),
        help="V1.2 对比使用的 V1.1 Hybrid 报告",
    )
    return parser.parse_args()
