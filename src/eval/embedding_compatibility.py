"""Compare local and remote Qwen query vectors against one Milvus collection."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from dataclasses import replace
from pathlib import Path
from typing import Any, Sequence

from agenticrag.rag.integrations.embeddings import EmbeddingConfig, create_embeddings
from agenticrag.rag.integrations.milvus import MilvusConfig
from eval.retrieval_runner import load_retrieval_dataset

DEFAULT_DATASET = Path("eval/datasets/retrieval_eval_v2.jsonl")
DEFAULT_QUERY_LIMIT = 10
DEFAULT_TOP_K = 20
DEFAULT_OUTPUT = Path("artifacts/profiling/embedding_compatibility_10q.json")


def evaluate_embedding_compatibility(
    *,
    dataset: Path = DEFAULT_DATASET,
    limit: int = DEFAULT_QUERY_LIMIT,
    top_k: int = DEFAULT_TOP_K,
) -> dict[str, Any]:
    """Compare local and remote embeddings and their rankings on one collection."""
    _validate_positive_int(limit, "limit")
    _validate_positive_int(top_k, "top_k")
    base_config = EmbeddingConfig.from_env()
    local_config = replace(base_config, backend="local")
    remote_config = replace(base_config, backend="remote")
    local_embeddings = create_embeddings(local_config)
    remote_embeddings = create_embeddings(remote_config)

    try:
        from langchain_milvus import Milvus
    except ImportError as exc:
        raise RuntimeError("兼容性评测需要可选依赖，请执行：uv sync --extra milvus") from exc

    milvus_config = MilvusConfig.from_env()
    if not milvus_config.collection_name:
        raise ValueError("兼容性评测需要 MILVUS_COLLECTION")
    vector_store = Milvus(
        embedding_function=local_embeddings,
        collection_name=milvus_config.collection_name,
        connection_args={"uri": milvus_config.uri},
    )

    samples = load_retrieval_dataset(dataset)[:limit]
    if not samples:
        raise ValueError(f"评测集不能为空：{dataset}")
    query_reports: list[dict[str, Any]] = []
    for sample in samples:
        query = sample["query"]
        local_vector = local_embeddings.embed_query(query)
        remote_vector = remote_embeddings.embed_query(query)
        _validate_comparable_dimensions(local_vector, remote_vector)
        local_ids = _search_ids(vector_store, local_vector, top_k)
        remote_ids = _search_ids(vector_store, remote_vector, top_k)
        query_reports.append(
            {
                "id": sample["id"],
                "query": query,
                "local_norm": _norm(local_vector),
                "remote_norm": _norm(remote_vector),
                "cosine_similarity": _cosine(local_vector, remote_vector),
                "max_abs_element_error": max(
                    abs(float(left) - float(right))
                    for left, right in zip(local_vector, remote_vector, strict=True)
                ),
                "top1_same": local_ids[:1] == remote_ids[:1],
                "top5_set_overlap": _set_overlap(local_ids[:5], remote_ids[:5]),
                "top20_set_overlap": _set_overlap(local_ids, remote_ids),
            }
        )

    return {
        "benchmark": "local_vs_remote_embedding_compatibility",
        "dataset": str(dataset),
        "query_count": len(query_reports),
        "top_k": top_k,
        "local_embedding": local_config.to_record(),
        "remote_embedding": remote_embeddings.model_record(),
        "queries": query_reports,
        "summary": {
            "top1_same_rate": statistics.fmean(
                float(row["top1_same"]) for row in query_reports
            ),
            "mean_top5_set_overlap": statistics.fmean(
                row["top5_set_overlap"] for row in query_reports
            ),
            "mean_top20_set_overlap": statistics.fmean(
                row["top20_set_overlap"] for row in query_reports
            ),
            "mean_cosine_similarity": statistics.fmean(
                row["cosine_similarity"] for row in query_reports
            ),
            "max_abs_element_error": max(
                row["max_abs_element_error"] for row in query_reports
            ),
        },
    }


def _search_ids(vector_store: Any, vector: Sequence[float], top_k: int) -> list[str]:
    results = vector_store.similarity_search_with_score_by_vector(vector, k=top_k)
    return [document.metadata["chunk_id"] for document, _ in results]


def _validate_comparable_dimensions(left: Sequence[float], right: Sequence[float]) -> None:
    if len(left) != len(right):
        raise ValueError(
            f"local/remote embedding dimension 不一致：{len(left)} != {len(right)}"
        )


def _norm(vector: Sequence[float]) -> float:
    return math.sqrt(sum(float(value) ** 2 for value in vector))


def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
    left_norm = _norm(left)
    right_norm = _norm(right)
    if left_norm == 0.0 or right_norm == 0.0:
        raise ValueError("embedding norm 不能为 0")
    return sum(float(a) * float(b) for a, b in zip(left, right, strict=True)) / (
        left_norm * right_norm
    )


def _set_overlap(left: Sequence[str], right: Sequence[str]) -> float:
    if not left or not right:
        return 1.0 if not left and not right else 0.0
    return len(set(left) & set(right)) / max(len(left), len(right))


def _validate_positive_int(value: int, field_name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field_name} 必须是正整数")


def main() -> None:
    args = _parse_args()
    report = evaluate_embedding_compatibility(
        dataset=args.dataset,
        limit=args.limit,
        top_k=args.top_k,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"报告已写入：{args.output}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="比较本地与远程 Qwen embedding 兼容性。")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--limit", type=int, default=DEFAULT_QUERY_LIMIT)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()
