"""Compare local and remote document embeddings before a remote re-index."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from dataclasses import replace
from pathlib import Path
from typing import Any, Sequence

from agenticrag.ingestion.chunk import load_documents_jsonl
from agenticrag.rag.integrations.embeddings import EmbeddingConfig, create_embeddings
from agenticrag.rag.integrations.remote_embeddings import DOCUMENT_PROMPT_PROFILE

DEFAULT_CHUNKS_DIR = Path("artifacts/chunks/pymupdf/v0")
DEFAULT_SAMPLE_SIZE = 20
DEFAULT_OUTPUT = Path("artifacts/profiling/document_embedding_compatibility_20chunks.json")


def evaluate_document_embedding_compatibility(
    *,
    chunks_dir: Path = DEFAULT_CHUNKS_DIR,
    sample_size: int = DEFAULT_SAMPLE_SIZE,
) -> dict[str, Any]:
    """Compare both backends on a deterministic, representative chunk sample."""
    if isinstance(sample_size, bool) or not isinstance(sample_size, int) or sample_size <= 0:
        raise ValueError("sample_size 必须是正整数")
    documents = _load_chunks(chunks_dir)
    selected, selection_categories = _select_chunks(documents, sample_size)

    base_config = EmbeddingConfig.from_env()
    local_config = replace(base_config, backend="local")
    remote_config = replace(base_config, backend="remote")
    local_embeddings = create_embeddings(local_config)
    remote_embeddings = create_embeddings(remote_config)
    texts = [document.page_content for document in selected]
    token_counts = remote_embeddings.document_token_lengths(texts)
    remote_limit = remote_embeddings.remote_config.max_model_len
    over_limit = [count > remote_limit for count in token_counts]
    if any(over_limit):
        raise ValueError(
            "document compatibility test 遇到超过 remote max_model_len 的 chunk："
            f"limit={remote_limit}, indexes={[i for i, value in enumerate(over_limit) if value]}"
        )

    local_vectors = local_embeddings.embed_documents(texts)
    remote_vectors = remote_embeddings.embed_documents(texts)
    rows: list[dict[str, Any]] = []
    for document, token_count, local_vector, remote_vector in zip(
        selected, token_counts, local_vectors, remote_vectors, strict=True
    ):
        if len(local_vector) != len(remote_vector):
            raise ValueError(
                f"dimension 不一致：{document.metadata['chunk_id']} "
                f"{len(local_vector)} != {len(remote_vector)}"
            )
        rows.append(
            {
                "chunk_id": document.metadata["chunk_id"],
                "language": _language(document.page_content),
                "char_count": len(document.page_content),
                "token_count": token_count,
                "over_remote_limit": token_count > remote_limit,
                "local_dimension": len(local_vector),
                "remote_dimension": len(remote_vector),
                "local_norm": _norm(local_vector),
                "remote_norm": _norm(remote_vector),
                "cosine_similarity": _cosine(local_vector, remote_vector),
                "max_abs_element_error": max(
                    abs(float(left) - float(right))
                    for left, right in zip(local_vector, remote_vector, strict=True)
                ),
            }
        )

    dimension_consistent = sum(
        row["local_dimension"] == row["remote_dimension"] for row in rows
    )
    return {
        "benchmark": "local_vs_remote_document_embedding_compatibility",
        "chunks_dir": str(chunks_dir),
        "sample_size": len(rows),
        "selection_categories": selection_categories,
        "preprocessing": "newline_to_space",
        "document_prompt_profile": DOCUMENT_PROMPT_PROFILE,
        "local_embedding": local_config.to_record(),
        "remote_embedding": remote_embeddings.model_record(),
        "remote_max_model_len": remote_limit,
        "chunks": rows,
        "summary": {
            "dimension_consistency_rate": dimension_consistent / len(rows),
            "mean_cosine_similarity": statistics.fmean(
                row["cosine_similarity"] for row in rows
            ),
            "min_cosine_similarity": min(row["cosine_similarity"] for row in rows),
            "max_abs_element_error": max(
                row["max_abs_element_error"] for row in rows
            ),
            "max_observed_tokens": max(row["token_count"] for row in rows),
            "over_limit_count": sum(over_limit),
            "compatibility_pass": (
                dimension_consistent == len(rows)
                and all(not value for value in over_limit)
            ),
        },
    }


def _load_chunks(chunks_dir: Path) -> list[Any]:
    files = sorted(Path(chunks_dir).glob("doc_*.jsonl"))
    if not files:
        raise FileNotFoundError(f"chunk 目录中没有 doc_*.jsonl：{chunks_dir}")
    documents = [document for path in files for document in load_documents_jsonl(path)]
    if not documents:
        raise ValueError(f"chunk 目录为空：{chunks_dir}")
    return documents


def _select_chunks(documents: list[Any], sample_size: int) -> tuple[list[Any], dict[str, int]]:
    buckets: dict[str, list[Any]] = {
        "short": [],
        "long": [],
        "table": [],
        "numeric_dense": [],
        "english_or_mixed": [],
    }
    for document in documents:
        text = document.page_content
        if len(text) <= 300:
            buckets["short"].append(document)
        if len(text) >= 650:
            buckets["long"].append(document)
        if "表" in text or "项目" in text or "具体内容" in text:
            buckets["table"].append(document)
        if sum(character.isdigit() for character in text) >= 12:
            buckets["numeric_dense"].append(document)
        if _language(text) in {"en", "mixed"}:
            buckets["english_or_mixed"].append(document)

    selected: list[Any] = []
    selected_ids: set[str] = set()
    categories: dict[str, int] = {}
    for category, candidates in buckets.items():
        for document in candidates:
            chunk_id = document.metadata["chunk_id"]
            if chunk_id in selected_ids or len(selected) >= sample_size:
                continue
            selected.append(document)
            selected_ids.add(chunk_id)
            categories[category] = categories.get(category, 0) + 1
            break
    for document in documents:
        if len(selected) >= sample_size:
            break
        chunk_id = document.metadata["chunk_id"]
        if chunk_id not in selected_ids:
            selected.append(document)
            selected_ids.add(chunk_id)
    if len(selected) < sample_size:
        raise ValueError(
            f"chunk 数量不足：requested={sample_size}, available={len(documents)}"
        )
    return selected, categories


def _language(text: str) -> str:
    cjk = sum("\u4e00" <= character <= "\u9fff" for character in text)
    latin = sum(character.isascii() and character.isalpha() for character in text)
    if cjk == 0 and latin > 0:
        return "en"
    if latin == 0 and cjk > 0:
        return "zh"
    if cjk and latin:
        return "mixed"
    return "other"


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


def main() -> None:
    args = _parse_args()
    report = evaluate_document_embedding_compatibility(
        chunks_dir=args.chunks_dir,
        sample_size=args.sample_size,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["summary"], ensure_ascii=False))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="比较本地和远程 document embedding 兼容性。")
    parser.add_argument("--chunks-dir", type=Path, default=DEFAULT_CHUNKS_DIR)
    parser.add_argument("--sample-size", type=int, default=DEFAULT_SAMPLE_SIZE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


if __name__ == "__main__":
    main()
