"""Embed chunks and write a V0 collection to Milvus."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from agenticrag.ingestion.chunk import load_documents_jsonl
from agenticrag.rag.integrations.embeddings import (
    EmbeddingConfig,
    create_embeddings,
    embedding_dimension,
)
from agenticrag.rag.integrations.milvus import MilvusConfig


def build_milvus_index(
    chunks_dir: Path,
    *,
    milvus_config: MilvusConfig | None = None,
    embedding_config: EmbeddingConfig | None = None,
    report_output: Path | None = None,
) -> dict[str, Any]:
    """Create a collection and insert every chunk with a stable primary key."""
    chunks_dir = Path(chunks_dir)
    files = sorted(chunks_dir.glob("doc_*.jsonl"))
    if not files:
        raise FileNotFoundError(f"chunk 目录中没有 doc_*.jsonl：{chunks_dir}")

    milvus_config = milvus_config or MilvusConfig.from_env()
    milvus_config.validate()
    embedding_config = embedding_config or EmbeddingConfig.from_env()
    embedding_config.validate()
    documents = [
        document
        for path in files
        for document in load_documents_jsonl(path)
    ]
    if not documents:
        raise ValueError(f"chunk 目录中没有可写入的文本：{chunks_dir}")
    ids = [_chunk_id(document.metadata) for document in documents]
    if len(ids) != len(set(ids)):
        raise ValueError("chunk_id 不唯一，拒绝写入 Milvus")

    embeddings = create_embeddings(embedding_config)
    dimension = embedding_dimension(embeddings)
    collection_name = milvus_config.collection_name or default_collection_name(
        embedding_config,
        chunks_dir,
    )
    try:
        from langchain_milvus import Milvus
    except ImportError as exc:
        raise RuntimeError("Milvus 集成需要可选依赖，请执行：uv sync --extra milvus") from exc

    Milvus.from_documents(
        documents=documents,
        embedding=embeddings,
        ids=ids,
        collection_name=collection_name,
        connection_args={"uri": milvus_config.uri},
        drop_old=milvus_config.drop_old,
    )

    report_path = Path(report_output) if report_output is not None else chunks_dir / "milvus_report.json"
    report = {
        "chunks_dir": chunks_dir.as_posix(),
        "collection_name": collection_name,
        "milvus_uri": milvus_config.uri,
        "drop_old": milvus_config.drop_old,
        "embedding": {
            **embedding_config.to_record(),
            "dimension": dimension,
        },
        "documents": len(files),
        "chunks": len(documents),
        "report_output": report_path.as_posix(),
    }
    _write_json(report, report_path)
    return report


def default_collection_name(embedding_config: EmbeddingConfig, chunks_dir: Path) -> str:
    """Build a readable collection name from model and chunking configuration."""
    model_slug = _slug(embedding_config.model_name)
    chunk_report = chunks_dir / "report.json"
    chunk_size = "unknown"
    chunk_overlap = "unknown"
    if chunk_report.is_file():
        data = json.loads(chunk_report.read_text(encoding="utf-8"))
        chunk_size = str(data.get("config", {}).get("chunk_size", chunk_size))
        chunk_overlap = str(data.get("config", {}).get("chunk_overlap", chunk_overlap))
    return f"rag_v0_{model_slug}_{chunk_size}_{chunk_overlap}"


def _chunk_id(metadata: dict[str, Any]) -> str:
    value = metadata.get("chunk_id")
    if not isinstance(value, str) or not value.strip():
        raise ValueError("chunk 缺少合法 chunk_id")
    return value


def _slug(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", value).strip("_").lower()
    return slug or "embedding"


def _write_json(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.tmp")
    temporary_path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary_path.replace(path)


def main() -> None:
    args = _parse_args()
    config = MilvusConfig(
        uri=args.uri or MilvusConfig.from_env().uri,
        collection_name=args.collection_name or MilvusConfig.from_env().collection_name,
        drop_old=args.drop_old,
    )
    report = build_milvus_index(args.chunks_dir, milvus_config=config)
    print(json.dumps(report, ensure_ascii=False))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="将 chunks 向量化并写入 Milvus。")
    parser.add_argument("--chunks-dir", type=Path, required=True, help="chunk JSONL 所在目录")
    parser.add_argument("--uri", help="Milvus 地址，默认读取 MILVUS_URI")
    parser.add_argument("--collection-name", help="collection 名称，默认按模型和切块配置生成")
    parser.add_argument("--drop-old", action="store_true", help="写入前删除同名 collection")
    return parser.parse_args()
