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

EMBEDDING_COMPATIBILITY_FIELDS = (
    "embedding_model",
    "embedding_model_revision",
    "embedding_dimension",
    "normalize_embeddings",
    "query_prompt_name",
    "document_prompt_profile",
)


def build_milvus_index(
    chunks_dir: Path,
    *,
    milvus_config: MilvusConfig | None = None,
    embedding_config: EmbeddingConfig | None = None,
    report_output: Path | None = None,
    limit: int | None = None,
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
    if limit is not None:
        if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
            raise ValueError("limit 必须是正整数")
        documents = documents[:limit]
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
    report_path = Path(report_output) if report_output is not None else chunks_dir / "milvus_report.json"
    embedding_record = _build_embedding_record(embedding_config, embeddings, dimension)
    _guard_existing_collection(
        milvus_config,
        collection_name=collection_name,
        report_path=report_path,
        expected_embedding=embedding_record,
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

    report = {
        "chunks_dir": chunks_dir.as_posix(),
        "collection_name": collection_name,
        "milvus_uri": milvus_config.uri,
        "drop_old": milvus_config.drop_old,
        "embedding": embedding_record,
        "documents": len(files),
        "chunks": len(documents),
        "limit": limit,
        "report_output": report_path.as_posix(),
    }
    _write_json(report, report_path)
    return report


def _build_embedding_record(
    config: EmbeddingConfig,
    embeddings: Any,
    dimension: int,
) -> dict[str, Any]:
    """Build the compatibility metadata stored with every dense index report."""
    record = config.to_record()
    model_record = getattr(embeddings, "model_record", None)
    if callable(model_record):
        record.update(model_record())
    record.update(
        {
            "embedding_backend": config.backend,
            "embedding_model": config.model_name,
            "embedding_model_revision": record.get("model_revision"),
            "embedding_dimension": dimension,
            "dimension": dimension,
            "normalize_embeddings": config.normalize_embeddings,
            "query_prompt_name": config.effective_query_prompt_name(),
            "document_prompt_profile": "raw_document_v1",
            "batch_size": record.get("batch_size", config.batch_size),
            "remote_max_model_len": record.get("max_model_len"),
            "revision_verified": bool(record.get("model_revision")),
        }
    )
    return record


def _guard_existing_collection(
    milvus_config: MilvusConfig,
    *,
    collection_name: str,
    report_path: Path,
    expected_embedding: dict[str, Any],
) -> None:
    """Reject unsafe mixed-semantic writes to an existing dense collection."""
    if milvus_config.drop_old or not _collection_exists(milvus_config.uri, collection_name):
        return
    if not report_path.is_file():
        raise ValueError(
            f"已有 collection 缺少 embedding metadata report：{collection_name}；"
            "无法验证兼容性，拒绝混写"
        )
    previous_report = json.loads(report_path.read_text(encoding="utf-8"))
    previous_embedding = previous_report.get("embedding")
    if not isinstance(previous_embedding, dict):
        raise ValueError("已有 Milvus report 缺少 embedding metadata，拒绝混写")
    mismatches: dict[str, tuple[Any, Any]] = {}
    for field in EMBEDDING_COMPATIBILITY_FIELDS:
        expected = expected_embedding.get(field)
        actual = _legacy_embedding_field(previous_embedding, field)
        if actual != expected:
            mismatches[field] = (actual, expected)
    if mismatches:
        raise ValueError(
            "已有 collection 的 embedding semantics 不兼容，拒绝混写："
            f"{mismatches}；如需重建请显式设置 --drop-old"
        )


def _legacy_embedding_field(record: dict[str, Any], field: str) -> Any:
    aliases = {
        "embedding_model": ("embedding_model", "model_name", "model"),
        "embedding_model_revision": ("embedding_model_revision", "model_revision"),
        "embedding_dimension": ("embedding_dimension", "dimension"),
        "normalize_embeddings": ("normalize_embeddings",),
        "query_prompt_name": ("query_prompt_name",),
        "document_prompt_profile": ("document_prompt_profile",),
    }
    value = next((record.get(key) for key in aliases[field] if key in record), None)
    if field == "document_prompt_profile" and value is None:
        return "raw_document_v1"
    return value


def _collection_exists(uri: str, collection_name: str) -> bool:
    try:
        from pymilvus import MilvusClient
    except ImportError as exc:
        raise RuntimeError("collection compatibility guard 需要 pymilvus") from exc
    try:
        return bool(MilvusClient(uri=uri).has_collection(collection_name))
    except Exception as exc:  # noqa: BLE001 - cannot safely verify a write
        raise RuntimeError(
            f"无法检查 Milvus collection 是否存在：{collection_name}"
        ) from exc


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
    report = build_milvus_index(
        args.chunks_dir,
        milvus_config=config,
        report_output=args.report_output,
        limit=args.limit,
    )
    print(json.dumps(report, ensure_ascii=False))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="将 chunks 向量化并写入 Milvus。")
    parser.add_argument("--chunks-dir", type=Path, required=True, help="chunk JSONL 所在目录")
    parser.add_argument("--uri", help="Milvus 地址，默认读取 MILVUS_URI")
    parser.add_argument("--collection-name", help="collection 名称，默认按模型和切块配置生成")
    parser.add_argument("--drop-old", action="store_true", help="写入前删除同名 collection")
    parser.add_argument("--limit", type=int, help="仅写入前 N 个 chunk（用于 smoke test）")
    parser.add_argument("--report-output", type=Path, help="index report 输出路径")
    return parser.parse_args()
