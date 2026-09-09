"""Build the V1.1 Milvus built-in BM25 collection."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from langchain_core.documents import Document

from agenticrag.ingestion.chunk import load_documents_jsonl
from agenticrag.rag.integrations.milvus_bm25 import (
    BM25_INDEX_PARAMS,
    BM25_MULTI_ANALYZER_PARAMS,
    BM25_SEARCH_PARAMS,
    BM25MilvusConfig,
    normalise_manifest_language,
)

DEFAULT_MANIFEST = Path("artifacts/manifests/v0_documents.jsonl")
DEFAULT_REPORT = Path("artifacts/chunks/pymupdf/v0/bm25_milvus_report.json")


def build_bm25_index(
    chunks_dir: Path,
    *,
    manifest_path: Path = DEFAULT_MANIFEST,
    milvus_config: BM25MilvusConfig | None = None,
    report_output: Path = DEFAULT_REPORT,
) -> dict[str, Any]:
    """Create an isolated Milvus BM25 collection from the existing chunks."""
    chunks_dir = Path(chunks_dir)
    manifest_path = Path(manifest_path)
    files = sorted(chunks_dir.glob("doc_*.jsonl"))
    if not files:
        raise FileNotFoundError(f"chunk 目录中没有 doc_*.jsonl：{chunks_dir}")

    milvus_config = milvus_config or BM25MilvusConfig.from_env()
    milvus_config.validate()
    language_by_doc = _load_language_manifest(manifest_path)
    raw_documents = [
        document
        for path in files
        for document in load_documents_jsonl(path)
    ]
    documents = _prepare_documents(raw_documents, language_by_doc)
    ids = [_chunk_id(document.metadata) for document in documents]
    if len(ids) != len(set(ids)):
        raise ValueError("chunk_id 不唯一，拒绝写入 BM25 collection")

    if _collection_exists(milvus_config.uri, milvus_config.collection_name) and not milvus_config.drop_old:
        raise ValueError(
            f"BM25 collection 已存在：{milvus_config.collection_name}；"
            "如需重建请显式设置 BM25_MILVUS_DROP_OLD=true 或传入 --drop-old"
        )

    try:
        from langchain_milvus import BM25BuiltInFunction, Milvus
        from pymilvus import DataType
    except ImportError as exc:
        raise RuntimeError(
            "Milvus BM25 建库需要可选依赖，请执行：uv sync --extra milvus"
        ) from exc

    bm25_function = BM25BuiltInFunction(
        multi_analyzer_params=BM25_MULTI_ANALYZER_PARAMS,
        function_name="v1_1_bm25_function",
    )
    metadata_schema = _metadata_schema(DataType)
    Milvus.from_documents(
        documents=documents,
        embedding=None,
        ids=ids,
        collection_name=milvus_config.collection_name,
        connection_args={"uri": milvus_config.uri},
        drop_old=milvus_config.drop_old,
        builtin_function=bm25_function,
        vector_field="sparse",
        index_params=BM25_INDEX_PARAMS,
        search_params=BM25_SEARCH_PARAMS,
        metadata_schema=metadata_schema,
    )

    report = {
        "version": "v1.1",
        "retriever": "milvus_bm25",
        "chunks_dir": chunks_dir.as_posix(),
        "manifest_path": manifest_path.as_posix(),
        "collection_name": milvus_config.collection_name,
        "milvus_uri": milvus_config.uri,
        "drop_old": milvus_config.drop_old,
        "documents": len(files),
        "chunks": len(documents),
        "languages": dict(sorted(Counter(_language(document) for document in documents).items())),
        "bm25": {
            "analyzer": BM25_MULTI_ANALYZER_PARAMS,
            "index_params": BM25_INDEX_PARAMS,
            "search_params": BM25_SEARCH_PARAMS,
        },
        "report_output": Path(report_output).as_posix(),
    }
    _write_json(report, Path(report_output))
    return report


def _load_language_manifest(path: Path) -> dict[str, str]:
    if not path.is_file():
        raise FileNotFoundError(f"manifest 不存在：{path}")

    languages: dict[str, str] = {}
    with path.open("r", encoding="utf-8") as file:
        for line_number, raw_line in enumerate(file, start=1):
            if not raw_line.strip():
                continue
            try:
                record = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"manifest 第 {line_number} 行不是合法 JSON：{path}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"manifest 第 {line_number} 行必须是 JSON 对象：{path}")
            doc_id = record.get("doc_id")
            if not isinstance(doc_id, str) or not doc_id.strip():
                raise ValueError(f"manifest 第 {line_number} 行缺少合法 doc_id：{path}")
            key = doc_id.strip()
            if key in languages:
                raise ValueError(f"manifest 存在重复 doc_id：{key}")
            languages[key] = normalise_manifest_language(record.get("language"))
    if not languages:
        raise ValueError(f"manifest 不能为空：{path}")
    return languages


def _prepare_documents(
    documents: list[Document],
    language_by_doc: dict[str, str],
) -> list[Document]:
    prepared: list[Document] = []
    for document in documents:
        metadata = document.metadata
        doc_id = _required_string(metadata.get("doc_id"), "doc_id")
        language = language_by_doc.get(doc_id)
        if language is None:
            raise ValueError(f"chunk 的 doc_id 不在 manifest 中：{doc_id}")
        prepared.append(
            Document(
                page_content=document.page_content,
                metadata={
                    "doc_id": doc_id,
                    "source": _required_string(metadata.get("source"), "source"),
                    "page_number": _positive_int(metadata.get("page_number"), "page_number"),
                    "chunk_id": _required_string(metadata.get("chunk_id"), "chunk_id"),
                    "language": language,
                },
            )
        )
    return prepared


def _metadata_schema(data_type: Any) -> dict[str, dict[str, Any]]:
    return {
        "doc_id": {"dtype": data_type.VARCHAR, "max_length": 256},
        "source": {"dtype": data_type.VARCHAR, "max_length": 65_535},
        "page_number": {"dtype": data_type.INT64},
        "chunk_id": {"dtype": data_type.VARCHAR, "max_length": 256},
        "language": {"dtype": data_type.VARCHAR, "max_length": 2},
    }


def _collection_exists(uri: str, collection_name: str) -> bool:
    try:
        from pymilvus import MilvusClient
    except ImportError as exc:
        raise RuntimeError(
            "Milvus BM25 建库需要 pymilvus，请执行：uv sync --extra milvus"
        ) from exc
    return bool(MilvusClient(uri=uri).has_collection(collection_name))


def _chunk_id(metadata: dict[str, Any]) -> str:
    return _required_string(metadata.get("chunk_id"), "chunk_id")


def _language(document: Document) -> str:
    return _required_string(document.metadata.get("language"), "language")


def _required_string(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"chunk 缺少合法 {field_name}")
    return value.strip()


def _positive_int(value: object, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"chunk 缺少合法 {field_name}")
    return value


def _write_json(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.tmp")
    temporary_path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(path)


def main() -> None:
    args = _parse_args()
    base_config = BM25MilvusConfig.from_env()
    config = BM25MilvusConfig(
        uri=args.uri or base_config.uri,
        collection_name=args.collection_name or base_config.collection_name,
        drop_old=args.drop_old or base_config.drop_old,
    )
    report = build_bm25_index(
        args.chunks_dir,
        manifest_path=args.manifest,
        milvus_config=config,
        report_output=args.report_output,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="使用 Milvus 内置 BM25 建立 V1.1 collection。")
    parser.add_argument("--chunks-dir", type=Path, required=True, help="chunk JSONL 所在目录")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST, help="文档 manifest JSONL")
    parser.add_argument("--uri", help="Milvus 地址，默认读取 MILVUS_URI")
    parser.add_argument("--collection-name", help="BM25 collection，默认读取 BM25_MILVUS_COLLECTION")
    parser.add_argument("--drop-old", action="store_true", help="写入前删除同名 BM25 collection")
    parser.add_argument("--report-output", type=Path, default=DEFAULT_REPORT, help="建库报告路径")
    return parser.parse_args()
