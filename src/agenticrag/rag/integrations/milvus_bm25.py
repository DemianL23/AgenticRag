"""Configuration and analyzer settings for the V1.1 Milvus BM25 index."""

from __future__ import annotations

import os
from dataclasses import dataclass

from agenticrag.rag.integrations.milvus import (
    DEFAULT_MILVUS_URI,
    _read_bool,
    _read_optional,
)

DEFAULT_BM25_COLLECTION = "rag_v1_1_bm25_multilingual_800_120"

BM25_MULTI_ANALYZER_PARAMS = {
    "analyzers": {
        "english": {"type": "english"},
        "chinese": {"type": "chinese"},
        "default": {"tokenizer": "icu"},
    },
    "by_field": "language",
    "alias": {
        "cn": "chinese",
        "en": "english",
    },
}

BM25_INDEX_PARAMS = {
    "metric_type": "BM25",
    "index_type": "SPARSE_INVERTED_INDEX",
    "params": {"drop_ratio_build": 0.0},
}

BM25_SEARCH_PARAMS = {
    "metric_type": "BM25",
    "params": {"drop_ratio_search": 0.0},
}


@dataclass(frozen=True, slots=True)
class BM25MilvusConfig:
    """Connection and collection settings for the isolated BM25 index."""

    uri: str = DEFAULT_MILVUS_URI
    collection_name: str = DEFAULT_BM25_COLLECTION
    drop_old: bool = False

    @classmethod
    def from_env(cls) -> "BM25MilvusConfig":
        from dotenv import load_dotenv

        load_dotenv(override=False)
        return cls(
            uri=os.getenv("MILVUS_URI", DEFAULT_MILVUS_URI),
            collection_name=(
                _read_optional("BM25_MILVUS_COLLECTION") or DEFAULT_BM25_COLLECTION
            ),
            drop_old=_read_bool("BM25_MILVUS_DROP_OLD", default=False),
        )

    def validate(self) -> None:
        if not self.uri.strip():
            raise ValueError("BM25 Milvus URI 不能为空")
        if not self.collection_name.strip():
            raise ValueError("BM25 Milvus collection_name 不能为空")


def normalise_manifest_language(value: object) -> str:
    """Convert the manifest language code to the analyzer alias."""
    if not isinstance(value, str):
        raise ValueError("manifest language 必须是字符串")
    language = value.strip().lower()
    if language == "cn":
        return "cn"
    if language == "en":
        return "en"
    raise ValueError(f"BM25 暂不支持 manifest language：{value!r}，仅支持 CN/EN")

