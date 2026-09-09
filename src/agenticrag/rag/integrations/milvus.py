"""Milvus connection settings for the local RAG index."""

from __future__ import annotations

import os
from dataclasses import dataclass

DEFAULT_MILVUS_URI = "http://localhost:19530"


@dataclass(frozen=True, slots=True)
class MilvusConfig:
    """Connection and collection settings, kept separate from RAG logic."""

    uri: str = DEFAULT_MILVUS_URI
    collection_name: str | None = None
    drop_old: bool = False

    @classmethod
    def from_env(cls) -> "MilvusConfig":
        from dotenv import load_dotenv

        load_dotenv(override=False)
        return cls(
            uri=os.getenv("MILVUS_URI", DEFAULT_MILVUS_URI),
            collection_name=_read_optional("MILVUS_COLLECTION"),
            drop_old=_read_bool("MILVUS_DROP_OLD", default=False),
        )

    def validate(self) -> None:
        if not self.uri.strip():
            raise ValueError("Milvus URI 不能为空")
        if self.collection_name is not None and not self.collection_name.strip():
            raise ValueError("Milvus collection_name 不能为空字符串")


def _read_optional(name: str) -> str | None:
    value = os.getenv(name)
    return value.strip() if value and value.strip() else None


def _read_bool(name: str, *, default: bool) -> bool:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"{name} 必须是 true/false")
