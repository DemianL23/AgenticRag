"""Configurable local text-embedding integrations.

The model name is configuration, not application logic. This lets us compare
Qwen3-Embedding and BGE-M3 without changing the indexing or retrieval code.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from langchain_core.embeddings import Embeddings


DEFAULT_EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-0.6B"
DEFAULT_EMBEDDING_DEVICE = "cpu"
DEFAULT_EMBEDDING_BATCH_SIZE = 32
DEFAULT_EMBEDDING_NORMALIZE = True


@dataclass(frozen=True, slots=True)
class EmbeddingConfig:
    """Configuration shared by document and query embedding."""

    model_name: str = DEFAULT_EMBEDDING_MODEL
    device: str = DEFAULT_EMBEDDING_DEVICE
    batch_size: int = DEFAULT_EMBEDDING_BATCH_SIZE
    normalize_embeddings: bool = DEFAULT_EMBEDDING_NORMALIZE
    query_prompt_name: str | None = None
    cache_folder: str | None = None
    model_kwargs: dict[str, Any] | None = None

    @classmethod
    def from_env(cls, dotenv_path: Path | None = None) -> "EmbeddingConfig":
        """Read ``EMBEDDING_*`` settings from ``.env`` and process variables."""
        try:
            from dotenv import load_dotenv
        except ImportError as exc:  # pragma: no cover - protected by project dependency
            raise RuntimeError("读取 .env 需要 python-dotenv") from exc

        load_dotenv(dotenv_path=dotenv_path, override=False)
        model_name = os.getenv("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)
        configured_prompt = os.getenv("EMBEDDING_QUERY_PROMPT_NAME")
        query_prompt_name = (
            configured_prompt.strip()
            if configured_prompt is not None and configured_prompt.strip()
            else _default_query_prompt_name(model_name)
        )
        model_kwargs: dict[str, Any] | None = None
        if _read_bool("EMBEDDING_TRUST_REMOTE_CODE", default=False):
            model_kwargs = {"trust_remote_code": True}

        config = cls(
            model_name=model_name,
            device=os.getenv("EMBEDDING_DEVICE", DEFAULT_EMBEDDING_DEVICE),
            batch_size=_read_int("EMBEDDING_BATCH_SIZE", DEFAULT_EMBEDDING_BATCH_SIZE),
            normalize_embeddings=_read_bool(
                "EMBEDDING_NORMALIZE", default=DEFAULT_EMBEDDING_NORMALIZE
            ),
            query_prompt_name=query_prompt_name,
            cache_folder=_read_optional("EMBEDDING_CACHE_FOLDER"),
            model_kwargs=model_kwargs,
        )
        config.validate()
        return config

    def validate(self) -> None:
        if not self.model_name.strip():
            raise ValueError("model_name 不能为空")
        if not self.device.strip():
            raise ValueError("device 不能为空")
        if self.batch_size <= 0:
            raise ValueError("batch_size 必须大于 0")

    def resolved_model_kwargs(self) -> dict[str, Any]:
        self.validate()
        return {"device": self.device, **(self.model_kwargs or {})}

    def resolved_document_encode_kwargs(self) -> dict[str, Any]:
        self.validate()
        return {
            "batch_size": self.batch_size,
            "normalize_embeddings": self.normalize_embeddings,
        }

    def resolved_query_encode_kwargs(self) -> dict[str, Any]:
        kwargs = self.resolved_document_encode_kwargs()
        prompt_name = self.effective_query_prompt_name()
        if prompt_name:
            kwargs["prompt_name"] = prompt_name
        return kwargs

    def effective_query_prompt_name(self) -> str | None:
        return self.query_prompt_name or _default_query_prompt_name(self.model_name)

    def to_record(self) -> dict[str, Any]:
        record = asdict(self)
        record["model_kwargs"] = self.model_kwargs or {}
        record["query_prompt_name"] = self.effective_query_prompt_name()
        return record


def create_embeddings(config: EmbeddingConfig | None = None) -> Embeddings:
    """Create the configured local Hugging Face embedding model.

    Model weights are downloaded by ``sentence-transformers`` on first use and
    then reused from its local cache. The import is intentionally lazy so PDF
    parsing and chunking do not require the optional ML stack.
    """
    config = config or EmbeddingConfig.from_env()
    config.validate()
    try:
        from langchain_huggingface import HuggingFaceEmbeddings
    except ImportError as exc:
        raise RuntimeError(
            "本机 Embedding 需要可选依赖，请执行：uv sync --extra embeddings"
        ) from exc

    return HuggingFaceEmbeddings(
        model_name=config.model_name,
        cache_folder=config.cache_folder,
        model_kwargs=config.resolved_model_kwargs(),
        encode_kwargs=config.resolved_document_encode_kwargs(),
        query_encode_kwargs=config.resolved_query_encode_kwargs(),
    )


def create_embeddings_from_env(dotenv_path: Path | None = None) -> Embeddings:
    """Load ``.env`` settings and create the configured embedding model."""
    return create_embeddings(EmbeddingConfig.from_env(dotenv_path))


def embedding_dimension(embeddings: Embeddings) -> int:
    """Return the vector dimension by probing one short string."""
    vector = embeddings.embed_query("dimension probe")
    if not vector:
        raise ValueError("Embedding 模型返回了空向量")
    return len(vector)


def _default_query_prompt_name(model_name: str) -> str | None:
    if "qwen3-embedding" in model_name.lower():
        return "query"
    return None


def _read_optional(name: str) -> str | None:
    value = os.getenv(name)
    return value.strip() if value and value.strip() else None


def _read_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"{name} 必须是整数") from exc


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
