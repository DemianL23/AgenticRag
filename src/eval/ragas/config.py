"""Configuration dedicated to the RAGAS judge and evaluator embeddings."""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from agenticrag.integrations.embeddings import (
    DEFAULT_EMBEDDING_BATCH_SIZE,
    DEFAULT_EMBEDDING_DEVICE,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_EMBEDDING_NORMALIZE,
)

DEFAULT_RAGAS_EVALUATOR_TEMPERATURE = 0.0
DEFAULT_RAGAS_EVALUATOR_MAX_TOKENS = 4096
DEFAULT_RAGAS_EVALUATOR_TIMEOUT = 120.0
DEFAULT_RAGAS_EVALUATOR_MAX_RETRIES = 2
DEFAULT_RAGAS_EVALUATOR_ENABLE_THINKING = False


@dataclass(frozen=True, slots=True)
class RagasEvaluatorConfig:
    """Settings for the judge; independent from the RAG generation config."""

    model: str
    base_url: str
    api_key: str | None = None
    temperature: float = DEFAULT_RAGAS_EVALUATOR_TEMPERATURE
    max_tokens: int = DEFAULT_RAGAS_EVALUATOR_MAX_TOKENS
    timeout: float = DEFAULT_RAGAS_EVALUATOR_TIMEOUT
    max_retries: int = DEFAULT_RAGAS_EVALUATOR_MAX_RETRIES
    enable_thinking: bool = DEFAULT_RAGAS_EVALUATOR_ENABLE_THINKING
    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    embedding_device: str = DEFAULT_EMBEDDING_DEVICE
    embedding_batch_size: int = DEFAULT_EMBEDDING_BATCH_SIZE
    embedding_normalize: bool = DEFAULT_EMBEDDING_NORMALIZE
    embedding_query_prompt_name: str | None = None
    embedding_trust_remote_code: bool = False

    @classmethod
    def from_env(cls, dotenv_path: Path | None = None) -> RagasEvaluatorConfig:
        """Read evaluator settings without coupling them to GenerationConfig."""
        try:
            from dotenv import load_dotenv
        except ImportError as exc:  # pragma: no cover - project dependency
            raise RuntimeError("读取 .env 需要 python-dotenv") from exc

        load_dotenv(dotenv_path=dotenv_path, override=False)
        embedding_model = os.getenv(
            "RAGAS_EVALUATOR_EMBEDDING_MODEL",
            os.getenv("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL),
        )
        configured_prompt = _read_optional(
            "RAGAS_EVALUATOR_EMBEDDING_QUERY_PROMPT_NAME"
        )
        config = cls(
            model=_read_required("RAGAS_EVALUATOR_MODEL"),
            base_url=_read_required("RAGAS_EVALUATOR_BASE_URL"),
            api_key=_read_optional("RAGAS_EVALUATOR_API_KEY")
            or _read_optional("DASHSCOPE_API_KEY"),
            temperature=_read_float(
                "RAGAS_EVALUATOR_TEMPERATURE", DEFAULT_RAGAS_EVALUATOR_TEMPERATURE
            ),
            max_tokens=_read_int(
                "RAGAS_EVALUATOR_MAX_TOKENS", DEFAULT_RAGAS_EVALUATOR_MAX_TOKENS
            ),
            timeout=_read_float(
                "RAGAS_EVALUATOR_TIMEOUT", DEFAULT_RAGAS_EVALUATOR_TIMEOUT
            ),
            max_retries=_read_int(
                "RAGAS_EVALUATOR_MAX_RETRIES", DEFAULT_RAGAS_EVALUATOR_MAX_RETRIES
            ),
            enable_thinking=_read_bool(
                "RAGAS_EVALUATOR_ENABLE_THINKING",
                DEFAULT_RAGAS_EVALUATOR_ENABLE_THINKING,
            ),
            embedding_model=embedding_model,
            embedding_device=os.getenv(
                "RAGAS_EVALUATOR_EMBEDDING_DEVICE",
                os.getenv("EMBEDDING_DEVICE", DEFAULT_EMBEDDING_DEVICE),
            ),
            embedding_batch_size=_read_int(
                "RAGAS_EVALUATOR_EMBEDDING_BATCH_SIZE",
                _read_int("EMBEDDING_BATCH_SIZE", DEFAULT_EMBEDDING_BATCH_SIZE),
            ),
            embedding_normalize=_read_bool(
                "RAGAS_EVALUATOR_EMBEDDING_NORMALIZE",
                _read_bool("EMBEDDING_NORMALIZE", DEFAULT_EMBEDDING_NORMALIZE),
            ),
            embedding_query_prompt_name=(
                configured_prompt
                if configured_prompt is not None
                else _default_query_prompt_name(embedding_model)
            ),
            embedding_trust_remote_code=_read_bool(
                "RAGAS_EVALUATOR_EMBEDDING_TRUST_REMOTE_CODE", False
            ),
        )
        config.validate(require_api_key=False)
        return config

    def validate(self, *, require_api_key: bool = True) -> None:
        if require_api_key and not self.api_key:
            raise ValueError(
                "RAGAS_EVALUATOR_API_KEY 未配置，且没有可复用的 DASHSCOPE_API_KEY"
            )
        for field_name, value in (
            ("evaluator model", self.model),
            ("evaluator base_url", self.base_url),
            ("evaluator embedding_model", self.embedding_model),
            ("evaluator embedding_device", self.embedding_device),
        ):
            if not value.strip():
                raise ValueError(f"{field_name} 不能为空")
        if self.temperature < 0:
            raise ValueError("RAGAS evaluator temperature 不能小于 0")
        if self.max_tokens <= 0:
            raise ValueError("RAGAS evaluator max_tokens 必须大于 0")
        if self.timeout <= 0:
            raise ValueError("RAGAS evaluator timeout 必须大于 0")
        if self.max_retries < 0:
            raise ValueError("RAGAS evaluator max_retries 不能小于 0")
        if self.embedding_batch_size <= 0:
            raise ValueError("RAGAS evaluator embedding_batch_size 必须大于 0")

    def to_record(self) -> dict[str, Any]:
        """Return report-safe settings without exposing the API key."""
        record = asdict(self)
        record["api_key"] = None
        record["api_key_configured"] = bool(self.api_key)
        return record


def _default_query_prompt_name(model_name: str) -> str | None:
    return "query" if "qwen3-embedding" in model_name.lower() else None


def _read_required(name: str) -> str:
    value = _read_optional(name)
    if value is None:
        raise ValueError(f"{name} 未配置，请在 .env 中显式设置")
    return value


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


def _read_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(f"{name} 必须是数字") from exc


def _read_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"{name} 必须是 true/false")
