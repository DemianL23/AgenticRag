"""Configuration for the Qwen generation model."""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


DEFAULT_GENERATION_MODEL = "qwen-plus"
DEFAULT_GENERATION_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_GENERATION_TEMPERATURE = 0.0
DEFAULT_GENERATION_MAX_TOKENS = 1024
DEFAULT_GENERATION_TIMEOUT = 60.0
DEFAULT_GENERATION_MAX_RETRIES = 2
DEFAULT_GENERATION_ENABLE_THINKING = False


@dataclass(frozen=True, slots=True)
class GenerationConfig:
    """Provider and decoding settings shared by every Qwen request."""

    api_key: str | None = None
    model: str = DEFAULT_GENERATION_MODEL
    base_url: str = DEFAULT_GENERATION_BASE_URL
    temperature: float = DEFAULT_GENERATION_TEMPERATURE
    max_tokens: int = DEFAULT_GENERATION_MAX_TOKENS
    timeout: float = DEFAULT_GENERATION_TIMEOUT
    max_retries: int = DEFAULT_GENERATION_MAX_RETRIES
    enable_thinking: bool = DEFAULT_GENERATION_ENABLE_THINKING

    @classmethod
    def from_env(cls, dotenv_path: Path | None = None) -> "GenerationConfig":
        """Read Qwen settings from the process environment and ``.env``."""
        try:
            from dotenv import load_dotenv
        except ImportError as exc:  # pragma: no cover - project dependency
            raise RuntimeError("读取 .env 需要 python-dotenv") from exc

        load_dotenv(dotenv_path=dotenv_path, override=False)
        config = cls(
            api_key=_read_optional("DASHSCOPE_API_KEY"),
            model=os.getenv("GENERATION_MODEL", DEFAULT_GENERATION_MODEL),
            base_url=os.getenv("DASHSCOPE_BASE_URL", DEFAULT_GENERATION_BASE_URL),
            temperature=_read_float(
                "GENERATION_TEMPERATURE", DEFAULT_GENERATION_TEMPERATURE
            ),
            max_tokens=_read_int("GENERATION_MAX_TOKENS", DEFAULT_GENERATION_MAX_TOKENS),
            timeout=_read_float("GENERATION_TIMEOUT", DEFAULT_GENERATION_TIMEOUT),
            max_retries=_read_int(
                "GENERATION_MAX_RETRIES", DEFAULT_GENERATION_MAX_RETRIES
            ),
            enable_thinking=_read_bool(
                "GENERATION_ENABLE_THINKING", DEFAULT_GENERATION_ENABLE_THINKING
            ),
        )
        config.validate(require_api_key=False)
        return config

    def validate(self, *, require_api_key: bool = True) -> None:
        if require_api_key and not self.api_key:
            raise ValueError("DASHSCOPE_API_KEY 未配置")
        if not self.model.strip():
            raise ValueError("generation model 不能为空")
        if not self.base_url.strip():
            raise ValueError("DASHSCOPE_BASE_URL 不能为空")
        if self.temperature < 0:
            raise ValueError("generation temperature 不能小于 0")
        if self.max_tokens <= 0:
            raise ValueError("generation max_tokens 必须大于 0")
        if self.timeout <= 0:
            raise ValueError("generation timeout 必须大于 0")
        if self.max_retries < 0:
            raise ValueError("generation max_retries 不能小于 0")

    def to_record(self) -> dict[str, Any]:
        """Return safe configuration metadata without exposing the API key."""
        record = asdict(self)
        record["api_key"] = None
        record["api_key_configured"] = bool(self.api_key)
        return record


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
