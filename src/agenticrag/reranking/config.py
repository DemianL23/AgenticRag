"""Environment-driven configuration for the V1.2 local reranker."""

from __future__ import annotations

import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


DEFAULT_RERANK_MODEL = "BAAI/bge-reranker-v2-m3"
DEFAULT_RERANK_MODEL_REVISION = "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e"
DEFAULT_RERANK_DEVICE = "cpu"
DEFAULT_RERANK_BATCH_SIZE = 8
DEFAULT_RERANK_MAX_LENGTH = 512
DEFAULT_RERANK_USE_FP16 = False
DEFAULT_RERANK_LOCAL_FILES_ONLY = False
_DEVICE_PATTERN = re.compile(r"^(cpu|cuda(?::\d+)?)$")


@dataclass(frozen=True, slots=True)
class RerankerConfig:
    """Loading and inference settings for a replaceable reranker model."""

    model_name: str = DEFAULT_RERANK_MODEL
    model_revision: str | None = DEFAULT_RERANK_MODEL_REVISION
    device: str = DEFAULT_RERANK_DEVICE
    batch_size: int = DEFAULT_RERANK_BATCH_SIZE
    max_length: int = DEFAULT_RERANK_MAX_LENGTH
    use_fp16: bool = DEFAULT_RERANK_USE_FP16
    local_files_only: bool = DEFAULT_RERANK_LOCAL_FILES_ONLY
    cache_dir: str | None = None

    @classmethod
    def from_env(cls, dotenv_path: Path | None = None) -> "RerankerConfig":
        """Read ``RERANK_*`` settings from the environment and optional .env."""
        try:
            from dotenv import load_dotenv
        except ImportError as exc:  # pragma: no cover - project dependency
            raise RuntimeError("读取 .env 需要 python-dotenv") from exc

        load_dotenv(dotenv_path=dotenv_path, override=False)
        model_name = os.getenv("RERANK_MODEL", DEFAULT_RERANK_MODEL).strip()
        configured_revision = _read_optional("RERANK_MODEL_REVISION")
        model_revision = configured_revision
        if configured_revision is None and model_name == DEFAULT_RERANK_MODEL:
            model_revision = DEFAULT_RERANK_MODEL_REVISION

        config = cls(
            model_name=model_name,
            model_revision=model_revision,
            device=os.getenv("RERANK_DEVICE", DEFAULT_RERANK_DEVICE).strip().lower(),
            batch_size=_read_int("RERANK_BATCH_SIZE", DEFAULT_RERANK_BATCH_SIZE),
            max_length=_read_int("RERANK_MAX_LENGTH", DEFAULT_RERANK_MAX_LENGTH),
            use_fp16=_read_bool("RERANK_USE_FP16", DEFAULT_RERANK_USE_FP16),
            local_files_only=_read_bool(
                "RERANK_LOCAL_FILES_ONLY", DEFAULT_RERANK_LOCAL_FILES_ONLY
            ),
            cache_dir=_read_optional("RERANK_CACHE_DIR"),
        )
        config.validate()
        return config

    def validate(self) -> None:
        if not self.model_name:
            raise ValueError("RERANK_MODEL 不能为空")
        if self.model_revision is not None and not self.model_revision.strip():
            raise ValueError("RERANK_MODEL_REVISION 不能为空字符串")
        if not _DEVICE_PATTERN.fullmatch(self.device):
            raise ValueError("RERANK_DEVICE 必须是 cpu、cuda 或 cuda:<index>")
        if self.batch_size <= 0:
            raise ValueError("RERANK_BATCH_SIZE 必须大于 0")
        if self.max_length <= 0:
            raise ValueError("RERANK_MAX_LENGTH 必须大于 0")
        if self.use_fp16 and not self.device.startswith("cuda"):
            raise ValueError("RERANK_USE_FP16=true 只允许用于 CUDA")

    def to_record(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)


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
