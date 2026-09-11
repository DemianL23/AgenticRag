"""Select the configured V1.2 reranker implementation."""

from __future__ import annotations

from pathlib import Path

from agenticrag.reranking.base import BaseReranker
from agenticrag.reranking.bge import BGEReranker
from agenticrag.reranking.config import (
    RemoteRerankerConfig,
    RerankerConfig,
    reranker_backend_from_env,
)
from agenticrag.reranking.remote import RemoteBGEReranker


def create_reranker(dotenv_path: Path | None = None) -> BaseReranker:
    """Create local or remote V1.2 reranking from environment configuration."""
    backend = reranker_backend_from_env(dotenv_path)
    if backend == "remote":
        return RemoteBGEReranker(RemoteRerankerConfig.from_env(dotenv_path))
    return BGEReranker(RerankerConfig.from_env(dotenv_path))
