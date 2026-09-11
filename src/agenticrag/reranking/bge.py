"""FlagEmbedding-backed implementation of the local BGE reranker."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Sequence

from agenticrag.reranking.base import (
    BaseReranker,
    RerankerInferenceError,
    RerankerLoadError,
)
from agenticrag.reranking.config import RerankerConfig
from agenticrag.retrieval.schemas import HybridRetrievedChunk


class BGEReranker(BaseReranker):
    """Lazily load one pinned BGE snapshot and score query/chunk pairs."""

    def __init__(self, config: RerankerConfig | None = None) -> None:
        self.config = config or RerankerConfig.from_env()
        self.config.validate()
        self._model: Any | None = None
        self._load_error: Exception | None = None
        self._load_seconds: float | None = None
        self._resolved_snapshot_path: str | None = None
        self._resolved_revision: str | None = None

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def load(self) -> None:
        if self._model is not None:
            return
        if self._load_error is not None:
            raise RerankerLoadError("Reranker 模型此前加载失败") from self._load_error

        started = time.perf_counter()
        try:
            from FlagEmbedding import FlagReranker
            from huggingface_hub import snapshot_download

            snapshot_path = snapshot_download(
                repo_id=self.config.model_name,
                revision=self.config.model_revision,
                cache_dir=self.config.cache_dir,
                local_files_only=self.config.local_files_only,
            )
            self._resolved_snapshot_path = str(snapshot_path)
            self._resolved_revision = _snapshot_revision(snapshot_path)
            self._model = FlagReranker(
                str(snapshot_path),
                use_fp16=self.config.use_fp16,
                devices=self.config.device,
            )
        except Exception as exc:
            self._load_error = exc
            raise RerankerLoadError(
                "无法加载本地 Reranker；请安装 reranking extra 并确认模型缓存/网络配置"
            ) from exc
        finally:
            self._load_seconds = time.perf_counter() - started

    def score(
        self,
        query: str,
        candidates: Sequence[HybridRetrievedChunk],
    ) -> Sequence[object]:
        self.load()
        if not candidates:
            return []

        pairs = [[query, candidate.content] for candidate in candidates]
        try:
            raw_scores = self._model.compute_score(
                pairs,
                batch_size=self.config.batch_size,
                max_length=self.config.max_length,
                normalize=False,
            )
        except Exception as exc:
            raise RerankerInferenceError("BGE Reranker 推理失败") from exc

        if hasattr(raw_scores, "tolist"):
            raw_scores = raw_scores.tolist()
        if isinstance(raw_scores, (int, float)):
            return [raw_scores]
        if isinstance(raw_scores, Sequence) and not isinstance(
            raw_scores, (str, bytes, bytearray)
        ):
            return list(raw_scores)
        return [raw_scores]

    def model_record(self) -> dict[str, Any]:
        return {
            **self.config.to_record(),
            "backend": "FlagEmbedding.FlagReranker",
            "backend_type": "local",
            "query_max_length": None,
            "effective_default_query_max_length": self.config.max_length * 3 // 4,
            "resolved_revision": self._resolved_revision,
            "resolved_snapshot_path": self._resolved_snapshot_path,
            "load_seconds": self._load_seconds,
            "loaded": self.is_loaded,
            "load_failed": self._load_error is not None,
        }


def _snapshot_revision(snapshot_path: str | Path) -> str | None:
    path = Path(snapshot_path)
    if path.parent.name == "snapshots" and len(path.name) == 40:
        return path.name
    return None
