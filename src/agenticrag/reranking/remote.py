"""HTTP client implementation for the remote vLLM BGE reranker."""

from __future__ import annotations

import json
import math
import time
from collections.abc import Sequence
from numbers import Real
from typing import Any
from urllib.request import Request, urlopen

from agenticrag.reranking.base import BaseReranker, RerankerInferenceError
from agenticrag.reranking.config import RemoteRerankerConfig
from agenticrag.retrieval.schemas import HybridRetrievedChunk


class RemoteBGEReranker(BaseReranker):
    """Score every candidate through vLLM's ``/v1/rerank`` API."""

    def __init__(self, config: RemoteRerankerConfig | None = None) -> None:
        self.config = config or RemoteRerankerConfig.from_env()
        self.config.validate()
        self._loaded = False
        self._last_request_seconds: float | None = None
        self._last_candidate_count: int = 0
        self._last_error: str | None = None

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def load(self) -> None:
        """Mark the stateless HTTP client ready without making a network call."""
        self.config.validate()
        self._loaded = True

    def score(
        self,
        query: str,
        candidates: Sequence[HybridRetrievedChunk],
    ) -> Sequence[object]:
        self.load()
        if not candidates:
            return []

        payload = {
            "model": self.config.model,
            "query": query,
            "documents": [candidate.content for candidate in candidates],
            "top_n": len(candidates),
        }
        started = time.perf_counter()
        self._last_candidate_count = len(candidates)
        try:
            response = _post_json(
                self.config.endpoint,
                payload,
                timeout_seconds=self.config.timeout_seconds,
            )
            scores = _scores_from_response(response, expected=len(candidates))
            self._last_error = None
            return scores
        except RerankerInferenceError as exc:
            self._last_error = str(exc)
            raise
        except Exception as exc:  # noqa: BLE001 - normalize all remote failures
            error = RerankerInferenceError(
                f"远程 Reranker 请求或响应无效：{type(exc).__name__}: {exc}"
            )
            self._last_error = str(error)
            raise error from exc
        finally:
            self._last_request_seconds = time.perf_counter() - started

    def model_record(self) -> dict[str, Any]:
        return {
            **self.config.to_record(),
            "backend": "remote",
            "backend_type": "remote",
            "provider": "vllm",
            "request_seconds": self._last_request_seconds,
            "candidate_count": self._last_candidate_count,
            "last_error": self._last_error,
            "loaded": self.is_loaded,
            "load_failed": False,
        }


def _post_json(url: str, payload: dict[str, Any], *, timeout_seconds: float) -> Any:
    request = Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=timeout_seconds) as response:  # noqa: S310 - configured endpoint
        status = getattr(response, "status", None)
        if status is not None and not 200 <= status < 300:
            raise ValueError(f"HTTP status={status}")
        body = response.read()
    return json.loads(body.decode("utf-8"))


def _scores_from_response(response: Any, *, expected: int) -> list[float]:
    if not isinstance(response, dict):
        raise ValueError("响应 JSON 必须是对象")
    results = response.get("results")
    if not isinstance(results, list):
        raise ValueError("响应缺少合法 results 列表")
    if len(results) != expected:
        raise ValueError(
            f"results 数量异常：expected={expected}, actual={len(results)}"
        )

    scores: list[float | None] = [None] * expected
    for item in results:
        if not isinstance(item, dict):
            raise ValueError("results item 必须是对象")
        index = item.get("index")
        if isinstance(index, bool) or not isinstance(index, int):
            raise ValueError("results item 缺少合法 index")
        if index < 0 or index >= expected:
            raise ValueError(f"results index 越界：{index}")
        if scores[index] is not None:
            raise ValueError(f"results index 重复：{index}")

        raw_score = item.get("relevance_score")
        if isinstance(raw_score, bool) or not isinstance(raw_score, Real):
            raise ValueError(f"index={index} 缺少合法 relevance_score")
        score = float(raw_score)
        if not math.isfinite(score):
            raise ValueError(f"index={index} relevance_score 不是有限数值")
        scores[index] = score

    if any(score is None for score in scores):
        raise ValueError("并非所有 candidate 都获得了 score")
    return [score for score in scores if score is not None]
