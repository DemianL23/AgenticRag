"""Remote Qwen embedding integration for an OpenAI-compatible vLLM service."""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import asdict, dataclass
from numbers import Real
from typing import Any, Sequence
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from langchain_core.embeddings import Embeddings

from agenticrag.rag.integrations.embeddings import EmbeddingConfig

DEFAULT_REMOTE_EMBEDDING_TIMEOUT_SECONDS = 30.0
DEFAULT_REMOTE_EMBEDDING_DIMENSION = 1024
DEFAULT_REMOTE_MAX_MODEL_LEN = 4096
DEFAULT_REMOTE_EMBEDDING_BATCH_SIZE = 32
DOCUMENT_PROMPT_PROFILE = "raw_document_v1"
QWEN3_QUERY_PROMPT = (
    "Instruct: Given a web search query, retrieve relevant passages that answer the query\n"
    "Query:"
)


class RemoteEmbeddingError(RuntimeError):
    """The remote embedding service returned an unusable result or failed."""


@dataclass(frozen=True, slots=True)
class RemoteEmbeddingConfig:
    """Connection and compatibility settings for the remote Qwen service."""

    url: str = "http://127.0.0.1:8002"
    model: str = "Qwen/Qwen3-Embedding-0.6B"
    timeout_seconds: float = DEFAULT_REMOTE_EMBEDDING_TIMEOUT_SECONDS
    expected_dimension: int = DEFAULT_REMOTE_EMBEDDING_DIMENSION
    max_model_len: int = DEFAULT_REMOTE_MAX_MODEL_LEN
    batch_size: int = DEFAULT_REMOTE_EMBEDDING_BATCH_SIZE
    model_revision: str | None = None

    @classmethod
    def from_env(cls, *, fallback_model: str) -> "RemoteEmbeddingConfig":
        config = cls(
            url=os.getenv("EMBEDDING_REMOTE_URL", "http://127.0.0.1:8002"),
            model=os.getenv("EMBEDDING_REMOTE_MODEL", fallback_model),
            timeout_seconds=_read_float(
                "EMBEDDING_REMOTE_TIMEOUT_SECONDS",
                DEFAULT_REMOTE_EMBEDDING_TIMEOUT_SECONDS,
            ),
            expected_dimension=_read_int(
                "EMBEDDING_REMOTE_DIMENSION",
                DEFAULT_REMOTE_EMBEDDING_DIMENSION,
            ),
            max_model_len=_read_int(
                "EMBEDDING_REMOTE_MAX_MODEL_LEN",
                DEFAULT_REMOTE_MAX_MODEL_LEN,
            ),
            batch_size=_read_int(
                "EMBEDDING_REMOTE_BATCH_SIZE",
                DEFAULT_REMOTE_EMBEDDING_BATCH_SIZE,
            ),
            model_revision=_read_optional("EMBEDDING_REMOTE_MODEL_REVISION"),
        )
        config.validate()
        return config

    @property
    def endpoint(self) -> str:
        return f"{self.url.rstrip('/')}/v1/embeddings"

    def validate(self) -> None:
        parsed = urlparse(self.url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("EMBEDDING_REMOTE_URL 必须是合法的 http(s) URL")
        if not self.model.strip():
            raise ValueError("EMBEDDING_REMOTE_MODEL 不能为空")
        if not math.isfinite(self.timeout_seconds) or self.timeout_seconds <= 0:
            raise ValueError("EMBEDDING_REMOTE_TIMEOUT_SECONDS 必须是正数")
        if self.expected_dimension <= 0:
            raise ValueError("EMBEDDING_REMOTE_DIMENSION 必须是正整数")
        if self.max_model_len <= 0:
            raise ValueError("EMBEDDING_REMOTE_MAX_MODEL_LEN 必须是正整数")
        if self.batch_size <= 0:
            raise ValueError("EMBEDDING_REMOTE_BATCH_SIZE 必须是正整数")

    def to_record(self) -> dict[str, Any]:
        record = asdict(self)
        record["endpoint"] = self.endpoint
        return record


class RemoteQwenEmbeddings(Embeddings):
    """Call the vLLM embedding API while preserving the local Qwen semantics."""

    def __init__(
        self,
        *,
        embedding_config: EmbeddingConfig,
        remote_config: RemoteEmbeddingConfig,
    ) -> None:
        embedding_config.validate()
        remote_config.validate()
        if "qwen3-embedding" not in remote_config.model.lower():
            raise ValueError(
                "RemoteQwenEmbeddings 只支持 Qwen3-Embedding 模型，"
                f"当前为：{remote_config.model}"
            )
        self.embedding_config = embedding_config
        self.remote_config = remote_config
        self.last_request_seconds = 0.0
        self._last_error: str | None = None
        self._tokenizer: Any | None = None

    def embed_query(self, text: str) -> list[float]:
        """Embed one query with the same Qwen query prompt as local SentenceTransformer."""
        if not isinstance(text, str) or not text.strip():
            raise ValueError("text 不能为空")
        prepared = self._prepare_query(text)
        return self._embed([prepared])[0]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed documents in one batch and restore the service's index order."""
        if not isinstance(texts, list):
            raise TypeError("texts 必须是 list[str]")
        if not texts:
            self.last_request_seconds = 0.0
            return []
        if any(not isinstance(text, str) or not text.strip() for text in texts):
            raise ValueError("texts 中每个 document 都必须是非空字符串")
        prepared = [text.replace("\n", " ") for text in texts]
        self._validate_document_lengths(prepared)
        vectors: list[list[float]] = []
        for start in range(0, len(prepared), self.remote_config.batch_size):
            vectors.extend(self._embed(prepared[start : start + self.remote_config.batch_size]))
        return vectors

    def document_token_lengths(self, texts: list[str]) -> list[int]:
        """Return token lengths after the same preprocessing sent to vLLM."""
        if not isinstance(texts, list):
            raise TypeError("texts 必须是 list[str]")
        if any(not isinstance(text, str) or not text.strip() for text in texts):
            raise ValueError("texts 中每个 document 都必须是非空字符串")
        return self._token_lengths([text.replace("\n", " ") for text in texts])

    def model_record(self) -> dict[str, Any]:
        """Return safe backend and compatibility metadata for reports."""
        return {
            "backend": "remote",
            "backend_type": "remote",
            "provider": "vllm",
            "url": self.remote_config.url,
            "endpoint": self.remote_config.endpoint,
            "model": self.remote_config.model,
            "model_revision": self.remote_config.model_revision,
            "revision_verified": self.remote_config.model_revision is not None,
            "dimension": self.remote_config.expected_dimension,
            "normalize_embeddings": self.embedding_config.normalize_embeddings,
            "normalization": "unit_norm_with_client_guard",
            "query_prompt_name": self.embedding_config.effective_query_prompt_name(),
            "query_prompt_profile": "qwen3_web_search_instruction_v1",
            "query_prompt": QWEN3_QUERY_PROMPT,
            "max_model_len": self.remote_config.max_model_len,
            "effective_max_length": self.remote_config.max_model_len,
            "batch_size": self.remote_config.batch_size,
            "document_prompt_profile": DOCUMENT_PROMPT_PROFILE,
            "timeout_seconds": self.remote_config.timeout_seconds,
            "request_seconds": self.last_request_seconds,
            "last_error": self._last_error,
        }

    def _prepare_query(self, text: str) -> str:
        clean_text = text.replace("\n", " ")
        prompt_name = self.embedding_config.effective_query_prompt_name()
        if prompt_name == "query":
            return QWEN3_QUERY_PROMPT + clean_text
        if prompt_name is None:
            return clean_text
        raise RemoteEmbeddingError(
            "远程 Qwen backend 无法解析自定义 query_prompt_name："
            f"{prompt_name}；请提供与服务端一致的 prompt profile"
        )

    def _embed(self, inputs: Sequence[str]) -> list[list[float]]:
        payload = {
            "model": self.remote_config.model,
            "input": list(inputs),
        }
        response = self._post_json(payload)
        vectors = _validate_embedding_response(
            response,
            expected_count=len(inputs),
            expected_dimension=self.remote_config.expected_dimension,
        )
        return [self._normalize_if_needed(vector) for vector in vectors]

    def _validate_document_lengths(self, texts: list[str]) -> None:
        lengths = self._token_lengths(texts)
        over_limit = [
            (index, length)
            for index, length in enumerate(lengths)
            if length > self.remote_config.max_model_len
        ]
        if over_limit:
            raise RemoteEmbeddingError(
                "document 超过 remote max_model_len，拒绝 silent truncation："
                f"limit={self.remote_config.max_model_len}, over_limit={over_limit[:5]}"
            )

    def _token_lengths(self, texts: list[str]) -> list[int]:
        try:
            from transformers import AutoTokenizer
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RemoteEmbeddingError(
                "文档长度保护需要 transformers，请执行：uv sync --extra embeddings"
            ) from exc

        if self._tokenizer is None:
            self._tokenizer = _get_tokenizer(
                self.remote_config.model,
                self.remote_config.model_revision,
            )
        tokenizer = self._tokenizer
        encoded = tokenizer(
            texts,
            add_special_tokens=True,
            truncation=False,
            padding=False,
            return_length=True,
        )
        lengths = encoded.get("length")
        if not isinstance(lengths, list) or len(lengths) != len(texts):
            raise RemoteEmbeddingError("无法获得完整 document token length")
        return [int(length) for length in lengths]

    def _normalize_if_needed(self, vector: list[float]) -> list[float]:
        if not self.embedding_config.normalize_embeddings:
            return vector
        norm = math.sqrt(sum(value * value for value in vector))
        if not math.isfinite(norm) or norm == 0.0:
            raise RemoteEmbeddingError("embedding norm 非法")
        if math.isclose(norm, 1.0, rel_tol=1e-5, abs_tol=1e-5):
            return vector
        return [value / norm for value in vector]

    def _post_json(self, payload: dict[str, Any]) -> dict[str, Any]:
        request = Request(
            self.remote_config.endpoint,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={
                "Accept": "application/json",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        started = time.perf_counter()
        try:
            with urlopen(request, timeout=self.remote_config.timeout_seconds) as response:  # noqa: S310
                status = response.getcode()
                if not 200 <= status < 300:
                    raise RemoteEmbeddingError(f"HTTP status 非成功：{status}")
                raw = response.read()
            decoded = json.loads(raw.decode("utf-8"))
            if not isinstance(decoded, dict):
                raise RemoteEmbeddingError("响应 JSON 顶层必须是 object")
            self._last_error = None
            return decoded
        except RemoteEmbeddingError as exc:
            self._last_error = str(exc)
            raise
        except HTTPError as exc:
            error = RemoteEmbeddingError(f"HTTP error：{exc.code}")
            self._last_error = str(error)
            raise error from exc
        except (URLError, TimeoutError, OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            error = RemoteEmbeddingError(f"remote embedding request failed：{exc}")
            self._last_error = str(error)
            raise error from exc
        finally:
            self.last_request_seconds = time.perf_counter() - started


def _validate_embedding_response(
    response: dict[str, Any],
    *,
    expected_count: int,
    expected_dimension: int,
) -> list[list[float]]:
    data = response.get("data")
    if not isinstance(data, list):
        raise RemoteEmbeddingError("响应缺少合法 data list")
    if len(data) != expected_count:
        raise RemoteEmbeddingError(
            f"embedding 数量异常：expected={expected_count}, actual={len(data)}"
        )

    ordered: list[list[float] | None] = [None] * expected_count
    seen: set[int] = set()
    for item in data:
        if not isinstance(item, dict):
            raise RemoteEmbeddingError("data item 必须是 object")
        index = item.get("index")
        if isinstance(index, bool) or not isinstance(index, int):
            raise RemoteEmbeddingError("embedding index 必须是整数")
        if index < 0 or index >= expected_count:
            raise RemoteEmbeddingError(f"embedding index 越界：{index}")
        if index in seen:
            raise RemoteEmbeddingError(f"embedding index 重复：{index}")
        seen.add(index)

        embedding = item.get("embedding")
        if not isinstance(embedding, list) or len(embedding) != expected_dimension:
            actual = len(embedding) if isinstance(embedding, list) else None
            raise RemoteEmbeddingError(
                "embedding dimension 异常："
                f"expected={expected_dimension}, actual={actual}"
            )
        vector: list[float] = []
        for value in embedding:
            if isinstance(value, bool) or not isinstance(value, Real):
                raise RemoteEmbeddingError("embedding 元素必须是数值")
            numeric = float(value)
            if not math.isfinite(numeric):
                raise RemoteEmbeddingError("embedding 元素不能是 NaN 或 Inf")
            vector.append(numeric)
        ordered[index] = vector

    if len(seen) != expected_count or any(vector is None for vector in ordered):
        raise RemoteEmbeddingError("不是所有输入都获得了 embedding")
    return [vector for vector in ordered if vector is not None]


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


_TOKENIZER_CACHE: dict[tuple[str, str | None], Any] = {}


def _get_tokenizer(model: str, revision: str | None) -> Any:
    key = (model, revision)
    tokenizer = _TOKENIZER_CACHE.get(key)
    if tokenizer is not None:
        return tokenizer
    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model, revision=revision)
    except Exception as exc:  # noqa: BLE001 - fail closed for length safety
        raise RemoteEmbeddingError(f"无法加载 document tokenizer：{exc}") from exc
    _TOKENIZER_CACHE[key] = tokenizer
    return tokenizer
