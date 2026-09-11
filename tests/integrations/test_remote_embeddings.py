from __future__ import annotations

import json
import math
from urllib.error import HTTPError, URLError

import pytest

from agenticrag.rag.integrations.embeddings import EmbeddingConfig
from agenticrag.rag.integrations.remote_embeddings import (
    QWEN3_QUERY_PROMPT,
    RemoteEmbeddingConfig,
    RemoteEmbeddingError,
    RemoteQwenEmbeddings,
)


def _config() -> RemoteEmbeddingConfig:
    return RemoteEmbeddingConfig(
        url="http://embedding.test:8002",
        model="Qwen/Qwen3-Embedding-0.6B",
    )


def _unit_vector(value: float) -> list[float]:
    return [value] + [0.0] * 1023


class FakeResponse:
    def __init__(self, body: object, status: int = 200) -> None:
        self.body = body
        self.status = status

    def __enter__(self) -> "FakeResponse":
        return self

    def __exit__(self, *_: object) -> None:
        return None

    def getcode(self) -> int:
        return self.status

    def read(self) -> bytes:
        if isinstance(self.body, bytes):
            return self.body
        return json.dumps(self.body).encode("utf-8")


def _embedding_client() -> RemoteQwenEmbeddings:
    return RemoteQwenEmbeddings(
        embedding_config=EmbeddingConfig(backend="remote"),
        remote_config=_config(),
    )


def test_remote_query_sends_prompt_and_validates_dimension(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_urlopen(request: object, *, timeout: float) -> FakeResponse:
        captured["url"] = request.full_url  # type: ignore[attr-defined]
        captured["timeout"] = timeout
        captured["payload"] = json.loads(request.data)  # type: ignore[attr-defined]
        return FakeResponse(
            {
                "object": "list",
                "data": [{"index": 0, "embedding": _unit_vector(1.0)}],
            }
        )

    monkeypatch.setattr(
        "agenticrag.rag.integrations.remote_embeddings.urlopen",
        fake_urlopen,
    )

    result = _embedding_client().embed_query("问题")

    assert result == _unit_vector(1.0)
    assert captured == {
        "url": "http://embedding.test:8002/v1/embeddings",
        "timeout": 30.0,
        "payload": {
            "model": "Qwen/Qwen3-Embedding-0.6B",
            "input": [QWEN3_QUERY_PROMPT + "问题"],
        },
    }


def test_remote_documents_restore_order_by_index(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "agenticrag.rag.integrations.remote_embeddings.urlopen",
        lambda *_args, **_kwargs: FakeResponse(
            {
                "data": [
                    {"index": 1, "embedding": _unit_vector(-1.0)},
                    {"index": 0, "embedding": _unit_vector(1.0)},
                ]
            }
        ),
    )

    result = _embedding_client().embed_documents(["doc A", "doc B"])

    assert result == [_unit_vector(1.0), _unit_vector(-1.0)]


@pytest.mark.parametrize(
    ("data", "match"),
    [
        (
            [
                {"index": 0, "embedding": _unit_vector(1.0)},
                {"index": 0, "embedding": _unit_vector(2.0)},
            ],
            "重复",
        ),
        (
            [{"embedding": _unit_vector(1.0)}, {"index": 1, "embedding": _unit_vector(2.0)}],
            "必须是整数",
        ),
        (
            [
                {"index": 0, "embedding": _unit_vector(1.0)},
                {"index": 2, "embedding": _unit_vector(2.0)},
            ],
            "越界",
        ),
        (
            [{"index": 0, "embedding": [0.0] * 1023}],
            "数量异常",
        ),
        (
            [
                {"index": 0, "embedding": [0.0] * 1023},
                {"index": 1, "embedding": _unit_vector(1.0)},
            ],
            "dimension 异常",
        ),
        (
            [
                {"index": 0, "embedding": [math.nan] + [0.0] * 1023},
                {"index": 1, "embedding": _unit_vector(1.0)},
            ],
            "NaN",
        ),
        (
            [
                {"index": 0, "embedding": [math.inf] + [0.0] * 1023},
                {"index": 1, "embedding": _unit_vector(1.0)},
            ],
            "NaN",
        ),
    ],
)
def test_remote_response_validation(data: list[dict[str, object]], match: str, monkeypatch) -> None:
    monkeypatch.setattr(
        "agenticrag.rag.integrations.remote_embeddings.urlopen",
        lambda *_args, **_kwargs: FakeResponse({"data": data}),
    )

    with pytest.raises(RemoteEmbeddingError, match=match):
        _embedding_client().embed_documents(["doc A", "doc B"])


def test_remote_rejects_invalid_json(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "agenticrag.rag.integrations.remote_embeddings.urlopen",
        lambda *_args, **_kwargs: FakeResponse(b"not-json"),
    )

    with pytest.raises(RemoteEmbeddingError, match="request failed"):
        _embedding_client().embed_query("问题")


@pytest.mark.parametrize(
    "error",
    [TimeoutError("timeout"), URLError("connection refused"), OSError("refused")],
)
def test_remote_network_failures_are_explicit(error: Exception, monkeypatch) -> None:
    def raise_error(*_args, **_kwargs):
        raise error

    monkeypatch.setattr(
        "agenticrag.rag.integrations.remote_embeddings.urlopen",
        raise_error,
    )

    with pytest.raises(RemoteEmbeddingError, match="request failed"):
        _embedding_client().embed_query("问题")


def test_remote_rejects_http_error(monkeypatch: pytest.MonkeyPatch) -> None:
    error = HTTPError(
        "http://embedding.test:8002/v1/embeddings",
        503,
        "unavailable",
        hdrs=None,
        fp=None,
    )
    monkeypatch.setattr(
        "agenticrag.rag.integrations.remote_embeddings.urlopen",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(error),
    )

    with pytest.raises(RemoteEmbeddingError, match="HTTP error"):
        _embedding_client().embed_query("问题")
