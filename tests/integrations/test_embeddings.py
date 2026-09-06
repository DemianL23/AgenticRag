import sys
import types
from pathlib import Path

import pytest

from agenticrag.integrations.embeddings import (
    EmbeddingConfig,
    create_embeddings,
    embedding_dimension,
)


def test_embedding_config_defaults_to_qwen_and_adds_query_prompt() -> None:
    config = EmbeddingConfig()

    assert config.model_name == "Qwen/Qwen3-Embedding-0.6B"
    assert config.device == "cpu"
    assert config.resolved_model_kwargs() == {"device": "cpu"}
    assert config.resolved_document_encode_kwargs() == {
        "batch_size": 32,
        "normalize_embeddings": True,
    }
    assert config.resolved_query_encode_kwargs() == {
        "batch_size": 32,
        "normalize_embeddings": True,
        "prompt_name": "query",
    }


def test_embedding_config_from_env_can_switch_to_bge(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / ".env").write_text(
        "EMBEDDING_MODEL=BAAI/bge-m3\n"
        "EMBEDDING_DEVICE=cpu\n"
        "EMBEDDING_BATCH_SIZE=16\n"
        "EMBEDDING_NORMALIZE=true\n",
        encoding="utf-8",
    )
    monkeypatch.delenv("EMBEDDING_MODEL", raising=False)
    monkeypatch.delenv("EMBEDDING_QUERY_PROMPT_NAME", raising=False)
    monkeypatch.delenv("EMBEDDING_BATCH_SIZE", raising=False)

    config = EmbeddingConfig.from_env(tmp_path / ".env")

    assert config.model_name == "BAAI/bge-m3"
    assert config.batch_size == 16
    assert config.query_prompt_name is None


def test_embedding_config_from_env_uses_module_defaults_with_slots(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for name in (
        "EMBEDDING_MODEL",
        "EMBEDDING_DEVICE",
        "EMBEDDING_BATCH_SIZE",
        "EMBEDDING_NORMALIZE",
        "EMBEDDING_QUERY_PROMPT_NAME",
        "EMBEDDING_CACHE_FOLDER",
        "EMBEDDING_TRUST_REMOTE_CODE",
    ):
        monkeypatch.delenv(name, raising=False)

    config = EmbeddingConfig.from_env(tmp_path / "missing.env")

    assert config.model_name == "Qwen/Qwen3-Embedding-0.6B"
    assert config.device == "cpu"
    assert config.batch_size == 32
    assert config.normalize_embeddings is True


def test_embedding_config_rejects_invalid_batch_size() -> None:
    with pytest.raises(ValueError, match="batch_size"):
        EmbeddingConfig(batch_size=0).validate()


def test_create_embeddings_passes_document_and_query_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: dict[str, object] = {}

    class FakeEmbeddings:
        def __init__(self, **kwargs: object) -> None:
            calls.update(kwargs)

        def embed_query(self, text: str) -> list[float]:
            assert text == "dimension probe"
            return [0.1, 0.2, 0.3]

        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            return [[0.1, 0.2, 0.3] for _ in texts]

    fake_module = types.ModuleType("langchain_huggingface")
    fake_module.HuggingFaceEmbeddings = FakeEmbeddings  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "langchain_huggingface", fake_module)

    embeddings = create_embeddings(EmbeddingConfig(batch_size=8))

    assert calls == {
        "model_name": "Qwen/Qwen3-Embedding-0.6B",
        "cache_folder": None,
        "model_kwargs": {"device": "cpu"},
        "encode_kwargs": {"batch_size": 8, "normalize_embeddings": True},
        "query_encode_kwargs": {
            "batch_size": 8,
            "normalize_embeddings": True,
            "prompt_name": "query",
        },
    }
    assert embedding_dimension(embeddings) == 3
