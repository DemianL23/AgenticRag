from pathlib import Path

import pytest

from agenticrag.reranking.config import (
    DEFAULT_RERANK_MODEL,
    DEFAULT_RERANK_MODEL_REVISION,
    RemoteRerankerConfig,
    RerankerConfig,
)


def test_reranker_config_reads_v1_2_defaults(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    for name in (
        "RERANK_MODEL",
        "RERANK_MODEL_REVISION",
        "RERANK_DEVICE",
        "RERANK_BATCH_SIZE",
        "RERANK_MAX_LENGTH",
        "RERANK_USE_FP16",
        "RERANK_LOCAL_FILES_ONLY",
        "RERANK_CACHE_DIR",
    ):
        monkeypatch.delenv(name, raising=False)

    config = RerankerConfig.from_env(tmp_path / "missing.env")

    assert config.model_name == DEFAULT_RERANK_MODEL
    assert config.model_revision == DEFAULT_RERANK_MODEL_REVISION
    assert config.device == "cpu"
    assert config.batch_size == 8
    assert config.max_length == 512
    assert config.use_fp16 is False
    assert config.local_files_only is False


def test_custom_model_does_not_inherit_bge_revision(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("RERANK_MODEL", "example/other-reranker")
    monkeypatch.delenv("RERANK_MODEL_REVISION", raising=False)

    config = RerankerConfig.from_env(tmp_path / "missing.env")

    assert config.model_revision is None


def test_fp16_requires_cuda() -> None:
    with pytest.raises(ValueError, match="CUDA"):
        RerankerConfig(device="cpu", use_fp16=True).validate()


def test_reranker_config_accepts_explicit_cuda() -> None:
    RerankerConfig(device="cuda:0", use_fp16=True).validate()


def test_remote_reranker_config_reads_http_settings(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("RERANK_REMOTE_URL", "http://192.168.31.238:8001/")
    monkeypatch.setenv("RERANK_REMOTE_MODEL", "BAAI/bge-reranker-v2-m3")
    monkeypatch.setenv("RERANK_REMOTE_TIMEOUT_SECONDS", "45")

    config = RemoteRerankerConfig.from_env(tmp_path / "missing.env")

    assert config.endpoint == "http://192.168.31.238:8001/v1/rerank"
    assert config.timeout_seconds == 45.0
    assert config.to_record()["endpoint"] == config.endpoint


def test_reranker_backend_defaults_to_local_and_rejects_unknown(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from agenticrag.reranking.config import reranker_backend_from_env

    monkeypatch.delenv("RERANK_BACKEND", raising=False)
    assert reranker_backend_from_env(tmp_path / "missing.env") == "local"

    monkeypatch.setenv("RERANK_BACKEND", "unsupported")
    with pytest.raises(ValueError, match="local 或 remote"):
        reranker_backend_from_env(tmp_path / "missing.env")


def test_factory_selects_remote_backend(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from agenticrag.reranking.factory import create_reranker
    from agenticrag.reranking.remote import RemoteBGEReranker

    monkeypatch.setenv("RERANK_BACKEND", "remote")
    monkeypatch.setenv("RERANK_REMOTE_URL", "http://192.168.31.238:8001")

    reranker = create_reranker(tmp_path / "missing.env")

    assert isinstance(reranker, RemoteBGEReranker)
