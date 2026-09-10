from pathlib import Path

import pytest

from agenticrag.reranking.config import (
    DEFAULT_RERANK_MODEL,
    DEFAULT_RERANK_MODEL_REVISION,
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
