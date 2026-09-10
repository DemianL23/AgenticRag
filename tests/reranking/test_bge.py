import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from agenticrag.reranking.base import RerankerLoadError
from agenticrag.reranking.bge import BGEReranker
from agenticrag.reranking.config import RerankerConfig
from agenticrag.retrieval.schemas import HybridRetrievedChunk


def _candidate(chunk_id: str, content: str) -> HybridRetrievedChunk:
    return HybridRetrievedChunk(
        content=content,
        score=0.1,
        doc_id="doc_001",
        source="report.pdf",
        page=1,
        chunk_id=chunk_id,
        rrf_score=0.1,
        rrf_rank=1,
    )


def test_bge_loads_pinned_snapshot_once_and_scores_pairs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: dict[str, object] = {"loads": 0}
    revision = "a" * 40
    snapshot_path = f"/cache/models--demo/snapshots/{revision}"

    def snapshot_download(**kwargs: object) -> str:
        calls["snapshot_kwargs"] = kwargs
        return snapshot_path

    class FakeFlagReranker:
        def __init__(self, model_path: str, **kwargs: object) -> None:
            calls["loads"] = int(calls["loads"]) + 1
            calls["model_path"] = model_path
            calls["model_kwargs"] = kwargs

        def compute_score(self, pairs: object, **kwargs: object) -> list[float]:
            calls["pairs"] = pairs
            calls["score_kwargs"] = kwargs
            return [2.0, 1.0]

    monkeypatch.setitem(
        sys.modules,
        "FlagEmbedding",
        SimpleNamespace(FlagReranker=FakeFlagReranker),
    )
    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(snapshot_download=snapshot_download),
    )
    config = RerankerConfig(
        model_name="demo/model",
        model_revision=revision,
        device="cpu",
        batch_size=8,
        max_length=512,
        local_files_only=True,
    )
    reranker = BGEReranker(config)

    scores = reranker.score(
        "query",
        [_candidate("a", "alpha"), _candidate("b", "beta")],
    )
    reranker.load()

    assert scores == [2.0, 1.0]
    assert calls["loads"] == 1
    assert calls["snapshot_kwargs"] == {
        "repo_id": "demo/model",
        "revision": revision,
        "cache_dir": None,
        "local_files_only": True,
    }
    assert calls["model_kwargs"] == {"use_fp16": False, "devices": "cpu"}
    assert calls["pairs"] == [["query", "alpha"], ["query", "beta"]]
    assert calls["score_kwargs"] == {
        "batch_size": 8,
        "max_length": 512,
        "normalize": False,
    }
    assert reranker.model_record()["resolved_revision"] == revision
    assert reranker.model_record()["effective_default_query_max_length"] == 384


def test_bge_remembers_load_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = 0

    def snapshot_download(**kwargs: object) -> str:
        nonlocal calls
        calls += 1
        raise OSError("offline cache miss")

    monkeypatch.setitem(
        sys.modules,
        "FlagEmbedding",
        SimpleNamespace(FlagReranker=object),
    )
    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(snapshot_download=snapshot_download),
    )
    reranker = BGEReranker(RerankerConfig(local_files_only=True))

    with pytest.raises(RerankerLoadError):
        reranker.load()
    with pytest.raises(RerankerLoadError, match="此前加载失败"):
        reranker.load()

    assert calls == 1
    assert reranker.model_record()["load_failed"] is True
