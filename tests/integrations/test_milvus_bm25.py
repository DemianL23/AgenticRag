from agenticrag.rag.integrations.milvus_bm25 import (
    DEFAULT_BM25_COLLECTION,
    BM25MilvusConfig,
    normalise_manifest_language,
)


def test_bm25_config_reads_isolated_collection(monkeypatch) -> None:
    monkeypatch.setenv("MILVUS_URI", "http://example:19530")
    monkeypatch.setenv("BM25_MILVUS_COLLECTION", "bm25_demo")
    monkeypatch.setenv("BM25_MILVUS_DROP_OLD", "true")

    config = BM25MilvusConfig.from_env()

    assert config.uri == "http://example:19530"
    assert config.collection_name == "bm25_demo"
    assert config.drop_old is True


def test_bm25_config_uses_isolated_default(monkeypatch) -> None:
    monkeypatch.delenv("BM25_MILVUS_COLLECTION", raising=False)
    monkeypatch.delenv("BM25_MILVUS_DROP_OLD", raising=False)

    assert BM25MilvusConfig.from_env().collection_name == DEFAULT_BM25_COLLECTION


def test_manifest_language_is_normalised() -> None:
    assert normalise_manifest_language("CN") == "cn"
    assert normalise_manifest_language("en") == "en"


def test_manifest_language_rejects_unknown_value() -> None:
    try:
        normalise_manifest_language("JP")
    except ValueError as exc:
        assert "仅支持 CN/EN" in str(exc)
    else:
        raise AssertionError("expected unsupported language to fail")

