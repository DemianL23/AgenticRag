from pathlib import Path

import pytest

from agenticrag.generation.config import GenerationConfig
from agenticrag.generation.generator import CitationValidationError, QwenAnswerGenerator
from agenticrag.generation.prompts import build_messages
from agenticrag.generation.schemas import AnswerDraft, GeneratedAnswer, SourceCitation
from agenticrag.retrieval.schemas import RetrievedChunk


def _chunk(chunk_id: str = "doc_000:p0003:c000") -> RetrievedChunk:
    return RetrievedChunk(
        content="公司主营工程技术服务。",
        score=0.3,
        doc_id="doc_000",
        source="corpus/report.pdf",
        page=3,
        chunk_id=chunk_id,
    )


def test_generation_config_reads_env_without_slots_descriptor_bug(tmp_path: Path) -> None:
    dotenv = tmp_path / ".env"
    dotenv.write_text(
        "GENERATION_MODEL=qwen3-max\n"
        "GENERATION_TEMPERATURE=0.2\n"
        "GENERATION_ENABLE_THINKING=true\n",
        encoding="utf-8",
    )

    config = GenerationConfig.from_env(dotenv)

    assert config.model == "qwen3-max"
    assert config.temperature == 0.2
    assert config.enable_thinking is True
    assert config.max_tokens == 1024


def test_build_messages_preserves_citation_metadata() -> None:
    messages = build_messages("问题", [_chunk()])

    content = messages[1].content
    assert "doc_id: doc_000" in content
    assert "source: corpus/report.pdf" in content
    assert "page: 3" in content
    assert "chunk_id: doc_000:p0003:c000" in content


class FakeStructuredModel:
    def __init__(self, draft: AnswerDraft) -> None:
        self.draft = draft
        self.messages = None

    def with_structured_output(self, _: type[AnswerDraft]) -> "FakeStructuredModel":
        return self

    def invoke(self, messages: object) -> AnswerDraft:
        self.messages = messages
        return self.draft


def test_generator_resolves_only_real_citations() -> None:
    model = FakeStructuredModel(
        AnswerDraft(answer="答案[E1]", citation_ids=["e1", "E1"])
    )
    generator = QwenAnswerGenerator(
        config=GenerationConfig(api_key="test"),
        model=model,
    )

    result = generator.generate("问题", [_chunk()])

    assert result == GeneratedAnswer(
        answer="答案[E1]",
        citations=(
            SourceCitation(
                citation_id="E1",
                doc_id="doc_000",
                source="corpus/report.pdf",
                page=3,
                chunk_id="doc_000:p0003:c000",
            ),
        ),
    )
    assert "chunk_id: doc_000:p0003:c000" in model.messages[1].content


def test_generator_rejects_unknown_citation() -> None:
    model = FakeStructuredModel(
        AnswerDraft(answer="答案", citation_ids=["E9"])
    )
    generator = QwenAnswerGenerator(
        config=GenerationConfig(api_key="test"),
        model=model,
    )

    with pytest.raises(CitationValidationError, match="E9"):
        generator.generate("问题", [_chunk()])


def test_generator_returns_grounded_fallback_without_chunks() -> None:
    model = FakeStructuredModel(AnswerDraft(answer="不应调用", citation_ids=[]))
    generator = QwenAnswerGenerator(
        config=GenerationConfig(api_key="test"),
        model=model,
    )

    result = generator.generate("问题", [])

    assert result == GeneratedAnswer(
        answer="根据当前检索到的资料无法确定。",
        citations=(),
    )
    assert model.messages is None
