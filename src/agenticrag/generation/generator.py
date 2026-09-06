"""Generate grounded answers from caller-provided retrieved chunks."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from agenticrag.generation.config import GenerationConfig
from agenticrag.generation.prompts import build_messages
from agenticrag.generation.qwen import create_qwen_chat_model
from agenticrag.generation.schemas import AnswerDraft, GeneratedAnswer, SourceCitation
from agenticrag.retrieval.schemas import RetrievedChunk


class CitationValidationError(ValueError):
    """Raised when the model cites evidence that was not provided to it."""


class QwenAnswerGenerator:
    """Call Qwen with retrieved evidence and resolve citations in application code."""

    def __init__(
        self,
        *,
        config: GenerationConfig | None = None,
        model: Any | None = None,
    ) -> None:
        self.config = config or GenerationConfig.from_env()
        self.model = model or create_qwen_chat_model(self.config)
        self.structured_model = self.model.with_structured_output(AnswerDraft)

    def generate(
        self,
        query: str,
        chunks: Sequence[RetrievedChunk],
    ) -> GeneratedAnswer:
        """Generate one answer from exactly the chunks supplied by the caller."""
        clean_query = query.strip()
        if not clean_query:
            raise ValueError("query 不能为空")
        if not chunks:
            return GeneratedAnswer(
                answer="根据当前检索到的资料无法确定。",
                citations=(),
            )

        draft = self.structured_model.invoke(build_messages(clean_query, chunks))
        parsed = AnswerDraft.model_validate(draft)
        answer = parsed.answer.strip()
        if not answer:
            raise ValueError("生成模型返回了空答案")

        by_citation_id = {
            f"E{index}": chunk for index, chunk in enumerate(chunks, start=1)
        }
        citation_ids = _normalise_citation_ids(parsed.citation_ids)
        unknown_ids = [citation_id for citation_id in citation_ids if citation_id not in by_citation_id]
        if unknown_ids:
            raise CitationValidationError(
                f"模型返回了未提供的 citation_ids：{', '.join(unknown_ids)}"
            )

        citations = tuple(
            _to_source_citation(citation_id, by_citation_id[citation_id])
            for citation_id in citation_ids
        )
        return GeneratedAnswer(answer=answer, citations=citations)


def _normalise_citation_ids(values: Sequence[str]) -> list[str]:
    normalised: list[str] = []
    for value in values:
        if not isinstance(value, str) or not value.strip():
            raise CitationValidationError("citation_ids 必须是非空字符串列表")
        citation_id = value.strip().upper()
        if citation_id not in normalised:
            normalised.append(citation_id)
    return normalised


def _to_source_citation(citation_id: str, chunk: RetrievedChunk) -> SourceCitation:
    return SourceCitation(
        citation_id=citation_id,
        doc_id=chunk.doc_id,
        source=chunk.source,
        page=chunk.page,
        chunk_id=chunk.chunk_id,
    )
