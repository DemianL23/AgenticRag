"""Schemas for structured Qwen output and validated citations."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from pydantic import BaseModel, Field


class AnswerDraft(BaseModel):
    """The only fields the generation model is allowed to return."""

    answer: str = Field(description="基于证据生成的中文答案")
    citation_ids: list[str] = Field(
        default_factory=list,
        description="答案使用的证据编号，例如 E1、E2",
    )


@dataclass(frozen=True, slots=True)
class SourceCitation:
    """A citation resolved from a real retrieved chunk."""

    citation_id: str
    doc_id: str
    source: str
    page: int
    chunk_id: str

    def to_record(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class GeneratedAnswer:
    """Application response after citation IDs have been validated."""

    answer: str
    citations: tuple[SourceCitation, ...]

    def to_record(self) -> dict[str, Any]:
        return {
            "answer": self.answer,
            "citations": [citation.to_record() for citation in self.citations],
        }
