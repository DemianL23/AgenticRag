"""V1.2 retrieval-to-generation control path.

This module deliberately contains no Agentic RAG orchestration.  It only
connects the existing V1.2 reranking retriever to the existing answer
generator so that V2 experiments have a fair end-to-end control.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from agenticrag.generation.generator import QwenAnswerGenerator
from agenticrag.generation.schemas import GeneratedAnswer
from agenticrag.retrieval.reranking_retriever import (
    RerankingRetriever,
    RerankingSearchTrace,
)
from agenticrag.retrieval.schemas import RerankedChunk


@dataclass(frozen=True, slots=True)
class V12ControlTrace:
    """Generated answer plus the exact V1.2 retrieval trace used to make it."""

    query: str
    answer: GeneratedAnswer
    retrieved_chunks: tuple[RerankedChunk, ...]
    retrieval_trace: RerankingSearchTrace


class V12ControlAnswerService:
    """Run ``V1.2 RerankingRetriever -> existing Answer Generator``."""

    def __init__(
        self,
        retriever: RerankingRetriever,
        generator: QwenAnswerGenerator,
    ) -> None:
        self.retriever = retriever
        self.generator = generator

    def answer(self, query: str, *, k: int = 5) -> GeneratedAnswer:
        """Return the answer produced from the V1.2 final Top-K evidence."""
        return self.answer_with_trace(query, k=k).answer

    def answer_with_trace(self, query: str, *, k: int = 5) -> V12ControlTrace:
        """Run retrieval once, then generate from exactly its final results."""
        retrieval_trace = self.retriever.search_with_trace(query, k=k)
        answer = self.generator.generate(query, retrieval_trace.results)
        return V12ControlTrace(
            query=query,
            answer=answer,
            retrieved_chunks=retrieval_trace.results,
            retrieval_trace=retrieval_trace,
        )

    def retrieval_record(self) -> dict[str, Any]:
        """Return safe resolved reranker metadata for the end-to-end report."""
        return self.retriever.model_record()
