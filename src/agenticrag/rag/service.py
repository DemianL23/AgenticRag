"""Compose a retriever and an answer generator without coupling their internals."""

from __future__ import annotations

from dataclasses import dataclass

from agenticrag.generation.generator import QwenAnswerGenerator
from agenticrag.generation.schemas import GeneratedAnswer
from agenticrag.retrieval.base import BaseRetriever
from agenticrag.retrieval.schemas import RetrievedChunk


@dataclass(frozen=True, slots=True)
class RagAnswerTrace:
    """Observable output of one RAG run for evaluation and debugging."""

    query: str
    answer: GeneratedAnswer
    retrieved_chunks: tuple[RetrievedChunk, ...]


class RagAnswerService:
    """Run the V0 pipeline: retrieve evidence, then generate from that evidence."""

    def __init__(
        self,
        retriever: BaseRetriever,
        generator: QwenAnswerGenerator,
    ) -> None:
        self.retriever = retriever
        self.generator = generator

    def answer(self, query: str, *, k: int = 5) -> GeneratedAnswer:
        """Return an answer grounded in the chunks returned for this query."""
        return self.answer_with_trace(query, k=k).answer

    def answer_with_trace(self, query: str, *, k: int = 5) -> RagAnswerTrace:
        """Return the generated answer together with the exact retrieved chunks."""
        chunks = self.retriever.search(query, k=k)
        answer = self.generator.generate(query, chunks)
        return RagAnswerTrace(
            query=query,
            answer=answer,
            retrieved_chunks=tuple(chunks),
        )
