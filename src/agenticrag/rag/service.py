"""Compose a retriever and an answer generator without coupling their internals."""

from __future__ import annotations

from agenticrag.generation.generator import QwenAnswerGenerator
from agenticrag.generation.schemas import GeneratedAnswer
from agenticrag.retrieval.base import BaseRetriever


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
        chunks = self.retriever.search(query, k=k)
        return self.generator.generate(query, chunks)
