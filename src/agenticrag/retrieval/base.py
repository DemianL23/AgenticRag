"""Interfaces shared by retrieval implementations."""

from __future__ import annotations

from abc import ABC, abstractmethod

from agenticrag.retrieval.schemas import RetrievedChunk


class BaseRetriever(ABC):
    """Shared retrieval contract for Dense and lexical retrievers."""

    @abstractmethod
    def search(self, query: str, k: int = 20) -> list[RetrievedChunk]:
        """Return up to ``k`` relevant chunks for a natural-language query."""
