"""Interfaces shared by retrieval implementations."""

from __future__ import annotations

from abc import ABC, abstractmethod

from agenticrag.retrieval.schemas import RetrievedChunk


class BaseRetriever(ABC):
    """Minimal V0 contract for dense and future retrieval strategies."""

    @abstractmethod
    def search(self, query: str, k: int = 5) -> list[RetrievedChunk]:
        """Return up to ``k`` relevant chunks for a natural-language query."""

