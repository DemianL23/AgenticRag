"""Model-facing contract shared by reranker implementations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Sequence

from agenticrag.retrieval.schemas import HybridRetrievedChunk


class RerankerError(RuntimeError):
    """Base class for expected reranker failures."""


class RerankerLoadError(RerankerError):
    """The reranker model could not be loaded."""


class RerankerInferenceError(RerankerError):
    """The loaded reranker failed while scoring candidates."""


class BaseReranker(ABC):
    """Score query/chunk pairs without owning retrieval fallback policy."""

    @property
    @abstractmethod
    def is_loaded(self) -> bool:
        """Whether this instance has a ready model."""

    @abstractmethod
    def load(self) -> None:
        """Load the model once, or re-raise a remembered load failure."""

    @abstractmethod
    def score(
        self,
        query: str,
        candidates: Sequence[HybridRetrievedChunk],
    ) -> Sequence[object]:
        """Return one raw relevance score per candidate in input order."""

    @abstractmethod
    def model_record(self) -> dict[str, Any]:
        """Return reproducibility metadata safe to write into reports."""
