"""Stable data structures returned by retrieval implementations."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class RetrievedChunk:
    """One chunk returned by a retriever, including citation metadata."""

    content: str
    score: float
    doc_id: str
    source: str
    page: int
    chunk_id: str

    def to_record(self) -> dict[str, Any]:
        """Return a JSON-serializable representation for APIs and CLI output."""
        return asdict(self)
