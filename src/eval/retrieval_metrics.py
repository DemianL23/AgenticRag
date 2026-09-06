"""Metrics for evaluating whether retrieval returns gold chunks."""

from __future__ import annotations

from collections.abc import Iterable, Sequence


def recall_at_k(
    retrieved_chunk_ids: Sequence[str],
    relevant_chunk_ids: Iterable[str],
    *,
    k: int,
) -> float:
    """Return the fraction of relevant chunks found in the first ``k`` results."""
    _validate_k(k)
    relevant_ids = _normalise_ids(relevant_chunk_ids, field_name="relevant_chunk_ids")
    retrieved_ids = _normalise_ids(retrieved_chunk_ids, field_name="retrieved_chunk_ids")
    hits = set(retrieved_ids[:k]) & set(relevant_ids)
    return len(hits) / len(set(relevant_ids))


def reciprocal_rank(
    retrieved_chunk_ids: Sequence[str],
    relevant_chunk_ids: Iterable[str],
    *,
    k: int | None = None,
) -> float:
    """Return the reciprocal rank of the first relevant result, or zero."""
    if k is not None:
        _validate_k(k)
    relevant_ids = set(
        _normalise_ids(relevant_chunk_ids, field_name="relevant_chunk_ids")
    )
    retrieved_ids = _normalise_ids(retrieved_chunk_ids, field_name="retrieved_chunk_ids")
    candidates = retrieved_ids if k is None else retrieved_ids[:k]
    for rank, chunk_id in enumerate(candidates, start=1):
        if chunk_id in relevant_ids:
            return 1.0 / rank
    return 0.0


def mean_reciprocal_rank(reciprocal_ranks: Iterable[float]) -> float:
    """Return the arithmetic mean of per-query reciprocal ranks."""
    values = [float(value) for value in reciprocal_ranks]
    if not values:
        raise ValueError("至少需要一个 query 的 reciprocal rank")
    return sum(values) / len(values)


def calculate_retrieval_metrics(
    retrieved_chunk_ids: Sequence[str],
    relevant_chunk_ids: Iterable[str],
    *,
    k: int = 5,
) -> dict[str, float]:
    """Calculate per-query Recall@K values and reciprocal rank."""
    _validate_k(k)
    cutoffs = sorted({cutoff for cutoff in (1, 3, 5) if cutoff <= k} | {k})
    metrics = {
        f"Recall@{cutoff}": recall_at_k(
            retrieved_chunk_ids,
            relevant_chunk_ids,
            k=cutoff,
        )
        for cutoff in cutoffs
    }
    metrics["MRR"] = reciprocal_rank(
        retrieved_chunk_ids,
        relevant_chunk_ids,
        k=k,
    )
    return metrics


def _normalise_ids(ids: Iterable[str], *, field_name: str) -> list[str]:
    values = list(ids)
    if not values or any(not isinstance(value, str) or not value.strip() for value in values):
        raise ValueError(f"{field_name} 必须是非空字符串列表")
    return [value.strip() for value in values]


def _validate_k(k: int) -> None:
    if isinstance(k, bool) or not isinstance(k, int) or k <= 0:
        raise ValueError("k 必须是正整数")
