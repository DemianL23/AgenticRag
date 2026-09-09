"""Small adapter around RAGAS 0.4 collections metrics."""

from __future__ import annotations

import asyncio
import math
from dataclasses import dataclass
from typing import Any, Protocol


class RagasMetric(Protocol):
    async def ascore(self, **kwargs: Any) -> Any:
        ...


@dataclass(frozen=True, slots=True)
class MetricDefinition:
    """A metric instance and the RAGAS sample fields it consumes."""

    name: str
    metric: RagasMetric
    input_fields: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class RagasMetricResult:
    """Per-sample scores; one broken metric does not erase the others."""

    scores: dict[str, float | None]
    reasons: dict[str, str]
    errors: dict[str, str]


class RagasEvaluator:
    """Evaluate one materialized RAG sample with independent metric calls."""

    def __init__(self, definitions: list[MetricDefinition]) -> None:
        if not definitions:
            raise ValueError("至少需要一个 RAGAS metric")
        names = [definition.name for definition in definitions]
        if len(names) != len(set(names)):
            raise ValueError("RAGAS metric 名称不能重复")
        self.definitions = tuple(definitions)

    @property
    def metric_names(self) -> tuple[str, ...]:
        return tuple(definition.name for definition in self.definitions)

    async def evaluate(
        self,
        *,
        user_input: str,
        reference: str,
        response: str,
        retrieved_contexts: list[str],
    ) -> RagasMetricResult:
        payload = {
            "user_input": user_input,
            "reference": reference,
            "response": response,
            "retrieved_contexts": retrieved_contexts,
        }

        async def score_one(
            definition: MetricDefinition,
        ) -> tuple[str, float | None, str | None, str | None]:
            kwargs = {field: payload[field] for field in definition.input_fields}
            try:
                result = await definition.metric.ascore(**kwargs)
                raw_value = getattr(result, "value", result)
                value = float(raw_value)
                if not math.isfinite(value):
                    raise ValueError(f"metric 返回非有限数值：{raw_value!r}")
                reason = getattr(result, "reason", None)
                return definition.name, value, str(reason) if reason else None, None
            except Exception as exc:  # noqa: BLE001 - isolate remote metric failures
                return definition.name, None, None, _format_error(exc)

        outcomes = await asyncio.gather(
            *(score_one(definition) for definition in self.definitions)
        )
        scores: dict[str, float | None] = {}
        reasons: dict[str, str] = {}
        errors: dict[str, str] = {}
        for name, value, reason, error in outcomes:
            scores[name] = value
            if reason:
                reasons[name] = reason
            if error:
                errors[name] = error

        return RagasMetricResult(scores=scores, reasons=reasons, errors=errors)


def _format_error(exc: Exception) -> str:
    return f"{type(exc).__name__}: {exc}"
