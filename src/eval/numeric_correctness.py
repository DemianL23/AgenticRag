"""Deterministic numeric correctness for numeric QA answers."""

from __future__ import annotations

import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation


NUMERIC_CORRECTNESS = "numeric_correctness"

_NUMERIC_TASK_TYPES = {"Comparison", "Numerical_Comparison", "Explicit_Reasoning"}
_NARRATIVE_TASK_TYPES = {"Knowledge_Query", "MultiHop_Judgment", "MultiHop_Reasoning"}
_NUMBER_PATTERN = re.compile(
    r"(?<![A-Za-z0-9_])"
    r"(?P<sign>[+-]?)"
    r"(?P<number>(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:[eE][+-]?\d+)?)"
    r"(?P<unit>\s*(?:％|%|percentage\s+points?|pp|times?|million|billion|"
    r"trillion|thousand|亿元|万元|亿|万)?)",
    re.IGNORECASE,
)

_UNIT_SCALES = {
    "thousand": Decimal("1000"),
    "million": Decimal("1000000"),
    "billion": Decimal("1000000000"),
    "trillion": Decimal("1000000000000"),
    "万": Decimal("10000"),
    "万元": Decimal("10000"),
    "亿": Decimal("100000000"),
    "亿元": Decimal("100000000"),
}


@dataclass(frozen=True, slots=True)
class NumericValue:
    """A parsed number with its display precision and optional unit."""

    value: Decimal
    tolerance: Decimal
    unit: str
    percent: bool


def numeric_correctness(
    reference: str,
    prediction: str | None,
    *,
    task_type: str | None = None,
) -> float | None:
    """Return 1, 0, or ``None`` for a numeric or non-numeric answer.

    The reference determines whether the sample is numeric and supplies the
    decimal precision. Each expected number must match a distinct number in
    the generated answer. Extra numbers are allowed because generated answers
    commonly include intermediate calculations and explanatory percentages.
    """
    expected = extract_numeric_values(reference)
    if not _is_numeric_answer(reference, expected, task_type=task_type):
        return None
    if prediction is None:
        return 0.0

    actual = extract_numeric_values(prediction)
    return 1.0 if _has_distinct_matches(expected, actual) else 0.0


def extract_numeric_values(text: str) -> list[NumericValue]:
    """Extract answer numbers while ignoring years and citation IDs."""
    if not isinstance(text, str):
        raise TypeError("numeric text 必须是字符串")

    values: list[NumericValue] = []
    for match in _NUMBER_PATTERN.finditer(text):
        number_text = match.group("number")
        unit = match.group("unit").strip().lower()
        start = match.start()
        prefix = text[max(0, start - 3) : start].lower()
        if _is_year(number_text, unit, prefix):
            continue
        try:
            raw_value = Decimal(number_text.replace(",", ""))
        except InvalidOperation as exc:  # pragma: no cover - regex guards input
            raise ValueError(f"无法解析数值：{number_text}") from exc

        scale = _UNIT_SCALES.get(unit, Decimal("1"))
        values.append(
            NumericValue(
                value=raw_value * scale,
                tolerance=_tolerance(raw_value) * scale,
                unit=unit,
                percent=unit in {"%", "％", "percentage point", "percentage points", "pp"},
            )
        )
    return values


def _is_numeric_answer(
    reference: str,
    values: list[NumericValue],
    *,
    task_type: str | None,
) -> bool:
    if not values:
        return False
    if task_type in _NARRATIVE_TASK_TYPES:
        return False
    if task_type in _NUMERIC_TASK_TYPES:
        return True

    stripped = reference.strip()
    if _looks_like_numeric_list(stripped):
        return True
    if len(values) == 1:
        return True

    # A prose comparison such as the Chinese financial answers can still be a
    # numeric answer when every value carries the same monetary unit.
    units = {value.unit for value in values}
    return len(units) == 1 and next(iter(units), "") in {"万", "万元", "亿", "亿元"}


def _looks_like_numeric_list(text: str) -> bool:
    return bool(
        re.fullmatch(
            r"\s*[\[（(]?\s*[+-]?(?:\d[\d,]*(?:\.\d+)?)(?:\s*[，,;；]\s*[+-]?(?:\d[\d,]*(?:\.\d+)?))*\s*[\]）)]?\s*",
            text,
        )
    )


def _has_distinct_matches(
    expected: list[NumericValue], actual: list[NumericValue]
) -> bool:
    if not expected:
        return False
    candidates = [
        [index for index, value in enumerate(actual) if _equivalent(expected_value, value)]
        for expected_value in expected
    ]
    if any(not indexes for indexes in candidates):
        return False

    # Backtracking is small and deterministic for answer-sized numeric lists.
    order = sorted(range(len(expected)), key=lambda index: len(candidates[index]))

    def assign(position: int, used: set[int]) -> bool:
        if position == len(order):
            return True
        expected_index = order[position]
        return any(
            candidate_index not in used
            and assign(position + 1, used | {candidate_index})
            for candidate_index in candidates[expected_index]
        )

    return assign(0, set())


def _equivalent(expected: NumericValue, actual: NumericValue) -> bool:
    expected_variants = _variants(expected)
    actual_variants = _variants(actual)
    return any(
        abs(expected_value - actual_value) <= expected_tolerance
        for expected_value, expected_tolerance in expected_variants
        for actual_value, _ in actual_variants
    )


def _variants(value: NumericValue) -> tuple[tuple[Decimal, Decimal], ...]:
    variants = [(value.value, value.tolerance)]
    if value.percent:
        variants.append((value.value / Decimal("100"), value.tolerance / Decimal("100")))
    else:
        variants.append((value.value * Decimal("100"), value.tolerance * Decimal("100")))
    return tuple(variants)


def _tolerance(value: Decimal) -> Decimal:
    exponent = value.as_tuple().exponent
    decimal_places = max(0, -exponent)
    return max(Decimal("1e-12"), Decimal("0.5") * (Decimal("10") ** -decimal_places))


def _is_year(number_text: str, unit: str, prefix: str) -> bool:
    if "." in number_text or "," in number_text:
        return False
    try:
        number = int(number_text)
    except ValueError:  # pragma: no cover - regex only captures digits
        return False
    return 1900 <= number <= 2100 and (unit in {"年", ""} or "fy" in prefix)
