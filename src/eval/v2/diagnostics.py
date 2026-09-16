"""Safe, compact exception diagnostics for V2 evaluators.

Evaluator artifacts are intentionally inspectable, but must never become a
second channel for provider credentials or raw model payloads.  These helpers
retain enough typed information to distinguish validation, contract, and
provider failures without serialising exception inputs or response bodies.
"""

from __future__ import annotations

import re
from typing import Any

from pydantic import ValidationError


_SENSITIVE_ASSIGNMENT = re.compile(
    r"(?i)\b(api[_-]?key|authorization|token|password|secret)\b\s*[:=]\s*"
    r"(?:bearer\s+)?[^\s,;]+"
)
_BEARER_TOKEN = re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+\-/=]+")
_WHITESPACE = re.compile(r"\s+")
_MAX_SUMMARY_LENGTH = 500
_MAX_VALIDATION_ERRORS = 20


def safe_exception_diagnostic(exc: BaseException) -> dict[str, Any]:
    """Return a JSON-safe, redacted exception summary for evaluator artifacts.

    Pydantic errors deliberately expose only stable field locations and error
    types.  In particular, ``ValidationError.errors()`` includes ``input``;
    this helper never writes that potentially sensitive value.
    """

    if isinstance(exc, ValidationError):
        return {
            "exception_type": type(exc).__name__,
            "summary": "Pydantic validation failed",
            "validation_errors": [
                {
                    "loc": _safe_location(error.get("loc", ())),
                    "type": _safe_text(str(error.get("type", "unknown")), 120),
                }
                for error in exc.errors()[:_MAX_VALIDATION_ERRORS]
            ],
        }
    return {
        "exception_type": type(exc).__name__,
        "summary": _safe_exception_summary(exc),
        "validation_errors": [],
    }


def _safe_location(location: object) -> list[str]:
    if not isinstance(location, tuple | list):
        return [_safe_text(str(location), 120)]
    return [_safe_text(str(segment), 120) for segment in location]


def _safe_exception_summary(exc: BaseException) -> str:
    message = _safe_text(str(exc), _MAX_SUMMARY_LENGTH)
    # Provider exceptions sometimes stringify their entire JSON response.  A
    # report needs the exception class, not the raw response body.
    if message.startswith(("{", "[")) or any(
        marker in message.casefold()
        for marker in ('"choices"', '"response"', '"message"')
    ):
        return "exception detail omitted because it resembles a structured payload"
    return message or "exception raised without a message"


def _safe_text(value: str, limit: int) -> str:
    compact = _WHITESPACE.sub(" ", value).strip()
    compact = _SENSITIVE_ASSIGNMENT.sub(
        lambda match: f"{match.group(1)}=[REDACTED]", compact
    )
    compact = _BEARER_TOKEN.sub("Bearer [REDACTED]", compact)
    return compact[:limit]
