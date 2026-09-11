"""Deterministic request-internal identifiers for V2."""

from __future__ import annotations

from uuid import UUID, uuid4


def new_request_id() -> str:
    """Create the only random public identity used by a request."""
    return str(uuid4())


def task_id(ordinal: int) -> str:
    return f"SQ_{_ordinal(ordinal):03d}"


def query_revision_id(task: str, ordinal: int) -> str:
    return f"QR_{_task_token(task)}_{_ordinal(ordinal):03d}"


def retrieval_attempt_id(task: str, revision: str, ordinal: int) -> str:
    return f"ATT_{_task_token(task)}_{_revision_token(revision)}_{_ordinal(ordinal):03d}"


def grade_record_id(task: str, revision: str, ordinal: int) -> str:
    return f"GR_{_task_token(task)}_{_revision_token(revision)}_{_ordinal(ordinal):03d}"


def routing_decision_id(task: str, revision: str, ordinal: int) -> str:
    return f"ROUTE_{_task_token(task)}_{_revision_token(revision)}_{_ordinal(ordinal):03d}"


def decomposition_id(task: str, revision: str, ordinal: int) -> str:
    return f"DEC_{_task_token(task)}_{_revision_token(revision)}_{_ordinal(ordinal):03d}"


def hitl_request_id(ordinal: int = 1) -> str:
    return f"HITL_{_ordinal(ordinal):03d}"


def hitl_item_id(ordinal: int) -> str:
    return f"ITEM_{_ordinal(ordinal):03d}"


def scope_option_id(ordinal: int) -> str:
    return f"OPT_{_ordinal(ordinal):03d}"


def _ordinal(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError("ordinal 必须是正整数")
    return value


def _task_token(value: str) -> str:
    if not isinstance(value, str) or not value.startswith("SQ_"):
        raise ValueError("task ID 必须以 SQ_ 开头")
    return value.replace("_", "", 1)


def _revision_token(value: str) -> str:
    if not isinstance(value, str) or not value.startswith("QR_"):
        raise ValueError("query revision ID 必须以 QR_ 开头")
    parts = value.split("_")
    if len(parts) != 3 or not parts[1].startswith("SQ"):
        raise ValueError("query revision ID 格式非法")
    return f"QR{parts[1][2:]}"


def validate_request_id(value: str) -> str:
    try:
        UUID(value, version=4)
    except (ValueError, AttributeError, TypeError) as exc:
        raise ValueError("request_id 必须是 UUID4") from exc
    return value
