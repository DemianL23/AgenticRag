"""Safe serialization helpers for V2 domain objects and graph state."""

from __future__ import annotations

import json
from enum import Enum
from typing import Any, Mapping

from pydantic import BaseModel


def to_jsonable(value: Any) -> Any:
    """Convert only explicit domain values to JSON-compatible primitives."""
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"V2 state 禁止保存 runtime object：{type(value).__name__}")


def serialize_state(state: Mapping[str, Any]) -> dict[str, Any]:
    """Return a JSON-safe copy and verify that no pickle fallback is needed."""
    jsonable = to_jsonable(state)
    encoded = json.dumps(jsonable, ensure_ascii=False, allow_nan=False)
    decoded = json.loads(encoded)
    if not isinstance(decoded, dict):  # pragma: no cover - guarded by Mapping input
        raise TypeError("V2 state 必须序列化为 object")
    return decoded
