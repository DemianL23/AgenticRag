"""Load the project's QA JSONL and adapt ``gold`` into a RAGAS reference."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class QaSample:
    """The minimum supervised sample required by end-to-end RAG evaluation."""

    sample_id: str
    question: str
    reference: str
    task_type: str | None = None


def load_qa_dataset(path: Path, *, limit: int | None = None) -> list[QaSample]:
    """Read JSONL records containing ``question`` and ``gold``."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"QA 数据集不存在：{path}")
    if limit is not None and (isinstance(limit, bool) or limit <= 0):
        raise ValueError("limit 必须是正整数")

    samples: list[QaSample] = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, raw_line in enumerate(file, start=1):
            if not raw_line.strip():
                continue
            try:
                record = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"QA 数据集第 {line_number} 行不是合法 JSON：{path}") from exc
            samples.append(_to_sample(record, line_number, path))
            if limit is not None and len(samples) >= limit:
                break

    if not samples:
        raise ValueError(f"QA 数据集不能为空：{path}")
    return samples


def gold_to_reference(gold: Any) -> str:
    """Convert supported gold values into stable, human-readable reference text."""
    if isinstance(gold, str):
        reference = gold.strip()
    elif isinstance(gold, list):
        parts = [gold_to_reference(value) for value in gold]
        reference = parts[0] if len(parts) == 1 else "\n".join(
            f"- {part}" for part in parts
        )
    elif isinstance(gold, (int, float)) and not isinstance(gold, bool):
        reference = json.dumps(gold, ensure_ascii=False)
    elif isinstance(gold, dict):
        reference = json.dumps(gold, ensure_ascii=False, sort_keys=True)
    else:
        raise TypeError(f"gold 类型不受支持：{type(gold).__name__}")

    if not reference:
        raise ValueError("gold 转换后的 reference 不能为空")
    return reference


def _to_sample(record: Any, line_number: int, path: Path) -> QaSample:
    if not isinstance(record, dict):
        raise TypeError(f"QA 数据集第 {line_number} 行必须是 JSON 对象：{path}")
    question = record.get("question")
    if not isinstance(question, str) or not question.strip():
        raise ValueError(f"QA 数据集第 {line_number} 行缺少合法 question：{path}")
    if "gold" not in record:
        raise ValueError(f"QA 数据集第 {line_number} 行缺少 gold：{path}")

    raw_id = (
        record.get("finqa_id")
        or record.get("uid")
        or record.get("id")
        or f"line_{line_number}"
    )
    return QaSample(
        sample_id=str(raw_id).strip(),
        question=question.strip(),
        reference=gold_to_reference(record["gold"]),
        task_type=(
            str(record["task_type"]).strip()
            if isinstance(record.get("task_type"), str)
            else None
        ),
    )
