from __future__ import annotations

import hashlib
from pathlib import Path

from eval.v2.module4 import load_manifest


def test_module4_manifest_is_current_19_sample_contract() -> None:
    path = Path("eval/datasets/v2_module4_real_eval.jsonl")
    records = load_manifest(path)
    assert len(records) == 19
    assert sum(record["expected"]["route"] == "unsupported" for record in records) == 11
    assert sum(record["retrieval"]["expected_to_run"] is True for record in records) == 8
    assert sum(record["evaluation_incomplete"] is True for record in records) == 8


def test_module4_manifest_digest_is_recordable_without_modifying_source() -> None:
    path = Path("eval/datasets/v2_module4_real_eval.jsonl")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert len(digest) == 64
