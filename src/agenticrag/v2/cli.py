"""Small non-persistent V2 stage runner CLI."""

from __future__ import annotations

import argparse
import json

from .module4 import Module4Service
from .module6 import Module6Service


def main() -> None:
    parser = argparse.ArgumentParser(prog="agenticrag-v2-run")
    parser.add_argument("--stage", choices=("v2_1", "v2_2", "v2_3"), required=True)
    parser.add_argument("question")
    parser.add_argument("--response-language", choices=("zh", "en"))
    args = parser.parse_args()
    if args.stage == "v2_3":
        parser.error("V2.3 durable HITL 尚未在 Module 6 实现")
    if args.stage == "v2_1":
        result = Module4Service().run(
            args.question, response_language=args.response_language
        )
    else:
        result = Module6Service().run(
            args.question, response_language=args.response_language
        )
    print(json.dumps(result.stage_result.model_dump(mode="json"), ensure_ascii=False, indent=2))
