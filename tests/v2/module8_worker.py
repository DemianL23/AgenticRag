from __future__ import annotations

import argparse
import json
from pathlib import Path

from agenticrag.v2.config import V2Config, V2PersistenceConfig
from agenticrag.v2.durable import DurableV23Service
from module8_support import make_runtime


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("operation", choices=("start", "status", "resume"))
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--request-id")
    args = parser.parse_args()
    config = V2Config(persistence=V2PersistenceConfig(sqlite_path=str(args.db)))
    if args.operation == "status":
        service = DurableV23Service(config)
        try:
            print(json.dumps(service.status(args.request_id).to_json(), ensure_ascii=False))
        finally:
            service.close()
        return
    service, *_ = make_runtime(config, resume_mode=args.operation == "resume")
    try:
        if args.operation == "start":
            result = service.start("Which year?", request_id=args.request_id)
        else:
            result = service.resume(
                args.request_id,
                {"responses": [{"item_id": "ITEM_001", "clarify_values": {"year": "2019"}}]},
            )
        print(
            json.dumps(
                {
                    "pid": __import__("os").getpid(),
                    "request_id": result.request_id,
                    "execution_status": result.stage_result.execution_status,
                    "answer_outcome": result.stage_result.answer_outcome,
                    "resumable": result.stage_result.resumable,
                    "interrupted": result.interrupted,
                },
                ensure_ascii=False,
            )
        )
    finally:
        service.close()


if __name__ == "__main__":
    main()
