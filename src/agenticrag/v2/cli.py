"""Small non-persistent V2 stage runner CLI."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict

from .config import V2Config
from .durable import DurableV23Service
from .persistence import PersistenceError
from .module4 import Module4Service
from .module6 import Module6Service


def main() -> None:
    parser = argparse.ArgumentParser(prog="agenticrag-v2-run")
    parser.add_argument("--stage", choices=("v2_1", "v2_2", "v2_3"), required=True)
    parser.add_argument("question")
    parser.add_argument("--response-language", choices=("zh", "en"))
    parser.add_argument("--sqlite-path")
    args = parser.parse_args()
    if args.stage == "v2_3":
        service = DurableV23Service(config=_config_with_sqlite_path(args.sqlite_path))
        result = service.start(
            args.question, response_language=args.response_language
        )
        print(json.dumps(_durable_run_json(result), ensure_ascii=False, indent=2))
        service.close()
        return
    if args.stage == "v2_1":
        result = Module4Service().run(
            args.question, response_language=args.response_language
        )
    else:
        result = Module6Service().run(
            args.question, response_language=args.response_language
        )
    print(json.dumps(result.stage_result.model_dump(mode="json"), ensure_ascii=False, indent=2))


def status_main() -> None:
    parser = argparse.ArgumentParser(prog="agenticrag-v2-status")
    parser.add_argument("request_id")
    parser.add_argument("--sqlite-path")
    args = parser.parse_args()
    service = DurableV23Service(config=_config_with_sqlite_path(args.sqlite_path))
    try:
        try:
            output = service.status(args.request_id).to_json()
        except PersistenceError as exc:
            _print_persistence_error(exc)
            raise SystemExit(2) from exc
        print(json.dumps(output, ensure_ascii=False, indent=2))
    finally:
        service.close()


def resume_main() -> None:
    parser = argparse.ArgumentParser(prog="agenticrag-v2-resume")
    parser.add_argument("request_id")
    parser.add_argument("--input", required=True, help="JSON ResumeRequest responses payload")
    parser.add_argument("--sqlite-path")
    args = parser.parse_args()
    try:
        payload = json.loads(args.input)
    except json.JSONDecodeError as exc:
        parser.error(f"--input 必须是合法 JSON：{exc}")
    service = DurableV23Service(config=_config_with_sqlite_path(args.sqlite_path))
    try:
        try:
            result = service.resume(args.request_id, payload)
        except PersistenceError as exc:
            _print_persistence_error(exc)
            raise SystemExit(2) from exc
        print(json.dumps(_durable_run_json(result), ensure_ascii=False, indent=2))
    finally:
        service.close()


def cleanup_main() -> None:
    parser = argparse.ArgumentParser(prog="agenticrag-v2-checkpoint-cleanup")
    parser.add_argument("--sqlite-path")
    parser.add_argument(
        "--apply",
        "--execute",
        dest="apply",
        action="store_true",
        help="physically delete eligible expired metadata and checkpoints",
    )
    args = parser.parse_args()
    service = DurableV23Service(config=_config_with_sqlite_path(args.sqlite_path))
    try:
        print(json.dumps(asdict(service.cleanup(apply=args.apply)), ensure_ascii=False, indent=2))
    finally:
        service.close()


def _config_with_sqlite_path(path: str | None) -> V2Config:
    config = V2Config.from_env()
    if path is not None:
        config = config.model_copy(
            update={"persistence": config.persistence.model_copy(update={"sqlite_path": path})}
        )
    return config


def _durable_run_json(result) -> dict[str, object]:
    return {
        "request_id": result.request_id,
        "thread_id": result.thread_id,
        "interrupted": result.interrupted,
        "stage_result": result.stage_result.model_dump(mode="json"),
    }


def _print_persistence_error(error: PersistenceError) -> None:
    print(
        json.dumps(
            {"error": {"code": error.code, "message": str(error)[:1500]}},
            ensure_ascii=False,
        ),
        file=sys.stderr,
    )
