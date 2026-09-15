"""Subprocess worker for deterministic Module 8 acceptance evaluation."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from agenticrag.v2.answering import HITLContentGenerator
from agenticrag.v2.config import V2Config
from agenticrag.v2.durable import DurableV23Service
from agenticrag.v2.hitl import HITLResumeService
from agenticrag.v2.persistence import PersistenceError
from agenticrag.v2.retrieval import RetrievalFanoutService
from agenticrag.v2.schemas import (
    Evidence,
    EvidenceOccurrence,
    RetrievalLatency,
    RetrievalResult,
)

from .contract_harness import (
    DeterministicFinding,
    DeterministicGrader,
    DeterministicHitlModel,
    DeterministicPlanner,
    HarnessTelemetry,
)

UTC = timezone.utc


class _CrossProcessBackend:
    """Return stable-but-distinct Evidence across independent processes."""

    def retrieve(self, **kwargs: object) -> RetrievalResult:
        strategy = str(kwargs["strategy"])
        evidence_id = f"E_{strategy.upper()}"
        return RetrievalResult(
            evidence=[
                Evidence(
                    evidence_id=evidence_id,
                    chunk_id=evidence_id,
                    content=f"deterministic {strategy} evidence",
                    doc_id="cross-process-doc",
                    source="cross-process-fixture",
                    page=1,
                    occurrences=[
                        EvidenceOccurrence(
                            task_id=str(kwargs["task_id"]),
                            query_revision_id=str(kwargs["query_revision_id"]),
                            retrieval_attempt_id=str(kwargs["attempt_id"]),
                            strategy=strategy,
                            final_rank=1,
                        )
                    ],
                )
            ],
            latency=RetrievalLatency(total_seconds=0.001),
            trace_ref=str(kwargs["attempt_id"]),
        )


def _runtime(config: V2Config, *, start_mode: bool) -> DurableV23Service:
    telemetry = HarnessTelemetry()
    backend = _CrossProcessBackend()
    grader = DeterministicGrader(
        telemetry,
        route_sequence=["clarify"] if start_mode else ["answer"],
        strategy=None,
        fault="none",
    )
    finding = DeterministicFinding(telemetry)
    hitl = HITLContentGenerator(
        config,
        model=DeterministicHitlModel(telemetry, scope=False),
    )
    resume = HITLResumeService(
        config,
        backend=backend,
        grader=grader,
        finding=finding,
    )
    return DurableV23Service(
        config,
        planner=DeterministicPlanner(
            complexity="simple", capability="retrieval_synthesis"
        ),
        retrieval=RetrievalFanoutService(backend, config),
        grader=grader,
        finding=finding,
        hitl=hitl,
        resume_service=resume,
    )


def _valid_payload(year: str = "2019") -> dict[str, object]:
    return {
        "responses": [
            {"item_id": "ITEM_001", "clarify_values": {"year": year}}
        ]
    }


def _run(operation: str, request_id: str, config: V2Config) -> dict[str, object]:
    if operation == "status":
        service = DurableV23Service(config)
        try:
            result = service.status(request_id).to_json()
            return {"pid": os.getpid(), **result}
        finally:
            service.close()
    if operation == "expired":
        service = _runtime(config, start_mode=True)
        try:
            service.start("Which year?", request_id=request_id)
            service.repository.connection.execute(
                "UPDATE v2_requests SET expires_at=? WHERE request_id=?",
                (datetime(2000, 1, 1, tzinfo=UTC).isoformat(), request_id),
            )
        finally:
            service.close()
        service = _runtime(config, start_mode=False)
        try:
            try:
                service.resume(request_id, _valid_payload())
            except PersistenceError as exc:
                return {
                    "pid": os.getpid(),
                    "request_id": request_id,
                    "thread_id": request_id,
                    "error_code": exc.code,
                }
            return {"pid": os.getpid(), "request_id": request_id, "error_code": None}
        finally:
            service.close()
    if operation == "lease_recovery":
        service = _runtime(config, start_mode=True)
        try:
            service.start("Which year?", request_id=request_id)
            service.repository.acquire_lease(
                request_id,
                owner="simulated-crashed-worker",
                token="simulated-crashed-token",
                duration_seconds=60,
            )
            service.repository.connection.execute(
                "UPDATE v2_requests SET lease_until=? WHERE request_id=?",
                (datetime(2000, 1, 1, tzinfo=UTC).isoformat(), request_id),
            )
        finally:
            service.close()
        service = _runtime(config, start_mode=False)
        try:
            return _run_result(service.resume(request_id, _valid_payload()))
        finally:
            service.close()

    service = _runtime(config, start_mode=operation == "start")
    try:
        if operation == "start":
            return _run_result(
                service.start("Which year?", request_id=request_id)
            )
        if operation in {"resume", "duplicate"}:
            return _run_result(service.resume(request_id, _valid_payload()))
        if operation == "invalid":
            try:
                service.resume(
                    request_id,
                    {
                        "responses": [
                            {
                                "item_id": "ITEM_001",
                                "clarify_values": {"wrong_slot": "2019"},
                            }
                        ]
                    },
                )
            except PersistenceError as exc:
                status = service.status(request_id).to_json()
                return {"pid": os.getpid(), **status, "error_code": exc.code}
        if operation == "stale":
            try:
                service.resume(
                    request_id,
                    {
                        "request_id": request_id,
                        "hitl_request_id": "HITL_000",
                        "responses": [
                            {
                                "item_id": "ITEM_001",
                                "clarify_values": {"year": "2020"},
                            }
                        ],
                    },
                )
            except PersistenceError as exc:
                return {
                    "pid": os.getpid(),
                    "request_id": request_id,
                    "thread_id": request_id,
                    "error_code": exc.code,
                }
        return {
            "pid": os.getpid(),
            "request_id": request_id,
            "thread_id": request_id,
            "error_code": None,
        }
    finally:
        service.close()


def _run_result(result: object) -> dict[str, object]:
    state = result.state
    tasks = list(state.get("tasks", {}).values())
    revisions = [revision for task in tasks for revision in task.query_revisions]
    attempts = [
        attempt for revision in revisions for attempt in revision.retrieval_attempts
    ]
    return {
        "pid": os.getpid(),
        "request_id": result.request_id,
        "thread_id": result.thread_id,
        "execution_status": result.stage_result.execution_status,
        "answer_outcome": result.stage_result.answer_outcome,
        "resumable": result.stage_result.resumable,
        "interrupted": result.interrupted,
        "affected_task_count": len(tasks),
        "new_query_revision_count": sum(
            max(0, len(task.query_revisions) - 1) for task in tasks
        ),
        "user_clarified_retrieval_count": sum(
            attempt.strategy == "user_clarified" for attempt in attempts
        ),
        "hitl_rounds": int(state.get("hitl_rounds", 0)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "operation",
        choices=(
            "start",
            "status",
            "resume",
            "duplicate",
            "stale",
            "invalid",
            "expired",
            "lease_recovery",
        ),
    )
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--config-json", type=Path, required=True)
    parser.add_argument("--request-id", required=True)
    args = parser.parse_args()
    config = V2Config.model_validate_json(args.config_json.read_text(encoding="utf-8"))
    config = config.model_copy(
        update={
            "persistence": config.persistence.model_copy(
                update={"sqlite_path": str(args.db)}
            )
        }
    )
    print(json.dumps(_run(args.operation, args.request_id, config), ensure_ascii=False))


if __name__ == "__main__":
    main()
