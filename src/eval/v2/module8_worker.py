"""Subprocess worker for deterministic Module 8 acceptance evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from agenticrag.v2.answering import HITLContentGenerator
from agenticrag.v2.config import V2Config
from agenticrag.v2.durable import DurableV23Service
from agenticrag.v2.hitl import HITLResumeService
from agenticrag.v2.persistence import PersistenceError
from agenticrag.v2.recovery import RecoveryService
from agenticrag.v2.retrieval import RetrievalFanoutService
from agenticrag.v2.schemas import (
    Evidence,
    EvidenceOccurrence,
    RetrievalLatency,
    RetrievalResult,
)

from .audit import audit_v2_result
from .contract_harness import (
    DeterministicFinding,
    DeterministicGrader,
    DeterministicHitlModel,
    DeterministicPlanner,
    DeterministicRewrite,
    DeterministicSynthesis,
    HarnessTelemetry,
)

UTC = timezone.utc


class _CrossProcessBackend:
    """Return stable-but-distinct Evidence across independent processes."""

    def __init__(self, telemetry: HarnessTelemetry) -> None:
        self.telemetry = telemetry

    def retrieve(self, **kwargs: object) -> RetrievalResult:
        self.telemetry.backend_calls.append(dict(kwargs))
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


def _runtime(
    config: V2Config, *, start_mode: bool
) -> tuple[DurableV23Service, HarnessTelemetry]:
    telemetry = HarnessTelemetry()
    backend = _CrossProcessBackend(telemetry)
    grader = DeterministicGrader(
        telemetry,
        route_sequence=["clarify"] if start_mode else ["answer"],
        strategy=None,
        fault="none",
    )
    finding = DeterministicFinding(telemetry)
    synthesis = DeterministicSynthesis(telemetry)
    recovery = RecoveryService(
        config,
        generator=DeterministicRewrite("direct_rewrite"),
        backend=backend,
        grader=grader,
    )
    hitl = HITLContentGenerator(
        config,
        model=DeterministicHitlModel(telemetry, scope=False),
    )
    resume = HITLResumeService(
        config,
        backend=backend,
        grader=grader,
        recovery=recovery,
        finding=finding,
        synthesis=synthesis,
    )
    return DurableV23Service(
        config,
        planner=DeterministicPlanner(
            complexity="simple", capability="retrieval_synthesis"
        ),
        retrieval=RetrievalFanoutService(backend, config),
        grader=grader,
        recovery=recovery,
        finding=finding,
        synthesis=synthesis,
        hitl=hitl,
        resume_service=resume,
    ), telemetry


def _valid_payload(year: str = "2019") -> dict[str, object]:
    return {
        "responses": [
            {"item_id": "ITEM_001", "clarify_values": {"year": year}}
        ]
    }


def _run(operation: str, request_id: str, config: V2Config) -> dict[str, object]:
    if operation == "status":
        # Status must reopen the same durable state without constructing the
        # production default adapters; the acceptance fixture stays entirely
        # deterministic and its zero-call telemetry proves the read path.
        service, telemetry = _runtime(config, start_mode=False)
        try:
            before = _business_state(service, request_id)
            result = service.status(request_id).to_json()
            after = _business_state(service, request_id)
            audit = _checkpoint_audit(service, request_id, config)
            return _with_observability(
                {"pid": os.getpid(), **result, **audit},
                before=before,
                after=after,
                telemetries=(telemetry,),
            )
        finally:
            service.close()
    if operation == "expired":
        service, start_telemetry = _runtime(config, start_mode=True)
        try:
            service.start("Which year?", request_id=request_id)
            service.repository.connection.execute(
                "UPDATE v2_requests SET expires_at=? WHERE request_id=?",
                (datetime(2000, 1, 1, tzinfo=UTC).isoformat(), request_id),
            )
        finally:
            service.close()
        service, resume_telemetry = _runtime(config, start_mode=False)
        try:
            before = _business_state(service, request_id)
            try:
                service.resume(request_id, _valid_payload())
            except PersistenceError as exc:
                return _with_observability(
                    {
                        "pid": os.getpid(),
                        "request_id": request_id,
                        "thread_id": request_id,
                        "error_code": exc.code,
                    },
                    before=before,
                    after=_business_state(service, request_id),
                    telemetries=(resume_telemetry,),
                    fixture_telemetries=(start_telemetry,),
                )
            return _with_observability(
                {"pid": os.getpid(), "request_id": request_id, "error_code": None},
                before=before,
                after=_business_state(service, request_id),
                telemetries=(resume_telemetry,),
                fixture_telemetries=(start_telemetry,),
            )
        finally:
            service.close()
    if operation == "lease_recovery":
        service, start_telemetry = _runtime(config, start_mode=True)
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
        service, resume_telemetry = _runtime(config, start_mode=False)
        try:
            before = _business_state(service, request_id)
            return _run_result(
                service.resume(request_id, _valid_payload()),
                service=service,
                before=before,
                telemetries=(start_telemetry, resume_telemetry),
            )
        finally:
            service.close()

    service, telemetry = _runtime(config, start_mode=operation == "start")
    try:
        before = _business_state(service, request_id)
        if operation == "start":
            return _run_result(
                service.start("Which year?", request_id=request_id),
                service=service,
                before=before,
                telemetries=(telemetry,),
            )
        if operation in {"resume", "duplicate"}:
            return _run_result(
                service.resume(request_id, _valid_payload()),
                service=service,
                before=before,
                telemetries=(telemetry,),
            )
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
                return _with_observability(
                    {"pid": os.getpid(), **status, "error_code": exc.code},
                    before=before,
                    after=_business_state(service, request_id),
                    telemetries=(telemetry,),
                )
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
                status = service.status(request_id).to_json()
                return _with_observability(
                    {
                        "pid": os.getpid(),
                        **status,
                        "error_code": exc.code,
                    },
                    before=before,
                    after=_business_state(service, request_id),
                    telemetries=(telemetry,),
                )
        return _with_observability(
            {
                "pid": os.getpid(),
                "request_id": request_id,
                "thread_id": request_id,
                "error_code": None,
            },
            before=before,
            after=_business_state(service, request_id),
            telemetries=(telemetry,),
        )
    finally:
        service.close()


def _run_result(
    result: object,
    *,
    service: DurableV23Service,
    before: dict[str, object] | None,
    telemetries: tuple[HarnessTelemetry, ...],
) -> dict[str, object]:
    state = result.state
    tasks = list(state.get("tasks", {}).values())
    revisions = [revision for task in tasks for revision in task.query_revisions]
    attempts = [
        attempt for revision in revisions for attempt in revision.retrieval_attempts
    ]
    payload = {
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
        "technical_failure_count": sum(
            task.execution_status == "failed" for task in tasks
        )
        + int(result.stage_result.execution_status == "failed"),
        "error_code": (
            result.stage_result.error.code
            if result.stage_result.error is not None
            else None
        ),
        "error_message": (
            result.stage_result.error.message
            if result.stage_result.error is not None
            else None
        ),
    }
    return _with_observability(
        payload,
        before=before,
        after=_business_state(service, result.request_id),
        telemetries=telemetries,
    )


def _business_state(
    service: DurableV23Service, request_id: str
) -> dict[str, object] | None:
    try:
        metadata = service.status(request_id).metadata
        state = service._checkpoint_state(metadata.thread_id)
    except PersistenceError:
        return None
    if state is None:
        return None
    tasks = sorted(
        state.get("tasks", {}).values(), key=lambda item: (item.ordinal, item.id)
    )
    revisions = [revision for task in tasks for revision in task.query_revisions]
    attempts = [
        attempt for revision in revisions for attempt in revision.retrieval_attempts
    ]
    payload: dict[str, object] = {
        "request_id": state.get("request_id"),
        "execution_status": state.get("execution_status"),
        "answer_outcome": state.get("answer_outcome"),
        "task_ids": [task.id for task in tasks],
        "query_revision_ids": [revision.id for revision in revisions],
        "retrieval_attempt_ids": [attempt.id for attempt in attempts],
        "grade_record_ids": [
            record.id for task in tasks for record in task.grade_records
        ],
        "routing_decision_ids": [
            decision.id for task in tasks for decision in task.routing_decisions
        ],
        "finding_evidence_ids": {
            task.id: list(task.grounded_finding.evidence_ids)
            for task in tasks
            if task.grounded_finding is not None
        },
        "hitl_rounds": int(state.get("hitl_rounds", 0)),
        "pending_hitl_request_id": (
            state["pending_hitl_request"].id
            if state.get("pending_hitl_request") is not None
            else None
        ),
    }
    payload.update(
        query_revision_count=len(revisions),
        retrieval_attempt_count=len(attempts),
        grade_record_count=sum(len(task.grade_records) for task in tasks),
        routing_decision_count=sum(len(task.routing_decisions) for task in tasks),
        finding_count=sum(task.grounded_finding is not None for task in tasks),
    )
    return payload


def _with_observability(
    payload: dict[str, object],
    *,
    before: dict[str, object] | None,
    after: dict[str, object] | None,
    telemetries: tuple[HarnessTelemetry, ...] = (),
    fixture_telemetries: tuple[HarnessTelemetry, ...] = (),
) -> dict[str, object]:
    telemetry = _telemetry_counts(telemetries)
    result = {
        **payload,
        "business_state_before": before,
        "business_state_after": after,
        "business_state_before_digest": _business_digest(before),
        "business_state_after_digest": _business_digest(after),
        "telemetry": telemetry,
    }
    if fixture_telemetries:
        result["fixture_telemetry"] = _telemetry_counts(fixture_telemetries)
    return result


def _telemetry_counts(
    telemetries: tuple[HarnessTelemetry, ...],
) -> dict[str, int]:
    return {
        "retrieval_calls": sum(len(item.backend_calls) for item in telemetries),
        "grader_calls": sum(len(item.grader_calls) for item in telemetries),
        "finding_calls": sum(len(item.finding_calls) for item in telemetries),
        "synthesis_calls": sum(item.synthesis_calls for item in telemetries),
        "hitl_calls": sum(item.hitl_calls for item in telemetries),
    }


def _business_digest(value: dict[str, object] | None) -> str | None:
    if value is None:
        return None
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()


def _checkpoint_audit(
    service: DurableV23Service, request_id: str, config: V2Config
) -> dict[str, object]:
    try:
        state = service._checkpoint_state(request_id)
        if state is None or state.get("stage_result") is None:
            raise ValueError("durable checkpoint has no StageRunResult")
        counts = audit_v2_result(
            list(state.get("tasks", {}).values()),
            state.get("evidence", {}),
            state["stage_result"],
            config=config,
            state=state,
        )
        tasks = list(state.get("tasks", {}).values())
        stage = state["stage_result"]
    except Exception as exc:
        return {
            "audit_source": "unavailable",
            "audit_error": f"{type(exc).__name__}: {exc}",
        }
    return {
        "audit_source": "durable_checkpoint_state",
        "audit_counts": counts.model_dump(mode="json"),
        "technical_failure_count": sum(
            task.execution_status == "failed" for task in tasks
        )
        + int(stage.execution_status == "failed"),
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
