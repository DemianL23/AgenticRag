"""Durable V2.3 application service.

This module deliberately separates the LangGraph checkpoint from request
metadata.  The checkpoint owns graph continuation; ``v2_requests`` owns
lookup, lifecycle, lease, and idempotency information.
"""

from __future__ import annotations

import os
import re
import sqlite3
import threading
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable

from langgraph.types import Command

from .config import V2Config
from .graph import build_graph_v2_3, initial_v2_3_state
from .hitl import HITLResumeError, HITLResumeService
from .persistence import (
    CleanupReport,
    Lease,
    PersistenceError,
    RequestMetadata,
    RequestMetadataRepository,
    canonical_resume_digest,
)
from .policies import validate_resume_request
from .schemas import ExecutionError, ResumeRequest, StageRunResult
from .state import V2State, V2_STATE_SCHEMA_VERSION

UTC = timezone.utc


class PersistenceDependencyError(PersistenceError):
    def __init__(self) -> None:
        super().__init__(
            "persistence_dependency_missing",
            "V2.3 persistence requires the v2-persistence optional extra",
        )


@dataclass(frozen=True, slots=True)
class DurableRun:
    request_id: str
    thread_id: str
    state: V2State
    stage_result: StageRunResult
    interrupted: bool = False


@dataclass(frozen=True, slots=True)
class DurableStatus:
    metadata: RequestMetadata

    def to_json(self) -> dict[str, object]:
        metadata = self.metadata
        result: dict[str, object] = {
            "request_id": metadata.request_id,
            "thread_id": metadata.thread_id,
            "target_stage": metadata.target_stage,
            "state_schema_version": metadata.state_schema_version,
            "execution_status": metadata.execution_status,
            "answer_outcome": metadata.answer_outcome,
            "resumable": metadata.resumable,
            "pending_hitl_request_id": metadata.pending_hitl_request_id,
            "pending_hitl_request": (
                metadata.pending_hitl_request.model_dump(mode="json")
                if metadata.pending_hitl_request
                else None
            ),
            "schema_version": metadata.schema_version,
            "created_at": metadata.created_at.isoformat(),
            "updated_at": metadata.updated_at.isoformat(),
            "expires_at": metadata.expires_at.isoformat(),
            "expired": metadata.expired,
            "error_code": metadata.error_code,
            "error_message": metadata.error_message,
            "checkpoint_ref": metadata.checkpoint_ref,
        }
        if metadata.consumed_resume_digest:
            result["consumed_resume_digest"] = metadata.consumed_resume_digest
        return result


class _LeaseHeartbeat:
    """Bounded lease renewal worker used only around one durable execution."""

    def __init__(
        self,
        repository: RequestMetadataRepository,
        lease: Lease,
        *,
        duration_seconds: int,
        interval_seconds: float,
        on_renew: Callable[[Lease], None] | None = None,
    ) -> None:
        if interval_seconds <= 0 or interval_seconds >= duration_seconds:
            raise ValueError("lease heartbeat interval must be shorter than lease duration")
        self.repository = repository
        self._lease = lease
        self.duration_seconds = duration_seconds
        self.interval_seconds = interval_seconds
        self.on_renew = on_renew
        self._stop = threading.Event()
        self._lost: PersistenceError | None = None
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self.renewal_count = 0

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("lease heartbeat already started")
        self._thread = threading.Thread(
            target=self._run,
            name="agenticrag-v2-lease-heartbeat",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
            self._thread = None

    def __enter__(self) -> "_LeaseHeartbeat":
        self.start()
        return self

    def __exit__(self, *_: object) -> None:
        self.stop()

    def current_lease(self) -> Lease:
        with self._lock:
            return self._lease

    def raise_if_lost(self) -> None:
        with self._lock:
            failure = self._lost
        if failure is not None:
            raise failure

    def _run(self) -> None:
        while not self._stop.wait(self.interval_seconds):
            try:
                renewed = self.repository.renew_lease(
                    self.current_lease(), duration_seconds=self.duration_seconds
                )
                with self._lock:
                    self._lease = renewed
                    self.renewal_count += 1
                if self.on_renew is not None:
                    self.on_renew(renewed)
            except Exception as exc:
                failure = PersistenceError(
                    "resume_conflict",
                    _safe_error_message(exc) or "lease heartbeat failed",
                )
                with self._lock:
                    self._lost = failure
                return


class DurableV23Service:
    """Start, inspect, resume, and clean up durable V2.3 requests."""

    def __init__(
        self,
        config: V2Config | None = None,
        *,
        repository: RequestMetadataRepository | None = None,
        checkpointer: object | None = None,
        planner: object | None = None,
        retrieval: object | None = None,
        grader: object | None = None,
        recovery: object | None = None,
        finding: object | None = None,
        synthesis: object | None = None,
        hitl: object | None = None,
        resume_service: HITLResumeService | None = None,
        owner_id: str | None = None,
        now_fn: Callable[[], datetime] | None = None,
        heartbeat_interval_seconds: float | None = None,
    ) -> None:
        self.config = config or V2Config.from_env()
        self._now = now_fn or (lambda: datetime.now(UTC))
        path = repository.path if repository is not None else self.config.persistence.sqlite_path
        self._repository_owned = repository is None
        self.repository = repository or RequestMetadataRepository(
            path,
            schema_version=self.config.persistence.schema_version,
            busy_timeout_seconds=self.config.persistence.sqlite_busy_timeout_seconds,
            now_fn=self._now,
        )
        self._checkpointer_connection: sqlite3.Connection | None = None
        if checkpointer is None:
            self.checkpointer = self._open_checkpointer(path)
        else:
            self.checkpointer = checkpointer
        self.owner_id = owner_id or f"pid-{os.getpid()}-{uuid.uuid4()}"
        self.heartbeat_interval_seconds = heartbeat_interval_seconds
        self.planner = planner
        self.retrieval = retrieval
        self.grader = grader
        self.recovery = recovery
        self.finding = finding
        self.synthesis = synthesis
        self.hitl = hitl
        if resume_service is not None:
            self.resume_service = resume_service
        else:
            resume_backend = getattr(self.retrieval, "backend", None)
            self.resume_service = HITLResumeService(
                self.config,
                backend=resume_backend,
                grader=self.grader,
                recovery=self.recovery,
                finding=self.finding,
                synthesis=self.synthesis,
            )

    def close(self) -> None:
        if self._checkpointer_connection is not None:
            self._checkpointer_connection.close()
            self._checkpointer_connection = None
        if self._repository_owned:
            self.repository.close()
            self._repository_owned = False
        # A caller-owned repository remains usable after service shutdown.

    def __enter__(self) -> "DurableV23Service":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def start(
        self,
        question: str,
        *,
        request_id: str | None = None,
        response_language: str | None = None,
    ) -> DurableRun:
        initial = initial_v2_3_state(
            question,
            request_id=request_id,
            response_language=response_language,
        )
        request_id = initial["request_id"]
        now = self._utc_now()
        metadata = self.repository.create(
            request_id=request_id,
            thread_id=request_id,
            target_stage="v2_3",
            state_schema_version=initial["state_schema_version"],
            expires_at=now + timedelta(seconds=self.config.budgets.checkpoint_ttl_seconds),
            checkpoint_ref=request_id,
        )
        del metadata
        try:
            graph = self._graph()
            graph_config = self._graph_config(request_id)
            raw_result = graph.invoke(initial, graph_config)
            state = _state_without_graph_meta(raw_result)
            interrupts = raw_result.get("__interrupt__", [])
            if interrupts:
                stage = _waiting_stage(state)
                try:
                    # Persist the stage result before publishing resumable=true.
                    graph.update_state(graph_config, {"stage_result": stage})
                except Exception as exc:
                    failed = _failed_stage(request_id, "checkpoint_write_failed", exc)
                    self.repository.update_stage(request_id, failed)
                    return DurableRun(request_id, request_id, state, failed)
                state["stage_result"] = stage
                pending = state.get("pending_hitl_request")
                self.repository.update_stage(request_id, stage, pending=pending)
                return DurableRun(request_id, request_id, state, stage, interrupted=True)
            stage = state.get("stage_result")
            if not isinstance(stage, StageRunResult):
                raise PersistenceError(
                    "checkpoint_write_failed", "V2.3 graph did not produce StageRunResult"
                )
            self.repository.update_stage(
                request_id,
                stage,
                pending=state.get("pending_hitl_request"),
            )
            return DurableRun(request_id, request_id, state, stage)
        except PersistenceError:
            raise
        except Exception as exc:
            failed = _failed_stage(request_id, "durable_execution_failed", exc)
            self.repository.update_stage(request_id, failed)
            return DurableRun(request_id, request_id, initial, failed)

    def status(self, request_id: str) -> DurableStatus:
        return DurableStatus(self.repository.get(request_id))

    def resume(
        self,
        request_id: str,
        request: ResumeRequest | dict[str, object],
    ) -> DurableRun:
        metadata = self.repository.get(request_id)
        self._validate_metadata_state_schema(metadata)
        if metadata.error_code == "checkpoint_expired":
            raise PersistenceError("checkpoint_expired", "request TTL 已过期")
        if self._utc_now() >= metadata.expires_at and metadata.resumable:
            self.repository.mark_expired(request_id)
            raise PersistenceError("checkpoint_expired", "request TTL 已过期")
        # Validate the durable business-state version before taking a lease or
        # accepting a resume payload.  A missing checkpoint is left to the
        # existing checkpoint_not_found path below so its lifecycle semantics
        # remain unchanged.
        preflight_state = self._checkpoint_state(metadata.thread_id)
        self._validate_checkpoint_state_schema(metadata, preflight_state)
        pending = metadata.pending_hitl_request
        if pending is None:
            if metadata.consumed_resume_digest is not None:
                candidate = self._coerce_resume_checked(request, metadata, None)
                digest = canonical_resume_digest(candidate)
                if digest == metadata.consumed_resume_digest and metadata.consumed_result:
                    return self._idempotent_result(metadata)
                raise PersistenceError("resume_conflict", "HITL request 已被消费且 payload 不同")
            raise PersistenceError("request_not_resumable", "request 没有 pending HITLRequest")
        candidate = self._coerce_resume_checked(request, metadata, pending)
        if candidate.hitl_request_id != pending.id:
            raise PersistenceError("resume_conflict", "ResumeRequest 引用了 stale HITLRequest")
        try:
            validate_resume_request(candidate, pending)
        except Exception as exc:
            raise PersistenceError(
                "resume_payload_invalid", _safe_error_message(exc)
            ) from exc
        digest = canonical_resume_digest(candidate)
        if metadata.consumed_resume_digest is not None:
            if digest == metadata.consumed_resume_digest and metadata.consumed_result:
                return self._idempotent_result(metadata)
            raise PersistenceError("resume_conflict", "HITL request 已被消费且 payload 不同")
        if metadata.target_stage != "v2_3" or not metadata.resumable:
            raise PersistenceError("request_not_resumable", "只有可恢复的 V2.3 request 才能 resume")
        if self._utc_now() >= metadata.expires_at:
            self.repository.mark_expired(request_id)
            raise PersistenceError("checkpoint_expired", "request TTL 已过期")

        lease = self.repository.acquire_lease(
            request_id,
            owner=self.owner_id,
            token=str(uuid.uuid4()),
            duration_seconds=self.config.persistence.lease_duration_seconds,
        )
        active_lease = lease
        heartbeat: _LeaseHeartbeat | None = None
        try:
            current = self.repository.get(request_id)
            self._validate_metadata_state_schema(current)
            if current.consumed_resume_digest is not None:
                if digest == current.consumed_resume_digest and current.consumed_result:
                    return self._idempotent_result(current)
                raise PersistenceError("resume_conflict", "HITL request 已被消费且 payload 不同")
            if self._utc_now() >= current.expires_at:
                self.repository.mark_expired(request_id)
                raise PersistenceError("checkpoint_expired", "request TTL 已过期")
            checkpoint = self._checkpoint_tuple(request_id)
            if checkpoint is None:
                failed = _failed_stage(
                    request_id,
                    "checkpoint_not_found",
                    PersistenceError("checkpoint_not_found", "LangGraph checkpoint 不存在"),
                )
                self.repository.update_stage(request_id, failed, lease=lease)
                raise PersistenceError("checkpoint_not_found", "LangGraph checkpoint 不存在")
            checkpoint_state = self._checkpoint_state(current.thread_id)
            self._validate_checkpoint_state_schema(current, checkpoint_state)
            if checkpoint_state is None or not _checkpoint_matches_pending(
                checkpoint_state, current
            ):
                failed = _failed_stage(
                    request_id,
                    "request_not_resumable",
                    PersistenceError(
                        "request_not_resumable",
                        "metadata 与 LangGraph waiting checkpoint 不一致",
                    ),
                )
                self.repository.update_stage(request_id, failed, lease=lease)
                raise PersistenceError(
                    "request_not_resumable",
                    "metadata 与 LangGraph waiting checkpoint 不一致",
                )
            heartbeat = _LeaseHeartbeat(
                self.repository,
                active_lease,
                duration_seconds=self.config.persistence.lease_duration_seconds,
                interval_seconds=self._heartbeat_interval_seconds(),
            )
            with heartbeat:
                graph = self._graph()
                raw_result = graph.invoke(
                    Command(resume=candidate.model_dump(mode="json")),
                    self._graph_config(current.thread_id),
                )
                state = _state_without_graph_meta(raw_result)
                stage = state.get("stage_result")
                if not isinstance(stage, StageRunResult):
                    raise PersistenceError(
                        "checkpoint_write_failed", "resume graph did not produce StageRunResult"
                    )
                # Stop renewal before the authoritative metadata write.  This
                # closes the small race where a failed heartbeat could arrive
                # after the health check but before commit.
                heartbeat.stop()
                heartbeat.raise_if_lost()
                active_lease = heartbeat.current_lease()
                heartbeat.raise_if_lost()
                updated = self.repository.update_stage(
                    request_id,
                    stage,
                    pending=state.get("pending_hitl_request"),
                    consumed_resume_digest=digest,
                    consumed_hitl_request_id=candidate.hitl_request_id,
                    lease=active_lease,
                )
                del updated
                return DurableRun(request_id, current.thread_id, state, stage)
        except HITLResumeError:
            raise
        except PersistenceError as exc:
            if exc.code in {
                "checkpoint_not_found",
                "checkpoint_expired",
                "checkpoint_version_incompatible",
                "resume_conflict",
            }:
                raise
            failed = _failed_stage(request_id, exc.code, exc)
            self.repository.update_stage(
                request_id,
                failed,
                consumed_resume_digest=digest,
                lease=active_lease,
            )
            raise
        except Exception as exc:
            failed = _failed_stage(request_id, "durable_execution_failed", exc)
            self.repository.update_stage(
                request_id,
                failed,
                consumed_resume_digest=digest,
                lease=active_lease,
            )
            raise PersistenceError(
                "durable_execution_failed", _safe_error_message(exc)
            ) from exc
        finally:
            if heartbeat is not None:
                active_lease = heartbeat.current_lease()
            self.repository.release_lease(active_lease)

    def cleanup(self, *, apply: bool = False):
        report = self.repository.cleanup_expired(apply=False)
        if not apply:
            return report
        for request_id in report.expired_request_ids:
            delete = getattr(self.checkpointer, "delete_thread", None)
            if delete is not None:
                delete(request_id)
        self.repository.delete_requests(report.expired_request_ids)
        return CleanupReport(
            report.expired_request_ids, report.eligible_checkpoint_count, True
        )

    def _coerce_resume(
        self,
        request: ResumeRequest | dict[str, object],
        metadata: RequestMetadata,
        pending: Any,
    ) -> ResumeRequest:
        if isinstance(request, ResumeRequest):
            return request
        payload = dict(request)
        payload.setdefault("request_id", metadata.request_id)
        if pending is not None:
            payload.setdefault("hitl_request_id", pending.id)
        elif metadata.consumed_hitl_request_id is not None:
            payload.setdefault("hitl_request_id", metadata.consumed_hitl_request_id)
        return ResumeRequest.model_validate(payload)

    def _coerce_resume_checked(
        self,
        request: ResumeRequest | dict[str, object],
        metadata: RequestMetadata,
        pending: Any,
    ) -> ResumeRequest:
        try:
            return self._coerce_resume(request, metadata, pending)
        except Exception as exc:
            raise PersistenceError(
                "resume_payload_invalid", _safe_error_message(exc)
            ) from exc

    def _idempotent_result(self, metadata: RequestMetadata) -> DurableRun:
        stage = metadata.consumed_result
        if stage is None:
            raise PersistenceError("resume_conflict", "缺少已消费 resume result")
        state = self._checkpoint_state(metadata.thread_id)
        self._validate_checkpoint_state_schema(metadata, state)
        if state is None:
            # The business result is still deterministic, but a missing graph
            # checkpoint must remain visible to callers as a persistence error.
            raise PersistenceError("checkpoint_not_found", "LangGraph checkpoint 不存在")
        return DurableRun(metadata.request_id, metadata.thread_id, state, stage)

    def _heartbeat_interval_seconds(self) -> float:
        duration = float(self.config.persistence.lease_duration_seconds)
        if self.heartbeat_interval_seconds is not None:
            return self.heartbeat_interval_seconds
        # A fraction-based default stays below every supported lease duration;
        # the cap keeps long leases responsive without needless writes.
        return min(duration / 3.0, 30.0)

    @staticmethod
    def _validate_metadata_state_schema(metadata: RequestMetadata) -> None:
        if metadata.state_schema_version != V2_STATE_SCHEMA_VERSION:
            raise PersistenceError(
                "checkpoint_version_incompatible",
                "metadata state schema version 不兼容；不执行自动迁移",
            )

    @classmethod
    def _validate_checkpoint_state_schema(
        cls, metadata: RequestMetadata, state: V2State | None
    ) -> None:
        cls._validate_metadata_state_schema(metadata)
        if state is None:
            return
        if state.get("state_schema_version") != metadata.state_schema_version:
            raise PersistenceError(
                "checkpoint_version_incompatible",
                "checkpoint state schema version 与 metadata 不一致；不执行自动迁移",
            )

    def _graph(self):
        kwargs: dict[str, object] = {"config": self.config, "checkpointer": self.checkpointer}
        if self.planner is not None:
            kwargs["planner"] = self.planner
        if self.retrieval is not None:
            kwargs["retrieval"] = self.retrieval
        if self.grader is not None:
            kwargs["grader"] = self.grader
        if self.recovery is not None:
            kwargs["recovery"] = self.recovery
        if self.finding is not None:
            kwargs["finding"] = self.finding
        if self.synthesis is not None:
            kwargs["synthesis"] = self.synthesis
        if self.hitl is not None:
            kwargs["hitl"] = self.hitl
        kwargs["resume_service"] = self.resume_service
        return build_graph_v2_3(**kwargs)  # type: ignore[arg-type]

    def _open_checkpointer(self, path: str) -> object:
        try:
            from langgraph.checkpoint.sqlite import SqliteSaver
            from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
        except ModuleNotFoundError as exc:
            raise PersistenceDependencyError() from exc
        if path != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(
            path,
            timeout=self.config.persistence.sqlite_busy_timeout_seconds,
            check_same_thread=False,
        )
        connection.execute(
            f"PRAGMA busy_timeout={max(1, int(self.config.persistence.sqlite_busy_timeout_seconds * 1000))}"
        )
        if path != ":memory:":
            connection.execute("PRAGMA journal_mode=WAL")
        # V2 business models are explicitly allow-listed for msgpack
        # round-trips.  Pickle fallback stays disabled by the serializer
        # default; runtime clients/callables never enter checkpoint state.
        serde = JsonPlusSerializer(
            pickle_fallback=False,
            allowed_msgpack_modules=[
                ("agenticrag.v2.schemas", name)
                for name in (
                    "AnswerLimitation",
                    "ComplexityDecision",
                    "DecompositionResult",
                    "Evidence",
                    "EvidenceGrade",
                    "EvidenceOccurrence",
                    "ExecutionError",
                    "GradeRecord",
                    "GroundedFinding",
                    "HITLItem",
                    "HITLRequest",
                    "HITLResponse",
                    "QueryRevision",
                    "RetrievalAttempt",
                    "RetrievalLatency",
                    "RetrievalResult",
                    "RetrievalTask",
                    "ResumeRequest",
                    "RoutingDecision",
                    "ScopeOption",
                    "StageRunResult",
                    "SynthesizedAnswer",
                    "TaskDraft",
                )
            ]
            + [("agenticrag.v2.planning", "PlanningResult")],
        )
        saver = SqliteSaver(connection, serde=serde)
        saver.setup()
        self._checkpointer_connection = connection
        return saver

    def _checkpoint_tuple(self, thread_id: str):
        try:
            return self.checkpointer.get_tuple(self._graph_config(thread_id))
        except Exception as exc:
            raise PersistenceError("checkpoint_not_found", "LangGraph checkpoint 读取失败") from exc

    def _checkpoint_state(self, thread_id: str) -> V2State | None:
        checkpoint = self._checkpoint_tuple(thread_id)
        if checkpoint is None:
            return None
        values = checkpoint.checkpoint.get("channel_values", {})
        if not isinstance(values, dict):
            raise PersistenceError("checkpoint_version_incompatible", "checkpoint state 不是 object")
        return _state_without_graph_meta(values)

    def _graph_config(self, thread_id: str) -> dict[str, dict[str, str]]:
        return {"configurable": {"thread_id": thread_id}}

    def _utc_now(self) -> datetime:
        value = self._now()
        if value.tzinfo is None:
            raise ValueError("durable now_fn 必须返回 timezone-aware datetime")
        return value.astimezone(UTC)


def _state_without_graph_meta(value: dict[str, object]) -> V2State:
    return {key: item for key, item in value.items() if key != "__interrupt__"}  # type: ignore[return-value]


def _checkpoint_matches_pending(
    state: V2State, metadata: RequestMetadata
) -> bool:
    pending = state.get("pending_hitl_request")
    return (
        state.get("request_id") == metadata.request_id
        and state.get("target_stage") == "v2_3"
        and state.get("execution_status") == "waiting_user"
        and pending is not None
        and pending.id == metadata.pending_hitl_request_id
    )


def _waiting_stage(state: V2State) -> StageRunResult:
    pending = state.get("pending_hitl_request")
    if pending is None:
        raise PersistenceError("checkpoint_write_failed", "waiting state 缺少 pending HITLRequest")
    return StageRunResult(
        request_id=state["request_id"],
        target_stage="v2_3",
        execution_status="waiting_user",
        answer_outcome=None,
        final_answer=None,
        resumable=True,
        pending_hitl_request=pending,
        error=None,
    )


def _failed_stage(request_id: str, code: str, cause: Exception) -> StageRunResult:
    message = _safe_error_message(cause) or code
    return StageRunResult(
        request_id=request_id,
        target_stage="v2_3",
        execution_status="failed",
        answer_outcome=None,
        final_answer=None,
        resumable=False,
        pending_hitl_request=None,
        error=ExecutionError(
            code=code,
            message=message[:1000],
            stage="module8_persistence",
            details={"exception_type": type(cause).__name__},
        ),
    )


def _safe_error_message(exc: Exception, *, limit: int = 1500) -> str:
    """Keep durable diagnostics useful without persisting credentials/raw payloads."""
    message = str(exc).strip()
    message = re.sub(
        r"(?i)(authorization\s*[:=]\s*)([^\s,;]+)",
        r"\1[REDACTED]",
        message,
    )
    message = re.sub(r"(?i)(bearer\s+)[^\s,;]+", r"\1[REDACTED]", message)
    message = re.sub(
        r"(?i)(api[_-]?key|token|secret|password)\s*[:=]\s*[^\s,;]+",
        r"\1=[REDACTED]",
        message,
    )
    return message[:limit]
