"""SQLite-backed request metadata for the durable V2.3 boundary.

LangGraph owns execution checkpoints.  This module owns the small, explicit
request index used by status, lifecycle, leases, and resume idempotency.  It
never serializes runtime objects or the complete graph state.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable

from .config import V2PersistenceConfig
from .ids import validate_request_id
from .schemas import HITLRequest, ResumeRequest, StageRunResult

SCHEMA_VERSION = 1
UTC = timezone.utc


class PersistenceError(RuntimeError):
    """A durable lifecycle failure with a stable public error code."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(message)


class LeaseConflictError(PersistenceError):
    def __init__(self, message: str = "request lease is held by another worker") -> None:
        super().__init__("resume_conflict", message)


@dataclass(frozen=True, slots=True)
class RequestMetadata:
    request_id: str
    thread_id: str
    target_stage: str
    state_schema_version: str
    execution_status: str
    answer_outcome: str | None
    resumable: bool
    pending_hitl_request_id: str | None
    pending_hitl_request: HITLRequest | None
    schema_version: int
    created_at: datetime
    updated_at: datetime
    expires_at: datetime
    lease_owner: str | None = None
    lease_token: str | None = None
    lease_until: datetime | None = None
    consumed_resume_digest: str | None = None
    consumed_hitl_request_id: str | None = None
    consumed_result: StageRunResult | None = None
    error_code: str | None = None
    error_message: str | None = None
    checkpoint_ref: str | None = None

    @property
    def expired(self) -> bool:
        return datetime.now(UTC) >= self.expires_at


@dataclass(frozen=True, slots=True)
class Lease:
    request_id: str
    owner: str
    token: str
    lease_until: datetime


@dataclass(frozen=True, slots=True)
class CleanupReport:
    expired_request_ids: tuple[str, ...]
    eligible_checkpoint_count: int
    applied: bool


def utc_now() -> datetime:
    return datetime.now(UTC)


def canonical_resume_digest(request: ResumeRequest) -> str:
    """Return a cross-process stable identity for a validated resume payload."""

    encoded = json.dumps(
        request.model_dump(mode="json"),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class RequestMetadataRepository:
    """Small transaction-safe repository for ``v2_requests``."""

    def __init__(
        self,
        path: str | Path,
        *,
        schema_version: int = SCHEMA_VERSION,
        busy_timeout_seconds: float = 30.0,
        now_fn: Callable[[], datetime] = utc_now,
    ) -> None:
        self.path = str(path)
        self.schema_version = schema_version
        self.busy_timeout_seconds = busy_timeout_seconds
        self.now_fn = now_fn
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(
            self.path,
            timeout=busy_timeout_seconds,
            isolation_level=None,
            check_same_thread=False,
        )
        self.connection.row_factory = sqlite3.Row
        self.connection.execute(
            f"PRAGMA busy_timeout={max(1, int(busy_timeout_seconds * 1000))}"
        )
        if self.path != ":memory:":
            self.connection.execute("PRAGMA journal_mode=WAL")
        self.setup()

    def setup(self) -> None:
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS v2_requests (
                request_id TEXT PRIMARY KEY,
                thread_id TEXT NOT NULL,
                target_stage TEXT NOT NULL,
                state_schema_version TEXT NOT NULL,
                execution_status TEXT NOT NULL,
                answer_outcome TEXT,
                resumable INTEGER NOT NULL,
                pending_hitl_request_id TEXT,
                pending_hitl_request_json TEXT,
                schema_version INTEGER NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                expires_at TEXT NOT NULL,
                lease_owner TEXT,
                lease_token TEXT,
                lease_until TEXT,
                consumed_resume_digest TEXT,
                consumed_hitl_request_id TEXT,
                consumed_result_json TEXT,
                error_code TEXT,
                error_message TEXT,
                checkpoint_ref TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_v2_requests_expiry
                ON v2_requests(expires_at);
            """
        )
        columns = {
            str(row["name"])
            for row in self.connection.execute("PRAGMA table_info(v2_requests)")
        }
        required = {
            "request_id",
            "thread_id",
            "target_stage",
            "state_schema_version",
            "schema_version",
            "pending_hitl_request_json",
            "consumed_hitl_request_id",
        }
        if not required <= columns:
            raise PersistenceError(
                "checkpoint_version_incompatible",
                "v2_requests schema 不兼容；Module 8 不执行自动迁移",
            )

    def close(self) -> None:
        self.connection.close()

    def create(
        self,
        *,
        request_id: str,
        thread_id: str,
        target_stage: str,
        expires_at: datetime,
        state_schema_version: str = "v2_3",
        checkpoint_ref: str | None = None,
    ) -> RequestMetadata:
        validate_request_id(request_id)
        if target_stage != "v2_3":
            raise PersistenceError(
                "request_not_resumable", "durable metadata 只接受 target_stage=v2_3"
            )
        now = self._now()
        try:
            self.connection.execute(
                """
                INSERT INTO v2_requests (
                    request_id, thread_id, target_stage, execution_status,
                    state_schema_version,
                    answer_outcome, resumable, pending_hitl_request_id,
                    pending_hitl_request_json, schema_version, created_at,
                    updated_at, expires_at, checkpoint_ref
                ) VALUES (?, ?, ?, 'running', ?, NULL, 0, NULL, NULL, ?, ?, ?, ?, ?)
                """,
                (
                    request_id,
                    thread_id,
                    target_stage,
                    state_schema_version,
                    self.schema_version,
                    _format_time(now),
                    _format_time(now),
                    _format_time(expires_at),
                    checkpoint_ref or thread_id,
                ),
            )
        except sqlite3.IntegrityError as exc:
            raise PersistenceError("resume_conflict", "request_id 已存在") from exc
        return self.get(request_id)

    def get(self, request_id: str) -> RequestMetadata:
        row = self.connection.execute(
            "SELECT * FROM v2_requests WHERE request_id = ?", (request_id,)
        ).fetchone()
        if row is None:
            raise PersistenceError("checkpoint_not_found", "request metadata 不存在")
        if int(row["schema_version"]) != self.schema_version:
            raise PersistenceError(
                "checkpoint_version_incompatible",
                "request metadata schema version 不兼容",
            )
        return _metadata_from_row(row)

    def update_stage(
        self,
        request_id: str,
        stage: StageRunResult,
        *,
        pending: HITLRequest | None = None,
        consumed_resume_digest: str | None = None,
        consumed_hitl_request_id: str | None = None,
        lease: Lease | None = None,
    ) -> RequestMetadata:
        now = self._now()
        pending_json = (
            json.dumps(pending.model_dump(mode="json"), ensure_ascii=False, sort_keys=True)
            if pending is not None
            else None
        )
        result_json = json.dumps(
            stage.model_dump(mode="json"), ensure_ascii=False, sort_keys=True
        )
        where = "request_id = ?"
        params: list[object] = [
            stage.target_stage,
            stage.execution_status,
            stage.answer_outcome,
            int(stage.resumable),
            pending.id if pending else None,
            pending_json,
            _format_time(now),
            stage.error.code if stage.error else None,
            stage.error.message if stage.error else None,
            request_id,
        ]
        if consumed_resume_digest is not None:
            params.insert(-1, consumed_resume_digest)
            params.insert(-1, consumed_hitl_request_id)
            params.insert(-1, result_json)
            consumed_sql = (
                ", consumed_resume_digest = ?, consumed_hitl_request_id = ?, "
                "consumed_result_json = ?"
            )
        else:
            consumed_sql = ""
        if lease is not None:
            where += " AND lease_owner = ? AND lease_token = ?"
            params.extend([lease.owner, lease.token])
        cursor = self.connection.execute(
            f"""
            UPDATE v2_requests SET
                target_stage=?, execution_status=?, answer_outcome=?, resumable=?,
                pending_hitl_request_id=?, pending_hitl_request_json=?, updated_at=?,
                error_code=?, error_message=?{consumed_sql}
            WHERE {where}
            """,
            params,
        )
        if cursor.rowcount != 1:
            raise PersistenceError("resume_conflict", "metadata update 未获得 request lease")
        return self.get(request_id)

    def acquire_lease(
        self,
        request_id: str,
        *,
        owner: str,
        token: str,
        duration_seconds: int,
    ) -> Lease:
        now = self._now()
        lease_until = now + timedelta(seconds=duration_seconds)
        now_text = _format_time(now)
        self.connection.execute("BEGIN IMMEDIATE")
        try:
            cursor = self.connection.execute(
                """
                UPDATE v2_requests
                SET lease_owner=?, lease_token=?, lease_until=?, updated_at=?
                WHERE request_id=?
                  AND (lease_token IS NULL OR lease_until IS NULL OR lease_until <= ?)
                """,
                (
                    owner,
                    token,
                    _format_time(lease_until),
                    now_text,
                    request_id,
                    now_text,
                ),
            )
            if cursor.rowcount != 1:
                self.connection.rollback()
                self.get(request_id)
                raise LeaseConflictError()
            self.connection.commit()
        except Exception:
            if self.connection.in_transaction:
                self.connection.rollback()
            raise
        return Lease(request_id, owner, token, lease_until)

    def release_lease(self, lease: Lease) -> None:
        self.connection.execute(
            """
            UPDATE v2_requests
            SET lease_owner=NULL, lease_token=NULL, lease_until=NULL
            WHERE request_id=? AND lease_owner=? AND lease_token=?
            """,
            (lease.request_id, lease.owner, lease.token),
        )

    def mark_expired(self, request_id: str) -> RequestMetadata:
        now = self._now()
        self.connection.execute(
            """
            UPDATE v2_requests SET execution_status='failed', answer_outcome=NULL,
                resumable=0, pending_hitl_request_id=NULL,
                pending_hitl_request_json=NULL, updated_at=?,
                error_code='checkpoint_expired',
                error_message='request TTL 已过期'
            WHERE request_id=?
            """,
            (_format_time(now), request_id),
        )
        return self.get(request_id)

    def cleanup_expired(self, *, apply: bool = False) -> CleanupReport:
        now_text = _format_time(self._now())
        rows = self.connection.execute(
            """
            SELECT request_id FROM v2_requests
            WHERE expires_at <= ?
              AND (lease_until IS NULL OR lease_until <= ?)
            ORDER BY request_id
            """,
            (now_text, now_text),
        ).fetchall()
        request_ids = tuple(str(row["request_id"]) for row in rows)
        if apply and request_ids:
            self.connection.executemany(
                "DELETE FROM v2_requests WHERE request_id = ?",
                [(request_id,) for request_id in request_ids],
            )
        return CleanupReport(request_ids, len(request_ids), apply)

    def delete_requests(self, request_ids: tuple[str, ...]) -> None:
        """Delete only an already-inspected set of expired request rows."""
        if request_ids:
            self.connection.executemany(
                "DELETE FROM v2_requests WHERE request_id = ?",
                [(request_id,) for request_id in request_ids],
            )

    def _now(self) -> datetime:
        value = self.now_fn()
        if value.tzinfo is None:
            raise ValueError("durable clock 必须返回 timezone-aware UTC")
        return value.astimezone(UTC)


def _metadata_from_row(row: sqlite3.Row) -> RequestMetadata:
    pending = (
        HITLRequest.model_validate(json.loads(row["pending_hitl_request_json"]))
        if row["pending_hitl_request_json"]
        else None
    )
    consumed = (
        StageRunResult.model_validate(json.loads(row["consumed_result_json"]))
        if row["consumed_result_json"]
        else None
    )
    return RequestMetadata(
        request_id=str(row["request_id"]),
        thread_id=str(row["thread_id"]),
        target_stage=str(row["target_stage"]),
        state_schema_version=str(row["state_schema_version"]),
        execution_status=str(row["execution_status"]),
        answer_outcome=row["answer_outcome"],
        resumable=bool(row["resumable"]),
        pending_hitl_request_id=row["pending_hitl_request_id"],
        pending_hitl_request=pending,
        schema_version=int(row["schema_version"]),
        created_at=_parse_time(row["created_at"]),
        updated_at=_parse_time(row["updated_at"]),
        expires_at=_parse_time(row["expires_at"]),
        lease_owner=row["lease_owner"],
        lease_token=row["lease_token"],
        lease_until=_parse_time(row["lease_until"]) if row["lease_until"] else None,
        consumed_resume_digest=row["consumed_resume_digest"],
        consumed_hitl_request_id=row["consumed_hitl_request_id"],
        consumed_result=consumed,
        error_code=row["error_code"],
        error_message=row["error_message"],
        checkpoint_ref=row["checkpoint_ref"],
    )


def _format_time(value: datetime) -> str:
    if value.tzinfo is None:
        raise ValueError("durable timestamp 必须 timezone-aware")
    return value.astimezone(UTC).isoformat(timespec="microseconds")


def _parse_time(value: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        raise PersistenceError("checkpoint_version_incompatible", "存储 timestamp 缺少 timezone")
    return parsed.astimezone(UTC)
