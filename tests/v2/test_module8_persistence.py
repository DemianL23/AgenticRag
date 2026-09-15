from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import threading
from datetime import datetime, timedelta, timezone

import pytest

from agenticrag.v2.config import V2BudgetConfig, V2Config, V2PersistenceConfig
from agenticrag.v2.durable import DurableV23Service, _LeaseHeartbeat
from agenticrag.v2.hitl import HITLResumeService
from agenticrag.v2.retrieval import RetrievalFanoutService
from agenticrag.v2.persistence import (
    Lease,
    LeaseConflictError,
    PersistenceError,
    RequestMetadataRepository,
    canonical_resume_digest,
)
from agenticrag.v2.state import V2_STATE_SCHEMA_VERSION
from eval.v2.module8 import build_module8_report, evaluate_module8_start
from agenticrag.v2.schemas import (
    ExecutionError,
    HITLItem,
    HITLRequest,
    ResumeRequest,
    StageRunResult,
)
from module8_support import make_runtime

UTC = timezone.utc


def _stage(request_id: str, *, status: str = "waiting_user", resumable: bool = True) -> StageRunResult:
    pending = (
        HITLRequest(
            id="HITL_001",
            request_id=request_id,
            items=[
                HITLItem(
                    id="ITEM_001",
                    action="clarify",
                    affected_task_ids=["SQ_001"],
                    question="Which year?",
                    missing_slots=["year"],
                )
            ],
        )
        if status == "waiting_user"
        else None
    )
    return StageRunResult(
        request_id=request_id,
        target_stage="v2_3",
        execution_status=status,
        answer_outcome=None if status != "completed" else "complete",
        final_answer=None,
        resumable=resumable,
        pending_hitl_request=pending,
        error=(
            ExecutionError(code="test_failure", message="test")
            if status == "failed"
            else None
        ),
    )


def _resume(request_id: str, *, year: str = "2019") -> ResumeRequest:
    return ResumeRequest(
        request_id=request_id,
        hitl_request_id="HITL_001",
        responses=[
            {
                "item_id": "ITEM_001",
                "clarify_values": {"year": year},
            }
        ],
    )


def _config(path, **updates) -> V2Config:
    persistence = V2PersistenceConfig(sqlite_path=str(path))
    budgets = V2BudgetConfig(**updates)
    return V2Config(persistence=persistence, budgets=budgets)


def test_repository_create_reopen_and_schema_version(tmp_path) -> None:
    path = tmp_path / "requests.sqlite3"
    repo = RequestMetadataRepository(path)
    request_id = "4b7d9e2c-5f24-4b1d-8e6f-2c7a9b1d4e60"
    repo.create(
        request_id=request_id,
        thread_id=request_id,
        target_stage="v2_3",
        expires_at=datetime.now(UTC) + timedelta(days=1),
    )
    repo.close()
    reopened = RequestMetadataRepository(path)
    metadata = reopened.get(request_id)
    assert metadata.thread_id == request_id
    assert metadata.schema_version == 1
    reopened.close()


def test_repository_rejects_unknown_schema_version_without_migration(tmp_path) -> None:
    path = tmp_path / "requests.sqlite3"
    repo = RequestMetadataRepository(path)
    request_id = "4b7d9e2c-5f24-4b1d-8e6f-2c7a9b1d4e60"
    repo.create(
        request_id=request_id,
        thread_id=request_id,
        target_stage="v2_3",
        expires_at=datetime.now(UTC) + timedelta(days=1),
    )
    repo.connection.execute(
        "UPDATE v2_requests SET schema_version=99 WHERE request_id=?", (request_id,)
    )
    with pytest.raises(PersistenceError, match="不兼容") as exc_info:
        repo.get(request_id)
    assert exc_info.value.code == "checkpoint_version_incompatible"
    repo.close()


def test_repository_ttl_and_cleanup_dry_run_apply(tmp_path) -> None:
    now = datetime(2026, 1, 1, tzinfo=UTC)
    path = tmp_path / "requests.sqlite3"
    repo = RequestMetadataRepository(path, now_fn=lambda: now)
    request_id = "4b7d9e2c-5f24-4b1d-8e6f-2c7a9b1d4e60"
    repo.create(
        request_id=request_id,
        thread_id=request_id,
        target_stage="v2_3",
        expires_at=now,
    )
    dry = repo.cleanup_expired()
    assert dry.expired_request_ids == (request_id,)
    assert repo.get(request_id).request_id == request_id
    applied = repo.cleanup_expired(apply=True)
    assert applied.applied is True
    with pytest.raises(PersistenceError) as exc_info:
        repo.get(request_id)
    assert exc_info.value.code == "checkpoint_not_found"
    repo.close()


def test_lease_is_atomic_and_expired_lease_is_recoverable(tmp_path) -> None:
    now = datetime(2026, 1, 1, tzinfo=UTC)
    path = tmp_path / "requests.sqlite3"
    repo_a = RequestMetadataRepository(path, now_fn=lambda: now)
    repo_b = RequestMetadataRepository(path, now_fn=lambda: now)
    request_id = "4b7d9e2c-5f24-4b1d-8e6f-2c7a9b1d4e60"
    repo_a.create(
        request_id=request_id,
        thread_id=request_id,
        target_stage="v2_3",
        expires_at=now + timedelta(days=1),
    )
    lease = repo_a.acquire_lease(
        request_id, owner="a", token="token-a", duration_seconds=60
    )
    with pytest.raises(PersistenceError) as exc_info:
        repo_b.acquire_lease(
            request_id, owner="b", token="token-b", duration_seconds=60
        )
    assert exc_info.value.code == "resume_conflict"
    repo_a.connection.execute(
        "UPDATE v2_requests SET lease_until=? WHERE request_id=?",
        ((now - timedelta(seconds=1)).isoformat(), request_id),
    )
    recovered = repo_b.acquire_lease(
        request_id, owner="b", token="token-b", duration_seconds=60
    )
    assert recovered.owner == "b"
    repo_a.release_lease(lease)
    repo_b.release_lease(recovered)
    repo_a.close()
    repo_b.close()


def test_lease_heartbeat_renews_across_original_expiry(tmp_path) -> None:
    now = [datetime(2026, 1, 1, tzinfo=UTC)]
    path = tmp_path / "heartbeat.sqlite3"
    repo_a = RequestMetadataRepository(path, now_fn=lambda: now[0])
    repo_b = RequestMetadataRepository(path, now_fn=lambda: now[0])
    request_id = "4b7d9e2c-5f24-4b1d-8e6f-2c7a9b1d4e60"
    repo_a.create(
        request_id=request_id,
        thread_id=request_id,
        target_stage="v2_3",
        expires_at=now[0] + timedelta(days=1),
    )
    lease = repo_a.acquire_lease(
        request_id, owner="a", token="token-a", duration_seconds=1
    )
    original_expiry = lease.lease_until
    renewed_after_clock_advance = threading.Event()

    def on_renew(updated: Lease) -> None:
        if updated.lease_until > original_expiry:
            renewed_after_clock_advance.set()

    heartbeat = _LeaseHeartbeat(
        repo_a,
        lease,
        duration_seconds=1,
        interval_seconds=0.01,
        on_renew=on_renew,
    )
    heartbeat.start()
    now[0] += timedelta(milliseconds=500)
    assert renewed_after_clock_advance.wait(1)
    now[0] = datetime(2026, 1, 1, 0, 0, 1, 100000, tzinfo=UTC)
    with pytest.raises(PersistenceError) as exc_info:
        repo_b.acquire_lease(
            request_id, owner="b", token="token-b", duration_seconds=1
        )
    assert exc_info.value.code == "resume_conflict"
    heartbeat.stop()
    repo_a.release_lease(heartbeat.current_lease())
    repo_a.close()
    repo_b.close()


def test_stopped_heartbeat_allows_expired_lease_takeover(tmp_path) -> None:
    now = [datetime(2026, 1, 1, tzinfo=UTC)]
    path = tmp_path / "heartbeat-takeover.sqlite3"
    repo_a = RequestMetadataRepository(path, now_fn=lambda: now[0])
    repo_b = RequestMetadataRepository(path, now_fn=lambda: now[0])
    request_id = "4b7d9e2c-5f24-4b1d-8e6f-2c7a9b1d4e60"
    repo_a.create(
        request_id=request_id,
        thread_id=request_id,
        target_stage="v2_3",
        expires_at=now[0] + timedelta(days=1),
    )
    lease = repo_a.acquire_lease(
        request_id, owner="a", token="token-a", duration_seconds=1
    )
    heartbeat = _LeaseHeartbeat(
        repo_a, lease, duration_seconds=1, interval_seconds=0.01
    )
    heartbeat.start()
    heartbeat.stop()
    now[0] += timedelta(seconds=1)
    recovered = repo_b.acquire_lease(
        request_id, owner="b", token="token-b", duration_seconds=1
    )
    assert recovered.owner == "b"
    repo_b.release_lease(recovered)
    repo_a.close()
    repo_b.close()


def test_wrong_owner_or_token_cannot_renew_lease(tmp_path) -> None:
    now = datetime(2026, 1, 1, tzinfo=UTC)
    repo = RequestMetadataRepository(tmp_path / "renew-owner.sqlite3", now_fn=lambda: now)
    request_id = "4b7d9e2c-5f24-4b1d-8e6f-2c7a9b1d4e60"
    repo.create(
        request_id=request_id,
        thread_id=request_id,
        target_stage="v2_3",
        expires_at=now + timedelta(days=1),
    )
    lease = repo.acquire_lease(
        request_id, owner="a", token="token-a", duration_seconds=60
    )
    with pytest.raises(LeaseConflictError):
        repo.renew_lease(
            Lease(request_id, "other", "token-a", lease.lease_until),
            duration_seconds=60,
        )
    with pytest.raises(LeaseConflictError):
        repo.renew_lease(
            Lease(request_id, "a", "other-token", lease.lease_until),
            duration_seconds=60,
        )
    repo.release_lease(lease)
    repo.close()


def test_failed_heartbeat_cannot_commit_successful_resume(tmp_path) -> None:
    config = _config(
        tmp_path / "heartbeat-failure.sqlite3",
    ).model_copy(
        update={
            "persistence": V2PersistenceConfig(
                sqlite_path=str(tmp_path / "heartbeat-failure.sqlite3"),
                lease_duration_seconds=1,
            )
        }
    )
    service, *_ = make_runtime(config)
    request_id = "4b7d9e2c-5f24-4b1d-8e6f-2c7a9b1d4e60"
    service.start("Which year?", request_id=request_id)
    entered = threading.Event()
    release = threading.Event()
    heartbeat_failed = threading.Event()

    class BlockingGraph:
        def invoke(self, *_args: object, **_kwargs: object) -> dict[str, object]:
            entered.set()
            assert release.wait(1)
            return {"stage_result": _stage(request_id)}

    service._graph = lambda: BlockingGraph()  # type: ignore[method-assign]

    def fail_renewal(*_args: object, **_kwargs: object) -> Lease:
        heartbeat_failed.set()
        raise LeaseConflictError("simulated lost lease")

    service.repository.renew_lease = fail_renewal  # type: ignore[method-assign]
    errors: list[PersistenceError] = []

    def resume() -> None:
        try:
            service.resume(request_id, _resume(request_id))
        except PersistenceError as exc:
            errors.append(exc)

    worker = threading.Thread(target=resume)
    worker.start()
    assert entered.wait(1)
    assert heartbeat_failed.wait(1)
    release.set()
    worker.join(1)
    assert not worker.is_alive()
    assert errors and errors[0].code == "resume_conflict"
    metadata = service.status(request_id).metadata
    assert metadata.execution_status == "waiting_user"
    assert metadata.resumable is True
    assert metadata.consumed_resume_digest is None
    service.close()


def test_initial_v23_state_uses_supported_state_schema_version() -> None:
    from agenticrag.v2.graph import initial_v2_3_state

    assert initial_v2_3_state("Which year?")["state_schema_version"] == V2_STATE_SCHEMA_VERSION


def test_unknown_metadata_state_schema_is_rejected_before_resume(tmp_path) -> None:
    config = _config(tmp_path / "metadata-state-version.sqlite3")
    service, backend, grader, finding, _ = make_runtime(config)
    request_id = "4b7d9e2c-5f24-4b1d-8e6f-2c7a9b1d4e60"
    service.start("Which year?", request_id=request_id)
    service.repository.connection.execute(
        "UPDATE v2_requests SET state_schema_version='unknown' WHERE request_id=?",
        (request_id,),
    )
    with pytest.raises(PersistenceError) as exc_info:
        service.resume(request_id, _resume(request_id))
    assert exc_info.value.code == "checkpoint_version_incompatible"
    assert len(backend.calls) == 1
    assert len(grader.calls) == 1
    assert len(finding.calls) == 0
    metadata = service.status(request_id).metadata
    assert metadata.state_schema_version == "unknown"
    service.close()


def test_unknown_checkpoint_state_schema_is_rejected_before_resume(tmp_path) -> None:
    config = _config(tmp_path / "checkpoint-state-version.sqlite3")
    service, backend, grader, finding, _ = make_runtime(config)
    request_id = "4b7d9e2c-5f24-4b1d-8e6f-2c7a9b1d4e60"
    service.start("Which year?", request_id=request_id)
    service._graph().update_state(
        service._graph_config(request_id), {"state_schema_version": "unknown"}
    )
    with pytest.raises(PersistenceError) as exc_info:
        service.resume(request_id, _resume(request_id))
    assert exc_info.value.code == "checkpoint_version_incompatible"
    assert len(backend.calls) == 1
    assert len(grader.calls) == 1
    assert len(finding.calls) == 0
    service.close()


def test_metadata_and_checkpoint_state_schema_versions_must_match(tmp_path) -> None:
    config = _config(tmp_path / "checkpoint-mismatch.sqlite3")
    service, backend, grader, finding, _ = make_runtime(config)
    request_id = "4b7d9e2c-5f24-4b1d-8e6f-2c7a9b1d4e60"
    service.start("Which year?", request_id=request_id)
    service._graph().update_state(
        service._graph_config(request_id), {"state_schema_version": "v2_3-old"}
    )
    with pytest.raises(PersistenceError) as exc_info:
        service.resume(request_id, _resume(request_id))
    assert exc_info.value.code == "checkpoint_version_incompatible"
    assert len(backend.calls) == 1
    assert len(grader.calls) == 1
    assert len(finding.calls) == 0
    service.close()


def test_durable_v23_waits_then_reopens_and_resumes(tmp_path) -> None:
    config = _config(tmp_path / "durable.sqlite3")
    service, backend, grader, finding, hitl_model = make_runtime(config)
    run = service.start("Which year?", request_id="4b7d9e2c-5f24-4b1d-8e6f-2c7a9b1d4e60")
    assert run.stage_result.execution_status == "waiting_user"
    assert run.stage_result.resumable is True
    checkpoint_state = service._checkpoint_state(run.request_id)
    assert checkpoint_state is not None
    assert checkpoint_state["retrieval_results"]["SQ_001"].diagnostics == {
        "trace_ref": "ATT_SQ001_QR001_001"
    }
    request_id = run.request_id
    service.close()
    service2 = DurableV23Service(
        config,
        planner=service.planner,
        retrieval=RetrievalFanoutService(backend, config),
        grader=grader,
        finding=finding,
        resume_service=service.resume_service,
    )
    resumed = service2.resume(request_id, _resume(request_id))
    assert resumed.stage_result.execution_status == "completed"
    assert resumed.stage_result.answer_outcome == "complete"
    task = resumed.state["tasks"]["SQ_001"]
    assert len(task.query_revisions) == 2
    assert task.query_revisions[-1].source == "hitl"
    assert task.query_revisions[-1].retrieval_attempts[0].strategy == "user_clarified"
    assert len(backend.calls) == 2
    service2.close()


def test_duplicate_resume_is_idempotent_and_different_payload_conflicts(tmp_path) -> None:
    config = _config(tmp_path / "durable.sqlite3")
    service, backend, *_ = make_runtime(config)
    request_id = "4b7d9e2c-5f24-4b1d-8e6f-2c7a9b1d4e60"
    run = service.start("Which year?", request_id=request_id)
    first = service.resume(request_id, _resume(request_id))
    second = service.resume(request_id, _resume(request_id))
    assert second.stage_result.model_dump(mode="json") == first.stage_result.model_dump(mode="json")
    second_from_cli_shape = service.resume(
        request_id,
        {"responses": [{"item_id": "ITEM_001", "clarify_values": {"year": "2019"}}]},
    )
    assert second_from_cli_shape.stage_result.model_dump(mode="json") == first.stage_result.model_dump(mode="json")
    assert len(backend.calls) == 2
    with pytest.raises(PersistenceError) as exc_info:
        service.resume(request_id, _resume(request_id, year="2020"))
    assert exc_info.value.code == "resume_conflict"
    service.close()


def test_invalid_resume_does_not_mutate_or_call_services(tmp_path) -> None:
    config = _config(tmp_path / "durable.sqlite3")
    service, backend, grader, finding, hitl_model = make_runtime(config)
    request_id = "4b7d9e2c-5f24-4b1d-8e6f-2c7a9b1d4e60"
    run = service.start("Which year?", request_id=request_id)
    before = service.status(request_id).to_json()
    with pytest.raises(PersistenceError) as exc_info:
        service.resume(
            request_id,
            {
                "responses": [
                    {"item_id": "ITEM_001", "clarify_values": {"wrong": "2019"}}
                ]
            },
        )
    assert exc_info.value.code == "resume_payload_invalid"
    after = service.status(request_id).to_json()
    assert before == after
    assert len(backend.calls) == 1
    assert len(grader.calls) == 1
    assert len(finding.calls) == 0
    assert hitl_model.calls == 1
    service.close()


def test_v22_metadata_cannot_be_created_as_durable_request(tmp_path) -> None:
    repo = RequestMetadataRepository(tmp_path / "requests.sqlite3")
    with pytest.raises(PersistenceError) as exc_info:
        repo.create(
            request_id="4b7d9e2c-5f24-4b1d-8e6f-2c7a9b1d4e60",
            thread_id="4b7d9e2c-5f24-4b1d-8e6f-2c7a9b1d4e60",
            target_stage="v2_2",
            expires_at=datetime.now(UTC) + timedelta(days=1),
        )
    assert exc_info.value.code == "request_not_resumable"
    repo.close()


def test_durable_resume_rejects_v22_metadata_before_execution(tmp_path) -> None:
    config = _config(tmp_path / "v22.sqlite3")
    service, backend, grader, finding, _ = make_runtime(config)
    request_id = "4b7d9e2c-5f24-4b1d-8e6f-2c7a9b1d4e60"
    service.start("Which year?", request_id=request_id)
    service.repository.connection.execute(
        "UPDATE v2_requests SET target_stage='v2_2', resumable=1 WHERE request_id=?",
        (request_id,),
    )
    with pytest.raises(PersistenceError) as exc_info:
        service.resume(request_id, _resume(request_id))
    assert exc_info.value.code == "request_not_resumable"
    assert len(backend.calls) == 1
    assert len(grader.calls) == 1
    assert len(finding.calls) == 0
    service.close()


def test_active_lease_rejects_durable_resume_without_execution(tmp_path) -> None:
    config = _config(tmp_path / "lease.sqlite3")
    service, backend, *_ = make_runtime(config)
    request_id = "4b7d9e2c-5f24-4b1d-8e6f-2c7a9b1d4e60"
    service.start("Which year?", request_id=request_id)
    lease = service.repository.acquire_lease(
        request_id, owner="other-process", token="other-token", duration_seconds=60
    )
    with pytest.raises(PersistenceError) as exc_info:
        service.resume(request_id, _resume(request_id))
    assert exc_info.value.code == "resume_conflict"
    assert len(backend.calls) == 1
    service.repository.release_lease(lease)
    service.close()


def test_stale_hitl_request_is_rejected_without_mutation(tmp_path) -> None:
    config = _config(tmp_path / "stale.sqlite3")
    service, backend, *_ = make_runtime(config)
    request_id = "4b7d9e2c-5f24-4b1d-8e6f-2c7a9b1d4e60"
    service.start("Which year?", request_id=request_id)
    before = service.status(request_id).to_json()
    stale = _resume(request_id).model_copy(update={"hitl_request_id": "HITL_000"})
    with pytest.raises(PersistenceError) as exc_info:
        service.resume(request_id, stale)
    assert exc_info.value.code == "resume_conflict"
    assert service.status(request_id).to_json() == before
    assert len(backend.calls) == 1
    service.close()


def test_checkpoint_missing_is_reported_without_restart(tmp_path) -> None:
    config = _config(tmp_path / "durable.sqlite3")
    service, backend, *_ = make_runtime(config)
    request_id = "4b7d9e2c-5f24-4b1d-8e6f-2c7a9b1d4e60"
    service.start("Which year?", request_id=request_id)
    service.checkpointer.delete_thread(request_id)
    with pytest.raises(PersistenceError) as exc_info:
        service.resume(request_id, _resume(request_id))
    assert exc_info.value.code == "checkpoint_not_found"
    assert len(backend.calls) == 1
    assert service.status(request_id).metadata.error_code == "checkpoint_not_found"
    service.close()


def test_expired_resume_is_rejected_before_graph_execution(tmp_path) -> None:
    clock = [datetime(2026, 1, 1, tzinfo=UTC)]
    config = _config(tmp_path / "expired.sqlite3")
    config = config.model_copy(
        update={
            "budgets": config.budgets.model_copy(update={"checkpoint_ttl_seconds": 60})
        }
    )
    service, backend, grader, finding, _ = make_runtime(config)
    service._now = lambda: clock[0]
    request_id = "4b7d9e2c-5f24-4b1d-8e6f-2c7a9b1d4e60"
    service.start("Which year?", request_id=request_id)
    clock[0] = clock[0] + timedelta(seconds=60)
    with pytest.raises(PersistenceError) as exc_info:
        service.resume(request_id, _resume(request_id))
    assert exc_info.value.code == "checkpoint_expired"
    assert len(backend.calls) == 1
    assert len(grader.calls) == 1
    assert len(finding.calls) == 0
    assert service.status(request_id).metadata.error_code == "checkpoint_expired"
    service.close()


def test_metadata_checkpoint_mismatch_is_not_resumed(tmp_path) -> None:
    config = _config(tmp_path / "mismatch.sqlite3")
    service, backend, *_ = make_runtime(config)
    request_id = "4b7d9e2c-5f24-4b1d-8e6f-2c7a9b1d4e60"
    service.start("Which year?", request_id=request_id)
    service.repository.connection.execute(
        "UPDATE v2_requests SET pending_hitl_request_id='HITL_STALE' WHERE request_id=?",
        (request_id,),
    )
    with pytest.raises(PersistenceError) as exc_info:
        service.resume(request_id, _resume(request_id))
    assert exc_info.value.code == "request_not_resumable"
    assert len(backend.calls) == 1
    service.close()


def test_canonical_resume_digest_is_stable() -> None:
    request = _resume("4b7d9e2c-5f24-4b1d-8e6f-2c7a9b1d4e60")
    assert canonical_resume_digest(request) == canonical_resume_digest(
        ResumeRequest.model_validate(request.model_dump(mode="json"))
    )


def test_v23_stage_report_records_durable_roles_and_attempt_metrics(tmp_path) -> None:
    config = _config(tmp_path / "report.sqlite3")
    service, *_ = make_runtime(config)
    report = evaluate_module8_start(
        "Which year?",
        service=service,
        config=config,
        output_root=tmp_path / "reports",
        run_id="run-1",
    )
    assert report["target_stage"] == "v2_3"
    assert report["metrics"]["interrupt_count"] == 1
    assert report["metrics"]["technical_failure_count"] == 0
    assert set(report["model_configs"]) == {
        "router", "decomposer", "grader", "rewrite", "hitl",
        "simple_answer", "finding", "synthesis",
    }
    assert (tmp_path / "reports" / "run-1" / "report.json").exists()
    service.close()


def test_durable_v23_cross_process_start_status_resume_status(tmp_path) -> None:
    db = tmp_path / "cross-process.sqlite3"
    worker = str(__import__("pathlib").Path(__file__).with_name("module8_worker.py"))

    def run(operation: str, *extra: str) -> dict[str, object]:
        completed = subprocess.run(
            [os.sys.executable, worker, operation, "--db", str(db), *extra],
            check=True,
            capture_output=True,
            text=True,
        )
        return json.loads(completed.stdout)

    started = run("start", "--request-id", "4b7d9e2c-5f24-4b1d-8e6f-2c7a9b1d4e60")
    assert started["execution_status"] == "waiting_user"
    request_id = str(started["request_id"])
    waiting = run("status", "--request-id", request_id)
    assert waiting["execution_status"] == "waiting_user"
    resumed = run("resume", "--request-id", request_id)
    assert resumed["execution_status"] == "completed"
    assert resumed["answer_outcome"] == "complete"
    completed = run("status", "--request-id", request_id)
    assert completed["execution_status"] == "completed"
    assert completed["resumable"] is False
