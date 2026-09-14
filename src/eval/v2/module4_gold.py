"""Module 4 EvidenceGrade/route Gold authoring and frozen replay evaluation.

Gold authoring intentionally projects only task/revision/Final Top-5 evidence
from a clean workflow run.  Production grades and routes never enter the
authoring prompt or the frozen fixture.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

from pydantic import Field, model_validator

from agenticrag.v2.config import ModelRetryPolicy, V2Config
from agenticrag.v2.grading import EvidenceGrader, EvidenceGradingError
from agenticrag.v2.module4 import make_grade_record, route_for
from agenticrag.v2.planning import _invoke_structured
from agenticrag.v2.policies import routing_decision
from agenticrag.v2.schemas import (
    Evidence,
    EvidenceGrade,
    QueryRevision,
    RetrievalAttempt,
    RetrievalTask,
    V2Model,
)
from agenticrag.v2.types import RecoveryStrategy, Route


DEFAULT_SOURCE_REPORT = Path(
    "artifacts/eval/v2/module4_v2_1/278dedde-b6f1-4cd6-81d2-efdc7b7df655/report.json"
)
DEFAULT_GOLD_DATASET = Path("eval/datasets/v2_module4_grade_route_gold.jsonl")
DEFAULT_CANDIDATE_ROOT = Path("artifacts/eval/v2/module4_gold_candidates")
DEFAULT_OUTPUT_ROOT = Path("artifacts/eval/v2/module4_grade_route")


class Module4GoldJudgeConfig(V2Model):
    """Eval-only Gold authoring Judge config, separate from V2 runtime roles."""

    provider: str = "openai_compatible"
    model: str = "qwen-plus"
    model_revision: str | None = None
    endpoint_identifier: str = "eval-module4-gold-judge"
    temperature: float = Field(default=0.0, ge=0.0, le=0.0)
    timeout_seconds: float = Field(default=120.0, gt=0.0)
    max_tokens: int = Field(default=1024, gt=0)
    thinking: bool = False
    retry_policy: ModelRetryPolicy = Field(default_factory=ModelRetryPolicy)

    @classmethod
    def from_env(cls) -> "Module4GoldJudgeConfig":
        try:
            from dotenv import load_dotenv

            load_dotenv(override=False)
        except ImportError:  # pragma: no cover
            pass
        return cls(
            provider=os.getenv("V2_EVAL_MODULE4_GOLD_JUDGE_PROVIDER", "openai_compatible"),
            model=os.getenv("V2_EVAL_MODULE4_GOLD_JUDGE_MODEL", "qwen-plus"),
            model_revision=_optional_env("V2_EVAL_MODULE4_GOLD_JUDGE_MODEL_REVISION"),
            endpoint_identifier=os.getenv(
                "V2_EVAL_MODULE4_GOLD_JUDGE_ENDPOINT_IDENTIFIER",
                "eval-module4-gold-judge",
            ),
            timeout_seconds=_float_env(
                "V2_EVAL_MODULE4_GOLD_JUDGE_TIMEOUT_SECONDS", 120.0
            ),
            max_tokens=_int_env("V2_EVAL_MODULE4_GOLD_JUDGE_MAX_TOKENS", 1024),
            thinking=_bool_env("V2_EVAL_MODULE4_GOLD_JUDGE_THINKING", False),
            retry_policy=ModelRetryPolicy(
                max_attempts=_int_env("V2_EVAL_MODULE4_GOLD_JUDGE_MAX_ATTEMPTS", 2),
                retry_timeout=_bool_env(
                    "V2_EVAL_MODULE4_GOLD_JUDGE_RETRY_TIMEOUT", True
                ),
                retry_transient_provider_error=_bool_env(
                    "V2_EVAL_MODULE4_GOLD_JUDGE_RETRY_TRANSIENT", True
                ),
                retry_rate_limit=_bool_env(
                    "V2_EVAL_MODULE4_GOLD_JUDGE_RETRY_RATE_LIMIT", True
                ),
                retry_structured_output=_bool_env(
                    "V2_EVAL_MODULE4_GOLD_JUDGE_RETRY_STRUCTURED", True
                ),
            ),
        )

    def resolved_record(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


class GoldTaskInput(V2Model):
    query: str = Field(min_length=1)
    intent: str = Field(min_length=1)


class GoldQueryRevisionInput(V2Model):
    query: str = Field(min_length=1)
    source: str = Field(min_length=1)


class GoldEvidenceInput(V2Model):
    evidence_id: str = Field(min_length=1)
    content: str
    doc_id: str = Field(min_length=1)
    source: str = Field(min_length=1)
    page: int


class GoldExpected(V2Model):
    relevance: Literal["none", "weak", "strong"]
    answerability: Literal["none", "partial", "sufficient"]
    ambiguity: Literal["none", "missing_slot", "multiple_candidates"]
    recoverability: Literal["none", "likely"]
    failure_reason: Literal[
        "none",
        "irrelevant_evidence",
        "insufficient_coverage",
        "query_mismatch",
        "overly_specific",
        "terminology_gap",
    ]
    missing_information: list[str] = Field(default_factory=list)
    missing_slots: list[str] = Field(default_factory=list)
    supporting_evidence_ids: list[str] = Field(default_factory=list)
    route: Route
    recovery_strategy: RecoveryStrategy | None = None

    def as_grade(self) -> EvidenceGrade:
        return EvidenceGrade(
            relevance=self.relevance,
            answerability=self.answerability,
            ambiguity=self.ambiguity,
            recoverability=self.recoverability,
            failure_reason=self.failure_reason,
            reason="frozen independent Gold fixture",
            missing_information=self.missing_information,
            missing_slots=self.missing_slots,
            supporting_evidence_ids=self.supporting_evidence_ids,
        )


class Module4GoldFixture(V2Model):
    gold_id: str = Field(min_length=1)
    qa_id: str = Field(min_length=1)
    source_run_id: str = Field(min_length=1)
    source_git_commit: str = Field(min_length=1)
    source_task_id: str = Field(min_length=1)
    source_task_ordinal: int = Field(gt=0)
    task: GoldTaskInput
    query_revision: GoldQueryRevisionInput
    evidence: list[GoldEvidenceInput] = Field(min_length=1, max_length=5)
    expected: GoldExpected
    needs_review: bool = False
    review_notes: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_fixture_contract(self) -> "Module4GoldFixture":
        evidence_ids = [item.evidence_id for item in self.evidence]
        if len(evidence_ids) != len(set(evidence_ids)):
            raise ValueError("Gold fixture Evidence ID 必须唯一")
        grade = self.expected.as_grade()
        grade.validate_against_evidence_ids(set(evidence_ids))
        return self


def extract_gold_inputs(source_report: dict[str, Any]) -> list[dict[str, Any]]:
    """Explicitly project safe authoring inputs; never copy source dictionaries."""
    run_id = source_report.get("run_id")
    git_commit = source_report.get("git_commit")
    predictions = source_report.get("predictions")
    if not isinstance(run_id, str) or not isinstance(git_commit, str):
        raise ValueError("source report 缺少 run_id/git_commit")
    if not isinstance(predictions, list):
        raise ValueError("source report predictions 必须是 list")

    candidates: list[tuple[str, dict[str, Any], dict[str, Any], dict[str, Any]]] = []
    for sample in predictions:
        if not isinstance(sample, dict):
            continue
        qa_id = sample.get("qa_id")
        observed = sample.get("observed")
        if not isinstance(qa_id, str) or not isinstance(observed, dict):
            continue
        retrieval_results = observed.get("retrieval_results")
        tasks = observed.get("tasks")
        if not isinstance(retrieval_results, dict) or not isinstance(tasks, list):
            continue
        for task in tasks:
            if not isinstance(task, dict) or task.get("capability") != "retrieval_synthesis":
                continue
            task_id = task.get("id")
            revisions = task.get("query_revisions")
            # Presence of a grade record is used only to select the normal,
            # completed path; its contents are deliberately never projected.
            if (
                not isinstance(task_id, str)
                or not isinstance(revisions, list)
                or not revisions
                or not task.get("grade_records")
                or task.get("error") is not None
            ):
                continue
            retrieval = retrieval_results.get(task_id)
            if not isinstance(retrieval, dict):
                continue
            evidence = retrieval.get("evidence")
            revision = revisions[-1]
            if not isinstance(evidence, list) or not isinstance(revision, dict):
                continue
            candidates.append((qa_id, task, revision, {"evidence": evidence}))

    candidates.sort(key=lambda item: (item[0], int(item[1].get("ordinal", 0))))
    projected: list[dict[str, Any]] = []
    for fixture_ordinal, (qa_id, task, revision, retrieval) in enumerate(candidates, 1):
        raw_evidence = retrieval["evidence"]
        projected_evidence: list[dict[str, Any]] = []
        for item in raw_evidence:
            if not isinstance(item, dict):
                raise ValueError(f"{qa_id}/{task['id']} Evidence record 非法")
            projected_evidence.append(
                {
                    "evidence_id": item["evidence_id"],
                    "content": item["content"],
                    "doc_id": item["doc_id"],
                    "source": item["source"],
                    "page": item["page"],
                }
            )
        projected.append(
            {
                "gold_id": f"M4G_{qa_id}_{fixture_ordinal:03d}",
                "qa_id": qa_id,
                "source_run_id": run_id,
                "source_git_commit": git_commit,
                "source_task_id": task["id"],
                "source_task_ordinal": task["ordinal"],
                "task": {"query": task["query"], "intent": task["intent"]},
                "query_revision": {
                    "query": revision["query"],
                    "source": revision["source"],
                },
                "evidence": projected_evidence,
            }
        )
    if len(projected) != 14:
        raise ValueError(f"source run 应投影 14 个正常 retrieval task，实际 {len(projected)} 个")
    return projected


GOLD_JUDGE_PROMPT = """You are an independent offline Gold Authoring Judge for Agentic RAG V2 Module 4.
You are authoring an EvidenceGrade for one frozen task and its current V1.2 Final Top-5 Evidence.
You must not use or infer any production Grader output, production route, gold answer, expected label,
candidate pool, RRF diagnostics, or evidence from another task. Return only a complete EvidenceGrade;
do not return route or recovery_strategy.

Evaluate only these fields:
- relevance: none / weak / strong
- answerability: none / partial / sufficient
- ambiguity: none / missing_slot / multiple_candidates
- recoverability: none / likely
- failure_reason: none / irrelevant_evidence / insufficient_coverage / query_mismatch /
  overly_specific / terminology_gap
- missing_information, missing_slots, supporting_evidence_ids, and a concise reason

Semantic rules:
- missing_slot means the task query itself omits a required parameter value that the user must supply.
- multiple_candidates means the query contains an expression that resolves to multiple plausible candidates.
- A complete query with facts absent from Evidence is not missing_slot: use ambiguity=none,
  missing_slots=[], and describe the absent facts in missing_information.
- Answerability boundary: sufficient means the Evidence is enough to answer the task. partial means
  the Evidence establishes at least one explicitly requested target fact, but other necessary target
  facts are still missing. none means the Evidence establishes none of the requested target facts.
- Adjacent, proxy, or merely correlated metrics do not count as a requested target fact. For example,
  if a task asks for gross margin but the Evidence only contains operating margin, net margin, and
  EBITDA margin, answerability is none, not partial. Partial requires at least one actual requested
  gross-margin value (for example, one year of a multi-year request).
- supporting_evidence_ids must be copied exactly from allowed_supporting_evidence_ids.
- answerability=none requires supporting_evidence_ids=[]; partial/sufficient requires at least one valid ID.
- If another retrieval attempt could reasonably find missing facts, use recoverability=likely and a
  specific non-none failure_reason.

The final route will be calculated separately by the frozen deterministic policy. Do not output a route.
"""


class Module4GoldJudge:
    def __init__(
        self,
        config: Module4GoldJudgeConfig | None = None,
        *,
        model: Any | None = None,
    ) -> None:
        self.config = config or Module4GoldJudgeConfig.from_env()
        self._model = model

    def grade(
        self, fixture_input: dict[str, Any]
    ) -> tuple[EvidenceGrade, int]:
        evidence = fixture_input["evidence"]
        evidence_ids = [item["evidence_id"] for item in evidence]
        prompt = build_gold_judge_prompt(fixture_input)
        model = self._model or create_module4_gold_judge_model(self.config)
        return _invoke_structured(
            model,
            EvidenceGrade,
            prompt,
            role="module4_gold_judge",
            retry_policy=self.config.retry_policy,
            post_validate=lambda value: value.validate_against_evidence_ids(
                set(evidence_ids)
            ),
        )


def build_gold_judge_prompt(fixture_input: dict[str, Any]) -> str:
    """Build Judge input from the explicit projection only."""
    context = {
        "task": fixture_input["task"],
        "query_revision": fixture_input["query_revision"],
        "final_top5_evidence": fixture_input["evidence"],
        "allowed_supporting_evidence_ids": [
            item["evidence_id"] for item in fixture_input["evidence"]
        ],
    }
    return GOLD_JUDGE_PROMPT + "\nInput:\n" + json.dumps(
        context, ensure_ascii=False, indent=2
    )


def create_module4_gold_judge_model(config: Module4GoldJudgeConfig) -> Any:
    from agenticrag.generation.config import GenerationConfig

    try:
        from langchain_openai import ChatOpenAI
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "Module 4 Gold Judge 需要可选依赖，请执行：uv sync --extra generation"
        ) from exc
    generation_defaults = GenerationConfig.from_env()
    base_url = os.getenv(
        "V2_EVAL_MODULE4_GOLD_JUDGE_BASE_URL", generation_defaults.base_url
    )
    api_key = os.getenv("V2_EVAL_MODULE4_GOLD_JUDGE_API_KEY") or generation_defaults.api_key
    return ChatOpenAI(
        model=config.model,
        api_key=api_key,
        base_url=base_url,
        temperature=0,
        max_tokens=config.max_tokens,
        timeout=config.timeout_seconds,
        max_retries=0,
        extra_body={"enable_thinking": config.thinking},
    )


def derive_gold_route(
    fixture: Module4GoldFixture, grade: EvidenceGrade, config: V2Config | None = None
) -> tuple[Route, RecoveryStrategy | None]:
    """Derive route/strategy through the frozen production policy only."""
    config = config or V2Config()
    evidence_ids = [item.evidence_id for item in fixture.evidence]
    decision = routing_decision(
        decision_id=f"ROUTE_{fixture.gold_id}_001",
        grade_record_id=f"GR_{fixture.gold_id}_001",
        capability="retrieval_synthesis",
        grade=grade,
        input_evidence_ids=evidence_ids,
        retrieval_budget_available=True,
    )
    return decision.route, decision.recovery_strategy


def author_gold_dataset(
    source_report_path: Path = DEFAULT_SOURCE_REPORT,
    *,
    output_path: Path | None = None,
    judge: Module4GoldJudge | None = None,
    force: bool = False,
) -> dict[str, Any]:
    if output_path is None:
        output_path = DEFAULT_CANDIDATE_ROOT / str(uuid4()) / "candidate.jsonl"
    elif not force and (
        output_path.resolve() == DEFAULT_GOLD_DATASET.resolve()
        or output_path.exists()
    ):
        raise FileExistsError(
            f"Refusing to write protected Gold output: {output_path}. "
            "Pass force=True (CLI: --force) explicitly to overwrite."
        )
    source_report = json.loads(source_report_path.read_text())
    inputs = extract_gold_inputs(source_report)
    judge = judge or Module4GoldJudge()
    fixtures: list[Module4GoldFixture] = []
    for fixture_input in inputs:
        grade, _attempts = judge.grade(fixture_input)
        expected = GoldExpected(
            relevance=grade.relevance,
            answerability=grade.answerability,
            ambiguity=grade.ambiguity,
            recoverability=grade.recoverability,
            failure_reason=grade.failure_reason,
            missing_information=grade.missing_information,
            missing_slots=grade.missing_slots,
            supporting_evidence_ids=grade.supporting_evidence_ids,
            route="no_knowledge",
        )
        fixture = Module4GoldFixture(
            **fixture_input,
            expected=expected,
        )
        route, strategy = derive_gold_route(fixture, grade)
        fixture = fixture.model_copy(
            update={
                "expected": expected.model_copy(
                    update={"route": route, "recovery_strategy": strategy}
                )
            }
        )
        validate_gold_fixture(fixture)
        fixtures.append(fixture)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "".join(
            json.dumps(fixture.model_dump(mode="json"), ensure_ascii=False) + "\n"
            for fixture in fixtures
        )
    )
    return {
        "dataset_path": str(output_path),
        "dataset_sha256": sha256(output_path),
        "fixture_count": len(fixtures),
        "qa_ids": sorted({fixture.qa_id for fixture in fixtures}),
        "source_run_id": source_report["run_id"],
        "source_git_commit": source_report["git_commit"],
        "judge_config": judge.config.resolved_record(),
        "needs_review_count": sum(fixture.needs_review for fixture in fixtures),
        "route_distribution": dict(Counter(f.expected.route for f in fixtures)),
    }


def load_gold_dataset(path: Path = DEFAULT_GOLD_DATASET) -> list[Module4GoldFixture]:
    if not path.exists():
        raise FileNotFoundError(f"Module 4 Gold dataset 不存在：{path}")
    fixtures: list[Module4GoldFixture] = []
    seen: set[str] = set()
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            fixture = Module4GoldFixture.model_validate(json.loads(line))
        except Exception as exc:
            raise ValueError(f"Gold dataset 第 {line_number} 行非法") from exc
        if fixture.gold_id in seen:
            raise ValueError(f"Gold dataset gold_id 重复：{fixture.gold_id}")
        seen.add(fixture.gold_id)
        validate_gold_fixture(fixture)
        fixtures.append(fixture)
    if not fixtures:
        raise ValueError("Gold dataset 不能为空")
    return fixtures


def validate_gold_fixture(fixture: Module4GoldFixture) -> None:
    grade = fixture.expected.as_grade()
    grade.validate_against_evidence_ids({item.evidence_id for item in fixture.evidence})
    route, strategy = derive_gold_route(fixture, grade)
    if (route, strategy) != (
        fixture.expected.route,
        fixture.expected.recovery_strategy,
    ):
        raise ValueError(f"{fixture.gold_id} route/strategy 与 deterministic policy 不一致")
    if fixture.expected.route == "answer" and not (
        grade.relevance == "strong" and grade.answerability == "sufficient"
    ):
        raise ValueError(f"{fixture.gold_id} answer route contract 非法")
    if fixture.expected.route == "recover" and grade.recoverability != "likely":
        raise ValueError(f"{fixture.gold_id} recover route contract 非法")
    if fixture.expected.route == "clarify" and grade.ambiguity != "missing_slot":
        raise ValueError(f"{fixture.gold_id} clarify route contract 非法")
    if fixture.expected.route == "scope_select" and grade.ambiguity != "multiple_candidates":
        raise ValueError(f"{fixture.gold_id} scope_select route contract 非法")


def replay_module4_gold(
    gold_path: Path = DEFAULT_GOLD_DATASET,
    *,
    config: V2Config | None = None,
    grader: EvidenceGrader | None = None,
    run_id: str | None = None,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
) -> dict[str, Any]:
    config = config or V2Config.from_env()
    fixtures = load_gold_dataset(gold_path)
    grader = grader or EvidenceGrader(config)
    started = time.perf_counter()
    per_fixture: list[dict[str, Any]] = []
    counts = Counter()
    repair_used = 0
    technical_failures = 0
    invariant_violations = 0
    successful = 0
    field_correct = Counter()
    grade_exact_correct = route_correct = 0
    recover_total = recover_correct = 0
    missing_slot_contract_correct = 0
    missing_slot_contract_total = 0
    missing_info_contract_correct = 0
    missing_info_contract_total = 0

    for fixture in fixtures:
        item: dict[str, Any] = {
            "gold_id": fixture.gold_id,
            "qa_id": fixture.qa_id,
            "task": fixture.task.model_dump(mode="json"),
            "gold": fixture.expected.model_dump(mode="json"),
            "predicted": None,
            "attempts": None,
            "errors": [],
            "invariant_violations": [],
        }
        task, revision, evidence = _materialize_fixture(fixture)
        try:
            grade, attempts = grader.grade(
                task=task, revision=revision, evidence=evidence
            )
            record = make_grade_record(
                task=task, revision=revision, evidence=evidence, grade=grade
            )
            routed_task = task.model_copy(update={"grade_records": [record]})
            decision = route_for(
                task=routed_task,
                revision=revision,
                grade_record=record,
                config=config,
            )
            item["predicted"] = {
                "grade": grade.model_dump(mode="json"),
                "route": decision.route,
                "recovery_strategy": decision.recovery_strategy,
            }
            item["attempts"] = attempts
            successful += 1
            repair_used += int(attempts > 1)
            violations = _replay_invariants(fixture, grade, decision)
            item["invariant_violations"] = violations
            invariant_violations += len(violations)
            expected = fixture.expected
            for field in (
                "relevance",
                "answerability",
                "ambiguity",
                "recoverability",
                "failure_reason",
            ):
                field_correct[field] += int(getattr(grade, field) == getattr(expected, field))
            grade_exact = all(
                getattr(grade, field) == getattr(expected, field)
                for field in (
                    "relevance",
                    "answerability",
                    "ambiguity",
                    "recoverability",
                    "failure_reason",
                )
            )
            grade_exact_correct += int(grade_exact)
            route_correct += int(decision.route == expected.route)
            if expected.route == "recover":
                recover_total += 1
                recover_correct += int(
                    decision.recovery_strategy == expected.recovery_strategy
                )
            missing_slot_contract_total += 1
            missing_slot_contract_correct += int(
                (
                    expected.ambiguity == "missing_slot"
                    and grade.ambiguity == "missing_slot"
                    and bool(grade.missing_slots)
                )
                or (
                    expected.ambiguity != "missing_slot"
                    and grade.ambiguity != "missing_slot"
                )
            )
            if expected.answerability in {"none", "partial"} and expected.missing_information:
                missing_info_contract_total += 1
                missing_info_contract_correct += int(
                    bool(grade.missing_information) and not grade.missing_slots
                )
        except EvidenceGradingError as exc:
            technical_failures += 1
            item["errors"].append(exc.execution_error.model_dump(mode="json"))
        per_fixture.append(item)

    report = {
        "run_id": run_id or str(uuid4()),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(),
        "git_dirty": git_dirty(),
        "dataset": {
            "path": str(gold_path),
            "sha256": sha256(gold_path),
            "fixture_count": len(fixtures),
            "qa_id_count": len({fixture.qa_id for fixture in fixtures}),
            "source_run_id": fixtures[0].source_run_id,
            "source_git_commit": fixtures[0].source_git_commit,
        },
        "model_configs": {
            "grader": config.decision_models.grader.model_dump(mode="json"),
        },
        "metrics": {
            "task_count": len(fixtures),
            "successful_task_count": successful,
            "relevance_accuracy": _accuracy(field_correct["relevance"], successful),
            "answerability_accuracy": _accuracy(field_correct["answerability"], successful),
            "ambiguity_accuracy": _accuracy(field_correct["ambiguity"], successful),
            "recoverability_accuracy": _accuracy(field_correct["recoverability"], successful),
            "failure_reason_accuracy": _accuracy(field_correct["failure_reason"], successful),
            "grade_exact_match_accuracy": _accuracy(grade_exact_correct, successful),
            "route_accuracy": _accuracy(route_correct, successful),
            "recovery_strategy_accuracy": _accuracy(recover_correct, recover_total),
            "recovery_strategy_correct": recover_correct,
            "recovery_strategy_total": recover_total,
            "missing_slot_contract_accuracy": _accuracy(
                missing_slot_contract_correct, missing_slot_contract_total
            ),
            "missing_information_contract_accuracy": _accuracy(
                missing_info_contract_correct, missing_info_contract_total
            ),
            "technical_failure_count": technical_failures,
            "invariant_violation_count": invariant_violations,
            "repair_used_count": repair_used,
            "gold_route_distribution": dict(Counter(f.expected.route for f in fixtures)),
            "predicted_route_distribution": dict(
                Counter(
                    item["predicted"]["route"]
                    for item in per_fixture
                    if item["predicted"] is not None
                )
            ),
        },
        "evaluation_incomplete": technical_failures > 0
        or any(fixture.needs_review for fixture in fixtures),
        "per_fixture": per_fixture,
        "elapsed_seconds": time.perf_counter() - started,
    }
    report_dir = output_root / report["run_id"]
    if report_dir.exists():
        raise FileExistsError(f"拒绝覆盖 Gold replay report：{report_dir}")
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    )
    (report_dir / "predictions.jsonl").write_text(
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in per_fixture)
    )
    return report


def _materialize_fixture(
    fixture: Module4GoldFixture,
) -> tuple[RetrievalTask, QueryRevision, list[Evidence]]:
    evidence = [
        Evidence(
            evidence_id=item.evidence_id,
            chunk_id=item.evidence_id,
            content=item.content,
            doc_id=item.doc_id,
            source=item.source,
            page=item.page,
        )
        for item in fixture.evidence
    ]
    attempt_id = f"ATT_{fixture.source_task_id.replace('_', '')}_QR001_001"
    revision_id = f"QR_{fixture.source_task_id.replace('_', '')}_001"
    revision = QueryRevision(
        id=revision_id,
        ordinal=1,
        source=fixture.query_revision.source,
        query=fixture.query_revision.query,
        retrieval_attempts=[
            RetrievalAttempt(
                id=attempt_id,
                ordinal=1,
                strategy="original",
                retrieval_query=fixture.query_revision.query,
                evidence_ids=[item.evidence_id for item in evidence],
            )
        ],
    )
    task = RetrievalTask(
        id=fixture.source_task_id,
        ordinal=fixture.source_task_ordinal,
        query=fixture.task.query,
        intent=fixture.task.intent,
        capability="retrieval_synthesis",
        required=True,
        query_revisions=[revision],
        execution_status="running",
    )
    return task, revision, evidence


def _replay_invariants(
    fixture: Module4GoldFixture, grade: EvidenceGrade, decision: Any
) -> list[str]:
    violations: list[str] = []
    evidence_ids = {item.evidence_id for item in fixture.evidence}
    if not set(grade.supporting_evidence_ids) <= evidence_ids:
        violations.append("supporting_evidence_outside_input")
    expected_route, expected_strategy = derive_gold_route(fixture, grade)
    if (decision.route, decision.recovery_strategy) != (expected_route, expected_strategy):
        violations.append("route_not_deterministic_from_predicted_grade")
    if decision.route == "unsupported":
        violations.append("retrieval_synthesis_task_routed_unsupported")
    return violations


def _accuracy(correct: int, total: int) -> float | None:
    return correct / total if total else None


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_commit() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    )
    return result.stdout.strip() if result.returncode == 0 else None


def git_dirty() -> bool | None:
    result = subprocess.run(
        ["git", "status", "--porcelain"], capture_output=True, text=True, check=False
    )
    return result.stdout.strip() != "" if result.returncode == 0 else None


def _optional_env(name: str) -> str | None:
    value = os.getenv(name)
    return value.strip() if value and value.strip() else None


def _int_env(name: str, default: int) -> int:
    value = os.getenv(name)
    return int(value) if value and value.strip() else default


def _float_env(name: str, default: float) -> float:
    value = os.getenv(name)
    return float(value) if value and value.strip() else default


def _bool_env(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if not value or not value.strip():
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"{name} 必须是 true/false")


def main_author() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Author Module 4 Grade/Route Gold")
    parser.add_argument("--source-report", type=Path, default=DEFAULT_SOURCE_REPORT)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Candidate output path; existing files require --force. Default: unique artifact path.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Explicitly allow overwriting an existing output file.",
    )
    args = parser.parse_args()
    result = author_gold_dataset(
        args.source_report, output_path=args.output, force=args.force
    )
    print(json.dumps(result, ensure_ascii=False))


def main_replay() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Replay Module 4 Grade/Route Gold")
    parser.add_argument("--gold", type=Path, default=DEFAULT_GOLD_DATASET)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args()
    report = replay_module4_gold(args.gold, output_root=args.output_root)
    print(json.dumps({"run_id": report["run_id"], "metrics": report["metrics"]}, ensure_ascii=False))
