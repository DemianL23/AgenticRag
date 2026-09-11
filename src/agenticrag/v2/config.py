"""Validated, role-specific V2 configuration."""

from __future__ import annotations

import os
from typing import Literal

from pydantic import Field

from .schemas import V2Model

DecisionRole = Literal["router", "decomposer", "grader", "rewrite", "hitl"]
AnswerRole = Literal["simple_answer", "finding", "synthesis"]


class ModelRetryPolicy(V2Model):
    max_attempts: int = Field(default=2, ge=1, le=2)
    retry_timeout: bool = True
    retry_transient_provider_error: bool = True
    retry_rate_limit: bool = True
    retry_structured_output: bool = True


class DecisionModelConfig(V2Model):
    role: DecisionRole
    provider: str = "openai_compatible"
    model: str = "qwen-plus"
    model_revision: str | None = None
    endpoint_identifier: str = "default"
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    timeout_seconds: float = Field(default=120.0, gt=0.0)
    max_tokens: int = Field(default=1024, gt=0)
    thinking: bool = False
    retry_policy: ModelRetryPolicy = Field(default_factory=ModelRetryPolicy)


class AnswerModelConfig(V2Model):
    role: AnswerRole
    provider: str = "openai_compatible"
    model: str = "qwen-plus"
    model_revision: str | None = None
    endpoint_identifier: str = "default"
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    timeout_seconds: float = Field(default=120.0, gt=0.0)
    max_tokens: int = Field(default=1024, gt=0)
    thinking: bool = False
    retry_policy: ModelRetryPolicy = Field(default_factory=ModelRetryPolicy)


class DecisionModels(V2Model):
    router: DecisionModelConfig = Field(
        default_factory=lambda: DecisionModelConfig(role="router")
    )
    decomposer: DecisionModelConfig = Field(
        default_factory=lambda: DecisionModelConfig(role="decomposer")
    )
    grader: DecisionModelConfig = Field(
        default_factory=lambda: DecisionModelConfig(role="grader")
    )
    rewrite: DecisionModelConfig = Field(
        default_factory=lambda: DecisionModelConfig(role="rewrite")
    )
    hitl: DecisionModelConfig = Field(
        default_factory=lambda: DecisionModelConfig(role="hitl")
    )


class AnswerModels(V2Model):
    simple_answer: AnswerModelConfig = Field(
        default_factory=lambda: AnswerModelConfig(role="simple_answer")
    )
    finding: AnswerModelConfig = Field(
        default_factory=lambda: AnswerModelConfig(role="finding")
    )
    synthesis: AnswerModelConfig = Field(
        default_factory=lambda: AnswerModelConfig(role="synthesis")
    )


class V2BudgetConfig(V2Model):
    max_subqueries: int = Field(default=4, ge=1)
    max_concurrent_subqueries: int = Field(default=1, ge=1)
    max_retrieval_attempts_per_revision: int = Field(default=2, ge=1, le=2)
    max_query_revisions: int = Field(default=2, ge=1, le=2)
    max_hitl_rounds: int = Field(default=1, ge=0, le=1)
    max_evidence_per_finding: int = Field(default=3, ge=1, le=3)
    max_scope_options: int = Field(default=5, ge=2, le=5)
    checkpoint_ttl_seconds: int = Field(default=604800, ge=1)

    @classmethod
    def from_env(cls) -> "V2BudgetConfig":
        return cls(
            max_subqueries=_read_int("V2_MAX_SUBQUERIES", 4),
            max_concurrent_subqueries=_read_int("V2_MAX_CONCURRENT_SUBQUERIES", 1),
            max_retrieval_attempts_per_revision=_read_int(
                "V2_MAX_RETRIEVAL_ATTEMPTS_PER_REVISION", 2
            ),
            max_query_revisions=_read_int("V2_MAX_QUERY_REVISIONS", 2),
            max_hitl_rounds=_read_int("V2_MAX_HITL_ROUNDS", 1),
            max_evidence_per_finding=_read_int("V2_MAX_EVIDENCE_PER_FINDING", 3),
            max_scope_options=_read_int("V2_MAX_SCOPE_OPTIONS", 5),
            checkpoint_ttl_seconds=_read_int("V2_CHECKPOINT_TTL_SECONDS", 604800),
        )


class V2Config(V2Model):
    schema_version: str = "v2.1"
    default_response_language: Literal["zh", "en"] = "zh"
    budgets: V2BudgetConfig = Field(default_factory=V2BudgetConfig)
    decision_models: DecisionModels = Field(default_factory=DecisionModels)
    answer_models: AnswerModels = Field(default_factory=AnswerModels)

    @classmethod
    def from_env(cls) -> "V2Config":
        return cls(
            budgets=V2BudgetConfig.from_env(),
            default_response_language=os.getenv(
                "V2_DEFAULT_RESPONSE_LANGUAGE", "zh"
            ),
        )

    def resolved_record(self) -> dict[str, object]:
        return self.model_dump(mode="json")


def _read_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"{name} 必须是整数") from exc
