"""Module 2 planning service: Router followed by conditional Decomposer."""

from __future__ import annotations

import os
from typing import Any, Callable, Protocol, TypeVar

from pydantic import Field, ValidationError, model_validator

from .config import DecisionModelConfig, V2Config
from .policies import normalize_query, validate_decomposition
from .prompts import build_decomposer_prompt, build_router_prompt
from .schemas import (
    ComplexityDecision,
    DecompositionResult,
    ExecutionError,
    TaskDraft,
    V2Model,
)

SchemaT = TypeVar("SchemaT", bound=V2Model)


class StructuredRunnable(Protocol):
    def invoke(self, input: str) -> object: ...


class PlanningError(RuntimeError):
    """A technical failure in Router/Decomposer structured planning."""

    def __init__(self, *, role: str, attempts: int, cause: Exception) -> None:
        self.role = role
        self.attempts = attempts
        self.execution_error = ExecutionError(
            code="structured_output_invalid",
            message=f"{role} structured output failed after {attempts} attempt(s)",
            stage="module2_planning",
            retryable=False,
            details={"role": role, "attempts": str(attempts)},
        )
        self.cause = cause
        super().__init__(self.execution_error.message)


class StructuredOutputContractError(ValueError):
    """A parsed model response that violates the Module 2 contract."""


class PlanningResult(V2Model):
    """Validated output of the Module 2 application boundary."""

    question: str = Field(min_length=1)
    normalized_question: str = Field(min_length=1)
    complexity_decision: ComplexityDecision
    simple_task: TaskDraft | None = None
    decomposition: DecompositionResult | None = None
    router_attempts: int = Field(ge=1, le=2)
    decomposer_attempts: int = Field(default=0, ge=0, le=2)

    @model_validator(mode="after")
    def validate_plan_contract(self) -> "PlanningResult":
        if self.complexity_decision.complexity == "simple":
            if self.simple_task is None or self.decomposition is not None:
                raise ValueError("simple plan 必须只有一个 simple_task")
            if self.simple_task.capability != self.complexity_decision.capability:
                raise ValueError("simple task capability 必须与 Router 一致")
        elif self.simple_task is not None:
            raise ValueError("complex plan 不应有 simple_task")
        return self

    @classmethod
    def from_router(
        cls,
        *,
        question: str,
        decision: ComplexityDecision,
        router_attempts: int,
    ) -> "PlanningResult":
        normalized = normalize_query(question)
        if decision.complexity == "simple":
            assert decision.capability is not None
            return cls(
                question=question,
                normalized_question=normalized,
                complexity_decision=decision,
                simple_task=TaskDraft(
                    query=normalized,
                    intent="answer the original question",
                    capability=decision.capability,
                ),
                router_attempts=router_attempts,
            )
        return cls(
            question=question,
            normalized_question=normalized,
            complexity_decision=decision,
            router_attempts=router_attempts,
        )


class PlanningService:
    """Run the bounded, deterministic Router → optional Decomposer flow."""

    def __init__(
        self,
        config: V2Config | None = None,
        *,
        router_model: Any | None = None,
        decomposer_model: Any | None = None,
    ) -> None:
        self.config = config or V2Config.from_env()
        self._router_model = router_model
        self._decomposer_model = decomposer_model

    def plan(self, question: str) -> PlanningResult:
        normalized = normalize_query(question)
        router_model = self._router_model or create_decision_chat_model(
            self.config.decision_models.router
        )
        decision, router_attempts = _invoke_structured(
            router_model,
            ComplexityDecision,
            build_router_prompt(normalized),
            role="router",
            retry_policy=self.config.decision_models.router.retry_policy,
        )
        result = PlanningResult.from_router(
            question=question,
            decision=decision,
            router_attempts=router_attempts,
        )
        if decision.complexity == "simple":
            return result

        decomposer_model = self._decomposer_model or create_decision_chat_model(
            self.config.decision_models.decomposer
        )
        decomposition, decomposer_attempts = _invoke_structured(
            decomposer_model,
            DecompositionResult,
            build_decomposer_prompt(normalized, self.config.budgets.max_subqueries),
            role="decomposer",
            retry_policy=self.config.decision_models.decomposer.retry_policy,
            post_validate=lambda value: validate_decomposition(
                decision, value, self.config.budgets
            ),
        )
        return result.model_copy(
            update={
                "decomposition": decomposition,
                "decomposer_attempts": decomposer_attempts,
            }
        )


def plan(
    question: str,
    config: V2Config | None = None,
    *,
    router_model: Any | None = None,
    decomposer_model: Any | None = None,
) -> PlanningResult:
    """Convenience application boundary for one planning request."""
    return PlanningService(
        config,
        router_model=router_model,
        decomposer_model=decomposer_model,
    ).plan(question)


def _invoke_structured(
    model: Any,
    schema: type[SchemaT],
    prompt: str,
    *,
    role: str,
    retry_policy: Any,
    post_validate: Callable[[SchemaT], None] | None = None,
) -> tuple[SchemaT, int]:
    try:
        structured: StructuredRunnable = model.with_structured_output(schema)
    except Exception as exc:
        raise PlanningError(role=role, attempts=0, cause=exc) from exc

    max_attempts = retry_policy.max_attempts
    for attempt in range(1, max_attempts + 1):
        try:
            parsed = schema.model_validate(structured.invoke(prompt))
            if post_validate is not None:
                try:
                    post_validate(parsed)
                except Exception as exc:
                    raise StructuredOutputContractError(str(exc)) from exc
            return parsed, attempt
        except Exception as exc:
            if attempt < max_attempts and _retryable(exc, retry_policy):
                continue
            raise PlanningError(role=role, attempts=attempt, cause=exc) from exc
    raise AssertionError("unreachable")


def _retryable(exc: Exception, retry_policy: Any) -> bool:
    if isinstance(exc, (ValidationError, StructuredOutputContractError)):
        return retry_policy.retry_structured_output
    name = type(exc).__name__.lower()
    message = str(exc).lower()
    if isinstance(exc, TimeoutError) or "timeout" in name or "timeout" in message:
        return retry_policy.retry_timeout
    if isinstance(exc, ConnectionError) or "connection" in name or "connection" in message:
        return retry_policy.retry_transient_provider_error
    if "rate" in message or "429" in message:
        return retry_policy.retry_rate_limit
    if any(token in message for token in ("structured", "schema", "parse", "json")):
        return retry_policy.retry_structured_output
    return False


def create_decision_chat_model(config: DecisionModelConfig) -> Any:
    """Create an OpenAI-compatible model without adding a provider framework."""
    from agenticrag.generation.config import GenerationConfig

    try:
        from langchain_openai import ChatOpenAI
    except ImportError as exc:  # pragma: no cover - exercised only without extra
        raise RuntimeError(
            "Module 2 Decision Model 需要可选依赖，请执行：uv sync --extra generation"
        ) from exc

    generation_defaults = GenerationConfig.from_env()
    role_base_url = f"V2_{config.role.upper()}_BASE_URL"
    base_url = os.getenv(role_base_url) or os.getenv("V2_DECISION_BASE_URL") or generation_defaults.base_url
    api_key = os.getenv("V2_DECISION_API_KEY") or generation_defaults.api_key
    return ChatOpenAI(
        model=config.model,
        api_key=api_key,
        base_url=base_url,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
        timeout=config.timeout_seconds,
        max_retries=0,
        extra_body={"enable_thinking": config.thinking},
    )
