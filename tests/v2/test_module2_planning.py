from __future__ import annotations

import pytest

from agenticrag.v2.config import V2Config
from agenticrag.v2.planning import PlanningError, PlanningService


class FakeStructuredModel:
    def __init__(self, outputs: list[object]) -> None:
        self.outputs = list(outputs)
        self.calls = 0
        self.prompts: list[str] = []

    def with_structured_output(self, schema: object) -> "FakeStructuredModel":
        return self

    def invoke(self, prompt: str) -> object:
        self.calls += 1
        self.prompts.append(prompt)
        output = self.outputs[min(self.calls - 1, len(self.outputs) - 1)]
        if isinstance(output, Exception):
            raise output
        return output


def _simple_router(capability: str = "retrieval_synthesis") -> dict[str, object]:
    return {
        "complexity": "simple",
        "capability": capability,
        "reason": "one independent task",
    }


def _complex_router() -> dict[str, object]:
    return {"complexity": "complex", "capability": None, "reason": "multi-part"}


def _decomposition(*tasks: dict[str, str], complete: bool = True) -> dict[str, object]:
    return {"tasks": list(tasks), "decomposition_complete": complete}


def _task(query: str, intent: str, capability: str = "retrieval_synthesis") -> dict[str, str]:
    return {"query": query, "intent": intent, "capability": capability}


def test_simple_router_creates_one_task_without_calling_decomposer() -> None:
    router = FakeStructuredModel([_simple_router()])
    decomposer = FakeStructuredModel([RuntimeError("must not be called")])
    result = PlanningService(
        V2Config(), router_model=router, decomposer_model=decomposer
    ).plan("保险集团风险包括哪些？")

    assert result.complexity_decision.complexity == "simple"
    assert result.simple_task is not None
    assert result.simple_task.capability == "retrieval_synthesis"
    assert result.decomposition is None
    assert router.calls == 1
    assert decomposer.calls == 0
    assert "arithmetic" in router.prompts[0]


def test_simple_arithmetic_capability_is_preserved() -> None:
    result = PlanningService(
        V2Config(), router_model=FakeStructuredModel([_simple_router("arithmetic")])
    ).plan("根据 ROA 和 asset turnover 计算二者比值")

    assert result.complexity_decision.capability == "arithmetic"
    assert result.simple_task is not None
    assert result.simple_task.capability == "arithmetic"


def test_complex_router_calls_decomposer_once_and_preserves_task_capabilities() -> None:
    router = FakeStructuredModel([_complex_router()])
    decomposer = FakeStructuredModel(
        [
            _decomposition(
                _task("查询 A 的金额", "提取 A 金额"),
                _task("计算 A 与 B 的增长率", "计算增长率", "arithmetic"),
            )
        ]
    )
    result = PlanningService(
        V2Config(), router_model=router, decomposer_model=decomposer
    ).plan("查询 A 和 B，再计算增长率")

    assert result.complexity_decision.capability is None
    assert result.decomposition is not None
    assert [task.capability for task in result.decomposition.tasks] == [
        "retrieval_synthesis",
        "arithmetic",
    ]
    assert decomposer.calls == 1
    assert "depends_on" in decomposer.prompts[0]
    assert "synthesis" in decomposer.prompts[0]


def test_invalid_router_output_retries_once_then_fails_without_decomposer() -> None:
    router = FakeStructuredModel([{"complexity": "simple"}, {"complexity": "simple"}])
    decomposer = FakeStructuredModel([_decomposition(_task("a", "a"), _task("b", "b"))])
    with pytest.raises(PlanningError) as exc_info:
        PlanningService(
            V2Config(), router_model=router, decomposer_model=decomposer
        ).plan("问题")

    assert exc_info.value.execution_error.code == "structured_output_invalid"
    assert exc_info.value.attempts == 2
    assert router.calls == 2
    assert decomposer.calls == 0


def test_router_retry_can_recover_from_invalid_structured_output() -> None:
    router = FakeStructuredModel([{"complexity": "invalid"}, _simple_router()])
    result = PlanningService(V2Config(), router_model=router).plan("问题")

    assert result.complexity_decision.complexity == "simple"
    assert result.router_attempts == 2


def test_duplicate_decomposition_output_is_rejected_and_retried() -> None:
    duplicate = _decomposition(_task("same", "a"), _task("same", "b"))
    decomposer = FakeStructuredModel([duplicate, duplicate])
    with pytest.raises(PlanningError) as exc_info:
        PlanningService(
            V2Config(),
            router_model=FakeStructuredModel([_complex_router()]),
            decomposer_model=decomposer,
        ).plan("复杂问题")

    assert exc_info.value.role == "decomposer"
    assert exc_info.value.attempts == 2
    assert decomposer.calls == 2


def test_empty_decomposition_query_is_rejected() -> None:
    invalid = _decomposition(_task("", "empty query"), _task("valid", "valid"))
    decomposer = FakeStructuredModel([invalid, invalid])
    with pytest.raises(PlanningError) as exc_info:
        PlanningService(
            V2Config(),
            router_model=FakeStructuredModel([_complex_router()]),
            decomposer_model=decomposer,
        ).plan("复杂问题")

    assert exc_info.value.role == "decomposer"
    assert decomposer.calls == 2


def test_complete_decomposition_over_limit_is_rejected() -> None:
    tasks = tuple(_task(f"任务 {index}", f"提取任务 {index}") for index in range(5))
    decomposer = FakeStructuredModel([_decomposition(*tasks), _decomposition(*tasks)])
    with pytest.raises(PlanningError):
        PlanningService(
            V2Config(),
            router_model=FakeStructuredModel([_complex_router()]),
            decomposer_model=decomposer,
        ).plan("超过上限的问题")
    assert decomposer.calls == 2


def test_decomposition_limit_is_explicit_and_never_silently_truncated() -> None:
    tasks = tuple(_task(f"任务 {index}", f"提取任务 {index}") for index in range(5))
    result = PlanningService(
        V2Config(),
        router_model=FakeStructuredModel([_complex_router()]),
        decomposer_model=FakeStructuredModel(
            [_decomposition(*tasks, complete=False) | {"failure_reason": "decomposition_limit"}]
        ),
    ).plan("需要五个独立信息单元")

    assert result.decomposition is not None
    assert result.decomposition.decomposition_complete is False
    assert result.decomposition.failure_reason == "decomposition_limit"
    assert len(result.decomposition.tasks) == 5
