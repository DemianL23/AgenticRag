"""Factories for the independent RAGAS judge, embeddings, and metrics."""

from __future__ import annotations

from typing import Any

from .config import RagasEvaluatorConfig
from .evaluator import MetricDefinition, RagasEvaluator


def create_ragas_evaluator(config: RagasEvaluatorConfig) -> RagasEvaluator:
    """Build all V0 metrics with RAGAS 0.4's collections API."""
    config.validate()
    llm = create_evaluator_llm(config)
    embeddings = create_evaluator_embeddings(config)

    try:
        from ragas.metrics.collections import (
            AnswerCorrectness,
            AnswerRelevancy,
            ContextEntityRecall,
            ContextPrecisionWithReference,
            ContextRecall,
            Faithfulness,
        )
    except ImportError as exc:
        raise RuntimeError(
            "RAGAS 评测依赖未安装，请执行：uv sync --extra evaluation "
            "--extra generation --extra embeddings --extra milvus"
        ) from exc

    return RagasEvaluator(
        [
            MetricDefinition(
                "faithfulness",
                Faithfulness(llm=llm),
                ("user_input", "response", "retrieved_contexts"),
            ),
            MetricDefinition(
                "answer_relevancy",
                AnswerRelevancy(llm=llm, embeddings=embeddings),
                ("user_input", "response"),
            ),
            MetricDefinition(
                "answer_correctness",
                AnswerCorrectness(llm=llm, embeddings=embeddings),
                ("user_input", "response", "reference"),
            ),
            MetricDefinition(
                "context_recall",
                ContextRecall(llm=llm),
                ("user_input", "retrieved_contexts", "reference"),
            ),
            MetricDefinition(
                "context_precision",
                ContextPrecisionWithReference(llm=llm),
                ("user_input", "reference", "retrieved_contexts"),
            ),
            MetricDefinition(
                "context_entity_recall",
                ContextEntityRecall(llm=llm),
                ("reference", "retrieved_contexts"),
            ),
        ]
    )


def create_evaluator_llm(config: RagasEvaluatorConfig) -> Any:
    """Create the RAGAS judge through an OpenAI-compatible async client."""
    config.validate()
    try:
        from openai import AsyncOpenAI
        from ragas.llms import llm_factory
    except ImportError as exc:
        raise RuntimeError("创建 RAGAS Judge 需要 evaluation 可选依赖") from exc

    client = AsyncOpenAI(
        api_key=config.api_key,
        base_url=config.base_url,
        timeout=config.timeout,
        max_retries=config.max_retries,
    )
    return llm_factory(
        config.model,
        provider="openai",
        client=client,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
        extra_body={"enable_thinking": config.enable_thinking},
    )


def create_evaluator_embeddings(config: RagasEvaluatorConfig) -> Any:
    """Create modern local RAGAS embeddings without deprecated wrappers."""
    try:
        from ragas.embeddings import HuggingFaceEmbeddings
    except ImportError as exc:
        raise RuntimeError("创建 RAGAS Embedding 需要 evaluation 可选依赖") from exc

    model_kwargs: dict[str, Any] = {}
    if config.embedding_query_prompt_name:
        model_kwargs["default_prompt_name"] = config.embedding_query_prompt_name
    if config.embedding_trust_remote_code:
        model_kwargs["trust_remote_code"] = True
    return HuggingFaceEmbeddings(
        model=config.embedding_model,
        device=config.embedding_device,
        normalize_embeddings=config.embedding_normalize,
        batch_size=config.embedding_batch_size,
        **model_kwargs,
    )
