"""Qwen chat-model factory using Alibaba Model Studio's compatible API."""

from __future__ import annotations

from typing import Any

from agenticrag.generation.config import GenerationConfig


def create_qwen_chat_model(config: GenerationConfig | None = None) -> Any:
    """Create a LangChain chat model without loading it at module import time."""
    config = config or GenerationConfig.from_env()
    config.validate()
    try:
        from langchain_openai import ChatOpenAI
    except ImportError as exc:
        raise RuntimeError(
            "Qwen 生成需要可选依赖，请执行：uv sync --extra generation"
        ) from exc

    return ChatOpenAI(
        model=config.model,
        api_key=config.api_key,
        base_url=config.base_url,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
        timeout=config.timeout,
        max_retries=config.max_retries,
        extra_body={"enable_thinking": config.enable_thinking},
    )
