"""Composition layer for retrieval and generation pipelines.

The exports are lazy because retrieval integrations import this package while
the generation package is still being initialized.
"""

from typing import Any

__all__ = ["RagAnswerService", "V12ControlAnswerService", "V12ControlTrace"]


def __getattr__(name: str) -> Any:
    if name == "RagAnswerService":
        from agenticrag.rag.service import RagAnswerService

        return RagAnswerService
    if name in {"V12ControlAnswerService", "V12ControlTrace"}:
        from agenticrag.rag.v1_2_control import (
            V12ControlAnswerService,
            V12ControlTrace,
        )

        return {
            "V12ControlAnswerService": V12ControlAnswerService,
            "V12ControlTrace": V12ControlTrace,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
