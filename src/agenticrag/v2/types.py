"""Shared literal types for the V2 domain model."""

from __future__ import annotations

from typing import Literal, TypeAlias

TargetStage: TypeAlias = Literal["v2_1", "v2_2", "v2_3"]
ResponseLanguage: TypeAlias = Literal["zh", "en"]
Complexity: TypeAlias = Literal["simple", "complex"]

TaskCapability: TypeAlias = Literal[
    "retrieval_synthesis",
    "arithmetic",
    "statistical_computation",
    "sql",
    "other_unsupported",
]

TaskExecutionStatus: TypeAlias = Literal[
    "pending",
    "running",
    "waiting_user",
    "completed",
    "failed",
]

TaskAnswerOutcome: TypeAlias = Literal[
    "complete",
    "no_knowledge",
    "unsupported",
    "unresolved",
]

GlobalExecutionStatus: TypeAlias = Literal[
    "running",
    "waiting_user",
    "completed",
    "failed",
]

GlobalAnswerOutcome: TypeAlias = Literal[
    "complete",
    "partial",
    "no_knowledge",
    "unsupported",
    "unresolved",
]

Route: TypeAlias = Literal[
    "answer",
    "recover",
    "clarify",
    "scope_select",
    "no_knowledge",
    "unsupported",
]

RecoveryStrategy: TypeAlias = Literal["direct_rewrite", "step_back", "hyde"]

RetrievalStrategy: TypeAlias = Literal[
    "original",
    "user_clarified",
    "direct_rewrite",
    "step_back",
    "hyde",
]

EvidenceFailureReason: TypeAlias = Literal[
    "none",
    "irrelevant_evidence",
    "insufficient_coverage",
    "query_mismatch",
    "overly_specific",
    "terminology_gap",
]

HITLAction: TypeAlias = Literal["clarify", "scope_select"]

AnswerLimitationKind: TypeAlias = Literal[
    "no_knowledge",
    "unsupported",
    "unresolved",
    "technical_failure",
]
