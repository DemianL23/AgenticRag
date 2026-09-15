"""Agentic RAG V2 domain contracts and stage services."""

from .config import V2Config, V2PersistenceConfig
from .ids import new_request_id
from .planning import PlanningError, PlanningResult, PlanningService, plan
from .grading import EvidenceGrader, EvidenceGradingError
from .graph import build_graph_v2_1, initial_v2_1_state
from .graph import build_graph_v2_2, initial_v2_2_state, build_graph_v2_3
from .graph import initial_v2_3_state
from .state import V2_STATE_SCHEMA_VERSION
from .answering import (
    AnswerGenerationError,
    FindingGenerator,
    HITLContentGenerator,
    SynthesisGenerator,
    validate_synthesized_answer,
)
from .module4 import (
    MaterializationError,
    Module4Run,
    Module4Service,
    materialize_retrieval_tasks,
    unsupported_routing_decision,
)
from .module6 import Module6Run, Module6Service
from .hitl import HITLResumeError, HITLResumeResult, HITLResumeService, await_user_input
from .durable import DurableRun, DurableStatus, DurableV23Service, PersistenceDependencyError
from .persistence import (
    CleanupReport,
    LeaseConflictError,
    PersistenceError,
    RequestMetadata,
    RequestMetadataRepository,
)
from .retrieval import (
    RetrievalBackendError,
    RetrievalFanoutResult,
    RetrievalFanoutService,
    V12RetrievalAdapter,
    V12RetrievalBackend,
)
from .schemas import (
    ComplexityDecision,
    DecompositionResult,
    Evidence,
    EvidenceGrade,
    GroundedFinding,
    QueryRevision,
    RetrievalAttempt,
    RetrievalResult,
    RetrievalTask,
    RoutingDecision,
    StageRunResult,
    SynthesizedAnswer,
    TaskDraft,
)

__all__ = [
    "ComplexityDecision",
    "DecompositionResult",
    "Evidence",
    "EvidenceGrade",
    "GroundedFinding",
    "QueryRevision",
    "RetrievalAttempt",
    "RetrievalResult",
    "RetrievalTask",
    "RoutingDecision",
    "StageRunResult",
    "SynthesizedAnswer",
    "TaskDraft",
    "V2Config",
    "V2PersistenceConfig",
    "new_request_id",
    "PlanningError",
    "PlanningResult",
    "PlanningService",
    "plan",
    "EvidenceGrader",
    "EvidenceGradingError",
    "build_graph_v2_1",
    "initial_v2_1_state",
    "build_graph_v2_2",
    "initial_v2_2_state",
    "initial_v2_3_state",
    "build_graph_v2_3",
    "V2_STATE_SCHEMA_VERSION",
    "AnswerGenerationError",
    "FindingGenerator",
    "HITLContentGenerator",
    "SynthesisGenerator",
    "validate_synthesized_answer",
    "MaterializationError",
    "Module4Run",
    "Module4Service",
    "materialize_retrieval_tasks",
    "unsupported_routing_decision",
    "Module6Run",
    "Module6Service",
    "HITLResumeError",
    "HITLResumeResult",
    "HITLResumeService",
    "await_user_input",
    "DurableRun",
    "DurableStatus",
    "DurableV23Service",
    "PersistenceDependencyError",
    "CleanupReport",
    "LeaseConflictError",
    "PersistenceError",
    "RequestMetadata",
    "RequestMetadataRepository",
    "RetrievalBackendError",
    "RetrievalFanoutResult",
    "RetrievalFanoutService",
    "V12RetrievalAdapter",
    "V12RetrievalBackend",
]
