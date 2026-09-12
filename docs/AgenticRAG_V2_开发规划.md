# AgenticRAG V2 开发规范

## 1. 文档地位

本文是 AgenticRAG V2 的唯一开发规范，取代此前的 V2 概要。实现、测试、评测和 baseline 冻结均以本文为准。

V2 基于已经冻结的 V1.2 Retrieval Baseline 演进，不重写 Dense Retrieval、BM25、RRF 或本地 BGE Reranker。开发仍按模块进行：开始一个模块前先确认该模块的输入、输出、失败行为和验收条件；完成后由项目维护者阅读代码并提问，再进入下一模块。

本文固定参考以下外部项目中的架构思想，不复制其业务代码：

```text
Repository: https://github.com/icey1287/SuperMew
Commit:    e2ebabe81e066e0498bdb58000942ef5199d338c
```

参考范围限于 Evidence Grader、条件路由、查询恢复和 HITL 的设计思想。本项目的数据契约、实现和测试必须适配现有 V1.2。

## 2. 当前项目基线

V1.2 当前提供的是检索能力，而不是完整问答版本：

```text
Query
  ├─ Dense Top-20
  ├─ BM25 Top-20
  └─ Union + chunk_id Dedup
          ↓
      RRF (k=60)
          ↓
  Local BGE Reranker
          ↓
     Final Top-5 Evidence
```

V1.2 同时保留 RRF Top-20、完整 union candidate pool、fallback 和耗时信息。V2 默认只把每次检索的 Final Top-5 交给 Evidence Grader；RRF Top-20 与完整 union pool 只进入诊断 trace，不默认进入 LLM 上下文。

现有完整问答 CLI 仍使用 V0 Dense Retriever。因此在比较 V2 前，必须增加一个不含 Agentic 控制逻辑的 V1.2 End-to-End Control：

```text
Question
  → V1.2 RerankingRetriever Top-5
  → Existing Answer Generator
  → Answer + Citations
```

该 control 不加入 Router、Decomposer、Grader、Recovery 或 HITL，只用于隔离 V2 编排带来的效果和代价。

## 3. V2 目标

V2 将 V1.2 检索链路升级为一个确定性、有状态、有界、可恢复的端到端 RAG 工作流：

```text
Question
   ↓
Request Preparation
   ↓
Complexity Router
   ├─ simple  → SQ_001
   └─ complex → Query Decomposer → 2..configured limit RetrievalTasks
                                      ↓
                              Capability Policy
                         ┌────────────┴────────────┐
                         │                         │
              retrieval_synthesis          unsupported capability
                         │                         │
                         ↓                         └→ Task unsupported
               V1.2 Retrieval Fan-out
                         ↓
                  Evidence Grader
                         ↓
                Deterministic Routing
         ┌──────────┬───────────┬────────────────────┐
         │ answer   │ recover   │ clarify/scope_select│
         │          │           │                    │
         │          ↓           ↓                    │
         │   Recovery Strategy  HITLRequest           │
         │          ↓           ↓                    │
         │   Re-retrieval       Checkpoint/Interrupt  │
         │          ↓           ↓                    │
         │      Re-grade        Resume → QueryRevision│
         └──────────┴───────────┴────────────────────┘
                         ↓
                GroundedFinding(s)
                         ↓
                 Outcome Aggregation
                         ↓
        Simple Adapter / Complex Final Synthesizer
                         ↓
              Citation & Contract Validation
                         ↓
                  SynthesizedAnswer
```

V2 必须回答三个工程问题：

1. 原问题应作为一个任务检索，还是拆成多个独立任务？
2. 当前证据是否足以回答；如果不足，在预算内应采用哪一种恢复策略？
3. 问题语义不完整时，能否安全暂停、持久化，并只恢复受用户输入影响的任务？

## 4. 范围边界

### 4.1 V2 包含

- LangGraph `StateGraph` 显式编排
- Complexity Router
- Query Decomposition
- 每任务 Capability 声明与确定性 Capability Policy
- 有界 Multi-query Retrieval
- Evidence Grader
- 确定性 Routing Policy
- Direct Rewrite、Step-back、HyDE
- 每个 QueryRevision 最多一次 corrective retrieval
- GroundedFinding 与分层答案合成
- Citation/Provenance Validation
- `clarify` 与 `scope_select`
- SQLite checkpoint、跨进程 interrupt/resume
- append-only 决策历史和结构化 trace
- V1.2 End-to-End Control 与 V2 分阶段评测

### 4.2 V2 不包含

- Calculator 或 Python Tool
- 由 LLM 心算代替 Calculator
- SQL Agent
- 子查询之间的数据依赖
- 动态 Replanning
- Reflection
- 开放式或无界 Retry
- ReAct Agent 或完整 Tool Selection
- Query Translation
- Conversation Memory
- 异步 FastAPI Job、Worker 或 Queue
- PostgreSQL checkpointer

直接从文档提取已有数值属于 `retrieval_synthesis`。只要必须产生新的算术、比例、增长率、统计、回归或 OLS 结果，该任务就必须标记为不受 V2 支持，不能让 Answer Model 尝试心算。

## 5. 核心架构决策

### 5.1 使用 LangGraph StateGraph

V2 明确使用类型化 `StateGraph`、普通边和 conditional edges，不使用 Functional API，也不使用 `create_agent`。V2 是确定性工作流，不是开放式 Tool Agent。

`langgraph` 必须成为项目直接依赖，不能依赖 LangChain 的传递安装。SQLite checkpointer 相关依赖放入独立的 `v2-persistence` optional extra；V2.1/V2.2 不要求安装该 extra，V2.3 durable CLI 必须安装。

### 5.2 三个阶段共享实现

V2.1、V2.2、V2.3 共用同一套 state、schemas、nodes 和 policies，通过三个 graph builder 组装不同终点：

- `graph_v2_1`：到 proposed decision 为止。
- `graph_v2_2`：执行 recovery、Finding 和答案合成；HITL 路径只返回不可恢复的等待结果。
- `graph_v2_3`：加入 durable checkpoint、interrupt 和 resume。

不得复制节点，也不得通过大量 feature flag 在一个 Graph 中隐藏阶段差异。

### 5.3 V1.2 是唯一检索后端

V2 通过薄适配器 `V12RetrievalBackend` 依赖稳定的 `retrieve(...) -> RetrievalResult` 契约。适配器内部调用现有 `RerankingRetriever.search_with_trace()`，不得复制或修改 V1.2 的 Dense、BM25、RRF 和 Reranker 实现。

### 5.4 评估与路由分离

Evidence Grader 只描述证据事实，不直接决定 route。Routing Policy 根据经过验证的 EvidenceGrade 和执行预算产生确定性 route。Recovery Strategy 同样由确定性规则选择；Rewrite Model 只负责生成既定策略的 recovery artifact。

## 6. 公共领域模型

Graph 顶层状态使用 `TypedDict` 和显式 reducer；领域对象使用 Pydantic 模型。所有进入 Graph State 的值都必须能够安全地序列化为 JSON/MessagePack，不允许 pickle fallback。

模型实例、Retriever、数据库连接、网络 client、lock、semaphore 等运行时对象不得进入 Graph State。

### 6.1 枚举

```python
TargetStage = Literal["v2_1", "v2_2", "v2_3"]
ResponseLanguage = Literal["zh", "en"]
Complexity = Literal["simple", "complex"]

TaskCapability = Literal[
    "retrieval_synthesis",
    "arithmetic",
    "statistical_computation",
    "sql",
    "other_unsupported",
]

TaskExecutionStatus = Literal[
    "pending",
    "running",
    "waiting_user",
    "completed",
    "failed",
]

TaskAnswerOutcome = Literal[
    "complete",
    "no_knowledge",
    "unsupported",
    "unresolved",
]

GlobalExecutionStatus = Literal[
    "running",
    "waiting_user",
    "completed",
    "failed",
]

GlobalAnswerOutcome = Literal[
    "complete",
    "partial",
    "no_knowledge",
    "unsupported",
    "unresolved",
]

Route = Literal[
    "answer",
    "recover",
    "clarify",
    "scope_select",
    "no_knowledge",
    "unsupported",
]

RetrievalStrategy = Literal[
    "original",
    "user_clarified",
    "direct_rewrite",
    "step_back",
    "hyde",
]
```

`failed` 和 `waiting_user` 的任务，其 `answer_outcome` 必须为 `null`。正常业务终止的任务使用 `execution_status=completed`，并具有非空 TaskAnswerOutcome。Task 层不使用 `partial`；它只由多个 required task 聚合产生。

### 6.2 请求状态

```python
class V2State(TypedDict):
    request_id: str
    target_stage: TargetStage
    original_question: str
    normalized_query: str
    response_language: ResponseLanguage

    complexity_decision: ComplexityDecision | None
    task_order: list[str]
    tasks: dict[str, RetrievalTask]
    evidence: dict[str, Evidence]

    pending_hitl_request: HITLRequest | None
    hitl_rounds: int

    execution_status: GlobalExecutionStatus
    answer_outcome: GlobalAnswerOutcome | None
    final_answer: SynthesizedAnswer | None
    error: ExecutionError | None

    state_schema_version: str
```

Reducer 合并 fan-out 结果后必须按稳定 `task_order` 排序，不能依赖并发完成顺序。

### 6.3 Complexity 与 Decomposition

```python
class ComplexityDecision(BaseModel):
    complexity: Literal["simple", "complex"]
    capability: TaskCapability | None
    reason: str

class TaskDraft(BaseModel):
    query: str
    intent: str
    capability: TaskCapability

class DecompositionResult(BaseModel):
    tasks: list[TaskDraft]
    decomposition_complete: bool
    failure_reason: Literal["decomposition_limit"] | None
```

约束：

- simple：Router 必须输出 capability，Application 创建唯一 `SQ_001`。
- complex：Router capability 必须为空，Decomposer 生成 2 到 `V2_MAX_SUBQUERIES` 个 TaskDraft；baseline 上限为 4。
- Task query 必须非空、语义不重复、可独立检索。
- 所有 Task 都是回答原问题所必需的 required task；V2 没有 optional task。
- Baseline 无法在 4 个独立任务内完整表达问题时，必须返回 `decomposition_complete=false` 和 `decomposition_limit`，不得静默截断。实验即使覆盖上限，也必须保持一个经过校验的有限正整数。
- Task ID 由程序分配，LLM 不生成 ID。

### 6.4 RetrievalTask

```python
class RetrievalTask(BaseModel):
    id: str
    ordinal: int
    query: str
    intent: str
    capability: TaskCapability
    required: Literal[True] = True

    query_revisions: list[QueryRevision]
    grade_records: list[GradeRecord]
    routing_decisions: list[RoutingDecision]
    grounded_finding: GroundedFinding | None

    execution_status: TaskExecutionStatus
    answer_outcome: TaskAnswerOutcome | None
    terminal_reason: str | None
    error: ExecutionError | None
```

V2 不提供 `depends_on`。所有 RetrievalTask 都必须能够独立执行。

### 6.5 QueryRevision 与 RetrievalAttempt

```python
class QueryRevision(BaseModel):
    id: str
    ordinal: int
    source: Literal["original", "hitl"]
    query: str
    user_input: dict[str, str] | None
    retrieval_attempts: list[RetrievalAttempt]

class RetrievalAttempt(BaseModel):
    id: str
    ordinal: int
    strategy: RetrievalStrategy
    retrieval_query: str
    evidence_ids: list[str]
    retrieval_degraded: bool
    retrieval_degraded_reason: str | None
    latency: RetrievalLatency
    trace_ref: str | None
```

QueryRevision 只在有效 HITL 输入改变任务语义时创建。Direct Rewrite、Step-back 和 HyDE 是同一 QueryRevision 中的第二次 RetrievalAttempt，不创建 revision。

每个 QueryRevision 最多两个 RetrievalAttempt；每个任务最多两个 QueryRevision，因此理论上最多四次检索。新 revision 只使用自身 attempts 的 Evidence 进行评分；旧 revision Evidence 保留在历史中，但不能自动继续支持已经变化的新语义。

所有 `query_revisions`、`retrieval_attempts`、`grade_records` 和 `routing_decisions` 都是 append-only，不得覆盖旧记录。

### 6.6 Evidence 与 Provenance

```python
class EvidenceOccurrence(BaseModel):
    task_id: str
    query_revision_id: str
    retrieval_attempt_id: str
    strategy: RetrievalStrategy
    dense_rank: int | None
    bm25_rank: int | None
    rrf_rank: int | None
    final_rank: int

class Evidence(BaseModel):
    evidence_id: str       # stable chunk_id
    chunk_id: str
    content: str
    doc_id: str
    source: str
    page: int
    occurrences: list[EvidenceOccurrence]
```

Evidence 按 `chunk_id` 全局去重；正文和稳定 metadata 只保存一份，出现历史不去重。Prompt 内的 `E1/E2/...` 只是一次调用的临时标签，不得写成持久 Evidence ID。

### 6.7 EvidenceGrade

```python
class EvidenceGrade(BaseModel):
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
    reason: str
    missing_information: list[str]
    missing_slots: list[str]
    supporting_evidence_ids: list[str]
```

EvidenceGrade 不包含 route。必须执行以下确定性交叉校验：

- `supporting_evidence_ids` 必须是本次 Grader 输入 Evidence ID 的子集。
- `answerability=sufficient` 时至少包含一个合法 supporting ID。
- `answerability=partial` 时至少包含一个合法 supporting ID。
- `answerability=none` 时 supporting IDs 必须为空。
- `relevance=none` 时不允许 `answerability=sufficient`。
- `recoverability=likely` 时，answerability 不能是 sufficient，failure_reason 不能是 none。
- `failure_reason=none` 时，recoverability 必须是 none。
- `ambiguity=missing_slot` 时，`missing_slots` 必须非空。

`missing_information` 表示证据中仍缺少的事实；`missing_slots` 表示用户问题中缺少、必须由用户提供的参数。只有 missing slots 触发 clarify。

### 6.8 GradeRecord 与 RoutingDecision

```python
class GradeRecord(BaseModel):
    id: str
    query_revision_id: str
    input_attempt_ids: list[str]
    input_evidence_ids: list[str]
    grade: EvidenceGrade

class RoutingDecision(BaseModel):
    id: str
    grade_record_id: str | None
    route: Route
    recovery_strategy: Literal[
        "direct_rewrite", "step_back", "hyde"
    ] | None
    reason: str
```

当前状态由最后一条合法记录派生，但历史记录永远不覆盖。

### 6.9 GroundedFinding 与最终回答

```python
class GroundedFinding(BaseModel):
    task_id: str
    text: str
    evidence_ids: list[str]

class AnswerLimitation(BaseModel):
    task_id: str
    kind: Literal[
        "no_knowledge",
        "unsupported",
        "unresolved",
        "technical_failure",
    ]
    reason: str

class SynthesizedAnswer(BaseModel):
    answer: str
    citation_evidence_ids: list[str]
    limitations: list[AnswerLimitation]
```

每个有效 GroundedFinding 必须至少引用一条、最多引用三条 Evidence，且这些 Evidence 必须属于该任务最后一次合法 Grade 的 supporting IDs。程序必须做确定性 provenance 校验。

## 7. 稳定 ID 规则

全局 request ID 使用 UUID4。请求内部 ID 根据父对象身份和局部 ordinal 确定性生成，不能依赖 fan-out 完成顺序。例如：

```text
SQ_001
QR_SQ001_001
ATT_SQ001_QR001_001
GR_SQ001_QR001_001
DEC_SQ001_QR001_001
HITL_001
OPT_001
```

具体格式可以在实现前微调，但必须满足：

- 同一请求内唯一；
- 可从 ID 看出父级关系；
- 并行执行不改变 ID；
- resume 不重新分配已存在 ID；
- HITL 和 ScopeOption ID 在 payload 创建时一次性固定。

## 8. 请求准备

### 8.1 Query normalization

`normalized_query` 只能做无语义变化的确定性规范化，包括 trim、whitespace 归一和 Unicode 规范化。禁止在此阶段使用 LLM 补公司、年份、范围、翻译或改写。

所有语义变化必须显式表现为：

- RetrievalAttempt 的 recovery strategy；或
- HITL 后的新 QueryRevision。

### 8.2 Response language

Application Service 在进入 Graph 前冻结 `response_language`：

1. 请求显式提供 `zh|en` 时直接采用；
2. 未提供时，使用确定性的 CJK/Latin 主体比例检测；
3. 写入 state 后所有节点禁止修改。

V2 baseline 只支持中文和英文，不增加语言识别模型。检测只统计 CJK 与 Latin 字符；具体阈值必须通过独立的 zh/en/mixed-language unit cases 验证后冻结，并写入 resolved config 和 baseline report，不主要使用 19 条 QA 调参。

## 9. Capability Policy

Capability 是 TaskDraft 的结构化字段，不新增逐任务 CapabilityClassifier LLM 调用。

- simple：Complexity Router 同时输出 query-level capability。
- complex：Query Decomposer 为每个 TaskDraft 输出 capability。

确定性 Capability Policy：

- `retrieval_synthesis` → 进入 V1.2 retrieval fan-out。
- `arithmetic/statistical_computation/sql/other_unsupported` → 不执行检索，任务终止为 `answer_outcome=unsupported` 并记录能力原因。

一个复杂请求中，受支持的任务继续执行；不受支持的任务不能抹掉其他任务的有效结果。

## 10. Evidence Grading 与 Routing

### 10.1 初次 Grade 输入

初次 Grade 只接收当前 QueryRevision 第一次 RetrievalAttempt 的 V1.2 Final Top-5。完整 candidate pool、RRF Top-20 和 HyDE artifact 不得进入 Evidence。

### 10.2 Recovery 后 Re-grade

Recovery 后，将当前 QueryRevision 中：

```text
Attempt 1 Final Top-5
UNION
Attempt 2 Final Top-5
```

按 `chunk_id` 去重后一起交给同一 Evidence Grader，最多 10 条。不得合并完整 candidate pool，也不得跨 QueryRevision 自动合并 Evidence。

### 10.3 Execution Guard

Technical failure 在 Routing Policy 之前处理：

```text
task.execution_status = failed
task.answer_outcome = null
stop current task
```

Technical failure 不是业务 route，不能转换成 no_knowledge 或 unsupported。

### 10.4 确定性路由优先级

通过 Execution Guard 后，Routing Policy 按以下顺序执行：

1. unsupported capability → `unsupported`
2. `ambiguity=missing_slot` → `clarify`
3. `ambiguity=multiple_candidates` → `scope_select`
4. `relevance=strong` 且 `answerability=sufficient` 且 supporting evidence 合法 → `answer`
5. `recoverability=likely` 且当前 QueryRevision 仍有 retrieval budget → `recover`
6. 其他情况 → `no_knowledge`

LLM 不得覆盖该顺序。特别地，weak relevance 即使被模型标记 sufficient，也不能直接进入 answer。

## 11. Recovery Policy

只有 route 为 recover、recoverability 为 likely 且当前 QueryRevision 仍有 retrieval budget 时，才选择策略：

| failure_reason | recovery_strategy |
|---|---|
| `irrelevant_evidence` | `direct_rewrite` |
| `insufficient_coverage` | `direct_rewrite` |
| `query_mismatch` | `direct_rewrite` |
| `overly_specific` | `step_back` |
| `terminology_gap` | `hyde` |

不再调用独立 Rewrite Selector。Rewrite Model 只按已确定的 strategy 生成 recovery artifact。

Recovery artifact 必须：

- schema 合法且内容非空；
- 规范化后不能与当前 QueryRevision 已执行过的 retrieval query 重复；
- 保存 strategy 和生成内容；
- HyDE artifact 必须明确标记 `is_evidence=false`。

非法或重复 artifact 是 `invalid_recovery_artifact` 技术失败。不得自动回退原 query，也不得消耗一次相同查询来伪装 recovery。

HyDE 内容无论多像答案，都不能进入 Evidence、GroundedFinding citation 或 Final Answer citation。

## 12. 有界执行预算

V2 baseline 默认值：

```dotenv
V2_MAX_SUBQUERIES=4
V2_MAX_CONCURRENT_SUBQUERIES=1
V2_MAX_RETRIEVAL_ATTEMPTS_PER_REVISION=2
V2_MAX_QUERY_REVISIONS=2
V2_MAX_HITL_ROUNDS=1
V2_MAX_EVIDENCE_PER_FINDING=3
V2_MAX_SCOPE_OPTIONS=5
V2_CHECKPOINT_TTL_SECONDS=604800
```

所有值必须经过配置校验。Baseline 使用上述默认值；实验可以显式覆盖，但 resolved configuration 必须完整写入报告。

CPU + 本地 BGE baseline 的子查询并发数固定为 1。接口保留可配置能力，只有在 CUDA 环境单独验证模型实例、显存和结果稳定性后，才允许评估 2/4 等并发。

## 13. GroundedFinding 与 Answer Synthesis

### 13.1 Finding 生成

只有 answer route 才生成 GroundedFinding。Finding Generator 输入当前任务、最后一次合法 Grade 和 Grade 支持的 Evidence。

如果 Finding 生成超时、structured output 非法、引用非法或 provenance 校验失败：

- 任务记为 technical failure；
- 不保留未验证 Finding；
- 不转换为 no_knowledge。

### 13.2 Simple request

Simple request 只有一个 `SQ_001`，仍然只执行一次 Answer Model 调用。该调用产生带 Evidence IDs 的 grounded answer，并在状态中保存为唯一 GroundedFinding。

Application 使用确定性 adapter 将该 Finding 转为统一的 SynthesizedAnswer，不再调用 Final Synthesizer。

### 13.3 Complex request

Complex request 为每个 answerable task 生成 GroundedFinding，然后执行 Final Synthesizer。Synthesizer 只接收：

1. original question；
2. 所有 required task 的业务 outcome 或失败原因；
3. 已验证 GroundedFindings；
4. 仅被这些 Findings 引用且已经验证的原始 Evidence 内容。

不得把所有 Top-5、RRF Top-20 或 union pool 一次性塞给 Synthesizer。4 个任务、每个 Finding 最多 3 条 Evidence，因此 baseline 在去重前最多携带 12 条原始 Evidence，去重后可能更少。

### 13.4 最终校验

- `citation_evidence_ids` 必须是所有有效 Findings 引用 Evidence union 的子集。
- global outcome 为 complete 时，limitations 必须为空。
- global outcome 为 partial 时，limitations 必须覆盖所有未 complete 的 required tasks，包括 technical failure。
- Synthesizer 不能修改程序已经确定的 GlobalAnswerOutcome。

已有 Findings 但 Final Synthesizer 超时、schema 非法或 citation 非法时，整个请求终止为：

```text
execution_status = failed
answer_outcome = null
```

V2 baseline 不使用字符串拼接 Findings 的隐藏 fallback，也不返回未通过完整契约校验的半成品。

### 13.5 零 Finding

没有有效 Finding 时，不调用 Final Synthesizer。Application 根据确定性 outcome 生成与 `response_language` 一致的 no_knowledge、unsupported 或 unresolved 响应。

## 14. Global Outcome Aggregation

业务结果由程序确定，不交给 Synthesizer。

存在至少一个有效 GroundedFinding 时：

- 所有 required task 的 `answer_outcome=complete` → global `complete`
- 否则 → global `partial`

不存在任何有效 GroundedFinding 时：

1. 存在 required task technical failure → global execution failed，answer outcome 为 null
2. 否则存在 unresolved task → global `unresolved`
3. 否则存在 unsupported task → global `unsupported`
4. 否则 → global `no_knowledge`

终态只表示工作流已停止，不等于成功回答。

## 15. HITL 数据契约

### 15.1 HITLRequest

```python
class ScopeOption(BaseModel):
    id: str
    label: str
    value: str
    description: str
    evidence_ids: list[str]

class HITLItem(BaseModel):
    id: str
    action: Literal["clarify", "scope_select"]
    affected_task_ids: list[str]
    question: str
    missing_slots: list[str]
    scope_options: list[ScopeOption]

class HITLRequest(BaseModel):
    id: str
    request_id: str
    items: list[HITLItem]
```

多个任务同时需要用户输入时，必须合并成一次 HITLRequest，以 task ID 标识受影响任务。全局最多一次 HITL round。

确定性校验：

- clarify item 的 missing_slots 必须非空；
- scope_select item 至少有 2 个、最多有 5 个 options；
- affected_task_ids 必须存在；
- option ID 在请求内唯一；
- option evidence IDs 必须来自对应任务已检索 Evidence。

`DecisionModelConfig.hitl` 只生成澄清问题和 ScopeOption 的语义内容。Application 负责 validate、dedup、稳定排序和分配 `OPT_001...` ID。

### 15.2 Interrupt 节点边界

必须拆分两个节点：

```text
build_hitl_request
        ↓
await_user_input
```

只有 `await_user_input` 调用 `interrupt()`。HITLRequest 必须先写入 Graph State/checkpoint。由于 LangGraph resume 会从中断节点入口重新执行，等待节点不得在 interrupt 前调用模型或执行非幂等副作用。

### 15.3 ResumeRequest

```python
class HITLResponse(BaseModel):
    item_id: str
    clarify_values: dict[str, str] | None
    selected_option_id: str | None

class ResumeRequest(BaseModel):
    request_id: str
    hitl_request_id: str
    responses: list[HITLResponse]
```

Application Service 必须在调用 LangGraph resume 之前验证：

- request ID 与当前 durable request 一致；
- HITL request ID 与当前 pending request 一致；
- items 一一对应，不允许只回答一部分；
- scope_select 只能提交当前 option ID；
- clarify values 覆盖要求的 slots 且值合法。

非法输入不得修改 Graph State、不得创建 QueryRevision、不得消耗 HITL round 或 retrieval budget。

### 15.4 新 QueryRevision

合法用户输入由 Application/Policy 使用确定性模板合并到受影响任务语义中，不额外调用 LLM。受影响任务创建新的 QueryRevision，其首次 RetrievalAttempt 使用 `strategy=user_clarified`。未受影响任务和已完成 Finding 保持不变。

新 revision 仍可在证据不足时使用一次 direct_rewrite、step_back 或 HyDE recovery。

如果第二个 QueryRevision 再次得到 clarify/scope_select，不再 interrupt。该任务终止为：

```text
answer_outcome = unresolved
terminal_reason = hitl_budget_exhausted
```

如果其他任务已完成，全局可以聚合为 partial；如果所有 required tasks 都 unresolved，则全局为 unresolved。

## 16. V2.3 Persistence

### 16.1 SQLite 职责

默认数据库：

```text
artifacts/checkpoints/v2/agenticrag_v2.sqlite3
```

路径允许环境变量覆盖，必须被 Git 忽略，进程启动时禁止自动删除。

同一个 SQLite 文件包含两类互不混淆的表：

- LangGraph checkpoint tables：完全由官方 SQLite checkpointer 管理。
- `v2_requests`：由 Application Service 管理 request lifecycle metadata。

Application 不得重写 LangGraph checkpoint schema。

`v2_requests` 使用 Python 标准库 SQLite 能力，三个阶段都可以记录轻量 request lifecycle。V2.1/V2.2 不写 Graph checkpoint；V2.3 安装 `v2-persistence` extra 后，才在同一文件中启用官方 LangGraph SQLite checkpointer。这样 V2.2 的 request ID 可以被识别为 `resumable=false`，但不存在可供恢复的 Graph State。

### 16.2 Request metadata

`v2_requests` 至少记录：

- request_id / thread_id
- target_stage
- state_schema_version
- execution status / answer outcome
- current pending HITL request ID
- created_at / updated_at / expires_at
- expired flag
- metadata version
- execution lease owner / lease expiry
- latest error

Application 生成 UUID4 request ID，并一对一用作 LangGraph thread ID。Checkpoint ID 由 LangGraph 管理，只用于内部诊断，不进入公共业务契约；用户 resume 只依赖 request ID。

### 16.3 并发 claim 与幂等

Resume 前，Application 使用 metadata version 和有时限 lease 在 SQLite 中原子 claim request。两个进程同时 resume 时只有一个能够获得执行权。Lease 必须过期，避免进程异常退出后 request 永久处于 running。

同一 HITL request ID 的重复有效 Resume 必须幂等：不能创建重复 QueryRevision，已经处理时返回当前 request result。

Lease 时长必须可配置，并在 V2.3 baseline report 中记录 resolved value。

### 16.4 TTL 与清理

Checkpoint TTL 默认 7 天。Application 在 Graph 外、resume 前检查存在性、TTL、execution state、schema version 和 pending HITL request。

过期时：

```text
request metadata.execution_status = failed
request metadata.answer_outcome = null
request metadata.error = checkpoint_expired
```

不得为了写 terminal state 而 resume 已过期 Graph。旧 checkpoint 保留为历史，直到显式 cleanup。

Cleanup 默认 dry-run，只列出 expired 且 eligible 的 requests。只有显式 `--execute` 才物理删除 request metadata 和对应 LangGraph checkpoints。禁止启动时自动清理。

Schema 版本不兼容时返回明确版本错误；V2 不实现 checkpoint 自动迁移。

## 17. 模型角色与配置

### 17.1 DecisionModelConfig

独立角色：

```text
router
decomposer
grader
rewrite
hitl
```

### 17.2 AnswerModelConfig

独立角色：

```text
simple_answer
finding
synthesis
```

角色允许默认解析到同一个 Qwen endpoint，但 model、revision、endpoint identifier、temperature、timeout、max_tokens、thinking 和 retry policy 必须能独立覆盖。Decision 配置不得直接复用现有 Answer Generator 的 GenerationConfig 实例。

每个 LLM role 默认最多两次调用：一次正常调用和一次 call-level retry。只重试 transient provider error、timeout、rate limit 和偶发 structured-output failure；确定性配置错误不要求重试。

Structured output 最终失败后禁止从自然语言猜测字段，必须记为 technical failure。LLM call retry 不计入 RetrievalAttempt。

每次 eval/report 必须记录所有 role 的 resolved configuration，但绝不记录 API Key、Secret 或完整敏感 endpoint 凭据。

## 18. V1.2 Retrieval Adapter

建议契约：

```python
class V12RetrievalBackend(Protocol):
    def retrieve(
        self,
        *,
        task_id: str,
        query_revision_id: str,
        attempt_id: str,
        query: str,
    ) -> RetrievalResult:
        ...

class RetrievalResult(BaseModel):
    evidence: list[Evidence]        # final Top-5
    retrieval_degraded: bool
    degraded_reason: str | None
    latency: RetrievalLatency
    trace_ref: str | None
```

V1.2 Reranker fallback 继续保证普通运行可用。发生 fallback 时：

```text
retrieval_degraded = true
degraded_reason = reranker_fallback
```

V2 可继续 Grade 和回答，但不得伪装为正常 Reranker 输出。正式 V2 experimental baseline 要求 `retrieval_degraded_queries=0`。

## 19. Graph 与阶段行为

### 19.1 V2.1：Planning、Retrieval 与 Grade

目标：观察完整的 classify/decompose/retrieve/grade/proposed decision，不执行 recovery、Finding、合成或 HITL。

建议节点：

```text
prepare_state
detect_complexity
build_simple_task / decompose_query
apply_capability_policy
retrieve_tasks
grade_tasks
propose_routing_decisions
finalize_v2_1_stage
```

验收：

- simple 创建唯一 SQ_001，不调用 Decomposer。
- complex 产生 2 到配置上限个独立 required tasks；baseline 上限为 4。
- unsupported tasks 不调用 Retrieval。
- retrieval tasks 完整复用 V1.2 Final Top-5。
- 每个 retrieval task 有合法 Grade 和 proposed decision。
- StageRunResult 可结束，但 `answer_outcome` 和 `final_answer` 允许为 null。

### 19.2 V2.2：有界 Recovery 与 Answer

新增节点：

```text
apply_routing_policy
select_recovery_strategy
generate_recovery_artifact
reretrieve
regrade
generate_grounded_finding
build_hitl_request
aggregate_outcome
adapt_simple_answer / synthesize_complex_answer
validate_final_answer
```

V2.2 遇到 clarify/scope_select 时：

```text
execution_status = waiting_user
resumable = false
pending_hitl_request = structured payload
```

随后结束当前 run。不得转换为 no_knowledge，也不得实现临时内存 resume。

### 19.3 V2.3：Durable HITL

在 V2.2 已使用 `build_hitl_request` 生成不可恢复 payload 的基础上，V2.3 新增或正式启用：

```text
await_user_input
validate_resume_at_application_boundary
create_query_revision
resume_affected_tasks
```

V2.3 waiting result 使用 `resumable=true`。必须通过两个独立 CLI/Python 进程验证：

```text
start → interrupt → process exit → resume
```

Resume 只重新执行 affected tasks；未受影响的完成任务、Evidence、Grade 和 Finding 不得重复执行或丢失。

### 19.4 StageRunResult

三个 builder 统一返回：

```python
class StageRunResult(BaseModel):
    request_id: str
    target_stage: TargetStage
    execution_status: GlobalExecutionStatus
    answer_outcome: GlobalAnswerOutcome | None
    final_answer: SynthesizedAnswer | None
    resumable: bool
    pending_hitl_request: HITLRequest | None
    trace_ref: str | None
    error: ExecutionError | None
```

阶段完成不等于完整端到端回答完成。V2.1 允许 completed + null outcome；完整 V2.3 的业务终态必须遵守 Global Outcome Aggregation。

## 20. CLI

V2 baseline 提供 Python Application Service 和以下 CLI，不引入完整异步 HTTP 基础设施：

```text
agenticrag-v2-run --stage v2_1|v2_2|v2_3 ...
agenticrag-v2-status REQUEST_ID
agenticrag-v2-resume REQUEST_ID --input ...
agenticrag-v2-checkpoint-cleanup
```

能力边界：

- run：三个阶段均支持。
- status：面向有 durable request metadata 的请求。
- resume：只允许 V2.3 resumable request。
- 对 V2.2 `resumable=false` 请求调用 resume，明确返回 `request_not_resumable`，不得重新启动 Graph。
- cleanup：只面向 V2.3 durable persistence，默认 dry-run。

## 21. 错误语义

必须区分：

- `no_knowledge`：系统正常执行，但知识库证据不足。
- `unsupported`：任务要求 V2 未实现的能力。
- `unresolved`：需要用户确认，但 HITL 预算耗尽。
- `technical failure`：模型、检索、schema、persistence 或生成执行失败。

Technical failure 永远不能转换成前三种业务结果。任务级失败彼此隔离；存在其他有效 Finding 时，全局可以形成 partial，并在 limitations 中标记 technical_failure。不存在任何 Finding 且有 required task 技术失败时，全局 execution failed、answer outcome 为 null。

错误至少使用稳定 code，包括：

```text
structured_output_invalid
retrieval_failed
invalid_recovery_artifact
finding_generation_failed
citation_validation_failed
request_not_resumable
resume_payload_invalid
resume_conflict
checkpoint_not_found
checkpoint_expired
checkpoint_version_incompatible
```

## 22. 可观测性

Checkpoint 只保存恢复所必需的业务状态。以下内容写入本地结构化 trace/report：

- 每个节点进入/退出和 active latency；
- 每个 role 的模型调用次数、重试、token usage（provider 可提供时）和 latency；
- 每个 task/revision/attempt 的 query、strategy 和 Evidence IDs；
- V1.2 candidate latency、reranker fallback 和完整 diagnostics reference；
- Grade、RoutingDecision、Finding 和最终校验结果；
- budget consumption 与 invariant violations。

完整 RRF/union diagnostics 不重复写入 checkpoint。LangSmith 只作为未来可选集成，不能成为 baseline 唯一诊断来源。

HITL waiting wall time 必须与工作流 active execution time 分开记录。

## 23. 评测数据

### 23.1 V1.2 retrieval regression

`eval/datasets/retrieval_eval_v2.jsonl` 的 47 条数据保持完全不变，用于确认 V2 没有破坏 V1.2 Retrieval。

### 23.2 原始 19 条 QA

根目录 `qa.jsonl` 保持完全不变，维持历史 Answer/RAGAS 可比性。V2 的 complexity、capability、required information units 和 expected outcome 写入独立的 `eval/datasets/v2_qa_annotations.jsonl` sidecar，不写回原文件。

计算类样本在 V2 中评价 capability/outcome/abstention，不将正确的 unsupported 直接按普通 Answer Correctness 记错。

### 23.3 V2 workflow scenarios

新增固定的 `eval/datasets/v2_workflow_scenarios.jsonl`，覆盖：

- direct rewrite、step-back、HyDE；
- clarify、scope_select；
- complete、partial、no_knowledge、unsupported、unresolved；
- provider/timeout/schema/retrieval 等技术失败；
- 中文、英文、mixed-language；
- simple、complex、supported/unsupported mixed；
- 跨进程 interrupt/resume。

每种 recovery strategy、HITL action 和重要 terminal path 至少两个 scenario：一个正常成功 case，加一个边界、失败或不应触发 case。数据冻结时记录实际样本数量。

## 24. 指标

### 24.1 Planning

- complexity accuracy
- capability accuracy
- decomposition requirement coverage
- empty/duplicate/task-limit violations

Decomposition 由人工标注 required information units。结构约束由程序确定性检查；语义 coverage 由锁定配置、独立于被测 Decomposer 的 Judge 评价，不做字符串精确匹配，也不允许被测模型充当自己的 Judge。

### 24.2 Evidence 与 Routing

- EvidenceGrade 各字段准确率
- route accuracy
- recovery strategy accuracy
- recovery success rate
- retrieval attempts 与 budget violations

### 24.3 Answer

- supported subset 的 RAGAS
- citation/provenance validity
- outcome accuracy
- unsupported-computation recall
- abstention correctness

Mixed supported/unsupported 请求中，已支持部分单独评价 grounded answer/citation；不支持部分评价 capability 和 outcome。

### 24.4 Runtime

- per-role LLM calls/tokens/latency
- V1.2 retrieval/reranker latency
- retrieval attempts per task/revision
- degraded retrieval queries
- active end-to-end time
- HITL waiting time（单独）
- checkpoint/resume latency

## 25. Report 与 Baseline

### 25.1 不覆盖历史

每次 eval 在 `artifacts/eval/v2/<stage>/<run_id>/` 生成新目录，禁止覆盖历史结果。报告至少包含：

- run_id / timestamp / git_commit
- target stage
- dataset/model identifiers
- resolved configuration
- metrics 与 invariant counts
- artifact digest / configuration digest

`latest` 可以作为可变引用，但 frozen baseline 必须记录不可变 run ID 和 digest。

### 25.2 阶段报告

- V2.1 输出 Planning/Retrieval/Grade Stage Report。
- V2.2 输出 Recovery/Answer Stage Report。
- V2.3 输出 Durable HITL Stage Report。

阶段报告用于演进记录，不分别命名为完整 V2 baseline。

### 25.3 冻结条件

只有 V2.3 完成且以下硬条件全部满足，才能冻结为：

```text
Agentic RAG V2 Baseline
```

硬条件：

- V1.2 retrieval regression 全部通过；
- schema invariant violations = 0；
- provenance violations = 0；
- citation violations = 0；
- budget violations = 0；
- retrieval_degraded_queries = 0；
- computation capability 安全用例全部正确进入 unsupported；
- baseline scenarios 全部达到预期 terminal contract；
- V2.3 restart/interrupt/resume 跨进程测试通过；
- 普通 baseline run 没有未预期 technical failure。

专门注入 timeout/provider/schema failure 的负向测试按“是否正确处理失败”验收，不计入普通 baseline 的 zero-technical-failure 条件。

Recall、RAGAS、planning accuracy 和 recovery success 等质量指标必须如实记录，但第一版不要求优于 V1.2 才能冻结。允许冻结质量没有提升的有效实验，并明确结论。

## 26. 测试策略

### 26.1 默认快速测试

- Pydantic schema 与跨字段校验
- ID 稳定性和 fan-out 顺序
- Capability、Routing、Recovery、Aggregation Policy
- Evidence merge 与 provenance validator
- Citation validator
- budget/revision/HITL terminal rules
- response language detection
- request metadata、TTL、lease 与 resume payload validator

### 26.2 Mock Graph 集成测试

使用 Mock LLM、Mock RetrievalBackend 和内存中的测试状态覆盖全部 route、recovery strategy、HITL action、预算和终态，不加载真实 BGE、不调用 Qwen。

### 26.3 真实集成测试

固定小集合运行真实 V1.2、Qwen 和本地 BGE。昂贵测试必须使用 pytest marker，不进入默认快速测试。

### 26.4 跨进程测试

使用 SQLite：

```text
Process A: run V2.3 → interrupt → exit
Process B: status → valid resume → complete
```

同时覆盖 duplicate resume、stale HITL request、invalid payload、expired checkpoint 和 lease recovery。

## 27. 建议目录

```text
src/agenticrag/v2/
  __init__.py
  config/
    budgets.py
    models.py
    persistence.py
  schemas/
    state.py
    planning.py
    evidence.py
    decisions.py
    answers.py
    hitl.py
  nodes/
    complexity_router.py
    query_decomposer.py
    retrieval.py
    evidence_grader.py
    recovery.py
    finding.py
    synthesis.py
    hitl.py
  policies/
    capability.py
    routing.py
    recovery.py
    outcome.py
  persistence/
    request_repository.py
    checkpointer.py
  graphs/
    v2_1.py
    v2_2.py
    v2_3.py
  application/
    service.py
    resume.py
  retrieval_backend.py
  ids.py
  language.py
  validators.py
  cli.py

src/eval/v2/
  datasets.py
  metrics.py
  runner.py
  report.py

tests/v2/
tests/eval/v2/
```

目录可在开发前按现有代码风格做小幅合并，但职责边界不得改变。V1.2 Retrieval 始终留在现有 `agenticrag/retrieval/`。

## 28. 分模块开发顺序

每个模块开始前再次确认契约，完成后停下来供项目维护者阅读。

### Module 0：V1.2 End-to-End Control

- 将 V1.2 Top-5 接入现有 Answer Generator。
- 生成不可覆盖的 control report。
- 不加入任何 V2 决策逻辑。

Module 0 的实现入口为：

```bash
uv run agenticrag-eval-v1-2-control
```

默认评测根目录 `qa.jsonl` 的全部 QA，最终送入 Generator 的结果为 V1.2
Reranker Top-5。报告写入 `artifacts/eval/v1_2_end_to_end_control/<run_id>/report.json`；
每次运行生成新的 UUID `run_id`，若指定的 `--output` 已存在则拒绝覆盖。
报告同时保存每条样本的答案、上下文、RRF/rerank trace、fallback/耗时、
resolved config、模型 revision，以及运行时的 `git_commit` / `git_dirty`。

### Module 1：V2 Schema、ID、Config 与 Policy

- Pydantic domain schemas 与 TypedDict state。
- 稳定层级 ID。
- budget、language、capability、routing、recovery、outcome policies。
- 全部 deterministic unit tests。

### Module 2：V2.1 Router 与 Decomposer

- role-specific DecisionModelConfig。
- simple/complex 和 capability structured output。
- 2 到配置上限个独立 TaskDraft（baseline 上限为 4）与 decomposition limit。
- Router 的 simple 输出必须带 top-level capability；complex 输出的 top-level capability 必须为 null，具体 capability 由 Decomposer 为每个 TaskDraft 声明。
- Planning gold 使用 `v2_qa_annotations.jsonl` sidecar，通过 `finqa_id` 与根目录 `qa.jsonl` 一一 join；simple annotation 使用 top-level capability 且 unit 的 `expected_capability` 为 null，complex annotation 使用 null top-level capability 且每个 unit 必须有 `expected_capability`。
- `required_information_units` 只描述可独立执行的业务信息单元，不包含跨 Task 的最终 synthesis、comparison 或 judgment；semantic coverage 由独立 Planning Judge 评估，结构约束由 deterministic code 检查。

### Module 3：V1.2 Adapter、Fan-out 与 Evidence Merge

- `V12RetrievalBackend`。
- bounded fan-out/reducer。
- Evidence canonicalization、occurrence history、degraded trace。

### Module 4：Evidence Grader 与 V2.1 Graph

- Grade prompt/schema/cross-field validator。
- proposed deterministic decisions。
- V2.1 CLI、Stage Report 和评测。

V2.1 通过验收后才能进入 V2.2。

### Module 5：Recovery

- 三种 Recovery artifact generator。
- attempt budget、duplicate query validator。
- re-retrieval 与同 revision Evidence union/re-grade。

### Module 6：GroundedFinding 与 Answer Synthesis

- simple single-call adapter。
- complex Finding → Synthesizer。
- outcome aggregation、limitations 和 citation/provenance validation。
- V2.2 Graph、CLI、Stage Report。

V2.2 的终止和失败语义稳定后才能进入 V2.3。

### Module 7：HITL Contract

- HITLRequestBuilder、ScopeOption 和 ResumeRequest validators。
- build/await 节点分离。
- QueryRevision 与 affected-task resume。

### Module 8：SQLite Persistence

- official LangGraph SQLite checkpointer。
- `v2_requests` metadata repository。
- TTL、schema version、lease、idempotent resume、cleanup。
- V2.3 Graph 与跨进程 CLI 测试。

### Module 9：最终评测与冻结

- 三套数据、四层指标、不可变 reports。
- V2.1/V2.2/V2.3 Stage Reports。
- 满足硬条件后冻结 Agentic RAG V2 Baseline。

## 29. Definition of Done

V2 完成必须同时满足：

- LangGraph StateGraph 显式描述所有节点和条件边；
- V1.2 只通过 adapter 复用，原 retrieval 回归不受破坏；
- simple 和 complex 请求都使用统一 RetrievalTask/Evidence/Finding 契约；
- 所有 LLM 输出经过 schema 与跨字段校验；
- 所有业务 route 由确定性 Policy 决定；
- 所有任务、revision、attempt、grade 和 decision 历史 append-only；
- Query Rewrite、Step-back、HyDE 均可执行且严格受预算限制；
- 计算、统计、SQL 等任务不会通过 LLM 偷偷执行；
- partial/no_knowledge/unsupported/unresolved/technical failure 语义清晰；
- Final Answer 只引用已验证的真实 Evidence；
- V2.3 能跨进程 interrupt/resume，并只恢复 affected tasks；
- checkpoint TTL、版本、lease、幂等和显式 cleanup 行为均有测试；
- 三阶段报告与最终 baseline report 不覆盖历史；
- 所有 baseline freeze 硬条件通过；
- Calculator/Python、SQL Agent、依赖型子查询、动态 Replanning、Reflection、多轮无界 Retry、完整 Tool Selection 和 HTTP Job 基础设施均未进入 V2。
