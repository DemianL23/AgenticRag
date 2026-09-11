# V2 使用 LangGraph 实现有界可恢复的状态工作流

## Context

V2 需要在既有 V1.2 检索后端之上加入独立任务拆分、证据评估、确定性条件路由、有限恢复检索以及可跨进程继续的人机确认。自研状态机或开放式 Agent 循环都会增加迁移成本，并使重试边界、暂停状态和局部恢复难以统一验证。

## Decision

V2 使用 LangGraph `StateGraph`、显式节点和条件边实现确定性工作流，不使用 Functional API、`create_agent` 或开放式 ReAct 循环。V1.2 通过单一 adapter 作为唯一检索后端；业务路由由结构化评估和确定性 Policy 驱动。工作流按任务、查询修订和检索尝试设置硬预算，并将执行状态与回答结果分开。V2.3 本地 baseline 使用官方 SQLite checkpointer 和独立 request metadata，支持跨进程 interrupt/resume；Graph State 只包含显式可序列化数据且禁止 pickle fallback。

## Consequences

项目必须把 `langgraph` 声明为直接依赖，并为 SQLite checkpointer 提供独立 persistence extra。节点需要保持职责单一和可重放，尤其要把 HITL payload 构建与 `interrupt()` 等待拆开；恢复只保证从持久化图状态继续，被中断节点仍可能从入口重新执行。该选择增加了状态、版本、TTL、lease 和幂等测试，但换来可审计的有限循环、局部失败隔离和跨进程恢复。PostgreSQL、多实例服务、HTTP Job/Worker、依赖型子查询和动态 replanning 不属于本 ADR 所覆盖的 V2 范围。
