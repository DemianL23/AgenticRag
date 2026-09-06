"""The first runnable backend: application setup and a liveness endpoint."""

from fastapi import FastAPI

app = FastAPI(
    title="AgenticRag API",
    description="财经文档问答项目。当前阶段提供服务存活检查，问答接口随后接入。",
    version="0.1.0",
)


@app.get("/api/health", tags=["health"], summary="检查 API 服务是否运行")
async def health() -> dict[str, str]:
    """Report process liveness; this does not check models or databases."""
    return {"status": "ok"}
