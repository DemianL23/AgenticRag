"""Prompt construction for grounded answers."""

from __future__ import annotations

from collections.abc import Sequence

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from agenticrag.retrieval.schemas import RetrievedChunk


SYSTEM_PROMPT = """你是一个财经文档问答助手。
你只能依据用户提供的证据回答问题，不能使用证据之外的知识补充答案。
如果证据不足以回答，请明确说明“根据当前检索到的资料无法确定”，并返回空的 citation_ids。
每个关键结论都必须引用一个或多个证据编号。
不要编造证据编号、文件名、页码或数字。
"""


def build_messages(query: str, chunks: Sequence[RetrievedChunk]) -> list[BaseMessage]:
    """Build messages while preserving every citation-relevant chunk field."""
    evidence_blocks: list[str] = []
    for index, chunk in enumerate(chunks, start=1):
        evidence_blocks.append(
            "\n".join(
                [
                    f"[E{index}]",
                    f"doc_id: {chunk.doc_id}",
                    f"source: {chunk.source}",
                    f"page: {chunk.page}",
                    f"chunk_id: {chunk.chunk_id}",
                    f"retrieval_score: {chunk.score}",
                    f"content:\n{chunk.content}",
                ]
            )
        )

    human_content = (
        f"问题：\n{query.strip()}\n\n"
        "证据：\n"
        f"{chr(10).join(evidence_blocks)}\n\n"
        "请返回结构化结果：answer 是答案文本，citation_ids 只能填写实际存在的证据编号。"
    )
    return [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=human_content),
    ]
