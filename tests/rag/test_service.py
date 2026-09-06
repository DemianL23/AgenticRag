from dataclasses import dataclass

from agenticrag.generation.schemas import GeneratedAnswer, SourceCitation
from agenticrag.rag.service import RagAnswerService
from agenticrag.retrieval.schemas import RetrievedChunk


def _chunk() -> RetrievedChunk:
    return RetrievedChunk(
        content="证据文本",
        score=0.2,
        doc_id="doc_000",
        source="report.pdf",
        page=2,
        chunk_id="doc_000:p0002:c000",
    )


@dataclass
class FakeRetriever:
    calls: list[tuple[str, int]]

    def search(self, query: str, k: int = 5) -> list[RetrievedChunk]:
        self.calls.append((query, k))
        return [_chunk()]


class FakeGenerator:
    def __init__(self) -> None:
        self.calls: list[tuple[str, list[RetrievedChunk]]] = []

    def generate(
        self, query: str, chunks: list[RetrievedChunk]
    ) -> GeneratedAnswer:
        self.calls.append((query, chunks))
        return GeneratedAnswer(
            answer="答案",
            citations=(
                SourceCitation(
                    citation_id="E1",
                    doc_id="doc_000",
                    source="report.pdf",
                    page=2,
                    chunk_id="doc_000:p0002:c000",
                ),
            ),
        )


def test_rag_answer_service_passes_retrieved_chunks_to_generator() -> None:
    retriever = FakeRetriever(calls=[])
    generator = FakeGenerator()
    service = RagAnswerService(retriever, generator)  # type: ignore[arg-type]

    result = service.answer("问题", k=3)

    assert result.answer == "答案"
    assert retriever.calls == [("问题", 3)]
    assert generator.calls == [("问题", [_chunk()])]
