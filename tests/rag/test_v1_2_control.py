from dataclasses import dataclass

from agenticrag.generation.schemas import GeneratedAnswer
from agenticrag.rag.v1_2_control import V12ControlAnswerService
from agenticrag.retrieval.reranking_retriever import RerankingSearchTrace
from agenticrag.retrieval.schemas import HybridRetrievedChunk, RerankedChunk


def _candidate() -> HybridRetrievedChunk:
    return HybridRetrievedChunk(
        content="V1.2 证据",
        score=1.0,
        doc_id="doc_001",
        source="report.pdf",
        page=3,
        chunk_id="doc_001:p0003:c000",
        dense_rank=2,
        bm25_rank=1,
        rrf_score=0.02,
        rrf_rank=1,
    )


def _trace() -> RerankingSearchTrace:
    candidate = _candidate()
    result = RerankedChunk(
        content=candidate.content,
        score=3.0,
        doc_id=candidate.doc_id,
        source=candidate.source,
        page=candidate.page,
        chunk_id=candidate.chunk_id,
        dense_rank=candidate.dense_rank,
        bm25_rank=candidate.bm25_rank,
        rrf_score=candidate.rrf_score,
        rrf_rank=candidate.rrf_rank,
        rerank_score=3.0,
        final_rank=1,
    )
    return RerankingSearchTrace(
        results=(result,),
        candidate_pool=(candidate,),
        rrf_top20=(candidate,),
        fallback_used=False,
        fallback_reason=None,
        invalid_scores=False,
        candidate_seconds=0.1,
        model_load_seconds=0.2,
        rerank_seconds=0.3,
        total_seconds=0.6,
    )


@dataclass
class FakeRetriever:
    trace: RerankingSearchTrace
    calls: list[tuple[str, int]]

    def search_with_trace(self, query: str, *, k: int = 5) -> RerankingSearchTrace:
        self.calls.append((query, k))
        return self.trace

    def model_record(self) -> dict[str, str]:
        return {"model_name": "fake-reranker"}


class FakeGenerator:
    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[RerankedChunk, ...]]] = []

    def generate(
        self, query: str, chunks: tuple[RerankedChunk, ...]
    ) -> GeneratedAnswer:
        self.calls.append((query, chunks))
        return GeneratedAnswer(answer="基于 V1.2 的答案", citations=())


def test_control_service_uses_v12_final_results_for_generation() -> None:
    retrieval_trace = _trace()
    retriever = FakeRetriever(retrieval_trace, [])
    generator = FakeGenerator()
    service = V12ControlAnswerService(retriever, generator)  # type: ignore[arg-type]

    trace = service.answer_with_trace("问题", k=5)

    assert retriever.calls == [("问题", 5)]
    assert generator.calls == [("问题", retrieval_trace.results)]
    assert trace.answer.answer == "基于 V1.2 的答案"
    assert trace.retrieved_chunks == retrieval_trace.results
    assert trace.retrieval_trace == retrieval_trace


def test_control_service_exposes_reranker_record() -> None:
    retriever = FakeRetriever(_trace(), [])
    service = V12ControlAnswerService(retriever, FakeGenerator())  # type: ignore[arg-type]

    assert service.retrieval_record() == {"model_name": "fake-reranker"}
