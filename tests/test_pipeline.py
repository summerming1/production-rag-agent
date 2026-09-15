from production_rag.agent import BoundedRAGAgent
from production_rag.hybrid import HybridRetriever
from production_rag.pipeline import RAGPipeline
from production_rag.types import Document, RetrievalHit


class StubGenerator:
    def __init__(self, answer: str) -> None:
        self.answer = answer
        self.calls = 0

    def generate(self, system: str, user: str) -> str:
        self.calls += 1
        return self.answer


class RecordingReranker:
    def __init__(self) -> None:
        self.seen = 0

    def rerank(self, query: str, hits: list[RetrievalHit], top_k: int) -> list[RetrievalHit]:
        self.seen = len(hits)
        return list(reversed(hits))[:top_k]


def test_pipeline_abstains_when_no_hits():
    p = RAGPipeline(HybridRetriever([Document("a", "alpha", "s")]))
    ans = p.answer("zzzzzz")
    assert not ans.hits
    assert ans.retrieval_coverage == 0.0
    assert ans.status == "abstain"


def test_valid_citation_is_returned():
    generator = StubGenerator("Supported answer [1]")
    p = RAGPipeline(
        HybridRetriever([Document("a", "alpha evidence", "a.md")]),
        generator=generator,
    )
    ans = p.answer("alpha")
    assert ans.status == "answer"
    assert ans.citations == ("[1] a.md",)
    assert generator.calls == 1


def test_unknown_citation_fails_closed():
    generator = StubGenerator("Unsupported answer [999]")
    p = RAGPipeline(
        HybridRetriever([Document("a", "alpha evidence", "a.md")]),
        generator=generator,
    )
    ans = p.answer("alpha")
    assert ans.status == "abstain"
    assert not ans.citations
    assert "unknown citation ranks: 999" in ans.citation_issues


def test_agent_abstention_does_not_call_generator():
    generator = StubGenerator("should not be called [1]")
    p = RAGPipeline(
        HybridRetriever([Document("a", "alpha evidence", "a.md")]),
        generator=generator,
    )
    decision, ans = BoundedRAGAgent(p, min_hits=2).run("alpha", top_k=1)
    assert decision.action == "abstain"
    assert ans.status == "abstain"
    assert generator.calls == 0


def test_reranker_receives_candidate_pool_before_final_top_k():
    docs = [
        Document(str(i), f"alpha evidence {i}", f"source-{i // 2}")
        for i in range(8)
    ]
    reranker = RecordingReranker()
    p = RAGPipeline(
        HybridRetriever(docs, max_per_source=2),
        reranker=reranker,
        rerank_candidate_k=6,
    )
    hits = p.retrieve("alpha", top_k=2)
    assert reranker.seen == 6
    assert len(hits) == 2
