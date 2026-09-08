from production_rag.hybrid import HybridRetriever
from production_rag.pipeline import RAGPipeline
from production_rag.types import Document


def test_pipeline_abstains_when_no_hits():
    p = RAGPipeline(HybridRetriever([Document("a", "alpha", "s")]))
    ans = p.answer("zzzzzz")
    assert not ans.hits
    assert ans.retrieval_coverage == 0.0
    assert "enough retrieved evidence" in ans.answer


def test_offline_generator_reports_only_cited_source():
    p = RAGPipeline(HybridRetriever([Document("a", "alpha evidence", "a.md")]))
    ans = p.answer("alpha")
    assert ans.citations == ("[1] a.md",)
