from production_rag.hybrid import HybridRetriever
from production_rag.pipeline import RAGPipeline
from production_rag.types import Document


def test_pipeline_abstains_when_no_hits():
    p = RAGPipeline(HybridRetriever([Document("a", "alpha", "s")]))
    ans = p.answer("zzzzzz")
    assert not ans.hits
    assert "enough retrieved evidence" in ans.answer
