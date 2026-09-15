from production_rag.eval import evaluate_pipeline_retrieval, evaluate_retrieval
from production_rag.hybrid import HybridRetriever
from production_rag.pipeline import RAGPipeline
from production_rag.types import Document


CASES = [
    {"query": "alpha", "relevant_ids": ["a"]},
    {"query": "beta", "relevant_ids": ["b"]},
]


def test_retriever_and_final_pipeline_can_be_evaluated_separately():
    retriever = HybridRetriever(
        [Document("a", "alpha evidence", "s1"), Document("b", "beta evidence", "s2")]
    )
    direct = evaluate_retrieval(retriever, CASES, top_k=1)
    final = evaluate_pipeline_retrieval(RAGPipeline(retriever), CASES, top_k=1)
    assert direct.recall_at_k == 1.0
    assert final.recall_at_k == 1.0
