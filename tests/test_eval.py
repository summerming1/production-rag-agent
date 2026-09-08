from production_rag.eval import evaluate_retrieval
from production_rag.hybrid import HybridRetriever
from production_rag.types import Document


def test_retrieval_metrics_are_bounded():
    docs = [Document("a", "bm25 sparse lexical retrieval", "s"), Document("b", "unrelated text", "s2")]
    m = evaluate_retrieval(HybridRetriever(docs), [{"query":"bm25 lexical", "relevant_ids":["a"]}], top_k=1)
    assert m.recall_at_k == 1.0
    assert m.mrr == 1.0
    assert m.ndcg_at_k == 1.0
