from production_rag.hybrid import HybridRetriever
from production_rag.types import Document, RetrievalHit


class DenseStub:
    def __init__(self, docs):
        self.docs = docs
    def search(self, query, top_k=10):
        return [RetrievalHit(self.docs[1], 0.99, rank=1, channel_scores={"dense":0.99}), RetrievalHit(self.docs[0], 0.5, rank=2, channel_scores={"dense":0.5})]


def test_rrf_merges_ranked_lists_without_score_addition():
    docs = [Document("a", "alpha lexical query", "s1"), Document("b", "semantic target", "s2")]
    hits = HybridRetriever(docs, dense=DenseStub(docs)).search("alpha query", top_k=2)
    assert {h.document.id for h in hits} == {"a", "b"}
    assert all(h.score < 0.1 for h in hits)
