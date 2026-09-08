from production_rag.bm25 import BM25Retriever
from production_rag.types import Document


def test_bm25_finds_lexical_match():
    docs = [Document("a", "vector databases store embeddings", "a"), Document("b", "bm25 uses term frequency and document length", "b")]
    hits = BM25Retriever(docs).search("term frequency bm25", top_k=1)
    assert hits[0].document.id == "b"
