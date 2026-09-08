from production_rag.citations import build_context, citation_label
from production_rag.types import Document, RetrievalHit


def test_citation_preserves_source_and_page():
    h = RetrievalHit(Document("x", "evidence", "manual.pdf", {"page":7}), 1.0, rank=2)
    assert citation_label(h) == "[2] manual.pdf, p. 7"
    assert "evidence" in build_context([h])
