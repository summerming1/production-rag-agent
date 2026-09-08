from production_rag.citations import build_context, citation_label, cited_labels
from production_rag.types import Document, RetrievalHit


def test_citation_preserves_source_and_page():
    h = RetrievalHit(Document("x", "evidence", "manual.pdf", {"page": 7}), 1.0, rank=2)
    assert citation_label(h) == "[2] manual.pdf, p. 7"
    assert "evidence" in build_context([h])


def test_only_actual_valid_citations_are_reported():
    hits = [
        RetrievalHit(Document("a", "a", "a.md"), 1.0, rank=1),
        RetrievalHit(Document("b", "b", "b.md"), 0.9, rank=2),
    ]
    assert cited_labels("Use [2]. Ignore invented [9]. Repeat [2].", hits) == ("[2] b.md",)
