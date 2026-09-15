from production_rag.citations import audit_citations, cited_labels
from production_rag.types import Document, RetrievalHit


def _hits():
    return [
        RetrievalHit(Document("a", "alpha", "a.md"), 1.0, rank=1),
        RetrievalHit(Document("b", "beta", "b.md"), 0.8, rank=2),
    ]


def test_cited_labels_deduplicate_and_ignore_unknown_for_compatibility():
    assert cited_labels("Use [1], [1] and [999]", _hits()) == ("[1] a.md",)


def test_audit_rejects_unknown_citation():
    audit = audit_citations("Use [1] and [999]", _hits())
    assert not audit.valid
    assert audit.labels == ("[1] a.md",)
    assert "unknown citation ranks: 999" in audit.issues


def test_audit_requires_at_least_one_valid_citation():
    audit = audit_citations("No source marker", _hits())
    assert not audit.valid
    assert audit.issues == ("answer contains no valid citation",)


def test_audit_is_structural_not_semantic():
    audit = audit_citations("A claim [2]", _hits())
    assert audit.valid
    assert audit.labels == ("[2] b.md",)
