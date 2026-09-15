from __future__ import annotations

import re
from dataclasses import dataclass

from .types import RetrievalHit

_CITATION_RE = re.compile(r"\[(\d+)\]")


@dataclass(frozen=True)
class CitationAudit:
    valid: bool
    labels: tuple[str, ...]
    issues: tuple[str, ...]


def citation_label(hit: RetrievalHit) -> str:
    """Return the stable human-facing label used in generated context."""
    page = hit.document.metadata.get("page")
    suffix = f", p. {page}" if page is not None else ""
    return f"[{hit.rank}] {hit.document.source}{suffix}"


def build_context(hits: list[RetrievalHit], max_chars_per_doc: int = 1800) -> str:
    """Build a bounded context while preserving rank/source/page provenance."""
    if max_chars_per_doc <= 0:
        raise ValueError("max_chars_per_doc must be positive")
    blocks: list[str] = []
    for hit in hits:
        label = citation_label(hit)
        text = hit.document.text[:max_chars_per_doc].strip()
        blocks.append(f"{label}\n{text}")
    return "\n\n".join(blocks)


def audit_citations(answer: str, hits: list[RetrievalHit], *, require_citation: bool = True) -> CitationAudit:
    """Check structural citation grounding against the frozen retrieved set.

    This proves that cited ranks exist in the supplied context. It does not prove
    that the cited text semantically supports every claim.
    """
    by_rank = {hit.rank: citation_label(hit) for hit in hits}
    raw_ranks = [int(value) for value in _CITATION_RE.findall(answer)]
    unknown = sorted({rank for rank in raw_ranks if rank not in by_rank})
    valid_ranks = []
    seen: set[int] = set()
    for rank in raw_ranks:
        if rank in by_rank and rank not in seen:
            valid_ranks.append(rank)
            seen.add(rank)

    issues: list[str] = []
    if unknown:
        issues.append("unknown citation ranks: " + ", ".join(map(str, unknown)))
    if require_citation and not valid_ranks:
        issues.append("answer contains no valid citation")

    return CitationAudit(
        valid=not issues,
        labels=tuple(by_rank[rank] for rank in valid_ranks),
        issues=tuple(issues),
    )


def cited_labels(answer: str, hits: list[RetrievalHit]) -> tuple[str, ...]:
    """Backward-compatible helper returning only valid cited labels."""
    return audit_citations(answer, hits, require_citation=False).labels
