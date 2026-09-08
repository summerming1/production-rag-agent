from __future__ import annotations

import re

from .types import RetrievalHit

_CITATION_RE = re.compile(r"\[(\d+)\]")


def citation_label(hit: RetrievalHit) -> str:
    """Return the stable human-facing label used in generated context."""
    page = hit.document.metadata.get("page")
    suffix = f", p. {page}" if page is not None else ""
    return f"[{hit.rank}] {hit.document.source}{suffix}"


def build_context(hits: list[RetrievalHit], max_chars_per_doc: int = 1800) -> str:
    """Build a bounded context while preserving rank/source/page provenance."""
    blocks: list[str] = []
    for hit in hits:
        label = citation_label(hit)
        text = hit.document.text[:max_chars_per_doc].strip()
        blocks.append(f"{label}\n{text}")
    return "\n\n".join(blocks)


def cited_labels(answer: str, hits: list[RetrievalHit]) -> tuple[str, ...]:
    """Return only retrieved sources that the generated answer actually cites.

    Unknown/out-of-range citation numbers are ignored instead of being silently
    accepted as grounded citations.
    """
    by_rank = {hit.rank: citation_label(hit) for hit in hits}
    seen: set[int] = set()
    labels: list[str] = []
    for raw_rank in _CITATION_RE.findall(answer):
        rank = int(raw_rank)
        if rank in by_rank and rank not in seen:
            labels.append(by_rank[rank])
            seen.add(rank)
    return tuple(labels)
