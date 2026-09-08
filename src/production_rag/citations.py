from __future__ import annotations

from .types import RetrievalHit


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
