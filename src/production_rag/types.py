from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Document:
    id: str
    text: str
    source: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RetrievalHit:
    document: Document
    score: float
    rank: int = 0
    channel_scores: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class RAGAnswer:
    question: str
    answer: str
    citations: tuple[str, ...]
    hits: tuple[RetrievalHit, ...]
    retrieval_coverage: float
