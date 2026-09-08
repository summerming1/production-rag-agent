from __future__ import annotations

from dataclasses import dataclass

from .pipeline import RAGPipeline
from .types import RAGAnswer


@dataclass(frozen=True)
class AgentDecision:
    action: str
    reason: str


class BoundedRAGAgent:
    """A deliberately bounded agent, not an open-ended autonomous loop.

    It has two explicit actions: retrieve/answer, or abstain. That makes the
    control flow deterministic and testable, which is useful for enterprise
    RAG where unsupported answers can be more costly than extra orchestration.
    """

    def __init__(self, pipeline: RAGPipeline, min_hits: int = 1) -> None:
        if min_hits < 1:
            raise ValueError("min_hits must be >= 1")
        self.pipeline = pipeline
        self.min_hits = min_hits

    def run(self, question: str, top_k: int = 6) -> tuple[AgentDecision, RAGAnswer]:
        hits = self.pipeline.retrieve(question, top_k)
        if len(hits) < self.min_hits:
            answer = self.pipeline.answer_from_hits(question, hits)
            return (
                AgentDecision("abstain", "retrieval returned too little evidence"),
                answer,
            )
        return (
            AgentDecision("answer", "retrieval returned sufficient evidence"),
            self.pipeline.answer_from_hits(question, hits),
        )
