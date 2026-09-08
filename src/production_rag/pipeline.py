from __future__ import annotations

from .citations import build_context, citation_label
from .generator import ExtractiveFallbackGenerator, Generator
from .hybrid import HybridRetriever
from .rerank import CrossEncoderReranker
from .types import RAGAnswer, RetrievalHit


SYSTEM_PROMPT = (
    "Answer only from the supplied evidence. If the evidence is insufficient, "
    "say so. Cite sources using the bracketed labels exactly as provided. "
    "Do not invent citations."
)


class RAGPipeline:
    def __init__(
        self,
        retriever: HybridRetriever,
        *,
        generator: Generator | None = None,
        reranker: CrossEncoderReranker | None = None,
    ) -> None:
        self.retriever = retriever
        self.generator = generator or ExtractiveFallbackGenerator()
        self.reranker = reranker

    def retrieve(self, question: str, top_k: int = 6) -> list[RetrievalHit]:
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        hits = self.retriever.search(
            question,
            top_k=top_k,
            candidate_k=max(20, top_k * 4),
        )
        if self.reranker is not None and hits:
            hits = self.reranker.rerank(question, hits, top_k)
        return hits[:top_k]

    @staticmethod
    def retrieval_coverage(hits: list[RetrievalHit]) -> float:
        """A bounded routing heuristic, explicitly not a calibrated probability."""
        if not hits:
            return 0.0
        return min(1.0, len(hits) / 3.0)

    # Backward-compatible public name used by the simple agent/API.
    confidence = retrieval_coverage

    def answer_from_hits(self, question: str, hits: list[RetrievalHit]) -> RAGAnswer:
        """Generate from an already-frozen retrieved set; never retrieves twice."""
        coverage = self.retrieval_coverage(hits)
        if not hits:
            return RAGAnswer(
                question,
                "I do not have enough retrieved evidence to answer.",
                (),
                (),
                0.0,
            )
        context = build_context(hits)
        user = (
            f"QUESTION:\n{question}\n\n"
            f"CONTEXT:\n{context}\n\n"
            "Answer with citations."
        )
        answer = self.generator.generate(SYSTEM_PROMPT, user)
        citations = tuple(citation_label(hit) for hit in hits)
        return RAGAnswer(question, answer, citations, tuple(hits), coverage)

    def answer(self, question: str, top_k: int = 6) -> RAGAnswer:
        return self.answer_from_hits(question, self.retrieve(question, top_k))
