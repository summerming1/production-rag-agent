from __future__ import annotations

from .citations import audit_citations, build_context
from .generator import ExtractiveFallbackGenerator, Generator
from .hybrid import HybridRetriever, apply_source_diversity
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
        rerank_candidate_k: int = 20,
    ) -> None:
        if rerank_candidate_k <= 0:
            raise ValueError("rerank_candidate_k must be positive")
        self.retriever = retriever
        self.generator = generator or ExtractiveFallbackGenerator()
        self.reranker = reranker
        self.rerank_candidate_k = rerank_candidate_k

    def retrieve(self, question: str, top_k: int = 6) -> list[RetrievalHit]:
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        if self.reranker is None:
            return self.retriever.search(
                question,
                top_k=top_k,
                candidate_k=max(20, top_k * 4),
            )

        pool_k = max(top_k, self.rerank_candidate_k)
        hits = self.retriever.search(
            question,
            top_k=pool_k,
            candidate_k=max(20, pool_k * 2),
            diversify=False,
        )
        reranked = self.reranker.rerank(question, hits, pool_k)
        return apply_source_diversity(
            reranked,
            top_k=top_k,
            max_per_source=self.retriever.max_per_source,
        )

    @staticmethod
    def retrieval_coverage(hits: list[RetrievalHit]) -> float:
        """A bounded routing heuristic, explicitly not a calibrated probability."""
        if not hits:
            return 0.0
        return min(1.0, len(hits) / 3.0)

    confidence = retrieval_coverage

    def abstain(
        self,
        question: str,
        hits: list[RetrievalHit],
        *,
        reason: str = "insufficient retrieved evidence",
        citation_issues: tuple[str, ...] = (),
    ) -> RAGAnswer:
        return RAGAnswer(
            question,
            f"I do not have enough grounded evidence to answer ({reason}).",
            (),
            tuple(hits),
            self.retrieval_coverage(hits),
            status="abstain",
            citation_issues=citation_issues,
        )

    def answer_from_hits(self, question: str, hits: list[RetrievalHit]) -> RAGAnswer:
        """Generate from an already-frozen retrieved set; never retrieves twice."""
        coverage = self.retrieval_coverage(hits)
        if not hits:
            return self.abstain(question, hits)
        context = build_context(hits)
        user = (
            f"QUESTION:\n{question}\n\n"
            f"CONTEXT:\n{context}\n\n"
            "Answer with citations."
        )
        answer = self.generator.generate(SYSTEM_PROMPT, user)
        audit = audit_citations(answer, hits, require_citation=True)
        if not audit.valid:
            return self.abstain(
                question,
                hits,
                reason="generated answer failed citation validation",
                citation_issues=audit.issues,
            )
        return RAGAnswer(
            question,
            answer,
            audit.labels,
            tuple(hits),
            coverage,
            status="answer",
        )

    def answer(self, question: str, top_k: int = 6) -> RAGAnswer:
        return self.answer_from_hits(question, self.retrieve(question, top_k))
