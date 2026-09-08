from __future__ import annotations

from collections import defaultdict
from typing import Protocol

from .bm25 import BM25Retriever
from .types import Document, RetrievalHit


class Searcher(Protocol):
    def search(self, query: str, top_k: int = 10) -> list[RetrievalHit]: ...


class HybridRetriever:
    """Hybrid retrieval using Reciprocal Rank Fusion (RRF).

    RRF is used instead of directly adding BM25 and cosine scores because the
    score scales are not calibrated to each other.
    """

    def __init__(self, documents: list[Document], *, dense: Searcher | None = None, rrf_k: int = 60, max_per_source: int = 3) -> None:
        self.documents = documents
        self.bm25 = BM25Retriever(documents)
        self.dense = dense
        self.rrf_k = rrf_k
        self.max_per_source = max_per_source

    def search(self, query: str, top_k: int = 6, candidate_k: int = 20) -> list[RetrievalHit]:
        channels: dict[str, list[RetrievalHit]] = {"bm25": self.bm25.search(query, candidate_k)}
        if self.dense is not None:
            channels["dense"] = self.dense.search(query, candidate_k)

        fused: dict[str, float] = defaultdict(float)
        channel_scores: dict[str, dict[str, float]] = defaultdict(dict)
        docs: dict[str, Document] = {}
        for channel, hits in channels.items():
            for rank, hit in enumerate(hits, 1):
                docs[hit.document.id] = hit.document
                fused[hit.document.id] += 1.0 / (self.rrf_k + rank)
                channel_scores[hit.document.id][channel] = hit.score

        ranked = sorted(fused.items(), key=lambda x: x[1], reverse=True)
        selected: list[RetrievalHit] = []
        source_counts: dict[str, int] = defaultdict(int)
        deferred: list[RetrievalHit] = []
        for doc_id, score in ranked:
            hit = RetrievalHit(docs[doc_id], score, channel_scores=channel_scores[doc_id])
            source = hit.document.source
            if source_counts[source] >= self.max_per_source:
                deferred.append(hit)
                continue
            selected.append(hit)
            source_counts[source] += 1
            if len(selected) == top_k:
                break
        if len(selected) < top_k:
            selected.extend(deferred[: top_k - len(selected)])
        return [RetrievalHit(h.document, h.score, rank=i, channel_scores=h.channel_scores) for i, h in enumerate(selected, 1)]
