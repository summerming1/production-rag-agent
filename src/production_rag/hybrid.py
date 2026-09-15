from __future__ import annotations

from collections import defaultdict
from typing import Protocol

from .bm25 import BM25Retriever
from .types import Document, RetrievalHit


class Searcher(Protocol):
    def search(self, query: str, top_k: int = 10) -> list[RetrievalHit]: ...


def apply_source_diversity(
    hits: list[RetrievalHit],
    *,
    top_k: int,
    max_per_source: int,
) -> list[RetrievalHit]:
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    if max_per_source <= 0:
        raise ValueError("max_per_source must be positive")
    selected: list[RetrievalHit] = []
    deferred: list[RetrievalHit] = []
    source_counts: dict[str, int] = defaultdict(int)
    for hit in hits:
        source = hit.document.source
        if source_counts[source] >= max_per_source:
            deferred.append(hit)
            continue
        selected.append(hit)
        source_counts[source] += 1
        if len(selected) == top_k:
            break
    if len(selected) < top_k:
        selected.extend(deferred[: top_k - len(selected)])
    return [
        RetrievalHit(h.document, h.score, rank=i, channel_scores=h.channel_scores)
        for i, h in enumerate(selected[:top_k], 1)
    ]


class HybridRetriever:
    """Hybrid retrieval using Reciprocal Rank Fusion (RRF).

    RRF is used instead of directly adding BM25 and cosine scores because the
    score scales are not calibrated to each other.
    """

    def __init__(
        self,
        documents: list[Document],
        *,
        dense: Searcher | None = None,
        rrf_k: int = 60,
        max_per_source: int = 3,
    ) -> None:
        if rrf_k <= 0:
            raise ValueError("rrf_k must be positive")
        if max_per_source <= 0:
            raise ValueError("max_per_source must be positive")
        self.documents = documents
        self.bm25 = BM25Retriever(documents)
        self.dense = dense
        self.rrf_k = rrf_k
        self.max_per_source = max_per_source

    def search(
        self,
        query: str,
        top_k: int = 6,
        candidate_k: int = 20,
        *,
        diversify: bool = True,
    ) -> list[RetrievalHit]:
        if top_k <= 0 or candidate_k <= 0:
            raise ValueError("top_k and candidate_k must be positive")
        candidate_k = max(candidate_k, top_k)
        channels: dict[str, list[RetrievalHit]] = {
            "bm25": self.bm25.search(query, candidate_k)
        }
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
        hits = [
            RetrievalHit(
                docs[doc_id],
                score,
                rank=i,
                channel_scores=channel_scores[doc_id],
            )
            for i, (doc_id, score) in enumerate(ranked[:candidate_k], 1)
        ]
        if not diversify:
            return [
                RetrievalHit(h.document, h.score, rank=i, channel_scores=h.channel_scores)
                for i, h in enumerate(hits[:top_k], 1)
            ]
        return apply_source_diversity(
            hits,
            top_k=top_k,
            max_per_source=self.max_per_source,
        )
