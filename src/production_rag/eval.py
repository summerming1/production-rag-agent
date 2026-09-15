from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

from .hybrid import HybridRetriever
from .types import RetrievalHit


@dataclass(frozen=True)
class RetrievalMetrics:
    recall_at_k: float
    mrr: float
    ndcg_at_k: float


def _dcg(relevances: list[int]) -> float:
    return sum(rel / math.log2(i + 2) for i, rel in enumerate(relevances))


def evaluate_ranked_retrieval(
    search: Callable[[str, int], list[RetrievalHit]],
    cases: list[dict],
    top_k: int = 5,
) -> RetrievalMetrics:
    if not cases:
        raise ValueError("cases must not be empty")
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    recalls, rrs, ndcgs = [], [], []
    for case in cases:
        gold = set(case["relevant_ids"])
        hits = search(case["query"], top_k)
        ids = [h.document.id for h in hits]
        recalls.append(len(gold.intersection(ids)) / max(1, len(gold)))
        rr = 0.0
        for i, doc_id in enumerate(ids, 1):
            if doc_id in gold:
                rr = 1.0 / i
                break
        rrs.append(rr)
        rel = [1 if doc_id in gold else 0 for doc_id in ids]
        ideal = [1] * min(len(gold), top_k)
        denom = _dcg(ideal)
        ndcgs.append(_dcg(rel) / denom if denom else 0.0)
    n = len(cases)
    return RetrievalMetrics(sum(recalls) / n, sum(rrs) / n, sum(ndcgs) / n)


def evaluate_retrieval(
    retriever: HybridRetriever,
    cases: list[dict],
    top_k: int = 5,
) -> RetrievalMetrics:
    return evaluate_ranked_retrieval(
        lambda query, k: retriever.search(query, top_k=k),
        cases,
        top_k,
    )


def evaluate_pipeline_retrieval(
    pipeline,
    cases: list[dict],
    top_k: int = 5,
) -> RetrievalMetrics:
    """Evaluate the final retrieval stage, including optional reranking."""
    return evaluate_ranked_retrieval(pipeline.retrieve, cases, top_k)
