from __future__ import annotations

import math
from collections import Counter

from .tokenize import tokenize
from .types import Document, RetrievalHit


class BM25Retriever:
    def __init__(self, documents: list[Document], *, k1: float = 1.5, b: float = 0.75) -> None:
        if not documents:
            raise ValueError("documents must not be empty")
        self.documents = documents
        self.k1, self.b = k1, b
        self.tokens = [tokenize(d.text) for d in documents]
        self.lengths = [len(t) for t in self.tokens]
        self.avgdl = sum(self.lengths) / len(self.lengths)
        self.tfs = [Counter(t) for t in self.tokens]
        df: Counter[str] = Counter()
        for toks in self.tokens:
            df.update(set(toks))
        n = len(documents)
        self.idf = {term: math.log(1.0 + (n - freq + 0.5) / (freq + 0.5)) for term, freq in df.items()}

    def search(self, query: str, top_k: int = 10) -> list[RetrievalHit]:
        q = tokenize(query)
        scored: list[tuple[float, int]] = []
        for i, tf in enumerate(self.tfs):
            dl = self.lengths[i]
            score = 0.0
            for term in q:
                freq = tf.get(term, 0)
                if not freq:
                    continue
                denom = freq + self.k1 * (1 - self.b + self.b * dl / max(self.avgdl, 1e-9))
                score += self.idf.get(term, 0.0) * (freq * (self.k1 + 1)) / denom
            if score > 0:
                scored.append((score, i))
        scored.sort(reverse=True)
        return [RetrievalHit(self.documents[i], score, rank=r, channel_scores={"bm25": score}) for r, (score, i) in enumerate(scored[:top_k], 1)]
