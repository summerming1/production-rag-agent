from __future__ import annotations

from .types import RetrievalHit


class CrossEncoderReranker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2", device: str | None = None) -> None:
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            raise RuntimeError("Install with: pip install -e '.[rerank]'") from exc
        self.model = CrossEncoder(model_name, device=device)

    def rerank(self, query: str, hits: list[RetrievalHit], top_k: int) -> list[RetrievalHit]:
        if not hits:
            return []
        scores = self.model.predict([(query, h.document.text) for h in hits])
        rows = sorted(zip(scores, hits), key=lambda x: float(x[0]), reverse=True)[:top_k]
        return [RetrievalHit(hit.document, float(score), rank=i, channel_scores={**hit.channel_scores, "rerank": float(score)}) for i, (score, hit) in enumerate(rows, 1)]
