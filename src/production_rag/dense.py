from __future__ import annotations

from .types import Document, RetrievalHit


class SentenceTransformerRetriever:
    """Optional dense retriever. Embeddings are normalized and scored by cosine similarity."""

    def __init__(self, documents: list[Document], model_name: str = "sentence-transformers/all-MiniLM-L6-v2", device: str | None = None) -> None:
        try:
            import numpy as np
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError("Install with: pip install -e '.[dense]'") from exc
        self.np = np
        self.documents = documents
        self.model = SentenceTransformer(model_name, device=device)
        self.embeddings = self.model.encode([d.text for d in documents], normalize_embeddings=True, convert_to_numpy=True)

    def search(self, query: str, top_k: int = 10) -> list[RetrievalHit]:
        q = self.model.encode([query], normalize_embeddings=True, convert_to_numpy=True)[0]
        scores = self.embeddings @ q
        order = self.np.argsort(-scores)[:top_k]
        return [RetrievalHit(self.documents[int(i)], float(scores[int(i)]), rank=r, channel_scores={"dense": float(scores[int(i)])}) for r, i in enumerate(order, 1)]
