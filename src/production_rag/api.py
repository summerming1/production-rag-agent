from __future__ import annotations

from pathlib import Path

from .corpus import load_jsonl_corpus
from .hybrid import HybridRetriever
from .pipeline import RAGPipeline


def create_app(corpus_path: str = "data/sample_corpus.jsonl"):
    try:
        from fastapi import FastAPI
        from pydantic import BaseModel
    except ImportError as exc:
        raise RuntimeError("Install with: pip install -e '.[api]'") from exc

    docs = load_jsonl_corpus(Path(corpus_path))
    pipeline = RAGPipeline(HybridRetriever(docs))
    app = FastAPI(title="Production RAG Agent Showcase", version="0.1.0")

    class Query(BaseModel):
        question: str
        top_k: int = 5

    @app.get("/health")
    def health():
        return {"status": "ok", "documents": len(docs)}

    @app.post("/retrieve")
    def retrieve(q: Query):
        return [{"id": h.document.id, "source": h.document.source, "score": h.score, "rank": h.rank} for h in pipeline.retrieve(q.question, q.top_k)]

    @app.post("/rag/chat")
    def chat(q: Query):
        ans = pipeline.answer(q.question, q.top_k)
        return {"answer": ans.answer, "citations": ans.citations, "retrieval_confidence": ans.retrieval_confidence}

    return app
