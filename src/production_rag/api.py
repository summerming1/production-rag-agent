from pathlib import Path

from .corpus import load_jsonl_corpus
from .hybrid import HybridRetriever
from .pipeline import RAGPipeline


def create_app(corpus_path: str = "data/sample_corpus.jsonl"):
    try:
        from fastapi import FastAPI
        from pydantic import BaseModel, Field
    except ImportError as exc:
        raise RuntimeError("Install with: pip install -e '.[api]'") from exc

    docs = load_jsonl_corpus(Path(corpus_path))
    pipeline = RAGPipeline(HybridRetriever(docs))
    app = FastAPI(title="Production RAG Agent Showcase", version="0.2.0")

    class Query(BaseModel):
        question: str = Field(min_length=1)
        top_k: int = Field(default=5, ge=1, le=50)

    @app.get("/health")
    def health():
        return {"status": "ok", "documents": len(docs)}

    @app.post("/retrieve")
    def retrieve(q: Query):
        return [
            {
                "id": h.document.id,
                "source": h.document.source,
                "score": h.score,
                "rank": h.rank,
            }
            for h in pipeline.retrieve(q.question, q.top_k)
        ]

    @app.post("/rag/chat")
    def chat(q: Query):
        ans = pipeline.answer(q.question, q.top_k)
        return {
            "status": ans.status,
            "answer": ans.answer,
            "citations": ans.citations,
            "citation_issues": ans.citation_issues,
            "retrieval_coverage": ans.retrieval_coverage,
        }

    return app
