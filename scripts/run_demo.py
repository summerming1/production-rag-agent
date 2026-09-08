from __future__ import annotations

import json
from pathlib import Path

from production_rag import BoundedRAGAgent, HybridRetriever, RAGPipeline, load_jsonl_corpus
from production_rag.eval import evaluate_retrieval


def main() -> None:
    docs = load_jsonl_corpus("data/sample_corpus.jsonl")
    retriever = HybridRetriever(docs)
    pipeline = RAGPipeline(retriever)
    agent = BoundedRAGAgent(pipeline)

    question = "Why is reciprocal rank fusion useful for hybrid retrieval?"
    decision, answer = agent.run(question, top_k=3)
    print("decision:", decision)
    print("answer:", answer.answer)
    print("citations:", answer.citations)

    cases = [json.loads(line) for line in Path("data/sample_eval.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    print("retrieval metrics:", evaluate_retrieval(retriever, cases, top_k=3))


if __name__ == "__main__":
    main()
