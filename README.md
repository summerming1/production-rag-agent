# Production RAG Agent

[![core-tests](https://github.com/summerming1/production-rag-agent/actions/workflows/ci.yml/badge.svg)](https://github.com/summerming1/production-rag-agent/actions/workflows/ci.yml)

A **runnable, inspectable RAG engineering showcase** with hybrid retrieval, citations, retrieval evaluation, optional reranking, OpenAI-compatible generation, and bounded agent orchestration.

This repository is intentionally designed so a technical reviewer can run the core retrieval and tests **without an API key, vector database, or GPU**. Optional dense retrieval, cross-encoder reranking, FastAPI serving, and vLLM/OpenAI-compatible generation can be enabled separately.

## What it demonstrates

- **BM25 lexical retrieval** implemented in the repository
- Optional **dense retrieval** with SentenceTransformers
- **Reciprocal Rank Fusion (RRF)** for hybrid ranking instead of naïvely adding incompatible score scales
- Per-source diversity caps to avoid one long document dominating the context
- Optional **cross-encoder reranking**
- Source-preserving **citations**
- Retrieval metrics: **Recall@k, MRR, nDCG@k**
- Explicit **abstention** when retrieval returns no evidence
- Optional FastAPI `/retrieve`, `/rag/chat`, `/health` endpoints
- Optional OpenAI-compatible generator suitable for **vLLM** or another local server
- A deliberately **bounded agent** rather than an untestable autonomous loop

## Quick start: no API key

```bash
python -m venv .venv
source .venv/bin/activate
# If pytest is already available:
PYTHONPATH=src pytest -q
PYTHONPATH=src python scripts/run_demo.py
```

For a normal editable install:

```bash
pip install -e '.[dev]'
pytest -q
python scripts/run_demo.py
```

## Example architecture

```text
Corpus
  +--> BM25 ------------------+
  |                           |
  +--> Dense (optional) ------+--> RRF --> source diversity
                                        |
                                        v
                               reranker (optional)
                                        |
                                        v
                                cited context
                                        |
                                        v
                           local vLLM / API model
                                        |
                                        v
                               answer or abstain
```

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for design trade-offs.

## Run retrieval evaluation

The included demo evaluates a frozen sample set using Recall@k, MRR and nDCG@k. The point is not the toy corpus score; the point is that retrieval quality is treated as a separately measurable subsystem.

```bash
PYTHONPATH=src python scripts/run_demo.py
```

## Enable dense retrieval

```bash
pip install -e '.[dense]'
```

```python
from production_rag.dense import SentenceTransformerRetriever
from production_rag.hybrid import HybridRetriever
from production_rag.corpus import load_jsonl_corpus

docs = load_jsonl_corpus("data/sample_corpus.jsonl")
dense = SentenceTransformerRetriever(docs)
retriever = HybridRetriever(docs, dense=dense)
```

## Enable reranking

```bash
pip install -e '.[rerank]'
```

Use `CrossEncoderReranker` in `RAGPipeline`. Reranking is intentionally second-stage so expensive cross-encoder scoring is applied only to a small candidate set.

## Serve through FastAPI

```bash
pip install -e '.[api]'
uvicorn 'production_rag.api:create_app' --factory --host 127.0.0.1 --port 8100
```

## Connect to a local vLLM/OpenAI-compatible model

```bash
pip install -e '.[llm]'
```

```python
from production_rag.generator import OpenAICompatibleGenerator

generator = OpenAICompatibleGenerator(
    base_url="http://127.0.0.1:8000/v1",
    model="local-model",
)
```

## Why this repository is intentionally not overbuilt

A portfolio repo becomes less credible when it claims production features it cannot actually demonstrate. This project therefore separates:

- **implemented and runnable now**: BM25, hybrid RRF, citations, source diversity, retrieval evaluation, abstention, unit tests;
- **optional but real adapters**: SentenceTransformers dense retrieval, cross-encoder reranking, FastAPI, OpenAI-compatible/vLLM generation;
- **documented production extensions**: persistent vector stores, ACL filters, incremental indexing, distributed workers, observability and release gates.

No private documents, prompts, answers, API credentials, or customer-specific logic are included.

## Relationship to private work

The design is informed by experience building RAG components inside a larger domain-data and LLM post-training workflow, including hybrid retrieval, reranking, metadata-aware filtering, source diversity and retrieval auditing. The implementation in this public repo was written from scratch as a generic showcase.

## Portfolio relevance

This repository supports work involving **RAG architecture, hybrid retrieval, reranking, RAG evaluation, local/open-weight LLM deployment, vLLM integration, grounded generation, and agent orchestration**.
