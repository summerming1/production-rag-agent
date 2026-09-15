# Production RAG Agent

[![core-tests](https://github.com/summerming1/production-rag-agent/actions/workflows/ci.yml/badge.svg)](https://github.com/summerming1/production-rag-agent/actions/workflows/ci.yml)

A runnable, inspectable RAG engineering showcase focused on **retrieval quality, grounded generation, failure handling, and bounded orchestration**.

The core path runs without an API key, vector database, or GPU. Optional dense retrieval, cross-encoder reranking, FastAPI serving, and OpenAI-compatible/vLLM generation are isolated behind adapters.

## 15-second reviewer map

| Capability | Verify here |
|---|---|
| BM25 lexical retrieval | [`src/production_rag/bm25.py`](src/production_rag/bm25.py) |
| Dense retrieval adapter | [`src/production_rag/dense.py`](src/production_rag/dense.py) |
| RRF fusion + source diversity | [`src/production_rag/hybrid.py`](src/production_rag/hybrid.py) |
| Cross-encoder reranking | [`src/production_rag/rerank.py`](src/production_rag/rerank.py) |
| Citation / grounding checks | [`src/production_rag/citations.py`](src/production_rag/citations.py) |
| Retrieval evaluation | [`src/production_rag/eval.py`](src/production_rag/eval.py) |
| Grounded answer pipeline | [`src/production_rag/pipeline.py`](src/production_rag/pipeline.py) |
| Bounded agent / abstention | [`src/production_rag/agent.py`](src/production_rag/agent.py) |
| OpenAI-compatible generation | [`src/production_rag/generator.py`](src/production_rag/generator.py) |
| FastAPI contract | [`src/production_rag/api.py`](src/production_rag/api.py), [`tests/test_api.py`](tests/test_api.py) |
| Runnable behavior | [`tests/`](tests/) and [`scripts/run_demo.py`](scripts/run_demo.py) |

## What this demonstrates

- BM25 and optional dense retrieval combined with **Reciprocal Rank Fusion**
- source-diversity constraints after final ranking
- optional cross-encoder reranking over a **larger candidate pool**, not only the final top-k
- retrieval metrics: **Recall@k, MRR, nDCG@k**
- separate evaluation of initial retrieval and final post-rerank retrieval
- bounded context construction with source/page provenance
- structural citation validation that fails closed on unknown or missing citations
- explicit abstention; the bounded agent does **not call the generator** when retrieval policy rejects a request
- optional FastAPI `/retrieve`, `/rag/chat`, `/health` endpoints with request validation
- optional OpenAI-compatible generator suitable for local **vLLM**

A valid citation number proves only that the cited item was present in the frozen context. It does **not** by itself prove semantic support for every generated claim; evidence-faithfulness evaluation remains a separate layer.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
pytest -q
python scripts/run_demo.py
```

To run API contract tests and serve the example API:

```bash
pip install -e '.[dev,api]'
pytest -q tests/test_api.py
uvicorn 'production_rag.api:create_app' --factory --host 127.0.0.1 --port 8100
```

## Architecture

```text
Corpus
  +--> BM25 ------------------+
  |                           |
  +--> Dense (optional) ------+--> RRF --> candidate pool
                                         |
                                         v
                                  reranker (optional)
                                         |
                                         v
                                  source diversity
                                         |
                                         v
                                    final top-k
                                         |
                                         v
                                cited bounded context
                                         |
                                         v
                              local vLLM / API model
                                         |
                                         v
                              citation validation
                                         |
                              answer or abstain
```

See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for design trade-offs.

## Retrieval evaluation

The included sample set is intentionally small and demonstrates the mechanics rather than claiming benchmark quality.

```bash
PYTHONPATH=src python scripts/run_demo.py
```

`evaluate_retrieval()` measures the retriever stage. `evaluate_pipeline_retrieval()` measures the final stage after optional reranking. Keeping these separate helps locate whether a regression came from recall, fusion/reranking, or answer generation.

## Optional dense retrieval and reranking

```bash
pip install -e '.[dense,rerank]'
```

`HybridRetriever` keeps BM25 and dense score scales separate and fuses ranks with RRF. When a reranker is configured, `RAGPipeline` retrieves a larger pool first, reranks it, then applies source diversity before the final context is built.

## OpenAI-compatible / vLLM generation

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

The adapter sends deterministic-temperature chat requests. Authentication, retries, rate limiting, production observability, and model-serving lifecycle belong in deployment-specific infrastructure rather than being faked in this showcase.

## Engineering boundaries

Implemented and tested here: lexical retrieval, hybrid fusion, source diversity, citation structure checks, retrieval evaluation, explicit abstention, API validation, and bounded orchestration.

Optional real adapters: SentenceTransformers dense retrieval, CrossEncoder reranking, FastAPI, and OpenAI-compatible/vLLM generation.

Documented rather than claimed: persistent vector stores, ACL-aware retrieval, ingestion backpressure, incremental indexing, production authorization, distributed workers, release gates, and full semantic citation-faithfulness judging.

No private documents, prompts, answers, API credentials, or customer-specific logic are included. The implementation is a generic public showcase informed by private engineering experience, not a copy of a customer repository.

## Portfolio relevance

This repository supports remote work involving **RAG architecture, hybrid retrieval, retrieval/reranking evaluation, grounded generation, failure analysis, local/open-weight LLM integration, vLLM, API engineering, and bounded agent workflows**.

Related public work:

- [27B LLM Fine-Tuning, Evaluation & vLLM Deployment](https://github.com/summerming1/llm-posttraining-case-study)
- [Industrial CV Production Pipeline](https://github.com/summerming1/industrial-cv-production-pipeline)
