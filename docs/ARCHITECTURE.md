# Architecture and engineering boundaries

This public repository is a clean-room reference implementation. It borrows **engineering ideas**, not source code, from experience with private RAG/data-generation systems.

```text
Documents
   |
   +--> lexical index (BM25)
   |
   +--> optional dense embeddings
             |
             v
       ranked candidate lists
             |
             v
 Reciprocal Rank Fusion (RRF)
             |
             v
 source-diversity constraint
             |
             v
 optional cross-encoder reranker
             |
             v
 cited context construction
             |
             v
 OpenAI-compatible generator / vLLM
             |
             v
 grounded answer or abstention
```

## Why RRF instead of score addition

Raw BM25 scores and dense cosine similarities are not naturally calibrated. Adding them directly creates an implicit, corpus-dependent weighting problem. Reciprocal Rank Fusion combines rank positions instead, which is robust and easy to audit.

## Why retrieval evaluation is first-class

A RAG system can fail because retrieval missed the evidence even when the language model is capable. This repo includes Recall@k, MRR and nDCG@k so retrieval can be frozen and evaluated independently from answer generation.

## Agent boundary

The included agent is intentionally bounded: retrieve, answer, or abstain. It is not an autonomous loop. For enterprise systems, explicit orchestration is often easier to test, secure and reason about than unrestricted tool recursion.

## What would change in a larger production system

- persistent vector store / FAISS or managed vector database
- incremental indexing and document-version lineage
- async ingestion and backpressure
- metadata ACL filtering before retrieval
- query rewriting and multi-query retrieval
- observability: latency, hit-rate, zero-hit rate, citation coverage, token cost
- offline evaluation gates tied to a frozen benchmark
- auth, rate limiting and secret management

These are deliberately documented rather than faked in a small showcase.
