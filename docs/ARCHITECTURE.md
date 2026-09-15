# Architecture and engineering boundaries

This public repository is a clean-room reference implementation. It borrows engineering ideas, not source code, from experience with private RAG and post-training systems.

```text
Documents
   +--> BM25
   +--> optional dense retrieval
             |
             v
     reciprocal-rank fusion
             |
             v
       candidate pool
             |
             v
 optional cross-encoder reranking
             |
             v
    source-diversity constraint
             |
             v
         final top-k
             |
             v
 bounded cited context
             |
             v
 OpenAI-compatible generator / vLLM
             |
             v
 structural citation audit
             |
      answer or abstain
```

## Why RRF instead of score addition

Raw BM25 and dense-cosine scores are not naturally calibrated. RRF combines rank positions instead of assuming their numeric scales are comparable.

## Why rerank before final truncation

A second-stage reranker only adds value if it sees more than the final top-k. The pipeline therefore preserves a larger fused candidate pool, applies the reranker, and only then enforces source diversity and final context size.

## Why evaluation is split by stage

A RAG failure can come from initial recall, fusion/reranking, context construction, generation, or grounding policy. The public API exposes evaluation helpers for both the retriever stage and the final retrieval stage so those failures are not collapsed into one score.

## Grounding boundary

Citation validation is fail-closed for missing or out-of-range citation labels. This is a structural check: it verifies that a cited rank existed in the frozen context. It does not prove the cited passage semantically entails a generated claim. Full evidence-faithfulness evaluation requires a separate benchmark or judge protocol.

## Agent boundary

The agent is deliberately bounded. A retrieval-policy rejection returns an abstention without calling the generator. If generation runs but citation validation fails, the generated text is not returned as a successful answer.

The included retrieval-coverage value is a routing heuristic, not a calibrated probability of correctness.

## API boundary

FastAPI request models validate non-empty questions and bounded `top_k`. CI includes real HTTP/OpenAPI contract tests. Production authentication, authorization, rate limiting, retries, tracing, and multi-tenant corpus isolation are deployment concerns and are not claimed here.

## Larger-system extensions

- persistent vector store / FAISS or managed vector database
- incremental indexing and document-version lineage
- metadata ACL filtering before retrieval
- async ingestion, backpressure and retry policy
- query rewriting / multi-query retrieval
- semantic citation-faithfulness evaluation
- observability: stage latency, hit rate, zero-hit rate, abstention rate, citation failures and token cost
- frozen offline benchmark gates tied to release decisions
- auth, rate limiting, secret management and tenant isolation

These are documented as extensions rather than represented as completed production features.
