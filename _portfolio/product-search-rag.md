---
title: "ShopSearch RAG"
collection: portfolio
date: 2026-06-03
tagline: "Natural-language product search for e-commerce — hybrid retrieval (BM25 + dense), structured attribute filtering, and an LLM re-ranker."
thumbnail: # /images/projects/shopsearch-rag.png    # optional: drop a screenshot or GIF in /images/projects/ and uncomment
tech:
  - RAG
  - LLM
  - Hybrid Retrieval
  - Qdrant
  - FastAPI
code_url:  "#"   # TODO: replace with https://github.com/hanntian/shopsearch-rag
demo_url:  "#"   # TODO: replace with live demo URL
# paper_url: ""    # leave empty to hide the Paper button
blog_url:  "#"   # TODO: replace with /posts/2026/06/blog-post-shopsearch/
excerpt: |
  🛒 **TL;DR** — A retrieval-augmented product search system for e-commerce that turns natural-language shopping queries (*"a waterproof backpack under $80 for hiking"*) into ranked product results by combining hybrid retrieval (BM25 + dense embeddings), structured attribute filtering, and an LLM re-ranker.
---

<div class="notice--info" markdown="1">
🛒 **TL;DR** — A retrieval-augmented product search system for e-commerce that turns natural-language shopping queries (*"a waterproof backpack under $80 for hiking"*) into ranked product results by combining hybrid retrieval (BM25 + dense embeddings), structured attribute filtering, and an LLM re-ranker.
</div>

## 1. Background & Motivation

Traditional keyword search on e-commerce sites struggles with descriptive, intent-rich queries such as *"a gift for a 6-year-old who likes dinosaurs"* or *"running shoes for flat feet, size 10, under $120"*. These queries mix unstructured intent, structured attributes (size, price, color), and implicit constraints. The goal of this project is to build a search pipeline that:

- understands the user's intent in natural language,
- respects hard constraints (price, in-stock, size availability),
- and returns a small, well-ranked list of candidate products with explanations.

## 2. Problem Definition

> Given a user query `q` (free text) and a product catalog of `N` items with structured attributes and unstructured descriptions, return the top-`k` products that best satisfy `q`, along with a short rationale for each.

Key challenges:
- High recall on long-tail / synonym-heavy queries.
- Hard constraints must be enforced, not "soft-ranked".
- Latency budget: end-to-end p95 < TBD ms.
- Cost: LLM calls per query must be bounded.

## 3. System Architecture

```
User Query
    │
    ▼
┌──────────────────────┐
│ Query Understanding  │  LLM extracts: intent, filters,
│ (LLM, function call) │  rewritten search query
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐    ┌──────────────────────┐
│ Hybrid Retrieval     │◀──▶│ Vector DB (dense)    │
│ BM25 + Dense + Filter│    │ Inverted index (BM25)│
└──────────┬───────────┘    └──────────────────────┘
           │
           ▼
┌──────────────────────┐
│ LLM Re-ranker        │  Cross-encoder / LLM scorer
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│ Answer Composer      │  Top-k results + rationale
└──────────────────────┘
```

## 4. Data

- **Catalog**: TBD products (source: ___), with fields {title, description, category, brand, price, attributes, image}.
- **Query set**: TBD synthetic + real anonymized queries for evaluation.
- **Labels**: relevance judgments collected via TBD.

## 5. Technical Stack

| Layer | Choice | Rationale |
|---|---|---|
| Embedding model | TBD (e.g., bge-large, gte) | balance recall vs. cost |
| Vector store | TBD (e.g., Qdrant / pgvector) | filter pushdown matters |
| Keyword index | OpenSearch / Elasticsearch | proven BM25 + filter |
| Re-ranker | TBD (cross-encoder or LLM) | precision@k boost |
| LLM | TBD (e.g., GPT-4o-mini / Claude Haiku) | latency + cost |
| Serving | FastAPI + async | streaming responses |

## 6. Implementation Notes

- *Query understanding*: prompt the LLM to output a JSON schema (rewritten query, filters, must/should). Function-calling preferred over free-text JSON for reliability.
- *Hybrid scoring*: linear combination `score = α · BM25 + (1−α) · cosine`, with `α` tuned per query type.
- *Filter enforcement*: hard filters applied before retrieval; soft preferences fed into re-ranker prompt.
- *Caching*: query → (rewritten query, candidates) cached to absorb repeats.

## 7. Evaluation

- **Retrieval**: Recall@50, MRR@10 on labeled query set.
- **Re-ranking**: nDCG@10, Precision@5.
- **Business proxy**: simulated click-through rate (CTR) and conversion lift vs. BM25-only baseline.
- **Latency / cost**: p50 / p95 end-to-end latency, $/1k queries.

| Metric | Baseline (BM25) | + Dense | + Re-rank |
|---|---|---|---|
| Recall@50 | TBD | TBD | TBD |
| nDCG@10  | TBD | TBD | TBD |
| p95 latency | TBD | TBD | TBD |

## 8. Lessons Learned

- TBD — what surprised me, what didn't work, design trade-offs.

## 9. Future Work

- Personalization signals (history, preferences).
- Multi-modal: image + text retrieval.
- Online learning from clicks.

## 10. Links

- **Code**: [GitHub repo](https://github.com/hanntian/REPO_NAME)
- **Demo**: [Live demo](https://example.com)
- **Write-up**: [Blog post]({{ site.baseurl }}/posts/2026/06/...)
