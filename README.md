# Hydra Compliance RAG

A retrieval-augmented question-answering system over multi-jurisdiction regulatory text. Answers compliance questions with grounded citations, distinguishes near-identical clauses across jurisdictions, and refuses to answer when the answer isn't in the corpus.

The emphasis of this project is **evaluation rigor**: the retrieval benchmark was rebuilt to measure something defensible, split into difficulty tiers, and used to catch bugs that an end-to-end test suite passed straight over.

---

## Architecture

```
Question
   │
   ▼
Scope guard ──► refuse if jurisdiction not in the indexed allowlist,
   │            or if a fictional regulation is named (runs before retrieval)
   ▼
Facet detection ──► jurisdiction + domain → metadata filter
   │                 (when the question names either)
   ▼
BGE query embedding ──► Pinecone (cosine, top-k, dedup by chunk_id)
   │
   ▼
Phi-3 generation (answer + article citation, grounded in retrieved context)
   │
   ▼
Response: answer, citations, regulation_ids, confidence,
          source_family, applies_to_jurisdictions
```

- **Embeddings:** `BAAI/bge-base-en-v1.5` (768-dim, contrastively trained; query-side instruction prefix applied, passages embedded plain)
- **Vector store:** Pinecone serverless, cosine
- **Generation:** `microsoft/Phi-3-mini-4k-instruct` via Hugging Face Inference API
- **Service:** FastAPI (serves both the API and the `/app` UI)
- **Eval:** pytest with tiered Retrieval@k scoring

---

## Why the embedding model matters here

The corpus originally used `legal-bert`, a domain-pretrained model with no retrieval head. It was a reasonable-sounding choice and the wrong one: masked-language-model pretraining teaches vocabulary, not query-passage similarity. Swapping to BGE (contrastively trained for retrieval) was the highest-impact change. See `docs/CASE_STUDY.md` for the debugging story, including why the swap initially *appeared* to make things worse.

---

## The corpus is non-trivial to evaluate

The 20 source "regulations" are **5 distinct bodies × jurisdictions**, plus a shared-obligations block identical across all 20. This breaks naive retrieval scoring:

- A clause about breach notification is valid in 4 regulations, not 1.
- A shared clause is valid in all 20.
- Two jurisdictions can carry **identical clause text**, distinguishable only by metadata.

So the benchmark scores **set membership**, not single-document match, and reports three tiers separately:

| Tier | n | What it tests | R@1 | R@3 | R@5 |
|---|---|---|---|---|---|
| family | 25 | clause valid across 4 regs | 0.48 | 1.00 | 1.00 |
| shared | 23 | clause valid across all 20 (near-trivial) | 0.70 | 0.87 | 1.00 |
| jurisdiction (raw) | 14 | semantic retrieval alone | 0.57 | 0.57 | 0.57 |
| jurisdiction (filtered) | 14 | + metadata filter (the live path) | **1.00** | **1.00** | **1.00** |

The jurisdiction rows are the point of the project: identical clause text across jurisdictions is structurally indistinguishable by embedding similarity (0.57, flat across k, more results never help), and a metadata filter resolves it completely (1.00). The contrast is the result, not either number alone.

Tiers are never averaged: they have different chance levels, and one number would hide the jurisdiction result.

---

## Metadata filtering, scope guards, and refusals

Three behaviours built on top of retrieval handle the cases pure semantic search gets wrong:

**Jurisdiction and domain filtering.** When a query names a jurisdiction ("...in Kenya") or a
domain ("data privacy fine"), that facet becomes a Pinecone metadata filter. This is the
legitimate use of keyword detection: applying a *structured constraint* the embedding cannot
infer, distinct from the keyword-routing anti-pattern that fakes semantic routing. It is what
turns the jurisdiction tier from 0.57 to 1.00.

**Out-of-scope guard (allowlist).** The indexed coverage is five jurisdictions (Brazil, EU,
Kenya, UK, USA). A query naming any other jurisdiction is refused *before retrieval*, so an
unindexed jurisdiction can never be answered with clauses from indexed ones. The rule is an
allowlist checked against the complete ISO country list, not a hand-maintained blocklist (a
blocklist leaks: any unlisted country slips through).

**Refusal handling.** Out-of-scope jurisdictions, fictional regulations, and no-match queries
return a clear refusal with zero confidence and no citations. The UI suppresses the
"verified / similarity" badge on refusals so a declined query never looks like an answer.

---

## Repo layout

```
src/
  ingest.py          build + upsert vectors from the gold corpus (per-jurisdiction)
  rag_chain.py       BGE query embedding, retrieval, dedup, prompt
  api.py             FastAPI endpoints (compliance-qa, search, summarize, metrics) + /app UI
  config.py
data/
  hydra_corpus_gold.json          30 deduplicated clause chunks (the source of truth)
  hydra_index_records.json        200 vectors (clauses expanded per jurisdiction)
  hydra_retrieval_gold.json       tiered retrieval benchmark
  hydra_generation_set.json       comparative/update questions (generation eval, not retrieval)
  retrieval_metrics.json          written by the eval run, served by /evaluation-metrics
  unindexed_countries.json        complete ISO country list (allowlist guard)
  hydra_regulations_cleaned.json  cleaned source regulations
tests/
  test_rag.py        tiered scope-split scoring + sanity + API structure
hydra_frontend.html  the /app UI (semantic search, Q&A, explorer, analytics)
diagnose.py          inspect failing tiers (shared failures, filtered jurisdiction)
isolate_retrieval.py strip the API stack, test raw retrieval alone
Dockerfile           Hugging Face Spaces deployment
docs/
  DATA.md            full data provenance (which source file became which output)
  CASE_STUDY.md      narrative version of the project and its debugging story
```

---

## Setup

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

cp .env.example .env        # fill in the keys
```

`.env`:

```
PINECONE_API_KEY=
PINECONE_INDEX_NAME=hydra-regulations
HUGGINGFACE_API_TOKEN=
HUGGINGFACE_MODEL=BAAI/bge-base-en-v1.5
LLM_MODEL=microsoft/Phi-3-mini-4k-instruct
```

`EMBEDDING_DIM=768` in `config.py` must match the embedding model. If you change models, delete and recreate the Pinecone index (dimension is fixed at creation).

---

## Run

```bash
python -m src.ingest                      # build + upsert 200 vectors
pytest tests/test_rag.py -v               # full suite
pytest tests/test_rag.py -k scope_split -s  # tiered metrics, writes retrieval_metrics.json

uvicorn src.api:app --reload --port 8000  # serve
```

Example, jurisdiction-filtered:

```bash
curl -s -X POST http://localhost:8000/compliance-qa \
  -H "Content-Type: application/json" \
  -d '{"question": "Under the Kenya data privacy regulation, what is the breach notification deadline?"}'
```

Returns the Kenya-specific regulation (reg 6), not whichever copy ranks first.

---

## Scope and honesty

The corpus is small (30 clauses) and synthetic; test queries are close paraphrases of clause text. Strong scores demonstrate the pipeline is **built and evaluated correctly**, not that it would generalise to a large, messy, real-world corpus. To move toward the latter, swap the synthetic bodies for real regulatory text; the pipeline is unchanged.

Two design notes worth knowing:

- **Facet detection (jurisdiction + domain) is keyword-based.** It sets a metadata filter, which is the legitimate use of keywords (applying a structured constraint semantics can't infer). This is distinct from the old keyword *router* that was removed, which faked retrieval routing and masked weak embeddings. A production version would use named-entity recognition for robust, language-agnostic detection.
- **`confidence` is raw cosine similarity.** It carries no calibrated probability meaning; a "low" 0.58 can still be a rank-1 correct retrieval. Displayed for transparency, not as a reliability score.
