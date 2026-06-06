# Hydra RAG Benchmark, rebuilt

This folder is a corrected, scorable version of the Hydra compliance retrieval
dataset. It was rebuilt to fix three structural problems in the original files:
duplicate-but-distinct regulations that made single-target scoring impossible,
gold labels paired with the wrong answers, and retrieval-unanswerable questions
mixed into the retrieval set.

---

## TL;DR, what to use

- Index: `hydra_corpus_gold.json`
- Score retrieval against: `hydra_retrieval_gold.json` using `score_retrieval.py`
- Hold for a later answer-quality eval: `hydra_generation_set.json`

Report Retrieval@1/3/5 **separately per scope**. Do not average across scopes.

---

## Provenance, which input became which output

Inputs are the files originally provided. Outputs are the files in this folder.

### Corpus

| Output | Records | Built from |
|---|---|---|
| `hydra_corpus_gold.json` | 30 clause chunks | `hydra_regulations_cleaned.json` (20 regs) |

The 20 regulations were 5 distinct bodies repeated across jurisdictions, plus one
shared Section-4 block identical in all 20. They collapse to 25 family clauses
(5 families x 5 clauses, each covering 4 regs) + 5 shared clauses (covering all 20).
`hydra_regulations.json` (the non-cleaned original) was used only for verification
and produced no output.

### Questions

Two input files were merged, then split by what each question actually is:

- `hydra_compliance_questions_cleaned.json` (50 questions)
- `hydra_rag_test_cleaned.json` (20 questions)

| Output | Records | Came from |
|---|---|---|
| `hydra_retrieval_gold.json` | 62 total | 27 compliance + 12 rag_test + 9 new disambiguated + 14 new jurisdiction hard cases |
| `hydra_generation_set.json` | 26 | 20 compliance + 6 rag_test (comparative/update) |
| `hydra_underspecified_quarantine.json` | 5 | 3 compliance + 2 rag_test (penalty/incident, no domain named) |

Reconciliation, nothing lost:
- compliance: 27 + 20 + 3 = 50
- rag_test: 12 + 6 + 2 = 20

The 9 disambiguated and 14 hard-case records are newly generated and trace to no
single input row.

### Retired inputs

Do not feed these to any evaluation. Their useful content now lives in the
outputs above.

- `hydra_compliance_questions.json` (original, mismatched labels)
- `hydra_rag_test.json` (original, mismatched labels)
- `hydra_compliance_questions_cleaned.json` (superseded by the three sorted files)
- `hydra_rag_test_cleaned.json` (superseded by the three sorted files)

Keep `hydra_regulations_cleaned.json` as the source of truth the corpus was built
from.

### Restored clause (Section 4.5)

The records-cooperation obligation ("provide requested records within ten business
days") was present verbatim in all 20 original regulations but dropped during
cleaning. It was restored from `hydra_regulations.json` as `SHARED:Section_4.5`
because 3 retrieval queries target it. Restoration, not fabrication: the clause
text is taken verbatim from the source. This took the corpus from 29 to 30 clauses
and the index from 180 to 200 vectors.

---

## File contents

### hydra_corpus_gold.json
Array of 29 chunks. Each chunk:
- `chunk_id` e.g. `DP:Article_2.3`, `SHARED:Section_4.1`
- `family`, `family_code`, `article_ref`, `text`
- `scope`: `family` (covers 4 regs) or `shared` (covers all 20)
- `covered_regulation_ids`: which original reg IDs contain this clause

### hydra_retrieval_gold.json
Object with two lists plus scoring notes.

`family_and_shared` (48 queries):
- 25 `family` scope, valid across 4 regs, your primary signal
- 23 `shared` scope, valid across all 20 regs, near-free, report as sanity only

`hard_jurisdiction` (14 queries):
- Same clause text in multiple jurisdictions. The retriever must disambiguate via
  jurisdiction metadata, not clause text.
- 8 `jurisdiction_exact` (one gold reg), 6 `jurisdiction_pair` (two gold regs,
  because some family+jurisdiction combos map to 2 regs).
- `is_uniquely_identifying` flags which is which.

Per-query fields: `query_text`, `gold_chunk_ids`, `valid_regulation_ids`,
`scope`, `difficulty`. Hard cases also carry `gold_regulation_ids`,
`jurisdiction`, `requires`.

Coverage note: the corpus does not place all 5 jurisdictions in every family, so
some family+jurisdiction hard cases cannot exist (e.g. no EU or USA GDPR
regulation). This is a data-generation gap in the source, not a labelling choice.

### hydra_generation_set.json
26 comparative/update queries with no static retrieval gold (they ask about
cross-jurisdiction comparison or version history the corpus does not contain).
Use for end-to-end answer-quality evaluation, not retrieval scoring.

---

## Scoring

Use `score_retrieval.py`. Wire the retriever into `retrieve(query_text, k)` so it
returns ranked `[{"chunk_id", "regulation_id"}]`. The harness computes
Retrieval@1/3/5 per scope:

- family: hit = any gold chunk in top-k
- shared: hit = the shared chunk in top-k (near-free, sanity check)
- jurisdiction: hit = gold chunk AND it resolves to a gold regulation_id

`build_gold.py` regenerates every output in this folder from the cleaned inputs.
