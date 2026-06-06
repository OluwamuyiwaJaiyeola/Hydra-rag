# Hydra: Building a Compliance RAG System You Can Actually Trust

**A retrieval-augmented question-answering system for multi-jurisdiction regulatory compliance, and the story of catching the bugs a passing test suite was hiding.**

---

## The one-line version

I built a RAG pipeline that answers regulatory compliance questions with cited sources across five jurisdictions. The interesting part wasn't the build. It was discovering that the evaluation set I'd been handed couldn't measure what it claimed to, rebuilding it from the ground up, and then catching three separate bugs that a green test suite was quietly hiding.

---

## The problem

Compliance teams ask questions like "what's the breach notification deadline?" and need an answer grounded in the actual regulation, with a citation, not a confident guess. The corpus spanned five jurisdictions (Brazil, EU, Kenya, UK, USA) across five regulatory domains (data privacy, financial crime, ESG, healthcare, cybersecurity). A working system has to return the right clause, attribute it to the right jurisdiction, and refuse to answer when the answer isn't there.

## What most people would have shipped

Wire an embedding model to a vector database, run the test suite, see it pass, call it done. I did exactly that, and the tests passed. Then I started pulling threads.

---

## Three things I caught that the tests didn't

### 1. The evaluation set was measuring the wrong thing

The benchmark scored retrieval against a single "correct" document. But the corpus wasn't 20 distinct regulations, it was 5 regulations repeated across jurisdictions, plus a block of shared text identical in all 20. So when the system correctly retrieved the EU copy of a clause and the answer key said "Brazil," it was marked wrong, for being right.

This is a non-identifiability problem: when four documents contain the same answer, you cannot score "did you find document 1" without penalising correct behaviour. I rebuilt the benchmark to score *set membership* (did you find any valid copy) and split it into difficulty tiers, because a clause that lives in all 20 regulations is trivially easy to find, while distinguishing two jurisdictions with identical text is genuinely hard. Averaging those into one number hides everything that matters.

### 2. I nearly reverted my best decision based on a false signal

I switched the embedding model to a stronger one, and the system's answers got *worse*. The obvious conclusion: the new model is bad, revert it.

I didn't trust the obvious conclusion. I wrote a ten-line script to test retrieval in isolation, stripped of every downstream processing step. The result: the new model retrieved the correct clause at rank one on every single failing query. The model was perfect. The corruption was happening *after* retrieval, in a filtering step I'd written myself, which was discarding the correct top-ranked result whenever lower-ranked noise happened to share a category. I deleted that step and every wrong answer became right.

The lesson I'd put on a wall: never diagnose from end-to-end output. Isolate the layer first. I was one commit away from reverting the single best fix in the project to cure a disease it didn't have.

### 3. The system contradicted itself in its own response

A jurisdiction-filtered query returned clean citations (all Kenya) but a summary field listing regulations from other jurisdictions entirely. Two fields in the same response, built from two different sources, disagreeing about which regulations were used. Correct retrieval, inconsistent assembly. The kind of thing that erodes trust in a demo the moment someone reads carefully. Fixed by deriving both fields from a single source.

---

## The structural insight that became a feature

Some questions can't be answered by meaning alone. "What does the Brazil regulation say about breaches?" and "What does the Kenya regulation say about breaches?" point at *identical clause text*. An embedding model sees the same sentence and cannot tell them apart, by design.

I measured this honestly: pure semantic search scored 57% on jurisdiction-specific questions and the score didn't improve no matter how many results I retrieved, because the information simply wasn't in the text. The fix wasn't a better model. It was recognising that jurisdiction is *structured metadata*, not semantic content, and applying it as a filter. With the filter, the same tier scored 100%. That contrast, 57% to 100%, is the difference between knowing when to reach for a model and when to reach for a constraint.

---

## What I'd claim, and what I wouldn't

**I'd claim:** this is a correctly engineered, rigorously evaluated RAG pipeline, and the evaluation is honest enough to show its own weak spots.

**I wouldn't claim** it's a high-accuracy production compliance system. The corpus is small and synthetic, and the test questions are close to the source text. Strong numbers here prove the pipeline is built correctly, not that it would survive a messy real-world corpus. Saying that plainly is part of the work, a benchmark you can't poke holes in is usually one you haven't looked at hard enough.

---

## Stack

Python, BGE embeddings (`bge-base-en-v1.5`), Pinecone vector database, Phi-3 for answer generation, FastAPI service layer, pytest evaluation harness with tiered scoring.
