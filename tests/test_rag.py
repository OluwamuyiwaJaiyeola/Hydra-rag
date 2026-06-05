import json
import pytest
from collections import defaultdict
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
from src.config import PINECONE_API_KEY, PINECONE_INDEX_NAME, HUGGINGFACE_MODEL
from src.rag_chain import embed_query

GOLD_PATH    = "data/hydra_retrieval_gold.json"
METRICS_PATH = "data/retrieval_metrics.json"


def load_gold():
    with open(GOLD_PATH) as f:
        return json.load(f)


@pytest.fixture(scope="session")
def embedding_model():
    return SentenceTransformer(HUGGINGFACE_MODEL)


@pytest.fixture(scope="session")
def pinecone_index():
    pc = Pinecone(api_key=PINECONE_API_KEY)
    return pc.Index(PINECONE_INDEX_NAME)


def retrieve(query: str, model, index, top_k: int, filter_dict=None) -> list:
    """Return ranked list of {chunk_id, regulation_id} using the BGE query prefix
    and dedup-by-chunk_id (faithful index, clean result list)."""
    qv = embed_query(query, model)
    res = index.query(vector=qv, top_k=50, include_metadata=True, filter=filter_dict)
    out, seen = [], set()
    for m in res["matches"]:
        cid = m["metadata"].get("chunk_id")
        if cid in seen:
            continue
        seen.add(cid)
        out.append({"chunk_id": cid, "regulation_id": int(m["metadata"]["regulation_id"])})
        if len(out) >= top_k:
            break
    return out


def hit_family_or_shared(q, results, k):
    gold = set(q["gold_chunk_ids"])
    return any(r["chunk_id"] in gold for r in results[:k])


def hit_jurisdiction(q, results, k):
    gold_chunks = set(q["gold_chunk_ids"])
    gold_regs = set(q["valid_regulation_ids"])
    return any(r["chunk_id"] in gold_chunks and r["regulation_id"] in gold_regs
               for r in results[:k])


def score_group(queries, model, index, ks=(1, 3, 5), jurisdiction=False, apply_filter=False):
    scores = {f"R@{k}": 0 for k in ks}
    for q in queries:
        # apply_filter mirrors what the live API does: when the query names a
        # jurisdiction, narrow retrieval to it via metadata. This is the engineered
        # path; without it we measure raw semantics (which cannot disambiguate
        # identical clause text across jurisdictions).
        fdict = {"jurisdiction": {"$eq": q["jurisdiction"]}} if (apply_filter and q.get("jurisdiction")) else None
        results = retrieve(q["query_text"], model, index, max(ks), filter_dict=fdict)
        for k in ks:
            ok = hit_jurisdiction(q, results, k) if jurisdiction else hit_family_or_shared(q, results, k)
            scores[f"R@{k}"] += int(ok)
    n = len(queries) or 1
    return {m: round(v / n, 3) for m, v in scores.items()} | {"n": len(queries)}


class TestRetrievalScopeSplit:
    """Scores the gold retrieval set in three difficulty tiers and writes the
    metrics file the /evaluation-metrics endpoint serves. Tiers are reported
    SEPARATELY, never averaged, because they have different chance levels."""

    def test_index_is_populated(self, pinecone_index):
        stats = pinecone_index.describe_index_stats()
        total = stats["total_vector_count"]
        assert total == 200, f"Expected 200 vectors (30 clauses expanded per jurisdiction), got {total}"

    def test_scope_split_and_write_metrics(self, embedding_model, pinecone_index):
        gold = load_gold()
        fam_shared = gold["family_and_shared"]
        hard = gold["hard_jurisdiction"]

        family = [q for q in fam_shared if q["scope"] == "family"]
        shared = [q for q in fam_shared if q["scope"] == "shared"]

        report = {
            "model": HUGGINGFACE_MODEL,
            "family":       score_group(family, embedding_model, pinecone_index),
            "shared":       score_group(shared, embedding_model, pinecone_index),
            "jurisdiction": score_group(hard, embedding_model, pinecone_index, jurisdiction=True),
            "jurisdiction_filtered": score_group(hard, embedding_model, pinecone_index, jurisdiction=True, apply_filter=True),
            "notes": {
                "family": "valid across 4 regs; primary signal",
                "shared": "valid across all 20 regs; near-trivial, sanity check only",
                "jurisdiction": "raw semantics, no filter; cannot disambiguate identical clause text across jurisdictions",
                "jurisdiction_filtered": "metadata filter applied (the engineered path); this is what the live API does",
                "scoring": "tiers reported separately, never averaged",
            },
        }

        with open(METRICS_PATH, "w") as f:
            json.dump(report, f, indent=2)

        print("\n=== Retrieval@k by scope ===")
        for tier in ("family", "shared", "jurisdiction", "jurisdiction_filtered"):
            r = report[tier]
            print(f"  {tier:22} R@1 {r['R@1']:.3f}  R@3 {r['R@3']:.3f}  R@5 {r['R@5']:.3f}  (n={r['n']})")

        # Real thresholds per tier. Family is the primary signal; filtered jurisdiction
        # is the engineered result and should be high; unfiltered is the honest baseline.
        assert report["family"]["R@5"] >= 0.70, f"Family R@5 {report['family']['R@5']} below 0.70"
        assert report["jurisdiction_filtered"]["R@5"] >= 0.90, f"Filtered jurisdiction R@5 {report['jurisdiction_filtered']['R@5']} below 0.90"


class TestSanity:
    def test_semantic_search_returns_results(self, embedding_model, pinecone_index):
        for query in ["breach notification requirements", "data retention period",
                      "annual independent review", "biometric data controls",
                      "cyber incident reporting"]:
            r = retrieve(query, embedding_model, pinecone_index, top_k=3)
            assert len(r) > 0, f"No results for: {query}"

    def test_jurisdiction_filter_works(self, pinecone_index, embedding_model):
        qv = embed_query("breach notification requirements", embedding_model)
        for jur in ["EU", "Brazil", "Kenya"]:
            res = pinecone_index.query(vector=qv, top_k=3, include_metadata=True,
                                       filter={"jurisdiction": {"$eq": jur}})
            for m in res["matches"]:
                assert m["metadata"]["jurisdiction"] == jur

    def test_relevance_scores_valid_and_sorted(self, embedding_model, pinecone_index):
        qv = embed_query("data breach notification", embedding_model)
        res = pinecone_index.query(vector=qv, top_k=5, include_metadata=True)
        scores = [m["score"] for m in res["matches"]]
        assert all(0.0 <= s <= 1.0 for s in scores)
        assert scores == sorted(scores, reverse=True)


class TestAPIEndpoints:
    def test_compliance_qa_structure(self, embedding_model, pinecone_index):
        from src.rag_chain import retrieve_chunks, format_context
        matches = retrieve_chunks("What are the breach notification requirements?",
                                  pinecone_index, embedding_model, top_k=2)
        context = format_context(matches)
        assert len(matches) > 0 and len(context) > 0
        for field in ("regulation_id", "chunk_id", "chunk_text"):
            assert field in matches[0]["metadata"], f"Missing {field}"
