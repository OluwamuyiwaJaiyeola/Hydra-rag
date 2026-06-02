import json
import pytest
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
from src.config import PINECONE_API_KEY, PINECONE_INDEX_NAME, HUGGINGFACE_MODEL


def load_test_data():
    with open("data/hydra_rag_test.json") as f:
        return json.load(f)


def load_compliance_questions():
    with open("data/hydra_compliance_questions.json") as f:
        return json.load(f)


@pytest.fixture(scope="session")
def embedding_model():
    return SentenceTransformer(HUGGINGFACE_MODEL)


@pytest.fixture(scope="session")
def pinecone_index():
    pc = Pinecone(api_key=PINECONE_API_KEY)
    return pc.Index(PINECONE_INDEX_NAME)


def retrieve_top_regulation(query: str, model, index, top_k: int = 3) -> list:
    query_vector = model.encode(query).tolist()
    results = index.query(vector=query_vector, top_k=top_k, include_metadata=True)
    return [int(m["metadata"]["regulation_id"]) for m in results["matches"]]


class TestRetrievalAccuracy:


    def test_specific_query_retrieval(self, embedding_model, pinecone_index):
        specific_queries = [
            {
                "query": "breach notification seventy-two hours supervisory authority",
                "expected_ids": [1, 6, 11, 16],
                "description": "72-hour breach notification"
            },
            {
                "query": "customer identity transaction records seven years retention",
                "expected_ids": [2, 7, 12, 17],
                "description": "7-year record retention"
            },
            {
                "query": "environmental incidents fifteen calendar days disclosure",
                "expected_ids": [3, 8, 13, 18],
                "description": "15-day environmental disclosure"
            },
            {
                "query": "biometric multifactor authentication clinical systems",
                "expected_ids": [4, 9, 14, 19],
                "description": "biometric MFA requirement"
            },
            {
                "query": "cyber incidents twenty-four hours essential services report",
                "expected_ids": [5, 10, 15, 20],
                "description": "24-hour cyber incident reporting"
            }
        ]

        passed = 0
        for case in specific_queries:
            retrieved = retrieve_top_regulation(
                case["query"], embedding_model, pinecone_index, top_k=3
            )
            if any(rid in retrieved for rid in case["expected_ids"]):
                passed += 1
                print(f"\nPASS: {case['description']} -> {retrieved}")
            else:
                print(f"\nFAIL: {case['description']}")
                print(f"  Expected any of: {case['expected_ids']}")
                print(f"  Retrieved: {retrieved}")

        accuracy = (passed / len(specific_queries)) * 100
        print(f"\nSpecific query accuracy: {accuracy:.1f}%")
        assert accuracy >= 60.0, f"Specific retrieval accuracy {accuracy:.1f}% below threshold"

    def test_index_is_populated(self, pinecone_index):
        stats = pinecone_index.describe_index_stats()
        total = stats["total_vector_count"]
        assert total == 240, f"Expected 240 vectors, found {total}"
        print(f"\nIndex vector count: {total}")

    def test_golden_dataset_retrieval(self, embedding_model, pinecone_index):
        test_data = load_test_data()
        compliance_questions = load_compliance_questions()
        question_map = {q["query_id"]: q["query_text"] for q in compliance_questions}

        # Build a map of query_text -> set of all valid regulation IDs
        # This handles the real-world case where one query applies to multiple regulations
        from collections import defaultdict
        query_to_valid_regulations = defaultdict(set)
        query_text_to_id = {}

        for case in test_data:
            qid = case["query_id"]
            query_text = question_map.get(qid)
            if query_text:
                query_to_valid_regulations[query_text].add(case["supporting_regulation_id"])
                query_text_to_id[query_text] = qid

        passed = 0
        failed = 0
        failed_cases = []

        for query_text, valid_ids in query_to_valid_regulations.items():
            retrieved_ids = retrieve_top_regulation(
                query_text, embedding_model, pinecone_index, top_k=5
            )

            if any(rid in retrieved_ids for rid in valid_ids):
                passed += 1
            else:
                failed += 1
                failed_cases.append({
                    "query": query_text,
                    "expected_any_of": sorted(valid_ids),
                    "retrieved_ids": retrieved_ids
                })

        total = passed + failed
        accuracy = (passed / total) * 100 if total > 0 else 0

        print(f"\nRetrieval Accuracy Results (deduplicated queries):")
        print(f"Unique queries tested: {total}")
        print(f"Passed: {passed}")
        print(f"Failed: {failed}")
        print(f"Accuracy: {accuracy:.1f}%")

        if failed_cases:
            print(f"\nFailed cases:")
            for fc in failed_cases:
                print(f"  Query: {fc['query'][:70]}...")
                print(f"    Expected any of: {fc['expected_any_of']}")
                print(f"    Retrieved: {fc['retrieved_ids']}")

        assert accuracy >= 20.0, f"Retrieval accuracy {accuracy:.1f}% is below minimum threshold of 20%"

    def test_semantic_search_returns_results(self, embedding_model, pinecone_index):
        test_queries = [
            "breach notification requirements",
            "data retention period",
            "annual independent review",
            "biometric data controls",
            "cyber incident reporting"
        ]

        for query in test_queries:
            retrieved = retrieve_top_regulation(query, embedding_model, pinecone_index, top_k=3)
            assert len(retrieved) > 0, f"No results returned for query: {query}"
            assert all(isinstance(r, int) for r in retrieved), "Regulation IDs must be integers"
            print(f"\nQuery: '{query}' -> Regulations: {retrieved}")

    def test_jurisdiction_filter_works(self, pinecone_index, embedding_model):
        query = "breach notification requirements"
        query_vector = embedding_model.encode(query).tolist()

        for jurisdiction in ["EU", "Brazil", "Kenya"]:
            results = pinecone_index.query(
                vector=query_vector,
                top_k=3,
                include_metadata=True,
                filter={"jurisdiction": {"$eq": jurisdiction}}
            )
            matches = results["matches"]
            if matches:
                for m in matches:
                    assert m["metadata"]["jurisdiction"] == jurisdiction, \
                        f"Filter failed: expected {jurisdiction}, got {m['metadata']['jurisdiction']}"
            print(f"\nJurisdiction filter '{jurisdiction}': {len(matches)} results returned")

    def test_relevance_scores_are_valid(self, embedding_model, pinecone_index):
        query_vector = embedding_model.encode("data breach notification").tolist()
        results = pinecone_index.query(vector=query_vector, top_k=5, include_metadata=True)

        for match in results["matches"]:
            score = match["score"]
            assert 0.0 <= score <= 1.0, f"Score {score} is outside valid range"

        scores = [m["score"] for m in results["matches"]]
        assert scores == sorted(scores, reverse=True), "Results not sorted by score descending"
        print(f"\nTop 5 scores: {[round(s, 4) for s in scores]}")


class TestAPIEndpoints:

    def test_compliance_qa_structure(self, embedding_model, pinecone_index):
        from src.rag_chain import retrieve_chunks, format_context
        question = "What are the breach notification requirements?"
        matches = retrieve_chunks(question, pinecone_index, embedding_model, top_k=2)
        context = format_context(matches)

        assert len(matches) > 0, "No chunks retrieved"
        assert len(context) > 0, "Context is empty"
        assert "regulation_id" in matches[0]["metadata"], "Missing regulation_id in metadata"
        assert "title" in matches[0]["metadata"], "Missing title in metadata"
        assert "chunk_text" in matches[0]["metadata"], "Missing chunk_text in metadata"
        print(f"\nRetrieved {len(matches)} chunks for compliance Q&A test")

    def test_search_returns_structured_results(self, embedding_model, pinecone_index):
        query_vector = embedding_model.encode("AML transaction monitoring").tolist()
        results = pinecone_index.query(vector=query_vector, top_k=3, include_metadata=True)
        matches = results["matches"]

        assert len(matches) > 0, "No search results returned"
        for m in matches:
            assert "regulation_id" in m["metadata"]
            assert "title" in m["metadata"]
            assert "jurisdiction" in m["metadata"]
            assert "category" in m["metadata"]
            assert "chunk_text" in m["metadata"]
        print(f"\nSearch returned {len(matches)} structured results")