import warnings
warnings.filterwarnings("ignore")

from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
from src.config import (
    PINECONE_API_KEY,
    PINECONE_INDEX_NAME,
    HUGGINGFACE_MODEL
)

# BGE retrieval instruction. BGE was trained with this exact query-side prefix;
# passages are embedded WITHOUT a prefix (see ingest.py). Using it is not optional,
# it is how the model was trained to separate query space from passage space.
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


def get_pinecone_index():
    pc = Pinecone(api_key=PINECONE_API_KEY)
    return pc.Index(PINECONE_INDEX_NAME)


def embed_query(query: str, model: SentenceTransformer) -> list:
    """
    Embed a query with the BGE instruction prefix.

    NOTE: the old detect_category_prefix() keyword router was removed. It hand-mapped
    query words to a domain string and effectively hardcoded retrieval routing, which
    inflates scores and is not a real retriever. Domain disambiguation now comes from
    the embedding model plus optional metadata filters (e.g. jurisdiction), not from
    keyword rules.
    """
    text = BGE_QUERY_PREFIX + query
    return model.encode(text, normalize_embeddings=True).tolist()


def retrieve_chunks(query: str, index, model, top_k: int = 5, filter_dict: dict = None) -> list:
    """
    Retrieve top-k UNIQUE clauses for a query.

    Stored vectors are one-per-(clause, jurisdiction), so the same clause can appear
    multiple times. We over-fetch, then deduplicate by chunk_id keeping the highest
    score per clause. This is the dedup-at-scoring pattern: the index stays faithful
    and filterable, the result list stays clean.
    """
    query_vector = embed_query(query, model)

    raw_results = index.query(
        vector=query_vector,
        top_k=50,
        include_metadata=True,
        filter=filter_dict
    )
    matches = raw_results["matches"]

    seen_chunks = set()
    deduplicated = []
    for m in matches:
        key = m["metadata"].get("chunk_id") or m["metadata"].get("chunk_text", "")[:80]
        if key not in seen_chunks:
            seen_chunks.add(key)
            deduplicated.append(m)
        if len(deduplicated) >= top_k:
            break

    # Relevance threshold. BGE cosine scores run lower than a keyword-prefixed setup,
    # so 0.65 is too aggressive and will return empty. Tune on real data; start lower.
    RELEVANCE_THRESHOLD = 0.45
    filtered = [m for m in deduplicated if m["score"] >= RELEVANCE_THRESHOLD]

    return filtered if filtered else deduplicated[:top_k]


def format_context(matches: list) -> str:
    context_parts = []
    for match in matches:
        meta = match["metadata"]
        article = meta.get("article_ref", "")
        chunk = meta["chunk_text"][:400]
        if article and article != "General provision":
            context_parts.append(f"{article}: {chunk}")
        else:
            context_parts.append(chunk)
    return "\n\n".join(context_parts)


def build_prompt(context: str, question: str) -> str:
    return f"""You are a regulatory compliance expert. Read the regulation text below and answer the question.
Rules:
1. Answer in one clear sentence
2. End your answer with the exact article reference in brackets like [Article 2.3] or [Section 5.7]
3. Use ONLY information from the regulation text provided

REGULATIONS:
{context[:800]}

QUESTION: {question}

ANSWER:"""