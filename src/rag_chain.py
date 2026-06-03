import warnings
import re
warnings.filterwarnings("ignore")

from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
from src.config import (
    PINECONE_API_KEY,
    PINECONE_INDEX_NAME,
    HUGGINGFACE_MODEL
)


def get_pinecone_index():
    pc = Pinecone(api_key=PINECONE_API_KEY)
    return pc.Index(PINECONE_INDEX_NAME)


def detect_category_prefix(query: str) -> str:
    """
    Add category context to query before embedding.
    Matches the prefix format used during ingestion so legal-bert
    can distinguish between compliance domains correctly.
    """
    query_lower = query.lower()
    if any(w in query_lower for w in [
        'data breach', 'material breach', 'breach notification',
        'data protection', 'privacy', 'gdpr',
        'personal information', 'data controllers', 'incident response'
    ]):
        return f"GDPR/Data Privacy: {query}"
    if any(w in query_lower for w in [
    'suspicious activity', 'aml', 'financial crime',
    'money laundering', 'transaction', 'financial intelligence',
    'due diligence', 'reporting threshold',
    'records retained', 'records be retained', 'retain records',
    'how long must', 'retention period',
    'maximum fine', 'civil penalties', 'license suspension',
    'twelve million', 'repeated reporting', 'reporting failures',
    'fine for violations', 'maximum fine for violations',
    'penalties for violations', 'fines for violations'
    ]):
        return f"AML/Financial Crime: {query}"  
    if any(w in query_lower for w in [
        'environmental', 'climate', 'esg', 'sustainability',
        'carbon', 'supply chain', 'disclosure'
    ]):
        return f"ESG Reporting: {query}"
    if any(w in query_lower for w in [
        'biometric', 'patient', 'healthcare', 'clinical',
        'hospital', 'medical', 'health'
    ]):
        return f"Healthcare Compliance: {query}"
    if any(w in query_lower for w in [
        'cyber', 'backup', 'vulnerability', 'network',
        'incident reporting', 'critical infrastructure', 'security controls'
    ]):
        return f"Cybersecurity: {query}"
    return query


def retrieve_chunks(query: str, index, model, top_k: int = 5) -> list:
    """
    Retrieve top-k unique regulation chunks for a query.
    Applies category prefix to query before embedding to improve
    legal-bert domain matching accuracy.
    """
    prefixed_query = detect_category_prefix(query)
    query_vector = model.encode(prefixed_query, normalize_embeddings=True).tolist()

    raw_results = index.query(
        vector=query_vector,
        top_k=20,
        include_metadata=True
    )
    matches = raw_results["matches"]

    # Deduplicate by chunk_text keeping highest score per unique clause
    seen_texts = set()
    deduplicated = []
    for m in matches:
        key = m["metadata"].get("chunk_text", "")[:80]
        if key not in seen_texts:
            seen_texts.add(key)
            deduplicated.append(m)
        if len(deduplicated) >= top_k:
            break

    # Apply relevance threshold
    RELEVANCE_THRESHOLD = 0.65
    filtered = [m for m in deduplicated if m["score"] >= RELEVANCE_THRESHOLD]

    return filtered if filtered else []


def format_context(matches: list) -> str:
    """Format retrieved chunks into a readable context string for the LLM."""
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
    """Build the prompt sent to Phi-3 for answer generation."""
    return f"""You are a regulatory compliance expert. Read the regulation text below and answer the question.
Rules:
1. Answer in one clear sentence
2. End your answer with the exact article reference in brackets like [Article 2.3] or [Section 5.7]
3. Use ONLY information from the regulation text provided

REGULATIONS:
{context[:800]}

QUESTION: {question}

ANSWER:"""