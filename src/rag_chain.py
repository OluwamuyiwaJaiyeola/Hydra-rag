import warnings
import re
warnings.filterwarnings("ignore")

from sentence_transformers import SentenceTransformer
from transformers import pipeline
from pinecone import Pinecone
from src.config import (
    PINECONE_API_KEY,
    PINECONE_INDEX_NAME,
    HUGGINGFACE_MODEL
)

def get_pinecone_index():
    pc = Pinecone(api_key=PINECONE_API_KEY)
    return pc.Index(PINECONE_INDEX_NAME)

def retrieve_chunks(query: str, index, model, top_k: int = 5) -> list:
    query_vector = model.encode(query, normalize_embeddings=True).tolist()

    # Fetch extra to allow for deduplication across jurisdictions
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
    RELEVANCE_THRESHOLD = 0.35
    filtered = [m for m in deduplicated if m["score"] >= RELEVANCE_THRESHOLD]

    return filtered if filtered else []

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
    return f"""You are a compliance expert. Read the regulation text below and answer the question in one clear sentence. Cite the specific obligation.

Regulation text:
{context[:600]}

Question: {question}

Answer in one sentence:"""

def load_local_llm():
    return pipeline(
        "text2text-generation",
        model="google/flan-t5-base",
        max_new_tokens=256
    )

def answer_compliance_question(question: str, llm_pipeline=None) -> dict:
    embedding_model = SentenceTransformer(HUGGINGFACE_MODEL)
    index = get_pinecone_index()

    matches = retrieve_chunks(question, index, embedding_model, top_k=5)

    if not matches:
        return {
            "question": question,
            "answer": "I cannot find relevant regulations to answer this question. The query may be outside the scope of indexed regulations.",
            "sources": []
        }

    context = format_context(matches)
    prompt_text = build_prompt(context, question)

    if llm_pipeline is None:
        llm_pipeline = load_local_llm()

    result = llm_pipeline(prompt_text, max_new_tokens=128)
    answer = result[0]["generated_text"]

    sources = [
        {
            "regulation_id": int(m["metadata"]["regulation_id"]),
            "title": m["metadata"]["title"],
            "jurisdiction": m["metadata"]["jurisdiction"],
            "score": round(m["score"], 4),
            "article_ref": m["metadata"].get("article_ref", "General provision")
        }
        for m in matches
    ]

    return {
        "question": question,
        "answer": answer,
        "sources": sources
    }