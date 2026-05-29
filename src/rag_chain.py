import warnings
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

def retrieve_chunks(query: str, index, model: SentenceTransformer, top_k: int = 2) -> list:
    query_vector = model.encode(query).tolist()
    results = index.query(vector=query_vector, top_k=top_k, include_metadata=True)
    return results["matches"]

def format_context(matches: list) -> str:
    context_parts = []
    for match in matches:
        meta = match["metadata"]
        context_parts.append(
            f"Regulation: {meta['title']}\n"
            f"Jurisdiction: {meta['jurisdiction']}\n"
            f"Content: {meta['chunk_text'][:300]}\n"
            f"Source ID: regulation_{int(meta['regulation_id'])}"
        )
    return "\n---\n".join(context_parts)

def build_prompt(context: str, question: str) -> str:
    return f"""Answer using the regulations below. Cite the regulation title.

REGULATIONS:
{context}

QUESTION:
{question}

ANSWER:"""

def load_local_llm():
    return pipeline(
        "text2text-generation",
        model="google/flan-t5-base",
        max_new_tokens=128
    )

def answer_compliance_question(question: str, llm_pipeline=None) -> dict:
    embedding_model = SentenceTransformer(HUGGINGFACE_MODEL)
    index = get_pinecone_index()

    matches = retrieve_chunks(question, index, embedding_model, top_k=2)
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
            "score": round(m["score"], 4)
        }
        for m in matches
    ]

    return {
        "question": question,
        "answer": answer,
        "sources": sources
    }