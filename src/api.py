from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
from transformers import pipeline as hf_pipeline
from src.rag_chain import answer_compliance_question, retrieve_chunks, format_context
from src.config import HUGGINGFACE_MODEL
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
from src.config import PINECONE_API_KEY, PINECONE_INDEX_NAME
import logging
import time

logging.basicConfig(
    filename="hydra_queries.log",
    level=logging.INFO,
    format="%(asctime)s - %(message)s"
)

REGULATIONS = [
    {"id": 1, "title": "Digital Identity Protection Regulation", "jurisdiction": "Brazil", "category": "GDPR/Data Privacy"},
    {"id": 2, "title": "Suspicious Activity Disclosure Framework", "jurisdiction": "EU", "category": "AML/Financial Crime"},
    {"id": 3, "title": "Climate Risk Transparency Regulation", "jurisdiction": "USA", "category": "ESG Reporting"},
    {"id": 4, "title": "Clinical Information Governance Act", "jurisdiction": "USA", "category": "Healthcare Compliance"},
    {"id": 5, "title": "Critical Infrastructure Cyber Defense Regulation", "jurisdiction": "Brazil", "category": "Cybersecurity"},
    {"id": 6, "title": "Electronic Consent and Retention Framework", "jurisdiction": "Kenya", "category": "GDPR/Data Privacy"},
    {"id": 7, "title": "Financial Transaction Monitoring Order", "jurisdiction": "EU", "category": "AML/Financial Crime"},
    {"id": 8, "title": "Climate Risk Transparency Regulation", "jurisdiction": "EU", "category": "ESG Reporting"},
    {"id": 9, "title": "Clinical Information Governance Act", "jurisdiction": "USA", "category": "Healthcare Compliance"},
    {"id": 10, "title": "Network Resilience and Breach Notification Code", "jurisdiction": "Kenya", "category": "Cybersecurity"},
    {"id": 11, "title": "Electronic Consent and Retention Framework", "jurisdiction": "Brazil", "category": "GDPR/Data Privacy"},
    {"id": 12, "title": "Cross-Institutional AML Governance Act", "jurisdiction": "USA", "category": "AML/Financial Crime"},
    {"id": 13, "title": "Environmental Impact Governance Standard", "jurisdiction": "Kenya", "category": "ESG Reporting"},
    {"id": 14, "title": "Medical Provider Compliance Directive", "jurisdiction": "EU", "category": "Healthcare Compliance"},
    {"id": 15, "title": "Digital Systems Integrity Directive", "jurisdiction": "Brazil", "category": "Cybersecurity"},
    {"id": 16, "title": "Cross-Border Personal Information Control Act", "jurisdiction": "UK", "category": "GDPR/Data Privacy"},
    {"id": 17, "title": "Anti-Illicit Banking Standards Regulation", "jurisdiction": "USA", "category": "AML/Financial Crime"},
    {"id": 18, "title": "Corporate Sustainability Disclosure Code", "jurisdiction": "UK", "category": "ESG Reporting"},
    {"id": 19, "title": "Patient Safety and Audit Framework", "jurisdiction": "EU", "category": "Healthcare Compliance"},
    {"id": 20, "title": "Enterprise Security Governance Framework", "jurisdiction": "EU", "category": "Cybersecurity"}
]

app = FastAPI(
    title="Hydra Analytics Compliance Intelligence API",
    description="AI-powered regulatory compliance search and Q&A platform",
    version="1.0.0"
)

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load models once at startup, not per request
embedding_model = SentenceTransformer(HUGGINGFACE_MODEL)
from huggingface_hub import InferenceClient
from src.config import HUGGINGFACE_API_TOKEN, LLM_MODEL
remote_llm_client = InferenceClient(api_key=HUGGINGFACE_API_TOKEN)

pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX_NAME)

# Request and response models

class QueryRequest(BaseModel):
    question: str
    jurisdiction: Optional[str] = None
    top_k: Optional[int] = 2

class Citation(BaseModel):
    regulation_id: int
    title: str
    jurisdiction: str
    score: float
    article_ref: Optional[str] = "General provision"

class QueryResponse(BaseModel):
    question: str
    answer: str
    citations: list[Citation]
    regulation_ids: list[int]
    confidence: float = 0.0

class SearchRequest(BaseModel):
    query: str
    jurisdiction: Optional[str] = None
    top_k: Optional[int] = 5

class SearchResult(BaseModel):
    regulation_id: int
    title: str
    jurisdiction: str
    category: str
    chunk_text: str
    score: float

class SearchResponse(BaseModel):
    query: str
    results: list[SearchResult]
    total_found: int

class SummarizeRequest(BaseModel):
    regulation_id: int

class SummarizeResponse(BaseModel):
    regulation_id: int
    title: str
    summary: str

# Health check
@app.get("/")
def root():
    return {
        "platform": "Hydra Analytics Compliance Intelligence API",
        "status": "running",
        "version": "1.0.0",
        "endpoints": ["/compliance-qa", "/search", "/summarize", "/docs"]
    }

# Endpoint 1: Compliance Q&A
@app.post("/compliance-qa", response_model=QueryResponse)
def compliance_qa(request: QueryRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    start_time = time.time()

    try:
        filter_dict = None
        if request.jurisdiction:
            filter_dict = {"jurisdiction": {"$eq": request.jurisdiction}}

        # Use retrieve_chunks which includes deduplication
        matches = retrieve_chunks(
            query=request.question,
            index=index,
            model=embedding_model,
            top_k=request.top_k or 5
        )

        if not matches or matches[0]["score"] < 0.3:
            elapsed = round(time.time() - start_time, 3)
            logging.info(
                f"query='{request.question}' jurisdiction='{request.jurisdiction}' "
                f"top_score=0 result=no_match elapsed={elapsed}s"
            )
            return QueryResponse(
                question=request.question,
                answer="I cannot find relevant regulations to answer this question. Please refine your query or check the jurisdiction filter.",
                citations=[],
                regulation_ids=[],
                confidence=0.0
            )

        confidence = round(matches[0]["score"], 4)
        context = format_context(matches)

        prompt = f"""You are a regulatory compliance expert. Read the regulation text and answer the question in one clear sentence. Always cite the specific article or provision.

REGULATIONS:
{context}

QUESTION:
{request.question}

ANSWER:"""

        try:
            llm_result = remote_llm_client.chat_completion(
                messages=[
                    {
                        "role": "system",
                        "content": "You are a regulatory compliance expert. Answer using only the provided regulation text. Cite the specific article."
                    },
                    {"role": "user", "content": prompt}
                ],
                model=LLM_MODEL,
                max_tokens=300,
                temperature=0.1
            )
            answer = llm_result.choices[0].message.content
        except Exception as e:
            answer = f"Generation failed: {str(e)}"

        citations = [
            Citation(
                regulation_id=int(m["metadata"]["regulation_id"]),
                title=m["metadata"]["title"],
                jurisdiction=m["metadata"]["jurisdiction"],
                score=round(m["score"], 4),
                article_ref=m["metadata"].get("article_ref", "General provision"),
                chunk_text=m["metadata"].get("chunk_text", "")
            )
            for m in matches
        ]

        elapsed = round(time.time() - start_time, 3)
        logging.info(
            f"query='{request.question}' jurisdiction='{request.jurisdiction}' "
            f"top_score={confidence} "
            f"regulation_id={int(matches[0]['metadata']['regulation_id'])} "
            f"elapsed={elapsed}s"
        )

        return QueryResponse(
            question=request.question,
            answer=answer,
            citations=citations,
            regulation_ids=[int(m["metadata"]["regulation_id"]) for m in matches],
            confidence=confidence
        )

    except Exception as e:
        logging.error(f"query='{request.question}' error='{str(e)}'")
        raise HTTPException(status_code=500, detail=str(e))

# Endpoint 3: Regulation Summarization
@app.post("/summarize", response_model=SummarizeResponse)
def summarize_regulation(request: SummarizeRequest):
    try:
        # Retrieve chunks for this specific regulation
        results = index.query(
            vector=[0.0] * 384,
            top_k=100,
            include_metadata=True,
            filter={"regulation_id": {"$eq": float(request.regulation_id)}}
        )
        matches = results["matches"]

        if not matches:
            raise HTTPException(
                status_code=404,
                detail=f"Regulation ID {request.regulation_id} not found in index."
            )

        title = matches[0]["metadata"]["title"]
        combined_text = " ".join([m["metadata"]["chunk_text"] for m in matches])[:600]

        prompt = f"""Summarize this regulation in 3 clear sentences for a compliance analyst.

REGULATION: {title}
TEXT: {combined_text}

SUMMARY:"""

        try:
            llm_result = remote_llm_client.chat_completion(
                messages=[
                    {"role": "system", "content": "You are a regulatory compliance expert. Answer using only the provided regulation text. Cite the specific article."},
                    {"role": "user", "content": prompt}
                ],
                model=LLM_MODEL,
                max_tokens=128,
                temperature=0.1
            )
            summary = llm_result.choices[0].message.content
        except Exception as e:
            summary = f"Generation failed: {str(e)}"

        return SummarizeResponse(
            regulation_id=request.regulation_id,
            title=title,
            summary=summary
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

# Additional endpoint to list all regulations (for frontend dropdowns, etc.)
@app.get("/regulations")
def list_regulations():
    return {
        "regulations": REGULATIONS,
        "total": len(REGULATIONS)
    }