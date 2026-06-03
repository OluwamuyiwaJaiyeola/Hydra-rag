from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
from src.rag_chain import retrieve_chunks, format_context, detect_category_prefix
from src.config import HUGGINGFACE_MODEL
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
from src.config import PINECONE_API_KEY, PINECONE_INDEX_NAME
import logging
import time
import re
import re as _re
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import os

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
    top_k: Optional[int] = 5

class Citation(BaseModel):
    regulation_id: int
    title: str
    jurisdiction: str
    score: float
    article_ref: Optional[str] = "General provision"
    chunk_text: Optional[str] = ""

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
            top_k=request.top_k or 5,
            filter_dict=filter_dict
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

        detected = detect_category_prefix(request.question)
        if ':' in detected:
            cat = detected.split(':')[0].strip()
            cat_matches = [m for m in matches if m['metadata'].get('category') == cat]
            if len(cat_matches) >= 2:  # Only filter if we have enough category matches
                matches = cat_matches

        context = format_context(matches)

        # Detect if user asked about a specific jurisdiction
        
        jurisdiction_mentions = {
            'kenya': 'Kenya', 'KENYA': 'Kenya', 'eu': 'EU', 'european union': 'EU', 'European Union': 'EU',
            'brazil': 'Brazil', 'uk': 'UK', 'united kingdom': 'UK', 'United Kingdom': 'UK',
            'usa': 'USA', 'united states': 'USA', 'america': 'USA', 'United States': 'USA', 'America': 'USA',
        }
        question_lower = request.question.lower()
        asked_jurisdiction = None
        for term, jname in jurisdiction_mentions.items():
            if term in question_lower:
                asked_jurisdiction = jname
                break

        # Check if retrieved regulations match the asked jurisdiction
        retrieved_jurisdictions = list(set([m["metadata"]["jurisdiction"] for m in matches]))
        jurisdiction_mismatch = (
            asked_jurisdiction and
            asked_jurisdiction not in retrieved_jurisdictions
        )


        # Hallucination guard: detect fictional regulations and unknown jurisdictions
        FICTIONAL_SIGNALS = [
            'hydra compliance directive',
            'hydra directive',
            'australia', 'australian',
            'canada', 'canadian',
            'india', 'indian',
            'china', 'chinese',
            'japan', 'japanese',
            'singapore',
            'new zealand',
            'south africa',
        ]

        is_fictional_regulation = any(
            f' {signal} ' in f' {question_lower} ' or
            question_lower.startswith(signal) or
            question_lower.endswith(signal)
            for signal in FICTIONAL_SIGNALS
        )

        is_unknown_jurisdiction = (
            asked_jurisdiction and
            asked_jurisdiction not in {'Brazil', 'EU', 'Kenya', 'UK', 'USA'}
        )

        if is_fictional_regulation or is_unknown_jurisdiction:
            elapsed = round(time.time() - start_time, 3)
            logging.info(
                f"query='{request.question}' result=hallucination_blocked elapsed={elapsed}s"
            )
            if is_unknown_jurisdiction:
                answer_msg = f"There are no {asked_jurisdiction} regulations in the indexed database. The Hydra Analytics platform currently covers Brazil, EU, Kenya, UK, and USA only."
            else:
                answer_msg = "The regulation referenced in your query does not exist in the indexed database. The Hydra Analytics platform covers regulations for Brazil, EU, Kenya, UK, and USA only. Please refine your query."
            return QueryResponse(
                question=request.question,
                answer=answer_msg,
                citations=[],
                regulation_ids=[],
                confidence=0.0
            )

        jurisdiction_note = ""

        if jurisdiction_mismatch:
            closest_jurisdiction = matches[0]["metadata"]["jurisdiction"]
            closest_title = matches[0]["metadata"]["title"]
            jurisdiction_note = f"""
        IMPORTANT: The user asked about {asked_jurisdiction} specifically.
        There is NO {asked_jurisdiction}-specific regulation for this topic in the indexed database.
        The closest relevant regulation is: {closest_title} ({closest_jurisdiction}).
        You MUST begin your answer exactly like this:
        "There is no {asked_jurisdiction}-specific regulation for this topic in the indexed database. The closest relevant regulation is the {closest_title} ({closest_jurisdiction}), which states: [then give the answer and cite the article]"
        Do NOT say "and states". Do NOT mention any other jurisdiction."""

        prompt = f"""You are a regulatory compliance expert. Read the regulation text and answer the question accurately.
        Rules:
        1. Answer in one clear sentence
        2. End your answer with the EXACT reference as it appears in the source text. If the text says 'Clause 4.6' write [Clause 4.6]. If it says 'Section 2.1' write [Section 2.1]. If it says 'Provision 3.1' write [Provision 3.1]. Never change the reference type.
        3. Use ONLY information from the regulation text provided
        {jurisdiction_note}

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

        # Extract article reference from the generated answer
        answer_ref_match = re.search(
            r'(Article|Section|Clause|Provision|Rule)\s+[\d\.]+',
            answer,
            re.IGNORECASE
        )

        if answer_ref_match:
            used_ref = answer_ref_match.group(0).strip().lower()
            matching = [c for c in citations if used_ref in c.article_ref.lower()]
            if matching and asked_jurisdiction:
                # Prefer citation from the asked jurisdiction
                jurisdiction_match = [c for c in matching if c.jurisdiction == asked_jurisdiction]
                display_citations = jurisdiction_match if jurisdiction_match else matching
            else:
                display_citations = matching if matching else citations[:1]
        else:
            display_citations = citations[:1]

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
            citations=display_citations,
            regulation_ids=[int(m["metadata"]["regulation_id"]) for m in matches],
            confidence=confidence
        )

    except Exception as e:
        logging.error(f"query='{request.question}' error='{str(e)}'")
        raise HTTPException(status_code=500, detail=str(e))

# Endpoint 2: Semantic Search
@app.post("/search", response_model=SearchResponse)
def semantic_search(request: SearchRequest):
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    try:
        filter_dict = None
        if request.jurisdiction:
            filter_dict = {"jurisdiction": {"$eq": request.jurisdiction}}
        
        prefixed_query = detect_category_prefix(request.query)
        query_vector = embedding_model.encode(
            prefixed_query,
            normalize_embeddings=True
        ).tolist()

        raw_results = index.query(
            vector=query_vector,
            top_k=50,
            include_metadata=True,
            filter=filter_dict
        )
        matches = raw_results["matches"]

        # Deduplicate by regulation_id keeping all unique regulations first
        seen_reg_ids = set()
        deduplicated = []
        for m in matches:
            reg_id = m["metadata"].get("regulation_id")
            if reg_id not in seen_reg_ids:
                seen_reg_ids.add(reg_id)
                deduplicated.append(m)
        # Cap after category filtering, not before
        # If query has a detectable category, filter results to that category
        detected_prefix = detect_category_prefix(request.query)
        if ':' in detected_prefix:
            detected_category = detected_prefix.split(':')[0].strip()
            category_filtered = [m for m in deduplicated if m["metadata"].get("category") == detected_category]
            if len(category_filtered) >= 2:
                deduplicated = category_filtered[:request.top_k or 10]
            elif len(category_filtered) == 1:
                others = [m for m in deduplicated if m["metadata"].get("category") != detected_category]
                deduplicated = (category_filtered + others)[:request.top_k or 10]
            else:
                deduplicated = deduplicated[:request.top_k or 10]
        else:
            deduplicated = deduplicated[:request.top_k or 10]

        results = [
            SearchResult(
                regulation_id=int(m["metadata"]["regulation_id"]),
                title=m["metadata"]["title"],
                jurisdiction=m["metadata"]["jurisdiction"],
                category=m["metadata"]["category"],
                chunk_text=m["metadata"].get("chunk_text", ""),
                score=round(m["score"], 4)
            )
            for m in deduplicated
        ]

        return SearchResponse(
            query=request.query,
            results=results,
            total_found=len(results)
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Endpoint 3: Regulation Summarization
@app.post("/summarize", response_model=SummarizeResponse)
def summarize_regulation(request: SummarizeRequest):
    try:
        results = index.query(
            vector=[0.0] * 768,
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
        combined_text = " ".join([m["metadata"]["chunk_text"] for m in matches])[:800]

        prompt = f"""Summarize this regulation in 3 clear sentences for a compliance analyst.

REGULATION: {title}
TEXT: {combined_text}

SUMMARY:"""

        try:
            llm_result = remote_llm_client.chat_completion(
                messages=[
                    {
                        "role": "system",
                        "content": "You are a regulatory compliance expert. Summarize regulation documents clearly and concisely in exactly 3 sentences."
                    },
                    {"role": "user", "content": prompt}
                ],
                model=LLM_MODEL,
                max_tokens=300,
                temperature=0.1
            )
            summary = llm_result.choices[0].message.content
        except Exception as e:
            summary = f"Summarization failed: {str(e)}"

        return SummarizeResponse(
            regulation_id=request.regulation_id,
            title=title,
            summary=summary
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Endpoint 4: List all regulations
@app.get("/regulations")
def list_regulations():
    return {
        "regulations": REGULATIONS,
        "total": len(REGULATIONS)
    }


@app.get("/compliance-questions")
def get_compliance_questions():
    try:
        with open("data/hydra_compliance_questions_cleaned.json") as f:
            questions = json.load(f)
        seen = set()
        unique = []
        for q in questions:
            if q["query_text"] not in seen:
                seen.add(q["query_text"])
                unique.append(q["query_text"])
        return {"questions": unique, "total": len(unique)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

# Endpoint 5: Live index stats
@app.get("/stats")
def get_stats():
    stats = index.describe_index_stats()
    return {
        "total_vectors": stats["total_vector_count"],
        "total_regulations": 20,
        "jurisdictions": 5,
        "categories": 5,
        "embedding_model": HUGGINGFACE_MODEL,
        "dimensions": 768
    }

@app.get("/analytics")
def get_analytics():
    try:
        total_queries = 0
        hallucination_blocked = 0
        no_match = 0
        response_times = []
        similarity_scores = []

        with open("hydra_queries.log", "r") as f:
            for line in f:
                if "query='" not in line:
                    continue
                total_queries += 1
                if "hallucination_blocked" in line:
                    hallucination_blocked += 1
                if "no_match" in line:
                    no_match += 1
                # Extract elapsed time
                if "elapsed=" in line:
                    try:
                        elapsed = float(line.split("elapsed=")[1].split("s")[0])
                        response_times.append(elapsed)
                    except:
                        pass
                # Extract top_score
                if "top_score=" in line:
                    try:
                        score = float(line.split("top_score=")[1].split(" ")[0])
                        if score > 0:
                            similarity_scores.append(score)
                    except:
                        pass

        hallucination_rate = round(hallucination_blocked / total_queries * 100, 1) if total_queries > 0 else 0
        avg_response_time = round(sum(response_times) / len(response_times), 2) if response_times else 0
        avg_similarity = round(sum(similarity_scores) / len(similarity_scores), 3) if similarity_scores else 0

        return {
            "total_queries": total_queries,
            "hallucination_blocked": hallucination_blocked,
            "hallucination_rate": hallucination_rate,
            "no_match_rate": round(no_match / total_queries * 100, 1) if total_queries > 0 else 0,
            "avg_response_time": avg_response_time,
            "avg_similarity_score": avg_similarity
        }
    except Exception as e:
        return {
            "total_queries": 0,
            "hallucination_blocked": 0,
            "hallucination_rate": 0,
            "no_match_rate": 0,
            "avg_response_time": 0,
            "avg_similarity_score": 0
        }


# Serve frontend
@app.get("/app")
def serve_frontend():
    return FileResponse("hydra_frontend.html")