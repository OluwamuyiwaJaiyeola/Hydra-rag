from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
from src.rag_chain import retrieve_chunks, format_context, embed_query
from src.config import HUGGINGFACE_MODEL
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
from src.config import PINECONE_API_KEY, PINECONE_INDEX_NAME
import logging
import time
import re
from fastapi.responses import FileResponse
import json as _json
import os
from pathlib import Path

LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)
LOG_FILE = LOG_DIR / "hydra_queries.log"

logging.basicConfig(
    filename=str(LOG_FILE),
    level=logging.INFO,
    format="%(asctime)s - %(message)s"
)

with open("data/hydra_regulations_cleaned.json") as _f:
    _raw = __import__('json').load(_f)
REGULATIONS = [
    {
        "id": r["regulation_id"],
        "title": r["title"],
        "jurisdiction": r["jurisdiction"],
        "category": r["category"]
    }
    for r in _raw
]

# Lookup: which jurisdictions (and which family) carry each clause. Built from the
# index records, the authoritative map of clause -> regulations. Used to tell the
# user a clause is multi-jurisdictional instead of arbitrarily naming one copy.
import json as _json_boot
CHUNK_JURISDICTIONS = {}   # chunk_id -> sorted list of jurisdictions
CHUNK_FAMILY = {}          # chunk_id -> family/category name
_DEFINING_CLAUSE = {}      # regulation_id -> its unique family clause (for previews)
try:
    with open("data/hydra_index_records.json") as _irf:
        for _rec in _json_boot.load(_irf):
            _cid = _rec["chunk_id"]
            CHUNK_JURISDICTIONS.setdefault(_cid, set()).add(_rec["jurisdiction"])
            CHUNK_FAMILY[_cid] = _rec.get("category") or _rec.get("family")
            # the defining clause is the regulation's FAMILY-scope clause, not the
            # shared Section 4.x boilerplate that is identical across all regs
            if _rec.get("scope") == "family" and _rec["regulation_id"] not in _DEFINING_CLAUSE:
                _DEFINING_CLAUSE[_rec["regulation_id"]] = _rec["text"]
    CHUNK_JURISDICTIONS = {k: sorted(v) for k, v in CHUNK_JURISDICTIONS.items()}

    # For each regulation, find the chunk_id of its defining (family) clause, then
    # list every jurisdiction that shares that clause. This lets the UI label a card
    # honestly: "shared across N jurisdictions in this family" rather than looking
    # like accidental duplication.
    _REG_DEFINING_CHUNK = {}
    with open("data/hydra_index_records.json") as _irf2:
        for _rec in _json_boot.load(_irf2):
            if _rec.get("scope") == "family" and _rec["regulation_id"] not in _REG_DEFINING_CHUNK:
                _REG_DEFINING_CHUNK[_rec["regulation_id"]] = _rec["chunk_id"]

    for _r in REGULATIONS:
        _r["defining_clause"] = _DEFINING_CLAUSE.get(_r["id"], "")
        _chunk = _REG_DEFINING_CHUNK.get(_r["id"])
        _r["family_jurisdictions"] = CHUNK_JURISDICTIONS.get(_chunk, []) if _chunk else []
except FileNotFoundError:
    pass  # enrichment unavailable; responses still work

# Complete set of country/nationality terms that are NOT indexed jurisdictions.
# Generated from the full ISO country list (pycountry) minus the five indexed
# jurisdictions and their aliases. This replaces a hand-maintained partial list,
# which leaked: any unlisted country (e.g. Fiji) slipped through and got answered
# with clauses from indexed jurisdictions. With the complete list, the allowlist
# (Brazil/EU/Kenya/UK/USA) is the only thing that passes; every other country is
# refused.
UNINDEXED_COUNTRY_TERMS = set()
try:
    with open("data/unindexed_countries.json") as _ucf:
        UNINDEXED_COUNTRY_TERMS = set(_json_boot.load(_ucf))
except FileNotFoundError:
    pass  # falls back to allowlist-only check; see guard below

app = FastAPI(
    title="Hydra Analytics Compliance Intelligence API",
    description="AI-powered regulatory compliance search and Q&A platform",
    version="1.0.0"
)

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8000", "https://thatblvck-hydra-analytics.hf.space"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

from fastapi.responses import RedirectResponse

@app.get("/")
def root():
    return RedirectResponse(url="/app")

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
    confidence_calibrated: float = 0.0   # 0-1, mapped from BGE's score band for UI thresholding
    source_family: Optional[str] = None          # e.g. "GDPR/Data Privacy"
    applies_to_jurisdictions: list[str] = []      # all jurisdictions carrying the cited clause

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
        "endpoints": ["/compliance-qa", "/search", "/summarize", "/regulations", "/stats", "/analytics", "/evaluation-metrics", "/docs"]
    }

# Endpoint 1: Compliance Q&A
@app.post("/compliance-qa", response_model=QueryResponse)
def compliance_qa(request: QueryRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    start_time = time.time()

    try:
        # Resolve the jurisdiction filter. Priority: explicit request field first,
        # otherwise auto-detect from the question text. Naming a jurisdiction in the
        # question ("...under the EU regulation") narrows retrieval to that
        # jurisdiction's vectors, which is what makes near-duplicate clauses across
        # jurisdictions distinguishable (unfiltered ~0.57 vs filtered 1.0 on the
        # jurisdiction tier; identical clause text, only metadata differs).
        _JURISDICTION_TERMS = {
            'kenya': 'Kenya', 'eu': 'EU', 'european union': 'EU',
            'brazil': 'Brazil', 'uk': 'UK', 'united kingdom': 'UK',
            'usa': 'USA', 'united states': 'USA', 'america': 'USA',
        }
        _q_lower = request.question.lower()
        detected_jurisdiction = None
        for _term, _jname in _JURISDICTION_TERMS.items():
            if _term in _q_lower:
                detected_jurisdiction = _jname
                break

        active_jurisdiction = request.jurisdiction or detected_jurisdiction

        # ── Out-of-scope jurisdiction guard (runs BEFORE retrieval and the penalty
        # handler, so an unindexed jurisdiction cannot slip through and get answered
        # with clauses from indexed ones). The rule is INVERTED: the five indexed
        # jurisdictions are the allowlist; any other named country is refused. The
        # country set is the COMPLETE ISO list (loaded at startup) minus the indexed
        # five, so there are no gaps, Fiji, Bhutan, anything unlisted is caught.
        INDEXED = {'Brazil', 'EU', 'Kenya', 'UK', 'USA'}
        # Normalize the query: strip punctuation so a trailing "?" or "." cannot
        # defeat the match (e.g. "...in Fiji?" must still match the token "fiji").
        # Surround with spaces so token checks are word-boundary safe.
        _q_norm = ' ' + ' '.join(re.sub(r'[^a-z0-9 ]+', ' ', _q_lower).split()) + ' '
        named_unindexed = any((' ' + tok + ' ') in _q_norm for tok in UNINDEXED_COUNTRY_TERMS)
        out_of_scope = (
            (active_jurisdiction is not None and active_jurisdiction not in INDEXED)
            or (named_unindexed and active_jurisdiction is None)
        )
        if out_of_scope:
            elapsed = round(time.time() - start_time, 3)
            logging.info(f"query='{request.question}' result=out_of_scope_jurisdiction elapsed={elapsed}s")
            return QueryResponse(
                question=request.question,
                answer="That jurisdiction is not currently indexed in the Hydra Analytics platform. Current coverage is limited to Brazil, EU, Kenya, UK, and USA. Please rephrase your question for one of these jurisdictions.",
                citations=[], regulation_ids=[], confidence=0.0
            )

        # Detect a regulatory domain named in the question, so retrieval can be
        # constrained to that family. "fine for DATA PRIVACY violation" must not
        # surface AML or cyber penalties. Single source of truth, reused by the
        # penalty handler below.
        _DOMAIN_TERMS = {
            'data privacy': 'GDPR/Data Privacy', 'data protection': 'GDPR/Data Privacy',
            'gdpr': 'GDPR/Data Privacy',
            'aml': 'AML/Financial Crime', 'money laundering': 'AML/Financial Crime',
            'financial crime': 'AML/Financial Crime',
            'esg': 'ESG Reporting', 'environmental': 'ESG Reporting',
            'sustainability': 'ESG Reporting', 'climate': 'ESG Reporting',
            'healthcare': 'Healthcare Compliance', 'clinical': 'Healthcare Compliance',
            'patient': 'Healthcare Compliance', 'biometric': 'Healthcare Compliance',
            'cyber': 'Cybersecurity', 'cybersecurity': 'Cybersecurity',
        }
        detected_domain = None
        for _term, _fam in _DOMAIN_TERMS.items():
            if _term in _q_lower:
                detected_domain = _fam
                break

        filter_dict = {}
        if active_jurisdiction:
            filter_dict["jurisdiction"] = {"$eq": active_jurisdiction}
        if detected_domain:
            filter_dict["category"] = {"$eq": detected_domain}
        filter_dict = filter_dict or None

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
        # Calibrated confidence. Raw BGE cosine for correct rank-1 retrievals on this
        # corpus sits roughly in 0.55-0.78. A frontend thresholding raw cosine at,
        # say, 0.7 wrongly flags correct answers as "low confidence". Map BGE's
        # observed band to a 0-1 scale so the warning fires only on genuinely weak
        # retrieval. Linear map: <=0.45 -> 0, >=0.75 -> 1.
        _lo, _hi = 0.45, 0.75
        confidence_calibrated = round(min(1.0, max(0.0, (matches[0]["score"] - _lo) / (_hi - _lo))), 4)

        # Define question_lower early - used by multiple blocks below
        question_lower = request.question.lower()

        # NOTE: an earlier majority-category narrowing block was REMOVED here.
        # It took the plurality category among the top matches and discarded the
        # rest, which threw away correct rank-1 results when lower-ranked noise
        # happened to share a category (e.g. a #1 AML clause discarded because two
        # cyber clauses sat at #2 and #5). Raw retrieval already ranks the correct
        # clause first; we trust the ranking and do not second-guess it by category.

        context = format_context(matches)

        # ── Penalty/fine handler ───────────────────────────────────────────
        # Phi-3 cannot reliably combine multiple penalty clauses in one answer.
        # For broad penalty queries with no jurisdiction filter, build the answer
        # programmatically from the retrieved chunks.
        PENALTY_SIGNALS = [
            'maximum fine', 'maximum penalty', 'what is the fine',
            'what are the penalties', 'fine for violations',
            'penalties for violations', 'what penalties'
        ]
        # Detected domain is computed before retrieval (single source of truth).
        is_broad_penalty = (
            any(s in question_lower for s in PENALTY_SIGNALS)
            and not request.jurisdiction
            and not any(j in question_lower for j in ['kenya', 'eu', 'brazil', 'uk', 'usa', 'united states', 'united kingdom'])
        )

        if is_broad_penalty:
            penalty_keywords = ['percent', 'million', 'suspension', 'fine', 'penalt', 'sanction']
            penalty_chunks = [
                m for m in matches
                if any(kw in m['metadata'].get('chunk_text', '').lower() for kw in penalty_keywords)
            ]
            # If the question named a domain, keep only that family's penalty clauses.
            if detected_domain:
                domain_chunks = [m for m in penalty_chunks
                                 if m['metadata'].get('category') == detected_domain]
                if domain_chunks:
                    penalty_chunks = domain_chunks
            if len(penalty_chunks) >= 2:
                parts = []
                seen_refs = set()
                penalty_cites = []
                for chunk in penalty_chunks[:3]:
                    ref = chunk['metadata'].get('article_ref', '')
                    if ref in seen_refs:
                        continue
                    seen_refs.add(ref)
                    text = chunk['metadata'].get('chunk_text', '')
                    for sentence in text.split('.'):
                        if any(kw in sentence.lower() for kw in penalty_keywords):
                            parts.append(sentence.strip() + f' [{ref}].')
                            penalty_cites.append(Citation(
                                regulation_id=int(chunk['metadata']['regulation_id']),
                                title=chunk['metadata']['title'],
                                jurisdiction=chunk['metadata']['jurisdiction'],
                                score=round(chunk['score'], 4),
                                article_ref=ref,
                                chunk_text=text
                            ))
                            break
                if parts:
                    combined_answer = ' '.join(parts)
                    elapsed = round(time.time() - start_time, 3)
                    logging.info(
                        f"query='{request.question}' jurisdiction='{request.jurisdiction}' "
                        f"top_score={confidence} result=penalty_combined elapsed={elapsed}s"
                    )
                    return QueryResponse(
                        question=request.question,
                        answer=combined_answer,
                        citations=penalty_cites,
                        regulation_ids=[int(m['metadata']['regulation_id']) for m in penalty_chunks[:3]],
                        confidence=confidence
                    )
        # ── End penalty handler ────────────────────────────────────────────

        # Reuse the jurisdiction detected before retrieval (single source of truth).
        # active_jurisdiction reflects an explicit request field OR text detection.
        asked_jurisdiction = active_jurisdiction
        retrieved_jurisdictions = list(set([m["metadata"]["jurisdiction"] for m in matches]))
        jurisdiction_mismatch = (
            asked_jurisdiction and
            asked_jurisdiction not in retrieved_jurisdictions
        )


        # Hallucination guard: fictional/non-existent named regulations. (Out-of-scope
        # JURISDICTIONS are already handled by the early guard above, before retrieval.)
        FICTIONAL_REGULATIONS = ['hydra compliance directive', 'hydra directive']
        is_fictional_regulation = any(
            f' {signal} ' in f' {question_lower} ' or
            question_lower.startswith(signal) or
            question_lower.endswith(signal)
            for signal in FICTIONAL_REGULATIONS
        )

        if is_fictional_regulation:
            elapsed = round(time.time() - start_time, 3)
            logging.info(f"query='{request.question}' result=hallucination_blocked elapsed={elapsed}s")
            return QueryResponse(
                question=request.question,
                answer="The regulation referenced in your query does not exist in the indexed database. The Hydra Analytics platform covers regulations for Brazil, EU, Kenya, UK, and USA only.",
                citations=[], regulation_ids=[], confidence=0.0
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
        1. If the question asks about fines, penalties, or sanctions, mention ALL penalty figures found in the regulations provided. Use two sentences if needed.
        2. For all other questions, answer in one clear sentence.
        3. End your answer with the EXACT reference as it appears in the source text. If the text says 'Clause 4.6' write [Clause 4.6]. If it says 'Section 2.1' write [Section 2.1]. If it says 'Provision 3.1' write [Provision 3.1]. Never change the reference type.
        4. Use ONLY information from the regulation text provided.
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

        # Extract article references from the generated answer.
        # Non-capturing group so findall returns the FULL ref ("Section 3.4"),
        # not just the keyword ("Section"). The grouped version returned only
        # "Section", which then substring-matched every "Section x.y" clause and
        # attached unrelated citations.
        answer_refs = re.findall(
            r'(?:Article|Section|Clause|Provision|Rule)\s+\d+(?:\.\d+)*',
            answer,
            re.IGNORECASE
        )

        if answer_refs:
            display_citations = []
            seen_refs = set()
            for ref_match in answer_refs:
                used_ref = ref_match.strip().lower()
                if used_ref in seen_refs:
                    continue
                seen_refs.add(used_ref)
                matching = [c for c in citations if used_ref == c.article_ref.lower()]
                if not matching:
                    # fall back to prefix match only if no exact match exists
                    matching = [c for c in citations if c.article_ref.lower().startswith(used_ref)]
                if matching and asked_jurisdiction:
                    jurisdiction_match = [c for c in matching if c.jurisdiction == asked_jurisdiction]
                    display_citations.extend((jurisdiction_match or matching)[:1])
                elif matching:
                    display_citations.extend(matching[:1])
            if not display_citations:
                display_citations = citations[:1]
        else:
            display_citations = citations[:1]

        elapsed = round(time.time() - start_time, 3)
        logging.info(
            f"query='{request.question}' jurisdiction='{request.jurisdiction}' "
            f"top_score={confidence} "
            f"regulation_id={int(matches[0]['metadata']['regulation_id'])} "
            f"elapsed={elapsed}s"
        )

        # regulation_ids must reflect what the user actually sees in citations,
        # not the raw retrieved set. Building it from `matches` previously leaked
        # unfiltered, duplicated reg IDs (e.g. [6,10,6,6,13] for a Kenya-filtered
        # query). Derive it from display_citations, order-preserving dedup.
        _seen_reg = set()
        _reg_ids = []
        for c in display_citations:
            if c.regulation_id not in _seen_reg:
                _seen_reg.add(c.regulation_id)
                _reg_ids.append(c.regulation_id)

        # Multi-jurisdiction enrichment. For the primary cited clause, look up every
        # jurisdiction that carries it. When no specific jurisdiction was asked and
        # the clause spans several, this tells the user the obligation is shared
        # rather than implying the one arbitrarily-highest-scoring copy is special.
        source_family = None
        applies_to = []
        if display_citations:
            primary = display_citations[0]
            # find the chunk_id of the primary citation via the retrieved matches
            primary_chunk_id = None
            for m in matches:
                if (m["metadata"].get("article_ref", "").lower() == primary.article_ref.lower()
                        and int(m["metadata"]["regulation_id"]) == primary.regulation_id):
                    primary_chunk_id = m["metadata"].get("chunk_id")
                    break
            if primary_chunk_id:
                applies_to = CHUNK_JURISDICTIONS.get(primary_chunk_id, [])
                source_family = CHUNK_FAMILY.get(primary_chunk_id)
            # if a jurisdiction was explicitly asked/detected, scope the display to it
            if asked_jurisdiction and asked_jurisdiction in applies_to:
                applies_to = [asked_jurisdiction]

        return QueryResponse(
            question=request.question,
            answer=answer,
            citations=display_citations,
            regulation_ids=_reg_ids,
            confidence=confidence,
            confidence_calibrated=confidence_calibrated,
            source_family=source_family,
            applies_to_jurisdictions=applies_to,
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
        
        # BGE query embedding (instruction prefix applied inside embed_query).
        query_vector = embed_query(request.query, embedding_model)

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
        # Cap to top_k. Earlier code narrowed by plurality category here too; removed
        # for the same reason as /compliance-qa (it discarded correct rank-1 results).
        # The ranking is trusted as-is.
        topk = request.top_k or 10
        deduplicated = deduplicated[:topk]

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
        # Pure metadata fetch: the filter does the work, the vector only needs to be
        # valid. Use embed_query for consistency with the rest of the pipeline.
        dummy_vector = embed_query(f"regulation {request.regulation_id}", embedding_model)
        results = index.query(
            vector=dummy_vector,
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

# Endpoint 5: Live index stats
@app.get("/stats")
def get_stats():
    stats = index.describe_index_stats()
    sample = embedding_model.encode("dimension check", normalize_embeddings=True)
    return {
        "total_vectors": stats["total_vector_count"],
        "total_regulations": len(REGULATIONS),
        "jurisdictions": len(set(r["jurisdiction"] for r in REGULATIONS)),
        "categories": len(set(r["category"] for r in REGULATIONS)),
        "embedding_model": HUGGINGFACE_MODEL,
        "dimensions": len(sample)
    }

@app.get("/analytics")
def get_analytics():
    try:
        total_queries = 0
        hallucination_blocked = 0
        no_match = 0
        response_times = []
        similarity_scores = []

        with open(LOG_FILE, "r") as f:
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

@app.get("/evaluation-metrics")
def get_evaluation_metrics():
    try:
        metrics_path = "data/retrieval_metrics.json"
        if not os.path.exists(metrics_path):
            return {
                "available": False,
                "message": "No evaluation metrics available yet. Run the scope-split test to populate.",
                "family": None,
                "shared": None,
                "jurisdiction": None,
            }
        with open(metrics_path) as f:
            metrics = _json.load(f)

        # Scope-split metrics: report each tier separately, never averaged.
        # family = primary signal (4-reg valid sets)
        # shared = near-trivial sanity check (valid across all 20 regs)
        # jurisdiction = hard metadata disambiguation
        return {
            "available": True,
            "model": metrics.get("model"),
            "family": metrics.get("family"),
            "shared": metrics.get("shared"),
            "jurisdiction": metrics.get("jurisdiction"),
            "jurisdiction_filtered": metrics.get("jurisdiction_filtered"),
            "notes": metrics.get("notes"),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Serve frontend
@app.get("/app")
def serve_frontend():
    return FileResponse("hydra_frontend.html")