# api.py
# VAKALAT PRO: SECURE CLOUD BUILD (v7.1 - Clean)
# Features: Agentic Search + Verifier + Access Control
# NO HARDCODED SECRETS.

from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Optional
import os
from dotenv import load_dotenv
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from datetime import date

# Citation verification imports
from src.verification import (
    CitationVerifier,
    CitationGate,
    SectionValidator,
    LegalCodeMapper,
    OutdatedCodeDetector,
)
from src.verification.audit import configure_audit_logging

# 1. SETUP & SECRET LOADING
load_dotenv() # Loads .env for local testing

# CRITICAL: Verify Secrets Exist
required_keys = ["OPENAI_API_KEY", "PINECONE_API_KEY", "PINECONE_INDEX_NAME", "ACCESS_CODE"]
missing_keys = [k for k in required_keys if not os.getenv(k)]

if missing_keys:
    print(f"⚠️  CRITICAL WARNING: Missing Environment Variables: {missing_keys}")
    print("    System may fail to start or authenticate.")

# Get Access Code from Env (No default fallback)
ACCESS_CODE = os.getenv("ACCESS_CODE")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. THE BOUNCER (Security Middleware)
@app.middleware("http")
async def verify_access(request: Request, call_next):
    # Allow Health Checks & Pre-flight
    if request.url.path == "/" or request.method == "OPTIONS":
        return await call_next(request)
    
    # Strict Check
    token = request.headers.get("x-access-token")
    if not ACCESS_CODE:
        # Failsafe if env var is missing on server
        return JSONResponse(status_code=500, content={"detail": "Server Misconfiguration: No ACCESS_CODE set."})
        
    if token != ACCESS_CODE:
        return JSONResponse(status_code=401, content={"detail": "⛔ ACCESS DENIED: Invalid Security Code"})
    
    return await call_next(request)

# 3. BRAIN CONFIG
llm = ChatOpenAI(model="gpt-4o", temperature=0)
embeddings = OpenAIEmbeddings()

if os.getenv("PINECONE_API_KEY"):
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    vector_db = PineconeVectorStore(index_name=os.getenv("PINECONE_INDEX_NAME"), embedding=embeddings)
else:
    vector_db = None # Handle gracefully if offline

# Initialize verification infrastructure
code_mapper = LegalCodeMapper()
outdated_detector = OutdatedCodeDetector(code_mapper)

citation_verifier = None
section_validator = None
citation_gate = None
if vector_db:
    citation_verifier = CitationVerifier(vector_db)
    section_validator = SectionValidator(vector_db)
    citation_gate = CitationGate(citation_verifier, section_validator)

# Configure audit logging for verification events
configure_audit_logging()

# 4. BEHAVIORAL GATES
BEHAVIORAL_GATES = """
1. **The Four Corners Rule:** You must rely *exclusively* on the RETRIEVED CONTEXT.
2. **The Silence Protocol:** If the answer is not in the text, say "Information Not Found."
3. **Diagrams:** Use 

[Image of X]
 tags for processes.
"""

# 5. MODELS
class SearchPlan(BaseModel):
    queries: list[str] = Field(description="List of 3 distinct search queries")

class JuristResponse(BaseModel):
    direct_answer: str = Field(description="A short summary sentence.")
    analysis: str = Field(description="Detailed legal reasoning.")
    authorities: list[str] = Field(description="List of specific filenames/sections cited.")

class VerifierResponse(BaseModel):
    verdict: str = Field(description="'PASS' or 'FAIL'")
    reason: str = Field(description="Specific discrepancy found")
    sanitized_reply: str = Field(description="Safe fall-back reply if FAIL")

class QueryRequest(BaseModel):
    question: str

class DraftRequest(BaseModel):
    doc_type: str
    details: str

class ReviewRequest(BaseModel):
    doc_text: str

class VerifyCitationRequest(BaseModel):
    citation_type: str = Field(description="'case' or 'statute'")
    case_name: Optional[str] = None
    year: Optional[int] = None
    act_name: Optional[str] = None
    section: Optional[str] = None

class MapCodeRequest(BaseModel):
    old_code: str = Field(description="Old code name (IPC, CrPC, Evidence Act)")
    section: str = Field(description="Section number")

# 6. PROMPTS
PLANNER_PROMPT = ChatPromptTemplate.from_template("""
You are a Legal Search Strategist. Query: "{question}". Generate 3 search queries. Return JSON: {{ "queries": ["q1", "q2", "q3"] }}
""")
planner_parser = JsonOutputParser(pydantic_object=SearchPlan)

JURIST_PROMPT = ChatPromptTemplate.from_template("""
### SYSTEM INSTRUCTION: JURIST AGENT
TODAY: {current_date} | RULES: {gates} | LIBRARY: {context} | QUERY: {question}
OUTPUT JSON: {format_instructions}
""")
jurist_parser = JsonOutputParser(pydantic_object=JuristResponse)

VERIFIER_PROMPT = ChatPromptTemplate.from_template("""
### VERIFIER AGENT
SOURCE: {context} | DRAFT: {draft_answer}
AUDIT: 1. Hallucination Check. 2. Inference Check.
OUTPUT JSON: {format_instructions}
""")
verifier_parser = JsonOutputParser(pydantic_object=VerifierResponse)

DRAFT_PROMPT = ChatPromptTemplate.from_template("""
### ROLE: Senior Legal Drafter
DATE: {current_date} | DOC: {doc_type} | FACTS: {details} | LAW: {context}
AUDIT: If claim > 3 years old, warn TIME-BARRED.
OUTPUT: Pure HTML. Use inline CSS for Red Warning Box if needed.
""")

REVIEW_PROMPT = ChatPromptTemplate.from_template("""
### ROLE: Opposing Counsel
DOC: {doc_text} | LAW: {context}
OUTPUT: HTML bullet points. Red box for RISKS. Green box for COUNTER-ARGS.
""")

# 7. ENDPOINTS

@app.get("/")
def health(): return {"status": "VakalatOS Secure Cloud Online"}

@app.post("/research")
async def research(req: QueryRequest):
    if not vector_db: raise HTTPException(500, "Database Connection Failed")
    
    # 1. Plan
    plan = await (PLANNER_PROMPT | llm | planner_parser).ainvoke({"question": req.question})
    
    # 2. Retrieve
    unique_docs = {}
    for q in plan['queries']:
        results = vector_db.similarity_search(q, k=5)
        for doc in results:
            unique_docs[doc.page_content[:50]] = doc
    ctx = "\n\n".join([f"[Source: {d.metadata.get('source_id', 'Unknown')}] {d.page_content}" for d in unique_docs.values()])

    # 3. Reason
    jurist_out = await (JURIST_PROMPT | llm | jurist_parser).ainvoke({
        "context": ctx, "gates": BEHAVIORAL_GATES, "question": req.question, "current_date": str(date.today()), "format_instructions": jurist_parser.get_format_instructions()
    })

    # 4. Audit
    audit = await (VERIFIER_PROMPT | llm | verifier_parser).ainvoke({
        "draft_answer": str(jurist_out), "context": ctx, "format_instructions": verifier_parser.get_format_instructions()
    })

    if audit['verdict'] == "FAIL":
        return {"direct_answer": "Information Not Found", "analysis": audit['sanitized_reply'], "authorities": []}

    # 5. Detect outdated legal code references
    response = dict(jurist_out)
    if outdated_detector:
        outdated_result = outdated_detector.detect_outdated(jurist_out.get('analysis', ''))
        if outdated_result.has_outdated:
            response['outdated_codes'] = [
                {
                    'original': ref.original_text,
                    'suggestion': ref.suggestion
                }
                for ref in outdated_result.outdated_references
            ]

    return response

@app.post("/draft")
async def draft(req: DraftRequest):
    if not vector_db: raise HTTPException(500, "Database Connection Failed")

    ctx = "\n".join([d.page_content for d in vector_db.similarity_search(
        f"limitation mandatory requirements {req.doc_type}", k=8)])

    res = await (DRAFT_PROMPT | llm | StrOutputParser()).ainvoke({
        "doc_type": req.doc_type,
        "details": req.details,
        "context": ctx,
        "current_date": str(date.today())
    })

    draft_html = res.replace("```html", "").replace("```", "").strip()

    # Citation verification - filter and sanitize unverified citations
    verification_applied = False
    blocked_count = 0

    if citation_gate:
        # Filter both case and statute citations
        filtered = citation_gate.filter_all_citations(draft_html)
        if filtered.blocked:
            blocked_count = len(filtered.blocked)
            draft_html = citation_gate.sanitize_output(draft_html, filtered)
            verification_applied = True

    # Detect outdated codes in the generated draft
    outdated_codes_detected = False
    code_update_suggestions = ""
    if outdated_detector:
        outdated_in_draft = outdated_detector.detect_outdated(draft_html)
        if outdated_in_draft.has_outdated:
            outdated_codes_detected = True
            code_update_suggestions = outdated_detector.get_suggestions_html(outdated_in_draft)

    return {
        "draft": draft_html,
        "citations_verified": verification_applied,
        "citations_blocked": blocked_count,
        "outdated_codes_detected": outdated_codes_detected,
        "code_update_suggestions": code_update_suggestions
    }

@app.post("/analyze")
async def analyze(req: ReviewRequest):
    if not vector_db: raise HTTPException(500, "Database Connection Failed")
    ctx = "\n".join([d.page_content for d in vector_db.similarity_search(f"legality enforceability {req.doc_text[:200]}", k=10)])
    res = await (REVIEW_PROMPT | llm | StrOutputParser()).ainvoke({"doc_text": req.doc_text, "context": ctx})
    return {"analysis": res.replace("```html", "").replace("```", "").strip()}

# 8. VERIFICATION ENDPOINTS

@app.post("/verify-citation")
async def verify_citation(req: VerifyCitationRequest):
    """
    Verify a legal citation (case or statute).

    For case citations: Checks if the case exists in the database.
    For statute citations: Validates section exists and is not repealed.
    """
    if req.citation_type == "case":
        if not citation_verifier:
            raise HTTPException(500, "Citation verifier not available")
        result = citation_verifier.verify_case_citation(req.case_name, req.year)
        return result.model_dump()
    elif req.citation_type == "statute":
        if not section_validator:
            raise HTTPException(500, "Section validator not available")
        result = section_validator.validate_section(req.act_name, req.section)
        return result.model_dump()
    else:
        raise HTTPException(400, "citation_type must be 'case' or 'statute'")

@app.post("/map-code")
async def map_code(req: MapCodeRequest):
    """
    Map an old legal code section to its new equivalent.

    Converts IPC -> BNS, CrPC -> BNSS, Evidence Act -> BSA sections.
    """
    result = code_mapper.map_section(req.old_code, req.section)
    return result.model_dump()