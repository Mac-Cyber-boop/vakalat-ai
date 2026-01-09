# api.py
# VAKALAT PRO: SECURE CLOUD BUILD (v7.1 - Clean)
# Features: Agentic Search + Verifier + Access Control
# NO HARDCODED SECRETS.

from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import os
from dotenv import load_dotenv
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from datetime import date

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
    return jurist_out

@app.post("/draft")
async def draft(req: DraftRequest):
    if not vector_db: raise HTTPException(500, "Database Connection Failed")
    ctx = "\n".join([d.page_content for d in vector_db.similarity_search(f"limitation mandatory requirements {req.doc_type}", k=8)])
    res = await (DRAFT_PROMPT | llm | StrOutputParser()).ainvoke({
        "doc_type": req.doc_type, "details": req.details, "context": ctx, "current_date": str(date.today())
    })
    return {"draft": res.replace("```html", "").replace("```", "").strip()}

@app.post("/analyze")
async def analyze(req: ReviewRequest):
    if not vector_db: raise HTTPException(500, "Database Connection Failed")
    ctx = "\n".join([d.page_content for d in vector_db.similarity_search(f"legality enforceability {req.doc_text[:200]}", k=10)])
    res = await (REVIEW_PROMPT | llm | StrOutputParser()).ainvoke({"doc_text": req.doc_text, "context": ctx})
    return {"analysis": res.replace("```html", "").replace("```", "").strip()}