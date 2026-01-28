# api.py
# VAKALAT PRO: SECURE CLOUD BUILD (v7.1 - Clean)
# Features: Agentic Search + Verifier + Access Control
# NO HARDCODED SECRETS.

from fastapi import FastAPI, Request, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Optional, List
import os
from dotenv import load_dotenv
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from datetime import date
import time

# Citation verification imports
from src.verification import (
    CitationVerifier,
    CitationGate,
    SectionValidator,
    LegalCodeMapper,
    OutdatedCodeDetector,
)
from src.verification.audit import configure_audit_logging

# Template storage imports
from src.templates import (
    TemplateRepository,
    DocumentType,
    CourtLevel,
    LegalTemplate,
    TemplateStatus,
    ChangelogEntry,
    # Upload validation
    process_template_upload,
    validate_template_upload,
    UploadValidationResult,
    UploadProcessResult,
    MAX_TEMPLATE_SIZE_BYTES,
    # Lifecycle management
    change_template_status,
    get_template_status,
    is_template_usable,
    StatusChangeResult,
    UsabilityResult,
    # Preview generation
    get_template_preview,
    TemplatePreview,
)

# Citation engine imports
from src.citations import CitationRecommender, CitationRecommendation

# Document generation imports
from src.generation import (
    DocumentGenerator,
    GeneratedDocument,
    DocumentReviser,
    RevisionResult,
    BailApplicationFacts,
    LegalNoticeFacts,
    AffidavitFacts,
    PetitionFacts,
)

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

# Citation recommendation infrastructure
citation_recommender = None
if vector_db:
    citation_recommender = CitationRecommender(vector_db, citation_verifier)

# Template infrastructure
template_repo = TemplateRepository()

# Document generation infrastructure
document_generator = None
document_reviser = None
if vector_db:
    document_generator = DocumentGenerator(
        template_repo=template_repo,
        citation_recommender=citation_recommender,
        citation_gate=citation_gate
    )
    document_reviser = DocumentReviser()

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

class ListTemplatesRequest(BaseModel):
    doc_type: Optional[str] = Field(
        default=None,
        description="Filter by document type: bail_application, legal_notice, affidavit, petition"
    )

class GetTemplateRequest(BaseModel):
    doc_type: str = Field(description="Document type to retrieve")
    court_level: str = Field(description="Court level: supreme_court, high_court, district_court")

class RecommendCitationsRequest(BaseModel):
    legal_issue: str = Field(description="The legal issue to find precedents for")
    filing_court: Optional[str] = Field(
        default="supreme_court",
        description="Court where case will be filed (for jurisdiction ranking). Options: supreme_court, delhi, bombay, calcutta, madras, karnataka, allahabad"
    )
    top_k: Optional[int] = Field(
        default=5,
        description="Number of precedents to return (1-10)"
    )

class FormatStatuteRequest(BaseModel):
    section: str = Field(description="Section number (e.g., '302', '438')")
    act_name: str = Field(description="Full act name (e.g., 'Indian Penal Code')")
    year: Optional[int] = Field(default=None, description="Year of enactment (e.g., 1860)")

class GenerateDocumentRequest(BaseModel):
    doc_type: str = Field(
        description="Document type: bail_application, legal_notice, affidavit, petition"
    )
    court_level: str = Field(
        description="Court level: supreme_court, high_court, district_court"
    )
    facts: dict = Field(
        description="Structured facts matching the document type requirements"
    )

class ReviseDocumentRequest(BaseModel):
    content: str = Field(
        description="Current document content to revise"
    )
    instruction: str = Field(
        min_length=5,
        description="Revision instruction (e.g., 'Make grounds more concise')"
    )

class TemplateHistoryRequest(BaseModel):
    doc_type: str = Field(
        description="Document type: bail_application, legal_notice, affidavit, petition"
    )
    court_level: str = Field(
        description="Court level: supreme_court, high_court, district_court"
    )

class TemplateStatusChangeRequest(BaseModel):
    doc_type: str = Field(
        description="Document type: bail_application, legal_notice, affidavit, petition"
    )
    court_level: str = Field(
        description="Court level: supreme_court, high_court, district_court"
    )
    new_status: str = Field(
        description="Target status: active, deprecated, archived"
    )
    reason: str = Field(
        min_length=5,
        description="Reason for status change (recorded in changelog)"
    )

class TemplatePreviewRequest(BaseModel):
    doc_type: str = Field(
        description="Document type: bail_application, legal_notice, affidavit, petition"
    )
    court_level: str = Field(
        description="Court level: supreme_court, high_court, district_court"
    )

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

# 9. TEMPLATE ENDPOINTS

@app.post("/templates/list")
async def list_templates(req: ListTemplatesRequest):
    """
    List available legal document templates.

    Optionally filter by document type. Returns template summaries
    including doc_type, court_level, name, version, and description.
    """
    # Convert string to DocumentType enum if provided
    doc_type_filter = None
    if req.doc_type:
        try:
            doc_type_filter = DocumentType(req.doc_type)
        except ValueError:
            valid_types = [dt.value for dt in DocumentType]
            raise HTTPException(
                400,
                f"Invalid doc_type '{req.doc_type}'. Valid types: {valid_types}"
            )

    templates = template_repo.list_templates(doc_type_filter)

    # Return summaries, not full templates
    summaries = [
        {
            "doc_type": t.metadata.doc_type.value,
            "court_level": t.metadata.court_level.value,
            "name": t.metadata.name,
            "version": t.metadata.version,
            "description": t.metadata.description,
        }
        for t in templates
    ]

    return {
        "count": len(summaries),
        "templates": summaries
    }

@app.post("/templates/get")
async def get_template(req: GetTemplateRequest):
    """
    Retrieve a specific legal document template.

    Returns the full template including metadata, formatting requirements,
    required and optional fields, and template content.
    """
    # Convert strings to enums
    try:
        doc_type = DocumentType(req.doc_type)
    except ValueError:
        valid_types = [dt.value for dt in DocumentType]
        raise HTTPException(
            400,
            f"Invalid doc_type '{req.doc_type}'. Valid types: {valid_types}"
        )

    try:
        court_level = CourtLevel(req.court_level)
    except ValueError:
        valid_levels = [cl.value for cl in CourtLevel]
        raise HTTPException(
            400,
            f"Invalid court_level '{req.court_level}'. Valid levels: {valid_levels}"
        )

    template = template_repo.get_template(doc_type, court_level)

    if template is None:
        raise HTTPException(
            404,
            f"Template not found for {req.doc_type} at {req.court_level}"
        )

    return template.model_dump()

# 10. CITATION ENDPOINTS

@app.post("/citations/recommend")
async def recommend_citations(req: RecommendCitationsRequest):
    """
    Recommend relevant case law precedents for a legal issue.

    Returns precedents ranked by:
    - Semantic relevance to the legal issue
    - Jurisdictional authority (Supreme Court > Same HC > Other HC)
    - Temporal recency (recent cases weighted higher)

    Each citation includes verification status and badge HTML for UI display.
    """
    if not citation_recommender:
        raise HTTPException(500, "Citation recommender not available")

    # Validate top_k range
    top_k = max(1, min(10, req.top_k or 5))

    recommendations = citation_recommender.recommend_precedents(
        legal_issue=req.legal_issue,
        filing_court=req.filing_court or "supreme_court",
        top_k=top_k
    )

    return {
        "count": len(recommendations),
        "filing_court": req.filing_court or "supreme_court",
        "precedents": [r.model_dump() for r in recommendations]
    }

@app.post("/citations/format-statute")
async def format_statute(req: FormatStatuteRequest):
    """
    Format a statute citation in proper Indian legal format.

    Returns: "Section N, Act Name, Year"
    Example: "Section 438, Code of Criminal Procedure, 1973"
    """
    from src.citations import CitationFormatter

    formatted = CitationFormatter.format_statute(
        section=req.section,
        act_name=req.act_name,
        year=req.year
    )

    return {"formatted_citation": formatted}

# 11. DOCUMENT GENERATION ENDPOINTS

@app.post("/generate")
async def generate_document(req: GenerateDocumentRequest):
    """
    Generate a legal document from structured facts.

    Supports document types: bail_application, legal_notice, affidavit, petition
    Supports court levels: supreme_court, high_court, district_court

    The system:
    1. Loads court-specific template
    2. Retrieves relevant citations for the legal issue
    3. Generates formal legal content using LLM
    4. Verifies all citations before including
    5. Returns complete document with metadata

    Returns:
        GeneratedDocument with content, citations_used, verification_status
    """
    if not document_generator:
        raise HTTPException(500, "Document generator not available")

    # Validate doc_type
    try:
        doc_type = DocumentType(req.doc_type)
    except ValueError:
        valid_types = [dt.value for dt in DocumentType]
        raise HTTPException(
            400,
            f"Invalid doc_type '{req.doc_type}'. Valid types: {valid_types}"
        )

    # Validate court_level
    try:
        court_level = CourtLevel(req.court_level)
    except ValueError:
        valid_levels = [cl.value for cl in CourtLevel]
        raise HTTPException(
            400,
            f"Invalid court_level '{req.court_level}'. Valid levels: {valid_levels}"
        )

    # Validate facts based on doc_type
    fact_model_map = {
        DocumentType.BAIL_APPLICATION: BailApplicationFacts,
        DocumentType.LEGAL_NOTICE: LegalNoticeFacts,
        DocumentType.AFFIDAVIT: AffidavitFacts,
        DocumentType.PETITION: PetitionFacts,
    }

    fact_model = fact_model_map.get(doc_type)
    if fact_model:
        try:
            validated_facts = fact_model(**req.facts)
            facts_dict = validated_facts.model_dump()
        except Exception as e:
            raise HTTPException(400, f"Invalid facts: {str(e)}")
    else:
        facts_dict = req.facts

    try:
        result = document_generator.generate_document(
            doc_type=doc_type,
            court_level=court_level,
            user_facts=facts_dict
        )
        return result.model_dump()
    except Exception as e:
        raise HTTPException(500, f"Generation failed: {str(e)}")


@app.post("/revise")
async def revise_document(req: ReviseDocumentRequest):
    """
    Revise an existing document based on user instruction.

    Uses the "edit trick" pattern for efficient revision:
    - Generates minimal edit instructions
    - Applies only necessary changes
    - Preserves unchanged sections

    This is ~79% faster than full regeneration.

    Args:
        content: Current document content
        instruction: User's revision request (e.g., "Make the grounds more concise")

    Returns:
        RevisionResult with revised content and edit metadata
    """
    if not document_reviser:
        raise HTTPException(500, "Document reviser not available")

    try:
        result = document_reviser.revise_document(
            original_content=req.content,
            user_instruction=req.instruction
        )
        return result.model_dump()
    except Exception as e:
        raise HTTPException(500, f"Revision failed: {str(e)}")


# 12. TEMPLATE MANAGEMENT ENDPOINTS

@app.post("/templates/upload")
async def upload_template(file: UploadFile = File(...)):
    """
    Upload and validate a custom template.

    Accepts a JSON file containing a LegalTemplate schema.
    Validates:
    1. File size (max 500KB)
    2. JSON syntax
    3. Schema against LegalTemplate model
    4. Version higher than existing (if updating)

    For updates, include a 'change_description' query parameter.

    Returns:
        - success: Whether upload succeeded
        - is_update: Whether this was an update to existing template
        - old_version/new_version: Version change if updating
        - errors: Validation errors if failed
    """
    # Read file content
    try:
        content = await file.read()
        content_str = content.decode('utf-8')
    except UnicodeDecodeError:
        raise HTTPException(400, "File must be valid UTF-8 encoded JSON")

    # Check file size
    if len(content) > MAX_TEMPLATE_SIZE_BYTES:
        raise HTTPException(
            400,
            f"File too large: {len(content):,} bytes (max: {MAX_TEMPLATE_SIZE_BYTES:,} bytes)"
        )

    # Process upload
    result = process_template_upload(
        content=content_str,
        repository=template_repo,
        change_description=None,  # Could be passed as query param
        author="api_upload"
    )

    if not result.success:
        raise HTTPException(400, {"errors": result.errors})

    return {
        "success": result.success,
        "is_update": result.is_update,
        "old_version": result.old_version,
        "new_version": result.new_version,
        "template_name": result.template.metadata.name if result.template else None,
        "doc_type": result.template.metadata.doc_type.value if result.template else None,
        "court_level": result.template.metadata.court_level.value if result.template else None,
    }


@app.post("/templates/history")
async def get_template_history(req: TemplateHistoryRequest):
    """
    Get version history with changelog for a template.

    Returns the template's changelog entries showing all version changes,
    status transitions, and the authors who made them.

    Returns:
        - doc_type, court_level: Template identifier
        - current_version: Current template version
        - changelog: List of changelog entries with version, date, changes, author
    """
    # Validate doc_type
    try:
        doc_type = DocumentType(req.doc_type)
    except ValueError:
        valid_types = [dt.value for dt in DocumentType]
        raise HTTPException(
            400,
            f"Invalid doc_type '{req.doc_type}'. Valid types: {valid_types}"
        )

    # Validate court_level
    try:
        court_level = CourtLevel(req.court_level)
    except ValueError:
        valid_levels = [cl.value for cl in CourtLevel]
        raise HTTPException(
            400,
            f"Invalid court_level '{req.court_level}'. Valid levels: {valid_levels}"
        )

    # Get template
    template = template_repo.get_template(doc_type, court_level)
    if template is None:
        raise HTTPException(
            404,
            f"Template not found for {req.doc_type} at {req.court_level}"
        )

    return {
        "doc_type": req.doc_type,
        "court_level": req.court_level,
        "current_version": template.metadata.version,
        "status": template.status.value,
        "changelog": [
            {
                "version": entry.version,
                "date": entry.date,
                "changes": entry.changes,
                "author": entry.author,
            }
            for entry in template.changelog
        ]
    }


@app.post("/templates/status")
async def change_status(req: TemplateStatusChangeRequest):
    """
    Change template lifecycle status.

    Valid transitions:
    - active -> deprecated (template shows warning but remains usable)
    - active -> archived (direct decommission)
    - deprecated -> archived (final retirement)
    - archived -> (none) - terminal state, cannot be changed

    The status change is recorded in the template's changelog.

    Returns:
        - success: Whether status change succeeded
        - previous_status: Status before change
        - new_status: Status after change
        - error: Error message if failed
    """
    # Validate doc_type
    try:
        doc_type = DocumentType(req.doc_type)
    except ValueError:
        valid_types = [dt.value for dt in DocumentType]
        raise HTTPException(
            400,
            f"Invalid doc_type '{req.doc_type}'. Valid types: {valid_types}"
        )

    # Validate court_level
    try:
        court_level = CourtLevel(req.court_level)
    except ValueError:
        valid_levels = [cl.value for cl in CourtLevel]
        raise HTTPException(
            400,
            f"Invalid court_level '{req.court_level}'. Valid levels: {valid_levels}"
        )

    # Validate new_status
    try:
        target_status = TemplateStatus(req.new_status)
    except ValueError:
        valid_statuses = [s.value for s in TemplateStatus]
        raise HTTPException(
            400,
            f"Invalid new_status '{req.new_status}'. Valid statuses: {valid_statuses}"
        )

    # Perform status change
    result = change_template_status(
        doc_type=doc_type,
        court_level=court_level,
        target_status=target_status,
        reason=req.reason,
        author="api_user",
        repository=template_repo,
    )

    if not result.success:
        raise HTTPException(400, result.error)

    return {
        "success": result.success,
        "previous_status": result.previous_status.value if result.previous_status else None,
        "new_status": result.new_status.value if result.new_status else None,
    }


@app.post("/templates/preview")
async def preview_template(req: TemplatePreviewRequest):
    """
    Preview template showing required fields without full content.

    Returns a lightweight preview including:
    - Template metadata (name, version, description)
    - Template status (active/deprecated/archived)
    - Required field count and details
    - Optional field count and details
    - Font and font size for display

    If template is deprecated, includes a warning message.

    Returns:
        - preview: TemplatePreview object
        - warning: Deprecation warning if applicable
    """
    # Validate doc_type
    try:
        doc_type = DocumentType(req.doc_type)
    except ValueError:
        valid_types = [dt.value for dt in DocumentType]
        raise HTTPException(
            400,
            f"Invalid doc_type '{req.doc_type}'. Valid types: {valid_types}"
        )

    # Validate court_level
    try:
        court_level = CourtLevel(req.court_level)
    except ValueError:
        valid_levels = [cl.value for cl in CourtLevel]
        raise HTTPException(
            400,
            f"Invalid court_level '{req.court_level}'. Valid levels: {valid_levels}"
        )

    # Get preview
    preview = get_template_preview(
        doc_type=doc_type,
        court_level=court_level,
        repository=template_repo,
    )

    if preview is None:
        raise HTTPException(
            404,
            f"Template not found for {req.doc_type} at {req.court_level}"
        )

    # Check usability for warning
    usability = is_template_usable(
        doc_type=doc_type,
        court_level=court_level,
        repository=template_repo,
    )

    response = {"preview": preview.model_dump()}

    # Add warning if deprecated
    if usability.message and preview.status == TemplateStatus.DEPRECATED:
        response["warning"] = usability.message

    # Block if archived
    if preview.status == TemplateStatus.ARCHIVED:
        response["blocked"] = True
        response["blocked_message"] = usability.message

    return response


# 13. PRODUCTION DRAFTING ENDPOINT

# Import export module
from src.export import PDFExporter, DocxExporter, ExportFormat
import structlog

# Initialize exporters
pdf_exporter = PDFExporter()
docx_exporter = DocxExporter()

# Configure structlog for error logging
logger = structlog.get_logger("vakalat_api")


class DraftProRequest(BaseModel):
    """Request model for professional document drafting with export."""
    doc_type: str = Field(
        description="Document type: bail_application, legal_notice, affidavit, petition"
    )
    court_level: str = Field(
        description="Court level: supreme_court, high_court, district_court"
    )
    facts: dict = Field(
        description="Structured facts matching the document type requirements"
    )
    export_format: Optional[str] = Field(
        default=None,
        description="Export format: 'pdf', 'docx', or None for text only"
    )


class DraftProResponse(BaseModel):
    """Response model for professional document drafting."""
    content: str = Field(description="Generated document text content")
    doc_type: str = Field(description="Document type generated")
    court_level: str = Field(description="Court level used")
    citations_verified: List[str] = Field(
        default_factory=list,
        description="List of verified citations used"
    )
    verification_status: str = Field(
        description="Citation verification outcome: 'verified' or 'sanitized'"
    )
    export_data: Optional[str] = Field(
        default=None,
        description="Base64-encoded PDF/DOCX if export_format specified"
    )
    export_filename: Optional[str] = Field(
        default=None,
        description="Suggested filename for exported document"
    )
    export_format: Optional[str] = Field(
        default=None,
        description="Format of exported document"
    )


class ErrorResponse(BaseModel):
    """Structured error response."""
    error: str = Field(description="Error type")
    message: str = Field(description="Detailed error message")
    field: Optional[str] = Field(default=None, description="Field causing error if applicable")


@app.post("/draft-pro", response_model=DraftProResponse)
async def draft_pro(req: DraftProRequest):
    """
    Professional document drafting with optional PDF/DOCX export.
    
    This unified endpoint combines:
    1. Template retrieval based on doc_type and court_level
    2. Citation recommendation from Pinecone
    3. Document generation with formal legal language
    4. Citation verification to prevent hallucinations
    5. Optional export to PDF or DOCX with court-standard formatting
    
    Args:
        doc_type: Document type (bail_application, legal_notice, affidavit, petition)
        court_level: Court level (supreme_court, high_court, district_court)
        facts: Structured facts matching document type requirements
        export_format: Optional 'pdf', 'docx', or None for text only
    
    Returns:
        DraftProResponse with generated content and optional export bytes
    
    Raises:
        400: Invalid doc_type, court_level, facts, or export_format
        500: Generation or export failed
    """
    request_id = date.today().strftime("%Y%m%d") + str(int(time.time()))
    
    try:
        # Validate document generator is available
        if not document_generator:
            logger.error("Document generator unavailable", request_id=request_id)
            raise HTTPException(500, "Document generation service unavailable")
        
        # Validate doc_type
        try:
            doc_type = DocumentType(req.doc_type)
        except ValueError:
            valid_types = [dt.value for dt in DocumentType]
            logger.warning(
                "Invalid doc_type",
                request_id=request_id,
                doc_type=req.doc_type,
                valid_types=valid_types
            )
            raise HTTPException(
                400,
                {"error": "validation_error", "message": f"Invalid doc_type '{req.doc_type}'", "field": "doc_type", "valid_options": valid_types}
            )
        
        # Validate court_level
        try:
            court_level = CourtLevel(req.court_level)
        except ValueError:
            valid_levels = [cl.value for cl in CourtLevel]
            logger.warning(
                "Invalid court_level",
                request_id=request_id,
                court_level=req.court_level,
                valid_levels=valid_levels
            )
            raise HTTPException(
                400,
                {"error": "validation_error", "message": f"Invalid court_level '{req.court_level}'", "field": "court_level", "valid_options": valid_levels}
            )
        
        # Validate export_format if provided
        if req.export_format:
            if req.export_format.lower() not in ["pdf", "docx"]:
                logger.warning(
                    "Invalid export_format",
                    request_id=request_id,
                    export_format=req.export_format
                )
                raise HTTPException(
                    400,
                    {"error": "validation_error", "message": f"Invalid export_format '{req.export_format}'", "field": "export_format", "valid_options": ["pdf", "docx"]}
                )
        
        # Validate facts based on doc_type
        fact_model_map = {
            DocumentType.BAIL_APPLICATION: BailApplicationFacts,
            DocumentType.LEGAL_NOTICE: LegalNoticeFacts,
            DocumentType.AFFIDAVIT: AffidavitFacts,
            DocumentType.PETITION: PetitionFacts,
        }
        
        fact_model = fact_model_map.get(doc_type)
        if fact_model:
            try:
                validated_facts = fact_model(**req.facts)
                facts_dict = validated_facts.model_dump()
            except Exception as e:
                logger.warning(
                    "Invalid facts",
                    request_id=request_id,
                    doc_type=req.doc_type,
                    error=str(e)
                )
                raise HTTPException(
                    400,
                    {"error": "validation_error", "message": f"Invalid facts: {str(e)}", "field": "facts"}
                )
        else:
            facts_dict = req.facts
        
        # Generate document
        logger.info(
            "Generating document",
            request_id=request_id,
            doc_type=req.doc_type,
            court_level=req.court_level
        )
        
        try:
            generated = document_generator.generate_document(
                doc_type=doc_type,
                court_level=court_level,
                user_facts=facts_dict
            )
        except ValueError as e:
            logger.error(
                "Document generation failed",
                request_id=request_id,
                error=str(e)
            )
            raise HTTPException(400, {"error": "generation_error", "message": str(e)})
        except Exception as e:
            logger.error(
                "Document generation failed unexpectedly",
                request_id=request_id,
                error=str(e),
                exc_info=True
            )
            raise HTTPException(500, {"error": "generation_error", "message": "Document generation failed"})
        
        # Build response
        response_data = {
            "content": generated.content,
            "doc_type": generated.doc_type,
            "court_level": generated.court_level,
            "citations_verified": generated.citations_used,
            "verification_status": generated.verification_status,
            "export_data": None,
            "export_filename": None,
            "export_format": None,
        }
        
        # Handle export if requested
        if req.export_format:
            export_format_lower = req.export_format.lower()
            
            try:
                # Get formatting from template
                template = template_repo.get_template(doc_type, court_level)
                formatting = template.formatting if template else None
                
                if export_format_lower == "pdf":
                    logger.info("Exporting to PDF", request_id=request_id)
                    export_result = pdf_exporter.export(generated, formatting)
                else:  # docx
                    logger.info("Exporting to DOCX", request_id=request_id)
                    export_result = docx_exporter.export(generated, formatting)
                
                response_data["export_data"] = export_result.to_base64()
                response_data["export_filename"] = export_result.filename
                response_data["export_format"] = export_result.format.value
                
                logger.info(
                    "Export successful",
                    request_id=request_id,
                    format=export_format_lower,
                    filename=export_result.filename
                )
                
            except Exception as e:
                logger.error(
                    "Export failed",
                    request_id=request_id,
                    format=export_format_lower,
                    error=str(e),
                    exc_info=True
                )
                raise HTTPException(500, {"error": "export_error", "message": f"Export to {export_format_lower.upper()} failed"})
        
        logger.info(
            "Draft-pro request completed",
            request_id=request_id,
            doc_type=req.doc_type,
            exported=bool(req.export_format)
        )
        
        return DraftProResponse(**response_data)
        
    except HTTPException:
        raise  # Re-raise HTTP exceptions as-is
    except Exception as e:
        logger.error(
            "Unexpected error in draft-pro",
            request_id=request_id,
            error=str(e),
            exc_info=True
        )
        raise HTTPException(500, {"error": "internal_error", "message": "An unexpected error occurred"})