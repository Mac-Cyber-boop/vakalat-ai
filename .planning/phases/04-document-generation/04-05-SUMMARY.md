# 04-05 Summary: Document Generation API Endpoints

## What Was Built
FastAPI endpoints for document generation and revision, completing the Phase 4 integration. Exposes DocumentGenerator and DocumentReviser capabilities via `/generate` and `/revise` endpoints.

## Files Changed
| File | Change |
|------|--------|
| api.py | Added generation imports, initialization, request models, and endpoints |

## Key Implementation Details

### Imports Added
```python
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
```

### Infrastructure Initialization
```python
document_generator = None
document_reviser = None
if vector_db:
    document_generator = DocumentGenerator(
        template_repo=template_repo,
        citation_recommender=citation_recommender,
        citation_gate=citation_gate
    )
    document_reviser = DocumentReviser()
```

### Request Models

**GenerateDocumentRequest:**
- `doc_type`: Document type (bail_application, legal_notice, affidavit, petition)
- `court_level`: Court level (supreme_court, high_court, district_court)
- `facts`: Structured facts dictionary

**ReviseDocumentRequest:**
- `content`: Current document content to revise
- `instruction`: Revision instruction (min 5 chars)

### POST /generate Endpoint
1. Validates doc_type against DocumentType enum
2. Validates court_level against CourtLevel enum
3. Validates facts using appropriate Pydantic model (BailApplicationFacts, etc.)
4. Calls document_generator.generate_document()
5. Returns GeneratedDocument as JSON

### POST /revise Endpoint
1. Validates inputs
2. Calls document_reviser.revise_document()
3. Returns RevisionResult with revised content and edit metadata

### Error Handling
- 500: Generator/reviser not available (no vector_db)
- 400: Invalid doc_type, court_level, or facts
- 500: Generation/revision failed with specific error message

## Commits
- `4aec09f`: feat(04-05): add document generation API endpoints

## Verification Results
- [x] api.py syntax valid
- [x] Generation imports present
- [x] document_generator.generate_document() called
- [x] document_reviser.revise_document() called
- [x] Fact models used for validation
- [x] Both endpoints use existing auth middleware (x-access-token)

## Requirements Addressed
- DOC-01: User can generate documents with court-specific formatting via API
- DOC-03: System provides structured fact input interface (validated via Pydantic)
- DOC-04: User can revise generated documents via /revise endpoint
- DOC-07: Generated content uses formal legal language (via prompts integration)
