# Phase 2: Template Storage - Research

**Researched:** 2026-01-24
**Domain:** Legal document template storage and validation
**Confidence:** HIGH

## Summary

Template storage for Indian court documents requires adherence to strict formatting standards mandated by the Supreme Court and High Courts. The standard technical approach combines Pydantic models for schema validation, enum-based type safety for document types, and JSON storage with version control. Court-specific formatting metadata (A4 paper, Times New Roman 14pt, 1.5 line spacing, 4cm margins) must be embedded in each template.

The research reveals that Indian courts have standardized on A4 paper format across all jurisdictions (Supreme Court Rules 2013), with specific formatting requirements that must be captured in template metadata. Template storage architecture should use structured JSON with Pydantic validation rather than database storage, as templates are static configuration data that benefit from version control and file-based transparency.

Security considerations are critical: Jinja2 template injection vulnerabilities (CVE-2025-27516) require strict input sanitization when user data populates templates. The existing Vakalat AI codebase already uses Pydantic extensively (api.py lines 107-140), making extension to template schemas straightforward.

**Primary recommendation:** Use Pydantic models to define template schemas with enum-based document type validation, store templates as JSON files in a templates/ directory, and implement schema versioning from day 1 to support template evolution without breaking changes.

## Standard Stack

The established libraries/tools for legal document template storage in Python:

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| Pydantic | 2.x | Schema validation and data modeling | Industry standard for FastAPI data validation, already in codebase |
| FastAPI | Latest | REST API framework | Already in production (api.py), native Pydantic integration |
| Python Enum | stdlib | Document type enumeration | Type-safe constants, built-in Pydantic support |
| JSON | stdlib | Template storage format | Human-readable, version-control friendly, schema-based |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| python-docx-template | Latest | DOCX generation with Jinja2 | Future phase: actual document rendering |
| Jinja2 | 3.x | Template rendering engine | Future phase: fill templates with user data |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| JSON files | Database storage | Database adds complexity for static data; JSON enables version control and transparency |
| Pydantic enums | String literals | Enums provide type safety and autocomplete; strings prone to typos |
| File storage | Pinecone vectors | Vector DB for retrieval, not configuration storage; templates are static metadata |

**Installation:**
```bash
# Core already installed in existing requirements.txt
# python-docx-template for future phases
pip install python-docx-template
```

## Architecture Patterns

### Recommended Project Structure
```
src/
├── templates/
│   ├── schemas.py          # Pydantic models for template validation
│   ├── storage.py          # Template CRUD operations
│   └── data/
│       ├── bail_application_supreme_court.json
│       ├── bail_application_high_court.json
│       ├── legal_notice_district_court.json
│       ├── affidavit_supreme_court.json
│       └── petition_high_court.json
```

### Pattern 1: Enum-Based Document Type Validation
**What:** Use Python Enum with Pydantic for type-safe document type constants
**When to use:** Always for categorizing templates (prevents typos, enables autocomplete)
**Example:**
```python
# Source: https://docs.pydantic.dev/latest/api/standard_library_types/
from enum import Enum
from pydantic import BaseModel

class DocumentType(str, Enum):
    BAIL_APPLICATION = "bail_application"
    LEGAL_NOTICE = "legal_notice"
    AFFIDAVIT = "affidavit"
    PETITION = "petition"

class CourtLevel(str, Enum):
    SUPREME_COURT = "supreme_court"
    HIGH_COURT = "high_court"
    DISTRICT_COURT = "district_court"

class TemplateMetadata(BaseModel):
    doc_type: DocumentType
    court_level: CourtLevel
    version: str
```

### Pattern 2: Nested Pydantic Models for Template Schema
**What:** Define template structure with nested Pydantic models for formatting and fields
**When to use:** Every template definition to ensure validation and JSON schema generation
**Example:**
```python
# Source: https://docs.pydantic.dev/latest/concepts/models/
from pydantic import BaseModel, Field
from typing import List, Optional

class FormattingRequirements(BaseModel):
    paper_size: str = Field(default="A4", description="29.7cm X 21cm")
    font: str = Field(default="Times New Roman")
    font_size: int = Field(default=14)
    line_spacing: float = Field(default=1.5)
    margin_left_right: str = Field(default="4cm")
    margin_top_bottom: str = Field(default="2cm")
    paper_quality: str = Field(default="75 GSM")
    double_sided: bool = Field(default=True)

class TemplateField(BaseModel):
    field_name: str
    field_type: str  # "text", "date", "party", "relief"
    required: bool
    description: str
    example: Optional[str] = None

class LegalTemplate(BaseModel):
    metadata: TemplateMetadata
    formatting: FormattingRequirements
    required_fields: List[TemplateField]
    optional_fields: List[TemplateField] = []
    template_content: str  # Future: Jinja2 template string
    created_at: str
    updated_at: str
```

### Pattern 3: Repository Pattern for Template CRUD
**What:** Separate storage logic from business logic with repository class
**When to use:** Always - enables testability and future storage backend changes
**Example:**
```python
# Source: https://github.com/zhanymkanov/fastapi-best-practices
from typing import List, Optional
import json
from pathlib import Path

class TemplateRepository:
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir

    def get_template(self, doc_type: DocumentType, court_level: CourtLevel) -> Optional[LegalTemplate]:
        filename = f"{doc_type.value}_{court_level.value}.json"
        filepath = self.data_dir / filename
        if not filepath.exists():
            return None
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return LegalTemplate(**data)

    def list_templates(self, doc_type: Optional[DocumentType] = None) -> List[LegalTemplate]:
        templates = []
        for filepath in self.data_dir.glob("*.json"):
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            template = LegalTemplate(**data)
            if doc_type is None or template.metadata.doc_type == doc_type:
                templates.append(template)
        return templates

    def save_template(self, template: LegalTemplate) -> None:
        filename = f"{template.metadata.doc_type.value}_{template.metadata.court_level.value}.json"
        filepath = self.data_dir / filename
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(template.model_dump(), f, indent=2, ensure_ascii=False)
```

### Pattern 4: FastAPI Integration Pattern
**What:** Expose template operations as REST endpoints following existing api.py patterns
**When to use:** For all template CRUD operations accessible via API
**Example:**
```python
# Source: Existing api.py pattern (lines 107-140)
from fastapi import FastAPI, HTTPException
from typing import Optional

class ListTemplatesRequest(BaseModel):
    doc_type: Optional[DocumentType] = None

class GetTemplateRequest(BaseModel):
    doc_type: DocumentType
    court_level: CourtLevel

@app.post("/templates/list")
async def list_templates(req: ListTemplatesRequest):
    templates = template_repo.list_templates(req.doc_type)
    return {"templates": [t.model_dump() for t in templates]}

@app.post("/templates/get")
async def get_template(req: GetTemplateRequest):
    template = template_repo.get_template(req.doc_type, req.court_level)
    if not template:
        raise HTTPException(404, "Template not found")
    return template.model_dump()
```

### Anti-Patterns to Avoid
- **String-based document types:** Use enums instead - prevents typos and enables type checking
- **Hardcoded formatting in code:** Store all court requirements in template metadata for flexibility
- **Direct database storage:** Templates are static config; JSON files enable version control and transparency
- **Accepting unsanitized user input in templates:** Jinja2 injection vulnerability (CVE-2025-27516)

## Don't Hand-Roll

Problems that look simple but have existing solutions:

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| JSON schema validation | Custom validation functions | Pydantic BaseModel | Automatic validation, JSON schema generation, FastAPI integration |
| Template versioning | Manual version tracking | Semantic versioning + schema migration pattern | Industry standard, clear compatibility rules |
| Document type constants | String literals or config dict | Python Enum with Pydantic | Type safety, autocomplete, validation errors for invalid values |
| File-based storage CRUD | Raw file operations | Repository pattern with Pydantic | Separation of concerns, testability, future-proof for storage changes |
| Court formatting standards | Application-level constants | Template metadata fields | Different courts may have variations, keep flexible |

**Key insight:** Legal document templates are structured configuration data, not business logic. Use validation-first patterns (Pydantic) rather than imperative code to define and enforce template schemas.

## Common Pitfalls

### Pitfall 1: Template Injection Vulnerabilities
**What goes wrong:** User input populates Jinja2 templates without sanitization, enabling code execution
**Why it happens:** Jinja2 templates can execute arbitrary Python code if user input is treated as template syntax
**How to avoid:** Never use `template.from_string()` with user input; always use `render()` with variable substitution
**Warning signs:** CVE-2025-27516 warns about |attr filter vulnerability in Jinja2
**Source:** https://security.snyk.io/vuln/SNYK-PYTHON-JINJA2-9292516

### Pitfall 2: Ignoring Court-Specific Formatting Requirements
**What goes wrong:** Templates use generic formatting instead of court-mandated standards
**Why it happens:** Developers unaware that Supreme Court Rules 2013 mandate specific paper size, font, margins
**How to avoid:** Embed FormattingRequirements in every template with Supreme Court defaults (A4, Times New Roman 14, 1.5 spacing, 4cm margins)
**Warning signs:** User complaints about document rejection by court registry
**Source:** https://www.scconline.com/blog/post/2020/03/12/sc-a4-size-paper-shall-be-used-in-pleadings-petitions-affidavits-or-other-documents-to-be-filed-in-supreme-court/

### Pitfall 3: Breaking Changes in Template Schema
**What goes wrong:** Adding/removing required fields breaks existing template consumers
**Why it happens:** No versioning strategy for template schema evolution
**How to avoid:** Use semantic versioning; add new fields as optional; migrate in expand-contract pattern
**Warning signs:** Template validation errors after updates
**Source:** https://docs.confluent.io/platform/current/schema-registry/fundamentals/schema-evolution.html

### Pitfall 4: Failure to Customize Legal Templates
**What goes wrong:** Generic templates used without case-specific customization lead to invalid documents
**Why it happens:** Treating templates as final documents rather than starting points
**How to avoid:** Validate that all required_fields are populated; warn users that templates need customization
**Warning signs:** Legal disputes, document rejection, loss of cases
**Source:** https://legaltrunk.com/common-mistakes-to-avoid-when-using-legal-document-templates/

### Pitfall 5: Database Storage for Static Configuration
**What goes wrong:** Templates stored in database complicate version control and transparency
**Why it happens:** Assumption that "all data goes in database"
**How to avoid:** Store templates as JSON files in version-controlled templates/data/ directory
**Warning signs:** Difficulty tracking template changes, no audit trail
**Source:** https://github.com/zhanymkanov/fastapi-best-practices (best practices recommend DB for dynamic data, files for config)

## Code Examples

Verified patterns from official sources:

### Complete Template Schema Definition
```python
# Source: https://docs.pydantic.dev/latest/concepts/models/
# Source: Supreme Court formatting requirements
from enum import Enum
from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime

class DocumentType(str, Enum):
    BAIL_APPLICATION = "bail_application"
    LEGAL_NOTICE = "legal_notice"
    AFFIDAVIT = "affidavit"
    PETITION = "petition"

class CourtLevel(str, Enum):
    SUPREME_COURT = "supreme_court"
    HIGH_COURT = "high_court"
    DISTRICT_COURT = "district_court"

class TemplateMetadata(BaseModel):
    doc_type: DocumentType
    court_level: CourtLevel
    version: str = Field(description="Semantic version (e.g., 1.0.0)")
    name: str
    description: str

class FormattingRequirements(BaseModel):
    """Supreme Court Rules 2013 formatting standards"""
    paper_size: str = Field(default="A4", description="29.7cm X 21cm")
    font: str = Field(default="Times New Roman")
    font_size: int = Field(default=14, ge=12, le=16)
    line_spacing: float = Field(default=1.5)
    margin_left_right: str = Field(default="4cm")
    margin_top_bottom: str = Field(default="2cm")
    paper_quality: str = Field(default="75 GSM")
    double_sided: bool = Field(default=True, description="Print both sides")

class TemplateField(BaseModel):
    field_name: str = Field(description="Unique field identifier")
    field_type: str = Field(description="text|date|party|case_number|relief")
    required: bool
    label: str = Field(description="Human-readable label")
    description: str
    example: Optional[str] = None
    validation_regex: Optional[str] = None

class LegalTemplate(BaseModel):
    metadata: TemplateMetadata
    formatting: FormattingRequirements
    required_fields: List[TemplateField]
    optional_fields: List[TemplateField] = []
    template_content: str = Field(description="Jinja2 template string (future use)")
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = Field(default_factory=lambda: datetime.now().isoformat())

    class Config:
        json_schema_extra = {
            "example": {
                "metadata": {
                    "doc_type": "bail_application",
                    "court_level": "supreme_court",
                    "version": "1.0.0",
                    "name": "Bail Application - Supreme Court",
                    "description": "Standard format for bail application under Section 439 CrPC"
                },
                "formatting": {
                    "paper_size": "A4",
                    "font": "Times New Roman",
                    "font_size": 14
                },
                "required_fields": [
                    {
                        "field_name": "applicant_name",
                        "field_type": "party",
                        "required": True,
                        "label": "Applicant Name",
                        "description": "Full legal name of the accused seeking bail",
                        "example": "Ramesh Kumar"
                    }
                ]
            }
        }
```

### FastAPI Endpoint Implementation
```python
# Source: https://fastapi.tiangolo.com/tutorial/body-fields/
# Pattern from existing api.py
from fastapi import FastAPI, HTTPException, Depends
from pathlib import Path

app = FastAPI()

# Initialize repository
TEMPLATES_DIR = Path(__file__).parent / "templates" / "data"
template_repo = TemplateRepository(TEMPLATES_DIR)

class ListTemplatesRequest(BaseModel):
    doc_type: Optional[DocumentType] = Field(
        default=None,
        description="Filter by document type"
    )

class GetTemplateRequest(BaseModel):
    doc_type: DocumentType = Field(description="Document type to retrieve")
    court_level: CourtLevel = Field(description="Court level (Supreme/High/District)")

@app.post("/templates/list")
async def list_templates(req: ListTemplatesRequest):
    """List available templates, optionally filtered by document type."""
    templates = template_repo.list_templates(req.doc_type)
    return {
        "count": len(templates),
        "templates": [
            {
                "doc_type": t.metadata.doc_type,
                "court_level": t.metadata.court_level,
                "name": t.metadata.name,
                "version": t.metadata.version
            }
            for t in templates
        ]
    }

@app.post("/templates/get")
async def get_template(req: GetTemplateRequest):
    """Retrieve a specific template by document type and court level."""
    template = template_repo.get_template(req.doc_type, req.court_level)
    if not template:
        raise HTTPException(
            status_code=404,
            detail=f"Template not found: {req.doc_type.value} for {req.court_level.value}"
        )
    return template.model_dump()
```

### Pydantic Field Validation
```python
# Source: https://fastapi.tiangolo.com/tutorial/body-fields/
from pydantic import BaseModel, Field, validator

class TemplateField(BaseModel):
    field_name: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Unique field identifier (snake_case)"
    )
    field_type: str = Field(
        ...,
        description="Field data type"
    )
    required: bool = Field(
        default=True,
        description="Whether field is mandatory"
    )

    @validator('field_type')
    def validate_field_type(cls, v):
        allowed_types = ['text', 'date', 'party', 'case_number', 'relief', 'court']
        if v not in allowed_types:
            raise ValueError(f'field_type must be one of {allowed_types}')
        return v

    @validator('field_name')
    def validate_snake_case(cls, v):
        if not v.islower() or ' ' in v:
            raise ValueError('field_name must be lowercase snake_case')
        return v
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| String-based document types | Enum with Pydantic | Pydantic 2.0 (2023) | Type safety, autocomplete, validation |
| Manual JSON validation | Pydantic BaseModel | FastAPI adoption | Automatic schema generation, OpenAPI docs |
| Database template storage | JSON files + version control | Git-first workflows | Transparency, audit trail, easy rollback |
| Green legal paper | A4 white paper (75 GSM) | Supreme Court 2020 directive | Environmental concerns, uniformity |
| Font size flexibility | Mandatory Times New Roman 14pt | Supreme Court Rules 2013 | Standardization across courts |

**Deprecated/outdated:**
- Green-colored legal paper: Replaced by A4 white paper (Supreme Court mandate 2020)
- Custom paper sizes: A4 (29.7cm X 21cm) now mandatory across all Indian courts
- Pydantic v1 validator syntax (@validator): Use v2 syntax (@field_validator) for new code

## Open Questions

Things that couldn't be fully resolved:

1. **District Court Format Variations**
   - What we know: Supreme Court and High Courts have standardized on A4, Times New Roman 14
   - What's unclear: Do all District Courts follow same formatting, or do some have local variations?
   - Recommendation: Start with Supreme Court standards as default, add court-specific overrides if needed

2. **Template Rendering Engine Choice**
   - What we know: python-docx-template combines Jinja2 with DOCX for rich formatting
   - What's unclear: Whether this phase needs rendering capability or just storage/retrieval
   - Recommendation: Phase 2 focuses on storage; defer rendering to Phase 3 (Document Generation)

3. **Version Migration Strategy**
   - What we know: Semantic versioning and expand-contract pattern are industry standard
   - What's unclear: What happens to documents generated from old template versions after schema update?
   - Recommendation: Store template version in generated documents; maintain backwards compatibility

4. **Multi-language Support**
   - What we know: Indian courts accept documents in English and regional languages
   - What's unclear: Whether templates need language variants (English/Hindi/regional)
   - Recommendation: Start with English templates; add language field to metadata for future expansion

## Sources

### Primary (HIGH confidence)
- Pydantic Official Documentation - https://docs.pydantic.dev/latest/concepts/models/
- FastAPI Official Documentation - https://fastapi.tiangolo.com/tutorial/body-fields/
- Supreme Court of India Forms - https://www.sci.gov.in/forms/
- Supreme Court Paper Standards (2020) - https://www.scconline.com/blog/post/2020/03/12/sc-a4-size-paper-shall-be-used-in-pleadings-petitions-affidavits-or-other-documents-to-be-filed-in-supreme-court/
- Pydantic Enum Support - https://docs.pydantic.dev/latest/api/standard_library_types/

### Secondary (MEDIUM confidence)
- Indian Court Formatting Requirements - https://www.latestlaws.com/latest-news/delhi-high-court-issues-directions-on-paper-size-and-print-fonts-read-text-190513/
- Bar and Bench: Court Uniformity Article - https://www.barandbench.com/columns/courts-and-uniformity
- Schema Evolution Best Practices - https://docs.confluent.io/platform/current/schema-registry/fundamentals/schema-evolution.html
- FastAPI Best Practices - https://github.com/zhanymkanov/fastapi-best-practices
- Jinja2 Official Documentation - https://jinja.palletsprojects.com/

### Tertiary (LOW confidence - for validation)
- Legal Template Mistakes - https://legaltrunk.com/common-mistakes-to-avoid-when-using-legal-document-templates/
- Jinja2 Template Injection CVE - https://security.snyk.io/vuln/SNYK-PYTHON-JINJA2-9292516
- Python-docx-template Documentation - https://docxtpl.readthedocs.io/

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - Pydantic and FastAPI verified from official docs and existing codebase
- Architecture: HIGH - Patterns verified from official Pydantic docs and FastAPI tutorials
- Court formatting: HIGH - Supreme Court official website and legal news sources
- Pitfalls: MEDIUM - Security issues verified (CVE), legal template mistakes from practitioner sources

**Research date:** 2026-01-24
**Valid until:** 2026-02-23 (30 days - stable domain, court rules change slowly)
