# Phase 1: Trust Foundation - Research

**Researched:** 2026-01-22
**Domain:** Citation Verification, Legal Code Mapping, Audit Logging
**Confidence:** MEDIUM-HIGH

## Summary

Phase 1 establishes the trust foundation for Vakalat AI's document drafting system. Research confirms that citation hallucination is the critical trust killer - "one hallucinated citation permanently destroys lawyer trust." The phase addresses five requirements: citation verification (TRUST-01, TRUST-02), legal code mapping (TRUST-03), section validation (TRUST-04), and outdated code flagging (TRUST-05).

The recommended approach leverages existing Pinecone infrastructure for citation verification via metadata-filtered vector search. For legal code mapping (IPC to BNS, CrPC to BNSS, Evidence Act to BSA), research identified comprehensive official mapping tables from BPRD and UP Police that should be ingested as structured JSON lookup tables. Audit logging should use structlog with contextvars for async-safe, JSON-structured logs integrated with the existing FastAPI middleware pattern.

**Primary recommendation:** Build a CitationVerifier service that performs exact metadata-filtered Pinecone queries, integrate code mapping as a separate LegalCodeMapper class with JSON-based lookup tables, and implement inline verification that blocks unverified citations before they reach output.

## Standard Stack

The established libraries/tools for this domain:

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| langchain-pinecone | existing | Vector search with metadata filtering | Already in use; Pinecone supports exact match filters |
| structlog | 24.x+ | Structured audit logging | Async-safe via contextvars; JSON output; FastAPI integration |
| pydantic | existing | Verification result models | Already in use for API models; validation built-in |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| python-json-logger | 3.x | JSON log formatting | Alternative to structlog if simpler setup needed |
| cachetools | 5.x | LRU caching for code mappings | Cache verified citations and code mappings |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Pinecone filtering | Full-text search on separate DB | Pinecone already has data; adding DB increases complexity |
| structlog | Standard logging + json-logger | structlog has better contextvars support for async |
| JSON mapping files | PostgreSQL tables | JSON is simpler for static mappings; DB needed only if dynamic updates required |

**Installation:**
```bash
pip install structlog cachetools
```

## Architecture Patterns

### Recommended Project Structure
```
src/
├── verification/           # Citation verification module
│   ├── __init__.py
│   ├── citation_verifier.py    # Core verification logic
│   ├── code_mapper.py          # IPC/BNS legal code mapping
│   ├── section_validator.py    # Section existence checking
│   └── models.py               # Pydantic verification models
├── audit/                  # Audit logging module
│   ├── __init__.py
│   └── logger.py               # Structlog configuration
└── data/
    └── mappings/           # Legal code mapping data
        ├── ipc_to_bns.json
        ├── crpc_to_bnss.json
        └── iea_to_bsa.json
```

### Pattern 1: Metadata-Filtered Citation Verification
**What:** Query Pinecone with exact metadata filters to verify citations exist in the database
**When to use:** Every time a case citation or statutory reference needs verification
**Example:**
```python
# Verified pattern from Pinecone documentation
from langchain_pinecone import PineconeVectorStore

class CitationVerifier:
    def __init__(self, vector_store: PineconeVectorStore):
        self.vector_store = vector_store

    def verify_case_citation(
        self,
        case_name: str,
        year: int,
        citation: str
    ) -> VerificationResult:
        """
        Verify case citation exists with exact metadata match.
        Pinecone filter syntax: {"$and": [{"field": {"$eq": value}}, ...]}
        """
        # Normalize for case-insensitive matching (store lowercase at ingestion)
        normalized_name = case_name.lower()

        # Metadata filter for exact match
        filter_dict = {
            "$and": [
                {"source_type": {"$eq": "case_law"}},
                {"case_name_normalized": {"$eq": normalized_name}},
                {"year": {"$eq": year}}
            ]
        }

        results = self.vector_store.similarity_search(
            query=case_name,  # Semantic backup
            k=3,
            filter=filter_dict
        )

        if results:
            return VerificationResult(
                status="VERIFIED",
                source_id=results[0].metadata.get("source_id"),
                confidence=1.0
            )
        return VerificationResult(
            status="UNVERIFIED",
            reason="Case not found in legal database"
        )
```

### Pattern 2: Legal Code Mapping with JSON Lookup
**What:** Static JSON mapping tables for IPC-to-BNS, CrPC-to-BNSS, Evidence Act-to-BSA conversions
**When to use:** When user cites old legal codes or system needs to suggest current equivalents
**Example:**
```python
import json
from pathlib import Path
from functools import lru_cache

class LegalCodeMapper:
    def __init__(self, mappings_dir: Path):
        self.mappings = {
            "IPC_BNS": self._load_mapping(mappings_dir / "ipc_to_bns.json"),
            "CRPC_BNSS": self._load_mapping(mappings_dir / "crpc_to_bnss.json"),
            "IEA_BSA": self._load_mapping(mappings_dir / "iea_to_bsa.json"),
        }

    def _load_mapping(self, path: Path) -> dict:
        with open(path) as f:
            return json.load(f)

    @lru_cache(maxsize=1000)
    def map_section(
        self,
        old_code: str,
        section: str
    ) -> CodeMappingResult:
        """
        Map old code section to new equivalent.
        Returns: new_code, new_section, confidence, notes
        """
        mapping_key = self._get_mapping_key(old_code)
        if mapping_key not in self.mappings:
            return CodeMappingResult(status="UNKNOWN_CODE")

        mapping = self.mappings[mapping_key].get(str(section))
        if mapping:
            return CodeMappingResult(
                status="MAPPED",
                old_code=old_code,
                old_section=section,
                new_code=mapping["new_code"],
                new_section=mapping["new_section"],
                confidence=mapping.get("confidence", 1.0),
                notes=mapping.get("notes", "")
            )
        return CodeMappingResult(
            status="NO_MAPPING",
            old_code=old_code,
            old_section=section,
            reason="Section has no direct equivalent"
        )

    def _get_mapping_key(self, old_code: str) -> str:
        code_upper = old_code.upper()
        if "IPC" in code_upper or "PENAL" in code_upper:
            return "IPC_BNS"
        elif "CRPC" in code_upper or "CRIMINAL PROCEDURE" in code_upper:
            return "CRPC_BNSS"
        elif "EVIDENCE" in code_upper or "IEA" in code_upper:
            return "IEA_BSA"
        return ""
```

### Pattern 3: Inline Verification with Fail-Safe Blocking
**What:** Verify each citation during generation, block unverified citations entirely
**When to use:** Per CONTEXT.md decision - "Block entirely" and "Inline verification"
**Example:**
```python
from pydantic import BaseModel, Field
from typing import Optional
from enum import Enum

class VerificationStatus(str, Enum):
    VERIFIED = "verified"
    UNVERIFIED = "unverified"
    BLOCKED = "blocked"

class VerifiedCitation(BaseModel):
    """Citation that passed verification - safe to include in output."""
    raw_citation: str
    formatted_citation: str
    status: VerificationStatus
    source_id: str
    verification_timestamp: str

class CitationGate:
    """
    Gate that blocks unverified citations from reaching output.
    Per CONTEXT.md: "omit and continue" when citation fails.
    """
    def __init__(self, verifier: CitationVerifier, logger):
        self.verifier = verifier
        self.logger = logger

    def filter_citations(
        self,
        citations: list[str]
    ) -> tuple[list[VerifiedCitation], list[str]]:
        """
        Filter citations through verification gate.
        Returns: (verified_citations, blocked_citations)
        """
        verified = []
        blocked = []

        for citation in citations:
            result = self.verifier.verify(citation)

            # Audit logging for every verification attempt
            self.logger.info(
                "citation_verification",
                citation=citation,
                status=result.status,
                reason=result.reason if result.status == "UNVERIFIED" else None
            )

            if result.status == "VERIFIED":
                verified.append(VerifiedCitation(
                    raw_citation=citation,
                    formatted_citation=result.formatted,
                    status=VerificationStatus.VERIFIED,
                    source_id=result.source_id,
                    verification_timestamp=datetime.utcnow().isoformat()
                ))
            else:
                blocked.append(citation)
                self.logger.warning(
                    "citation_blocked",
                    citation=citation,
                    reason=result.reason
                )

        return verified, blocked
```

### Anti-Patterns to Avoid
- **Fuzzy matching for citations:** Per CONTEXT.md, exact match is required. Do not use semantic similarity scores to "approximate" citation matches - if the exact case name + year + citation doesn't match, it's unverified.
- **Post-processing verification:** Per CONTEXT.md, verification must be inline during generation, not as an audit after document is complete.
- **Generating citations then verifying:** Never have LLM generate citations from scratch - always retrieve from database first, then verify and format.
- **Caching unverified status:** Don't cache "unverified" results permanently - the database may be updated with new cases.

## Don't Hand-Roll

Problems that look simple but have existing solutions:

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| JSON structured logging | Custom log formatters | structlog with JSON processor | Handles async contexts, correlation IDs, bound loggers properly |
| Case-insensitive search | String manipulation in queries | Normalize at ingestion time | Pinecone filters are case-sensitive; normalize once at ingestion |
| Code mapping tables | Database with CRUD | Static JSON files | Mappings are static (laws don't change often); JSON is simpler |
| Citation parsing | Regex patterns | Existing Verifier agent extension | Current Verifier already does citation extraction for verification |

**Key insight:** The existing Verifier agent in api.py (lines 117-123) already extracts citations and checks for hallucinations. Extend this pattern rather than building parallel verification logic.

## Common Pitfalls

### Pitfall 1: Case-Sensitive Metadata Filters
**What goes wrong:** Pinecone metadata filters are case-sensitive. Searching for "arnesh kumar" won't match "Arnesh Kumar"
**Why it happens:** User input varies in capitalization; LLM output varies in capitalization
**How to avoid:**
1. Normalize all case names to lowercase during Pinecone ingestion
2. Add `case_name_normalized` metadata field alongside original `case_name`
3. Always query against normalized field
**Warning signs:** Verification fails for cases that definitely exist in database

### Pitfall 2: Incomplete Code Mapping Data
**What goes wrong:** IPC section has no BNS equivalent in mapping table, system returns "no mapping" but BNS equivalent actually exists
**Why it happens:** Mapping tables from official sources may have gaps; some sections were split or merged
**How to avoid:**
1. Use multiple official sources (BPRD, UP Police, PIB) and cross-reference
2. Log all "no mapping" responses for manual review
3. Include fallback: "Section may have equivalent - verify manually"
**Warning signs:** Lawyers report mapping failures for common sections

### Pitfall 3: Blocking Too Aggressively
**What goes wrong:** System blocks citations that are valid but not in database (newer cases, unreported judgments)
**Why it happens:** Database coverage is incomplete; new cases published weekly
**How to avoid:**
1. Clear feedback: "Citation not found in database" (not "Citation is fake")
2. Allow lawyer override with explicit acknowledgment
3. Log all overrides for audit trail
4. Regular database updates from legal datasets
**Warning signs:** High volume of manual overrides; lawyer frustration

### Pitfall 4: Circular Verification Dependency
**What goes wrong:** Verification endpoint calls LLM which generates citations which need verification...
**Why it happens:** Trying to use LLM for citation extraction during verification
**How to avoid:**
1. Verification is pure database lookup - no LLM calls
2. Citation extraction from LLM output is regex/rule-based, not LLM-based
3. Clear separation: Extract (rules) -> Verify (database) -> Format (rules)
**Warning signs:** Verification latency spikes; recursive call patterns in logs

### Pitfall 5: Audit Log Explosion
**What goes wrong:** Logging every vector search detail creates massive log volume; searching logs becomes impossible
**Why it happens:** Over-logging during development; not configuring log levels
**How to avoid:**
1. Log verification decisions (VERIFIED/BLOCKED), not search internals
2. Use structured fields for querying (citation, status, request_id)
3. Configure log levels: DEBUG for search details, INFO for decisions
4. Include request correlation ID for tracing
**Warning signs:** Log storage costs spike; can't find relevant logs

## Code Examples

Verified patterns from official sources:

### Pinecone Metadata Filter Query
```python
# Source: Pinecone documentation - filter-with-metadata
# https://docs.pinecone.io/guides/data/filter-with-metadata

# Exact string matching (case-sensitive)
filter_dict = {"case_name_normalized": "arnesh kumar vs state of bihar"}

# Multiple conditions with $and
filter_dict = {
    "$and": [
        {"source_type": {"$eq": "case_law"}},
        {"year": {"$gte": 2020}},
        {"court": {"$in": ["Supreme Court", "Delhi High Court"]}}
    ]
}

# Query with filter
results = vector_store.similarity_search(
    query="arrest guidelines",
    k=5,
    filter=filter_dict
)
```

### Structlog Configuration for FastAPI
```python
# Source: structlog documentation + FastAPI integration patterns
# https://www.structlog.org/en/stable/

import structlog
from structlog.contextvars import bind_contextvars, clear_contextvars
from uuid import uuid4

def configure_logging():
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.JSONRenderer()
        ],
        wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )

# FastAPI middleware for request context
@app.middleware("http")
async def add_request_context(request: Request, call_next):
    clear_contextvars()
    request_id = str(uuid4())
    bind_contextvars(
        request_id=request_id,
        path=request.url.path,
        method=request.method
    )
    response = await call_next(request)
    return response
```

### Pydantic Models for Verification
```python
# Following existing api.py conventions
from pydantic import BaseModel, Field
from typing import Optional, Literal
from datetime import datetime

class CaseCitationInput(BaseModel):
    """Input for case citation verification."""
    case_name: str = Field(description="Full case name, e.g., 'Arnesh Kumar vs State of Bihar'")
    year: int = Field(description="Year of judgment")
    citation: Optional[str] = Field(None, description="Reporter citation, e.g., '(2014) 8 SCC 273'")

class StatuteCitationInput(BaseModel):
    """Input for statute section verification."""
    act_name: str = Field(description="Name of the Act, e.g., 'Indian Penal Code' or 'BNS'")
    section: str = Field(description="Section number, e.g., '302' or '103'")

class VerificationResult(BaseModel):
    """Output of verification process."""
    status: Literal["VERIFIED", "UNVERIFIED", "OUTDATED", "MAPPED"]
    source_id: Optional[str] = Field(None, description="Database ID if verified")
    confidence: float = Field(1.0, description="Confidence score 0-1")
    reason: Optional[str] = Field(None, description="Reason if not verified")
    suggested_update: Optional[str] = Field(None, description="For outdated codes")
    verification_time_ms: int = Field(description="Time taken to verify in milliseconds")

class CodeMappingResult(BaseModel):
    """Output of legal code mapping."""
    status: Literal["MAPPED", "NO_MAPPING", "UNKNOWN_CODE"]
    old_code: str
    old_section: str
    new_code: Optional[str] = None
    new_section: Optional[str] = None
    confidence: float = Field(1.0, description="Mapping confidence")
    notes: Optional[str] = Field(None, description="Additional context")
```

### IPC to BNS Mapping JSON Structure
```json
{
  "302": {
    "new_code": "BNS",
    "new_section": "103",
    "confidence": 1.0,
    "notes": "Punishment for murder - direct equivalent"
  },
  "304A": {
    "new_code": "BNS",
    "new_section": "106",
    "confidence": 1.0,
    "notes": "Causing death by negligence"
  },
  "420": {
    "new_code": "BNS",
    "new_section": "318",
    "confidence": 1.0,
    "notes": "Cheating and dishonestly inducing delivery of property"
  },
  "498A": {
    "new_code": "BNS",
    "new_section": "85",
    "confidence": 1.0,
    "notes": "Cruelty by husband or relatives"
  },
  "34": {
    "new_code": "BNS",
    "new_section": "3(5)",
    "confidence": 1.0,
    "notes": "Acts done in furtherance of common intention"
  }
}
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| IPC citations | BNS citations | July 1, 2024 | Crimes after Jul 2024 use BNS; both codes relevant for transition period |
| CrPC procedures | BNSS procedures | July 1, 2024 | Procedural law updated; section numbers changed |
| Indian Evidence Act | BSA | July 1, 2024 | Evidence rules updated; definitions expanded |
| Fuzzy citation matching | Exact metadata filtering | 2024 RAG best practices | 17-33% hallucination rate with semantic-only; exact match required |

**Deprecated/outdated:**
- **IPC, CrPC, IEA**: Still valid for pre-July 2024 cases, but new filings use BNS/BNSS/BSA
- **Semantic-only citation verification**: Research shows 17-33% hallucination rate in legal RAG systems; exact match required

## Open Questions

Things that couldn't be fully resolved:

1. **Pinecone Database Coverage**
   - What we know: Database contains 54+ Indian laws and Supreme Court judgments from 1950-2025
   - What's unclear: Complete list of case law; whether case_name metadata is normalized; coverage of High Court judgments
   - Recommendation: Inspect Pinecone metadata fields before building verifier; may need to add normalized fields

2. **Code Mapping Completeness**
   - What we know: Official BPRD and UP Police sources have IPC-BNS tables; structure identified
   - What's unclear: Whether mappings cover 100% of sections; handling of deleted/new sections
   - Recommendation: Start with official tables; log unmapped sections; build "partial mapping" handling

3. **External Database API Access**
   - What we know: IndianKanoon, SCC Online, Manupatra exist as authoritative sources
   - What's unclear: API availability, rate limits, authentication, cost
   - Recommendation: Start with Pinecone-only; defer external API integration to Phase 3 if needed

4. **Repeal Status Tracking**
   - What we know: TRUST-04 requires checking if sections are repealed
   - What's unclear: How to maintain current repeal status; frequency of updates
   - Recommendation: Add "status" field to mapping JSON (active/repealed/amended); manual updates initially

## Sources

### Primary (HIGH confidence)
- [Pinecone Metadata Filtering Documentation](https://docs.pinecone.io/guides/data/filter-with-metadata) - Filter syntax, operators, constraints
- [LangChain Pinecone Integration](https://python.langchain.com/api_reference/pinecone/vectorstores/langchain_pinecone.vectorstores.Pinecone.html) - Filter parameter usage
- Existing codebase api.py (lines 117-123) - Verifier agent pattern
- CONTEXT.md - User decisions on verification strictness

### Secondary (MEDIUM confidence)
- [BPRD IPC-BNS Comparison Summary](https://bprd.nic.in/uploads/pdf/COMPARISON%20SUMMARY%20BNS%20to%20IPC%20.pdf) - Official mapping tables
- [UP Police BNS-IPC Comparative Table](https://uppolice.gov.in/site/writereaddata/siteContent/Three%20New%20Major%20Acts/202406281710564823BNS_IPC_Comparative.pdf) - Section mappings
- [structlog Documentation](https://www.structlog.org/en/stable/) - Async logging patterns
- [Stanford Legal RAG Hallucinations Study](https://dho.stanford.edu/wp-content/uploads/Legal_RAG_Hallucinations.pdf) - 17-33% hallucination rate finding (unable to fetch full content)

### Tertiary (LOW confidence)
- LegalDesk AI, Vera Causa Legal - Online converter tools (pattern reference only)
- General legal AI best practices from training data

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - Pinecone already in use; structlog is established; patterns verified against docs
- Architecture: HIGH - Extends existing Verifier pattern; metadata filtering is documented Pinecone feature
- Code mappings: MEDIUM - Official sources identified but PDF content couldn't be fully extracted
- Pitfalls: HIGH - Based on documented legal AI failure modes and CONTEXT.md constraints

**Research date:** 2026-01-22
**Valid until:** 60 days (mappings stable; Pinecone API stable; legal codes fixed for now)

---

## Integration Points with Existing Code

### Existing Verifier Agent (api.py lines 117-123)
```python
# Current implementation
VERIFIER_PROMPT = ChatPromptTemplate.from_template("""
### VERIFIER AGENT
SOURCE: {context} | DRAFT: {draft_answer}
AUDIT: 1. Hallucination Check. 2. Inference Check.
OUTPUT JSON: {format_instructions}
""")
```
**Integration:** The new CitationVerifier should be called BEFORE the Verifier agent, not replace it. Flow:
1. Extract citations from Jurist response
2. CitationVerifier.verify_all() - database lookup
3. CitationGate.filter() - block unverified
4. Pass filtered response to existing Verifier agent

### Existing Metadata Structure (ingest_master.py)
```python
# Current metadata fields
chunk.metadata["source_dataset"] = config['name']
chunk.metadata["source_id"] = f"{config['id']}_{i}"
chunk.metadata["act"] = row["act_title"]  # statutes
chunk.metadata["section"] = str(row["section"])  # statutes
chunk.metadata["court"] = row["Court_Name"]  # judgments
chunk.metadata["title"] = row["Titles"]  # judgments
```
**Gap:** No `case_name_normalized` field. Recommend adding during re-ingestion or patch script.

### Existing Auth Middleware (api.py lines 43-58)
```python
@app.middleware("http")
async def verify_access(request: Request, call_next):
    # ... token verification
```
**Integration:** Audit logging middleware should be added similarly, binding request context before processing.
