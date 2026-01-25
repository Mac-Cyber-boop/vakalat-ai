# Phase 3: Citation Engine - Research

**Researched:** 2026-01-25
**Domain:** Legal citation retrieval, formatting, and verification
**Confidence:** HIGH (builds on existing Phase 1 infrastructure)

## Summary

Phase 3 extends the verification infrastructure from Phase 1 to create an intelligent citation recommendation engine. The core challenge is not just formatting citations correctly but retrieving relevant precedents based on legal issues rather than keyword matching.

**Key findings:**
1. **IndianKanoon API** is the most viable external legal database API for India, offering documented REST endpoints with JSON/XML responses and structured metadata including precedent classification
2. **Indian legal citation standards** follow well-defined formats: SCC format "(Year) Volume SCC Page" and AIR format "AIR Year Court Page"
3. **Jurisdiction-based ranking** should use a weighted scoring system combining semantic similarity, jurisdictional authority (Supreme Court > Same High Court > Other High Courts), and temporal recency
4. **Existing infrastructure** in `src/verification/` provides 80% of needed functionality; the citation engine primarily adds retrieval intelligence and formatting

**Primary recommendation:** Build the citation engine as a new module `src/citations/` that orchestrates existing verifiers with new retrieval and formatting logic, leveraging the existing Pinecone metadata for court and source filtering.

## Standard Stack

The established libraries/tools for this domain:

### Core (Already in Project)
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| langchain-pinecone | latest | Vector store with metadata filtering | Already integrated, supports jurisdiction filters |
| Pydantic | v2 | Data models for citations | Already used in verification models |
| regex (re) | stdlib | Citation pattern extraction | Already proven in CitationGate |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| python-dateutil | 2.8+ | Date parsing from citation strings | Parsing "(2014)" from various formats |
| httpx | 0.27+ | Async HTTP for IndianKanoon API | External API calls if implemented |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| IndianKanoon API | Build from HuggingFace data only | IndianKanoon has structured precedent metadata; HuggingFace data lacks citation relationships |
| Manual regex | spaCy NER for citations | Overkill - legal citations follow predictable patterns |

**Installation (if adding external API support):**
```bash
pip install httpx python-dateutil
```

## Architecture Patterns

### Recommended Project Structure
```
src/
├── citations/              # NEW: Citation engine module
│   ├── __init__.py
│   ├── retriever.py        # Precedent retrieval with ranking
│   ├── formatter.py        # Citation formatting (SCC, AIR)
│   ├── recommender.py      # Issue-based recommendation logic
│   └── models.py           # Citation data models
├── verification/           # EXISTING: Reuse heavily
│   ├── citation_verifier.py
│   ├── citation_gate.py
│   └── ...
└── ...
```

### Pattern 1: Retriever-Ranker Pipeline
**What:** Separate semantic retrieval from relevance ranking
**When to use:** All precedent suggestions
**Example:**
```python
# Source: Adapted from existing citation_verifier.py pattern
class PrecedentRetriever:
    def __init__(self, vector_store: PineconeVectorStore):
        self.vector_store = vector_store

    def retrieve_precedents(
        self,
        legal_issue: str,
        filing_court: str,
        top_k: int = 10
    ) -> list[RetrievedPrecedent]:
        # Step 1: Semantic retrieval with source_type filter
        results = self.vector_store.similarity_search(
            query=legal_issue,
            k=top_k * 2,  # Over-fetch for re-ranking
            filter={"source_type": {"$eq": "case_law"}}
        )

        # Step 2: Re-rank by jurisdiction affinity
        ranked = self._rank_by_jurisdiction(results, filing_court)
        return ranked[:top_k]

    def _rank_by_jurisdiction(
        self,
        results: list,
        filing_court: str
    ) -> list[RetrievedPrecedent]:
        # Jurisdiction scoring:
        # Supreme Court = 1.0 (binding on all)
        # Same High Court = 0.9 (binding on subordinates)
        # Other High Courts = 0.7 (persuasive only)
        JURISDICTION_WEIGHTS = {
            "supreme_court": 1.0,
            "same_hc": 0.9,
            "other_hc": 0.7,
            "tribunal": 0.5
        }
        # ... ranking logic
```

### Pattern 2: Citation Formatter Strategy
**What:** Strategy pattern for different citation formats
**When to use:** Rendering citations for output
**Example:**
```python
# Source: Based on Indian legal citation standards research
class CitationFormatter:
    """Formats citations according to Indian legal standards."""

    @staticmethod
    def format_scc(
        party_a: str,
        party_b: str,
        year: int,
        volume: int,
        page: int
    ) -> str:
        """Format as SCC: Party A vs Party B (2014) 8 SCC 273"""
        return f"{party_a} vs {party_b} ({year}) {volume} SCC {page}"

    @staticmethod
    def format_air(
        party_a: str,
        party_b: str,
        year: int,
        court: str,
        page: int
    ) -> str:
        """Format as AIR: Party A vs Party B, AIR 2014 SC 123"""
        court_abbr = {"supreme_court": "SC", "delhi": "Del", "bombay": "Bom"}
        return f"{party_a} vs {party_b}, AIR {year} {court_abbr.get(court, 'SC')} {page}"

    @staticmethod
    def format_statute(
        section: str,
        act_name: str,
        year: Optional[int] = None
    ) -> str:
        """Format statute: Section 438, Code of Criminal Procedure, 1973"""
        if year:
            return f"Section {section}, {act_name}, {year}"
        return f"Section {section}, {act_name}"
```

### Pattern 3: Verification Badge Decorator
**What:** Wrap citations with verification status for UI
**When to use:** All citation outputs to frontend
**Example:**
```python
# Source: Based on CitationGate filtering pattern
class CitationWithBadge(BaseModel):
    """Citation with verification status for UI display."""
    raw_citation: str
    formatted_citation: str
    verification_status: VerificationStatus  # VERIFIED, UNVERIFIED, OUTDATED
    badge_type: Literal["verified", "unverified", "outdated"]
    source_id: Optional[str] = None

    @property
    def badge_html(self) -> str:
        badges = {
            "verified": '<span class="badge verified" title="Verified in database">&#10003;</span>',
            "unverified": '<span class="badge unverified" title="Could not verify">&#9888;</span>',
            "outdated": '<span class="badge outdated" title="Uses repealed law">&#8635;</span>'
        }
        return badges.get(self.badge_type, "")
```

### Anti-Patterns to Avoid
- **Keyword-only retrieval:** Legal issues share keywords but differ in context; "anticipatory bail for economic offenses" is different from "anticipatory bail for violent crimes"
- **Flat ranking:** Treating all retrieved cases equally ignores binding authority hierarchy
- **Formatting in retrieval:** Keep retrieval and formatting separate for maintainability

## Don't Hand-Roll

Problems that look simple but have existing solutions:

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Citation extraction | Custom regex from scratch | Extend existing `CitationGate` patterns | Already handles SCC, AIR, Section patterns |
| Case verification | New verifier | `CitationVerifier` from Phase 1 | Already caches results, logs audits |
| Statute validation | New validator | `SectionValidator` from Phase 1 | Handles IPC/BNS mapping |
| Code modernization | Manual IPC->BNS lookup | `LegalCodeMapper` from Phase 1 | 250+ mappings already loaded |
| Date parsing | Regex extraction | `python-dateutil` | Handles "(2014)", "2014", etc. |

**Key insight:** Phase 1 invested heavily in verification infrastructure. Phase 3 should compose these components, not replace them.

## Common Pitfalls

### Pitfall 1: Treating Semantic Similarity as Legal Relevance
**What goes wrong:** High semantic similarity doesn't mean legal relevance; a case about "bail for murder" might semantically match "bail for theft" but the legal principles differ significantly
**Why it happens:** Vector embeddings capture textual similarity, not legal doctrine relationships
**How to avoid:** Post-filter retrieved results through legal issue classification; consider IndianKanoon's AI Tags if using their API
**Warning signs:** Users report irrelevant suggestions despite high similarity scores

### Pitfall 2: Ignoring Binding Authority Hierarchy
**What goes wrong:** Suggesting persuasive authority (other High Court) when binding authority (Supreme Court or same High Court) exists
**Why it happens:** Retrieval returns most similar, not most authoritative
**How to avoid:** Two-pass retrieval: first filter by jurisdiction hierarchy, then rank by similarity within each tier
**Warning signs:** Lawyers complain suggestions aren't "on point" for their court

### Pitfall 3: Over-relying on Metadata Filtering
**What goes wrong:** Pinecone metadata is inconsistent across data sources (HuggingFace datasets have different field names)
**Why it happens:** Ingestion from multiple sources without schema normalization
**How to avoid:** Build defensive retrieval that checks multiple metadata fields (existing pattern in `citation_verifier.py`)
**Warning signs:** Verified cases return UNVERIFIED because metadata field name differs

### Pitfall 4: Citation Format Ambiguity
**What goes wrong:** Same case may have multiple valid citations (SCC and AIR both)
**Why it happens:** Cases are reported in multiple reporters
**How to avoid:** Choose canonical format (SCC preferred per Indian bar conventions) but accept alternatives for verification
**Warning signs:** Verification fails for valid citations in alternate format

### Pitfall 5: Performance Degradation with Re-ranking
**What goes wrong:** Jurisdiction-based re-ranking adds latency on every request
**Why it happens:** Re-ranking requires post-processing all retrieved documents
**How to avoid:** Cache jurisdiction affinity scores; limit re-ranking to top-N candidates
**Warning signs:** Precedent suggestions take >2 seconds

## Code Examples

Verified patterns from existing codebase:

### Pinecone Metadata Filtering (From section_validator.py)
```python
# Source: src/verification/section_validator.py lines 316-330
# Filter by source_type=statute and source_book matching the act
filter_dict = {
    "$and": [
        {"source_type": {"$eq": "statute"}},
        {"source_book": {"$in": source_books}}
    ]
}

results = self.vector_store.similarity_search(
    query=query_text,
    k=3,
    filter=filter_dict
)
```

### Multi-Method Verification (From citation_verifier.py)
```python
# Source: src/verification/citation_verifier.py lines 145-179
# Check source filename, metadata fields, AND content
source_match = all(part in source_file for part in case_name_parts[:2])
metadata_match = normalized_name in doc_case_name or normalized_name in doc_title
content_match = all(part in content_lower for part in case_name_parts[:2]) and year_str in content_lower

if source_match or metadata_match or content_match:
    found_match = True
```

### Citation Extraction Regex (From citation_gate.py)
```python
# Source: src/verification/citation_gate.py lines 60-94
# Case name pattern
CASE_NAME_PATTERN = re.compile(
    r'([A-Z][a-zA-Z\.\s]+\s+(?:vs?\.?|versus)\s+[A-Z][a-zA-Z\.\s&]+)',
    re.IGNORECASE
)

# SCC citation pattern
SCC_PATTERN = re.compile(
    r'\((\d{4})\)\s*(\d+)\s*SCC\s*(\d+)',
    re.IGNORECASE
)

# AIR citation pattern
AIR_PATTERN = re.compile(
    r'AIR\s*(\d{4})\s*SC\s*(\d+)',
    re.IGNORECASE
)

# Statute section patterns
SECTION_ACT_PATTERN = re.compile(
    r'Section\s+(\d+[A-Za-z]?)\s*(?:,|of|under)\s+([A-Za-z\.\s]+)',
    re.IGNORECASE
)
```

### Existing API Endpoint Pattern (From api.py)
```python
# Source: api.py lines 298-318
# Pattern for verification endpoints
@app.post("/verify-citation")
async def verify_citation(req: VerifyCitationRequest):
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
```

## Indian Legal Citation Format Reference

### Supreme Court Cases (SCC) - Preferred Format
**Pattern:** `Party A vs Party B (Year) Volume SCC Page`
**Example:** `Arnesh Kumar vs State of Bihar (2014) 8 SCC 273`
**Components:**
- Party names separated by "vs"
- Year in parentheses
- Volume number
- "SCC" reporter abbreviation
- Starting page number

### All India Reporter (AIR) Format
**Pattern:** `Party A vs Party B, AIR Year Court Page`
**Example:** `K.S. Puttaswamy vs Union of India, AIR 2017 SC 4161`
**Components:**
- Party names
- "AIR" prefix
- Year (year of judgment, not publication)
- Court abbreviation (SC, Del, Bom, Cal, etc.)
- Page number

### Statute Citation Format
**Pattern:** `Section N, Act Name, Year`
**Example:** `Section 438, Code of Criminal Procedure, 1973`
**Components:**
- "Section" prefix
- Section number (may include sub-clauses like 302A)
- Full act name
- Year of enactment

### High Court Abbreviations
| Court | Abbreviation |
|-------|--------------|
| Supreme Court | SC |
| Delhi High Court | Del |
| Bombay High Court | Bom |
| Calcutta High Court | Cal |
| Madras High Court | Mad |
| Karnataka High Court | Kar |
| Allahabad High Court | All |

## External API Availability

### IndianKanoon API (RECOMMENDED)
**Status:** Production-ready with documented API
**Documentation:** https://api.indiankanoon.org/documentation/
**Features:**
- Search endpoint with metadata filters (court, date range, citation)
- Document retrieval by ID
- Precedent relationships (citing/cited-by)
- AI Tags for legal issue classification
- Structural analysis (Facts, Issues, Arguments, Conclusion)

**Authentication:** API Token (header-based) or RSA public-private key
**Pricing:** Not publicly documented; contact required

**Key endpoints:**
| Endpoint | Purpose |
|----------|---------|
| `/search/` | Query with filters (doctypes, fromdate, todate) |
| `/doc/<docid>/` | Get document content |
| `/docmeta/<docid>/` | Get metadata including citations |

**Integration recommendation:** Start with internal Pinecone database, add IndianKanoon API for citation relationship enrichment in future phase.

### SCC Online / Manupatra
**Status:** No documented public API
**Access:** Subscription-based web platform only
**Recommendation:** Not viable for programmatic access

## Jurisdiction Hierarchy (Indian Legal System)

### Binding Authority Rules
1. **Supreme Court** decisions bind ALL courts in India (Article 141)
2. **High Court** decisions bind subordinate courts in its territorial jurisdiction
3. High Court decisions from other states are **persuasive only**

### Recommendation Ranking Algorithm
```
Score = (SemanticSimilarity * 0.5) + (JurisdictionWeight * 0.3) + (RecencyWeight * 0.2)

JurisdictionWeight:
- Supreme Court: 1.0
- Same High Court as filing: 0.9
- Other High Court: 0.6
- Tribunal: 0.4

RecencyWeight:
- Last 5 years: 1.0
- 5-10 years: 0.8
- 10-20 years: 0.6
- Older: 0.4
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Keyword search | Semantic vector search | 2022-2023 | Better issue-based retrieval |
| Flat result lists | Precedent graphs with citation networks | 2024 | Understand case relationships |
| Manual citation formatting | Template-based formatters | 2020+ | Consistent professional output |

**Deprecated/outdated:**
- IPC, CrPC, IEA citations for matters after July 1, 2024 - use BNS, BNSS, BSA (existing code_mapper handles this)

## Open Questions

Things that couldn't be fully resolved:

1. **IndianKanoon API Pricing**
   - What we know: API exists, documentation is public, requires token
   - What's unclear: Cost per request, rate limits, terms of service
   - Recommendation: Start with internal Pinecone database; evaluate IndianKanoon for enrichment after launch

2. **Metadata Consistency Across Data Sources**
   - What we know: HuggingFace datasets use different field names (Court_Name vs court, Titles vs title)
   - What's unclear: Full schema variance across all ingested data
   - Recommendation: Audit existing Pinecone index metadata; build defensive multi-field checking (already pattern in citation_verifier.py)

3. **UI Component for Verification Badges**
   - What we know: Checkmarks/badges are standard UI pattern for trust indicators
   - What's unclear: Exact Streamlit/frontend framework capabilities for custom badges
   - Recommendation: Use HTML inline styling (already supported in api.py drafts); consider dedicated UI phase later

## Sources

### Primary (HIGH confidence)
- Existing codebase: `src/verification/citation_verifier.py`, `citation_gate.py`, `section_validator.py` - patterns for verification
- Pinecone documentation: https://docs.pinecone.io/guides/search/filter-by-metadata - metadata filtering syntax
- IndianKanoon API documentation: https://api.indiankanoon.org/documentation/ - external API capabilities

### Secondary (MEDIUM confidence)
- Indian legal citation standards (Bluebook Table T2.18, SCC format guide)
- Jurisdiction hierarchy (Article 141, stare decisis in India)

### Tertiary (LOW confidence - needs validation)
- IndianKanoon API pricing and availability for commercial use
- Exact field names in all Pinecone metadata (requires audit)

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - All libraries already in project or stdlib
- Architecture: HIGH - Patterns derived from existing codebase
- Pitfalls: HIGH - Based on Phase 1 implementation experience
- External APIs: MEDIUM - API exists but pricing/terms unclear
- Citation formats: HIGH - Well-documented legal standards

**Research date:** 2026-01-25
**Valid until:** 2026-03-25 (60 days - stable domain, existing infrastructure)
