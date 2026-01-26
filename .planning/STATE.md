# State: Vakalat AI

## Current Position

**Phase:** 3 of 6 - Citation Engine (COMPLETE)
**Plan:** 5 of 5 complete
**Status:** Phase complete
**Progress:** [####################] 12/12 plans (100%)

**Last activity:** 2026-01-26 - Completed 03-05-PLAN.md (Citation Verification UI)

## Project Reference

See: .planning/PROJECT.md (updated 2026-01-22)

**Core value:** Trustworthy legal document drafting that saves lawyers hours of work
**Current focus:** Milestone v1.0 - Professional Document Drafting

## Phase 3 Context (COMPLETE)

**Goal:** Citation engine for proper legal citation formatting and retrieval

**Requirements:**
- CITE-02: Statute citation formatting [DONE - 03-04]
- CITE-03: Case citation formatting [DONE - 03-01]
- CITE-01: Suggest relevant case law [DONE - 03-04]
- CITE-06: Prioritize filing court jurisdiction [DONE - 03-02]
- CITE-04: Verification badges [DONE - 03-05]

**Success Criteria:**
1. Citation models validate SCC, AIR, statute formats [DONE - 03-01]
2. CitationFormatter produces standard Indian legal citation strings [DONE - 03-01]
3. VerificationBadge integrates with verification system [DONE - 03-01]
4. PrecedentRetriever with jurisdiction-aware ranking [DONE - 03-02]
5. /citations/recommend API endpoint [DONE - 03-04]
6. /citations/format-statute API endpoint [DONE - 03-04]
7. Streamlit UI shows verification badges [DONE - 03-05]

## Accumulated Context

### Key Decisions
| Decision | Rationale | Plan |
|----------|-----------|------|
| Template-first architecture | LLM fills content into templates, not generates structure | PROJECT |
| Citation verification from day 1 | Hallucination is dealbreaker for lawyers | PROJECT |
| Reuse existing Pinecone + Verifier agent infrastructure | Avoid rebuilding what works | PROJECT |
| Code normalization via alias dictionary | Deterministic lookup for IPC/Indian Penal Code/I.P.C. variants | 01-01 |
| LRU cache on map_section() | Same sections looked up repeatedly during document analysis | 01-01 |
| Separate CodeMappingStatus from VerificationStatus | Different outcomes for mapping vs verification operations | 01-01 |
| Old codes return OUTDATED, not UNVERIFIED | IPC/CrPC/IEA still valid for pre-2024 cases, OUTDATED indicates should be updated | 01-03 |
| Module-level cached mapper for quick_check | Fast pre-filtering without Pinecone queries | 01-03 |
| Semantic search with post-filtering for cases | Database metadata varies, so verify match via filename/content | 01-02 |
| Manual dict cache for verification results | Allows cache clearing and 0ms timing for cache hits | 01-02 |
| Fail-closed citation gate | Unverified citations blocked by default, explicit allow-list | 01-04 |
| Omit and continue for blocked citations | Remove unverified citations, add footnote, don't fail generation | 01-04 |
| OUTDATED passes through CitationGate | Old code citations still valid for pre-2024 cases, not blocked | 01-04 |
| Regex patterns for both section orderings | Captures "Section 302, IPC" and "IPC Section 302" variants | 01-05 |
| Additive response fields | New fields don't break existing API clients | 01-05 |
| JSON file storage for templates | Version control transparency, human readable, no database overhead | 02-01 |
| Repository pattern for template CRUD | Abstraction layer for future storage backend changes | 02-01 |
| Required field counts by doc type | Bail (8), Legal notice (7), Affidavit (5), Petition (7) | 02-02 |
| Template summaries in list endpoint | Efficiency - don't return full templates for listing | 02-02 |
| Enum validation with helpful errors | 400 response includes list of valid options | 02-02 |
| Use "vs" per Indian legal convention | Not "v." or "versus" for case citations | 03-01 |
| Unicode icons for badges | Checkmark, warning, refresh icons are portable across platforms | 03-01 |
| 50/30/20 weighting for final score | Semantic (50%) + jurisdiction (30%) + recency (20%) | 03-02 |
| Over-fetch 2x then re-rank | Retrieve double results for quality re-ranking | 03-02 |
| Supreme Court = 1.0, Same HC = 0.9 | Jurisdiction weights reflect binding authority | 03-02 |
| top_k validation (1-10) | Prevent excessive queries while allowing reasonable results | 03-04 |
| Default filing_court = supreme_court | Supreme Court precedents are universally authoritative | 03-04 |
| Dark theme for precedent cards | Matches existing Streamlit app theme (#1E293B) | 03-05 |
| Expander for precedents | Keeps main response clean, user can expand if interested | 03-05 |
| Graceful degradation on import errors | App works even if citation modules unavailable | 03-05 |

### Technical Decisions
| Decision | Details | Plan |
|----------|---------|------|
| JSON mapping files with _metadata | Versioning and provenance tracking for legal data | 01-01 |
| src/ package structure | New code modules go in src/, keeps root clean | 01-01 |
| Pydantic models with Field descriptions | Follows api.py conventions for consistency | 01-01 |
| structlog for audit logging | JSON output for container compatibility | 01-02 |
| Pinecone filter with source_type and source_book | Efficient metadata-based filtering for section lookup | 01-03 |
| Multiple regex patterns for citation extraction | Case: vs format, SCC, AIR; Statute: Section N, Act formats | 01-04 |
| Response metadata for verification | citations_verified and citations_blocked fields in /draft response | 01-04 |
| HTML warning box styling | Yellow background with amber border for outdated code notices | 01-05 |
| Pydantic v2 field_validator | Used for snake_case and field_type validation in TemplateField | 02-01 |
| Template filename convention | {doc_type}_{court_level}.json for predictable file paths | 02-01 |
| Court-specific formatting | SC: 14pt/4cm, HC: 14pt/3.5cm, DC: 12pt/3cm margins | 02-02 |
| Pydantic computed_field | html_output computed lazily for FormattedCitation | 03-01 |
| CitationFormatter as static utility | No state, pure functions for formatting | 03-01 |
| HIGH_COURT_ABBREVIATIONS dict | 25+ court codes with multiple alias formats | 03-01 |
| Defensive metadata extraction | Try multiple field names for court/year since Pinecone metadata varies | 03-02 |
| COURT_NORMALIZATIONS dict | 50+ court name variants to canonical IDs | 03-02 |

### Blockers
(None)

### Issues Log
(None)

## Session Continuity

### What Was Done
- Requirements defined: 22 v1 requirements across TRUST, DOC, CITE, TMPL, PROD categories
- Research completed: Template-first architecture, citation verification critical path
- Roadmap created: 6 phases with 100% requirement coverage
- **Phase 1 complete**: All 5 plans executed successfully
- **Phase 2 complete**: Template storage layer with schemas, repository, and API
  - Plan 02-01: Template schemas with Pydantic models and TemplateRepository
  - Plan 02-02: 12 default templates and /templates/list, /templates/get endpoints
- **Phase 3 complete**: Citation engine
  - Plan 03-01: Citation models and formatter (SCC, AIR, statute formats)
  - Plan 03-02: Precedent retriever with jurisdiction-aware ranking
  - Plan 03-03: Citation recommender orchestrator
  - Plan 03-04: Citation API endpoints (/citations/recommend, /citations/format-statute)
  - Plan 03-05: Citation verification UI in Streamlit

### What's Next
1. Begin Phase 4: Core drafting integration
2. Then Phase 5: Document generation
3. Then Phase 6: Production readiness

### Open Questions
- External legal database API availability (IndianKanoon, SCC Online, Manupatra)
- WeasyPrint Windows compatibility (Phase 6 concern)

### Key Artifacts Created
- `src/verification/` - Full verification module with 8 files
- `src/data/mappings/*.json` - 283 section mappings across 3 code pairs
- `src/templates/schemas.py` - Pydantic models for template validation
- `src/templates/storage.py` - TemplateRepository for JSON file CRUD
- `src/templates/__init__.py` - Public exports for templates module
- `src/templates/data/*.json` - 12 default template files
- `src/citations/models.py` - Citation data models (CaseCitation, StatuteCitation, etc.)
- `src/citations/formatter.py` - CitationFormatter with SCC, AIR, statute formats
- `src/citations/__init__.py` - Citation module exports
- `src/citations/retriever.py` - PrecedentRetriever with jurisdiction-aware ranking
- `src/citations/recommender.py` - CitationRecommender orchestrator
- `api.py` - Updated with verification, template, and citation API integrations
- `main.py` - Updated with citation badge CSS and precedent display

---
*State initialized: 2026-01-22*
*Last updated: 2026-01-26 - Completed 03-05-PLAN.md (Citation Verification UI)*
