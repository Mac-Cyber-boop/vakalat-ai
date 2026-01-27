# State: Vakalat AI

## Current Position

**Phase:** 5 of 6 - Template Management (IN PROGRESS)
**Plan:** 3 of 5 complete
**Status:** In progress
**Progress:** [████████████████████] 20/22 plans (91%)

**Last activity:** 2026-01-27 - Completed 05-03-PLAN.md (Template Lifecycle Management)

## Project Reference

See: .planning/PROJECT.md (updated 2026-01-22)

**Core value:** Trustworthy legal document drafting that saves lawyers hours of work
**Current focus:** Milestone v1.0 - Professional Document Drafting

## Phase 5 Context (IN PROGRESS)

**Goal:** Template management - upload, versioning, and lifecycle control

**Requirements:**
- TMPL-01: Upload new template versions [IN PROGRESS]
- TMPL-02: Template lifecycle management [IN PROGRESS]
- TMPL-03: Template versioning [IN PROGRESS]

**Completed Plans:**
- 05-01: Template schema extension (status, changelog, semver) [DONE]
- 05-02: Upload validation and versioning utilities [DONE]
- 05-03: Template lifecycle management (status transitions, usability gating) [DONE]

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
| Field names mirror template required_fields exactly | 1:1 mapping between fact models and template schemas | 04-01 |
| BASE_LEGAL_TONE_PROMPT enforces Hon'ble Court, passive voice | Formal legal register requirements per DOC-02 | 04-02 |
| COURT_SPECIFIC_PROMPTS differentiate by court level | Article 136/142 for SC, 226/227 for HC per Indian practice | 04-02 |
| get_generation_prompt combines base + court + field + citations | Builder pattern for flexible prompt composition | 04-02 |
| Lifecycle state machine | Templates progress active -> deprecated -> archived, cannot go backwards | 05-01 |
| Direct archival allowed | Active templates can skip deprecated for immediate decommission | 05-01 |
| Backward compatible defaults | Existing templates load with status=ACTIVE, changelog=[] | 05-01 |
| Validation pipeline pattern | Early returns on validation failure for clear error messages | 05-02 |
| Result objects for outcomes | UploadValidationResult/UploadProcessResult provide structured results | 05-02 |
| Change description required for updates | Updates must include change_description for changelog | 05-02 |
| 500KB max template size | MAX_TEMPLATE_SIZE_BYTES prevents oversized uploads | 05-02 |
| UsabilityStatus enum separate from TemplateStatus | Clear separation between template status and usability check result | 05-03 |
| UsabilityResult with can_use flag | Boolean flag makes conditional checks simple | 05-03 |
| Descriptive messages for deprecated/archived | User-facing messages explain what's happening and suggest action | 05-03 |

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
| System prompt composition pattern | Base + court-specific + field-specific + citation integration | 04-02 |
| Triple-quoted strings for prompts | Multi-line formatting for system prompt readability | 04-02 |
| ALLOWED_TRANSITIONS dict pattern | State -> set of valid next states for lifecycle | 05-01 |
| Semver validation via field_validator | ChangelogEntry.version validated at parse time | 05-01 |
| Semver library for version comparison | compare_versions, bump_version use python-semver | 05-02 |
| Validation pipeline stages | File size -> JSON syntax -> schema -> version in sequence | 05-02 |
| Result objects for lifecycle | StatusChangeResult, UsabilityResult for structured outcomes | 05-03 |
| Usability gating pattern | WARNING for deprecated, BLOCKED for archived templates | 05-03 |

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
- **Phase 4 complete**: Document generation
  - Plan 04-01: Fact collection models (BailApplicationFacts, LegalNoticeFacts, AffidavitFacts, PetitionFacts)
  - Plan 04-02: Legal tone system prompts (BASE_LEGAL_TONE_PROMPT, COURT_SPECIFIC_PROMPTS, get_generation_prompt)
  - Plan 04-03: Document drafter with template + facts + prompts
  - Plan 04-04: Document reviser for iterative refinement
  - Plan 04-05: Document generation API endpoints
- **Phase 5 in progress**: Template management
  - Plan 05-01: Template schema extension (TemplateStatus, ChangelogEntry, ALLOWED_TRANSITIONS)
  - Plan 05-02: Upload validation and versioning utilities (versioning.py, upload.py)
  - Plan 05-03: Template lifecycle management (lifecycle.py with status transitions)

### What's Next
1. Continue Phase 5: Template management (2 plans remaining)
   - 05-04: Template management API endpoints
   - 05-05: Template admin UI
2. Then Phase 6: Production readiness

### Open Questions
- External legal database API availability (IndianKanoon, SCC Online, Manupatra)
- WeasyPrint Windows compatibility (Phase 6 concern)

### Key Artifacts Created
- `src/verification/` - Full verification module with 8 files
- `src/data/mappings/*.json` - 283 section mappings across 3 code pairs
- `src/templates/schemas.py` - Pydantic models for template validation (extended with lifecycle)
- `src/templates/storage.py` - TemplateRepository for JSON file CRUD
- `src/templates/__init__.py` - Public exports for templates module
- `src/templates/data/*.json` - 12 default template files
- `src/citations/models.py` - Citation data models (CaseCitation, StatuteCitation, etc.)
- `src/citations/formatter.py` - CitationFormatter with SCC, AIR, statute formats
- `src/citations/__init__.py` - Citation module exports
- `src/citations/retriever.py` - PrecedentRetriever with jurisdiction-aware ranking
- `src/citations/recommender.py` - CitationRecommender orchestrator
- `api.py` - Updated with verification, template, citation, and generation API integrations
- `main.py` - Updated with citation badge CSS and precedent display
- `src/generation/models.py` - Pydantic fact collection models for document generation
- `src/generation/__init__.py` - Generation module exports
- `src/generation/prompts.py` - System prompts for formal legal language generation
- `src/generation/drafter.py` - DocumentDrafter for template-based generation
- `src/generation/reviser.py` - DocumentReviser for iterative refinement
- `src/templates/versioning.py` - Semver version comparison and changelog utilities
- `src/templates/upload.py` - Upload validation pipeline and processing
- `src/templates/lifecycle.py` - Template lifecycle status transitions and usability gating

---
*State initialized: 2026-01-22*
*Last updated: 2026-01-27 - Completed 05-03-PLAN.md (Template Lifecycle Management)*
