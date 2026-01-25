# State: Vakalat AI

## Current Position

**Phase:** 2 of 6 - Template Storage
**Plan:** 1 of 2 complete
**Status:** In progress
**Progress:** [############--------] 6/11 plans (~55%)

**Last activity:** 2026-01-25 - Completed 02-01-PLAN.md (Template Schemas)

## Project Reference

See: .planning/PROJECT.md (updated 2026-01-22)

**Core value:** Trustworthy legal document drafting that saves lawyers hours of work
**Current focus:** Milestone v1.0 - Professional Document Drafting

## Phase 2 Context

**Goal:** Template storage layer with court-specific formatting and validation

**Requirements:**
- TMPL-01: Template schema with document type validation [IN PROGRESS - 02-01]
- TMPL-02: Court-level formatting requirements [DONE - 02-01]

**Success Criteria:**
1. Template schema validates document types and court levels [DONE - 02-01]
2. FormattingRequirements contain Supreme Court Rules 2013 defaults [DONE - 02-01]
3. TemplateRepository can load/list/save templates from JSON files [DONE - 02-01]

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
- **Plan 02-01 complete**: Template schemas with Pydantic models and TemplateRepository

### What's Next
1. Execute 02-02-PLAN.md: Create template JSON files for bail application
2. Continue Phase 2: Template storage layer
3. Begin Phase 3: Core Drafting integration

### Open Questions
- External legal database API availability (IndianKanoon, SCC Online, Manupatra)
- WeasyPrint Windows compatibility (Phase 6 concern)

### Key Artifacts Created
- `src/verification/` - Full verification module with 8 files
- `src/data/mappings/*.json` - 283 section mappings across 3 code pairs
- `src/templates/schemas.py` - Pydantic models for template validation
- `src/templates/storage.py` - TemplateRepository for JSON file CRUD
- `src/templates/__init__.py` - Public exports for templates module
- `api.py` - Updated with all verification integrations

---
*State initialized: 2026-01-22*
*Last updated: 2026-01-25 - Completed 02-01-PLAN.md*
