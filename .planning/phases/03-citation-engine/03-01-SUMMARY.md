---
phase: 03-citation-engine
plan: 01
subsystem: citations
tags: [pydantic, citations, scc, air, legal-formatting, indian-law]

# Dependency graph
requires:
  - phase: 01-trust-foundation
    provides: Pydantic model patterns (Field descriptions, type hints)
provides:
  - Citation data models (CaseCitation, StatuteCitation, VerificationBadge)
  - CitationFormatter class for SCC, AIR, statute formats
  - FormattedCitation with HTML output generation
affects: [03-citation-engine, 04-core-drafting]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - Pydantic computed_field for derived properties
    - Factory classmethods for badge creation (VerificationBadge.verified(), etc.)
    - Union types for polymorphic citation handling

key-files:
  created:
    - src/citations/models.py
    - src/citations/formatter.py
    - src/citations/__init__.py
  modified: []

key-decisions:
  - "Use 'vs' (not 'v.' or 'versus') per Indian legal convention"
  - "Unicode icons for badges (U+2713, U+26A0, U+21BB) for portability"
  - "HIGH_COURT_ABBREVIATIONS dict with 25+ courts for normalization"

patterns-established:
  - "CitationFormatter as static utility class (no state, pure functions)"
  - "format_with_badge() combines citation + badge into FormattedCitation"
  - "computed_field html_output generates HTML lazily"

# Metrics
duration: 8min
completed: 2026-01-25
---

# Phase 3 Plan 1: Citation Models and Formatter Summary

**Pydantic citation models (CaseCitation, StatuteCitation) with CitationFormatter outputting SCC "(Year) Volume SCC Page" and AIR "AIR Year Court Page" formats**

## Performance

- **Duration:** 8 min
- **Started:** 2026-01-25T17:43:18Z
- **Completed:** 2026-01-25T17:51:00Z
- **Tasks:** 3
- **Files created:** 3

## Accomplishments
- CitationBase, CaseCitation, StatuteCitation models with proper Pydantic validation
- VerificationBadge with factory classmethods for verified/unverified/outdated states
- FormattedCitation with computed html_output property combining text + badge HTML
- CitationFormatter with format_scc(), format_air(), format_statute() static methods
- HIGH_COURT_ABBREVIATIONS dictionary covering 25+ Indian courts

## Task Commits

Each task was committed atomically:

1. **Task 1: Create citation data models** - `68e4c4e` (feat)
2. **Task 2: Create citation formatter** - `8de297e` (feat)
3. **Task 3: Create module init and verify imports** - `0252413` (feat)

## Files Created/Modified
- `src/citations/models.py` - Pydantic models for citations and badges (172 lines)
- `src/citations/formatter.py` - Citation formatting utilities (274 lines)
- `src/citations/__init__.py` - Module exports (26 lines)

## Decisions Made
- Used "vs" (not "v." or "versus") per Indian legal convention for case citations
- Unicode icons for badges (checkmark U+2713, warning U+26A0, refresh U+21BB) for portability across platforms
- Extensive HIGH_COURT_ABBREVIATIONS dict with multiple alias formats (e.g., "delhi", "delhi_high_court", "del" all map to "Del")
- computed_field for html_output to avoid computing HTML when not needed

## Deviations from Plan
None - plan executed exactly as written.

## Issues Encountered
None.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- Citation models ready for retrieval integration (03-02)
- VerificationBadge integrates with existing verification module
- FormattedCitation.html_output ready for UI rendering
- Citation patterns established for document generation

---
*Phase: 03-citation-engine*
*Completed: 2026-01-25*
