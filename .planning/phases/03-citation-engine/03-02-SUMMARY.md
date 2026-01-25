---
phase: 03-citation-engine
plan: 02
subsystem: api
tags: [pinecone, langchain, pydantic, retriever, ranking]

# Dependency graph
requires:
  - phase: 02-template-storage
    provides: Template schemas and storage patterns
provides:
  - PrecedentRetriever with jurisdiction-aware ranking
  - RetrievedPrecedent Pydantic model with composite scoring
  - JurisdictionWeight constants (SC=1.0, Same HC=0.9, Other HC=0.6)
  - Court name normalization (50+ variants to canonical IDs)
affects: [03-drafting-pipeline, citation-suggestions, case-law-retrieval]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - Composite ranking (50% semantic + 30% jurisdiction + 20% recency)
    - Court name normalization via lookup dictionary
    - Over-fetch and re-rank retrieval pattern

key-files:
  created:
    - src/citations/retriever.py
  modified:
    - src/citations/__init__.py

key-decisions:
  - "50/30/20 weighting for final score (semantic/jurisdiction/recency)"
  - "Over-fetch 2x then re-rank for quality"
  - "Supreme Court = 1.0, Same HC = 0.9, Other HC = 0.6, Tribunal = 0.4"
  - "Recency: 5yr=1.0, 10yr=0.8, 20yr=0.6, older=0.4"

patterns-established:
  - "Jurisdiction weight class with named constants"
  - "Court normalization dictionary pattern"
  - "Defensive metadata extraction (try multiple field names)"

# Metrics
duration: 3min
completed: 2026-01-25
---

# Phase 3 Plan 02: Precedent Retriever Summary

**Jurisdiction-aware case law retriever with composite ranking (semantic + court authority + recency)**

## Performance

- **Duration:** 3 min
- **Started:** 2026-01-25T17:43:40Z
- **Completed:** 2026-01-25T17:46:49Z
- **Tasks:** 3
- **Files modified:** 2

## Accomplishments
- PrecedentRetriever class that fetches from Pinecone and re-ranks by jurisdiction authority
- RetrievedPrecedent Pydantic model with semantic_score, jurisdiction_weight, recency_weight, and final_score
- JurisdictionWeight constants: Supreme Court (1.0) > Same HC (0.9) > Other HC (0.6) > Tribunal (0.4)
- Comprehensive court name normalization (50+ Indian court variants to canonical IDs)
- Recency weighting that boosts cases from last 5 years

## Task Commits

Each task was committed atomically:

1. **Task 1+2: RetrievedPrecedent model, JurisdictionWeight, PrecedentRetriever** - `e50d455` (feat)
2. **Task 3: Export retriever from src.citations** - `96d5ed2` (feat)

## Files Created/Modified
- `src/citations/retriever.py` - PrecedentRetriever with ranking logic, RetrievedPrecedent model, JurisdictionWeight constants, court normalizations
- `src/citations/__init__.py` - Added retriever exports to module public API

## Decisions Made
- **50/30/20 score weighting**: Semantic similarity (50%) dominates but jurisdiction (30%) and recency (20%) adjust rankings
- **Over-fetch 2x**: Retrieve double requested results to have room for re-ranking
- **Defensive metadata extraction**: Try multiple field names (court, Court_Name, court_name, forum) since Pinecone metadata varies
- **Default year 2000**: When year cannot be extracted, default to 2000 (20+ years old = 0.4 recency weight)
- **Fall-back unfiltered search**: If source_type filter fails, do unfiltered search with post-filtering

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None - all tasks completed successfully.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness
- PrecedentRetriever ready for integration with drafting pipeline
- Can be used by /draft endpoint to suggest relevant case law
- CITE-01 (suggest case law) and CITE-06 (jurisdiction priority) requirements addressed

---
*Phase: 03-citation-engine*
*Completed: 2026-01-25*
