---
phase: 01-trust-foundation
plan: 01
subsystem: verification
tags: [pydantic, json, legal-codes, ipc, bns, crpc, bnss, evidence-act, bsa]

# Dependency graph
requires: []
provides:
  - LegalCodeMapper class for IPC->BNS, CrPC->BNSS, IEA->BSA mappings
  - Pydantic models for verification results
  - Comprehensive section mapping JSON data (283 mappings total)
affects: [01-02, 01-03, 01-04, 01-05]

# Tech tracking
tech-stack:
  added: [structlog, cachetools, pydantic>=2.0]
  patterns: [LRU caching for lookups, code normalization via aliases]

key-files:
  created:
    - src/verification/models.py
    - src/verification/code_mapper.py
    - src/data/mappings/ipc_to_bns.json
    - src/data/mappings/crpc_to_bnss.json
    - src/data/mappings/iea_to_bsa.json
  modified:
    - requirements.txt
    - src/verification/__init__.py

key-decisions:
  - "Code normalization via alias dictionary to handle variations like 'IPC', 'Indian Penal Code', 'I.P.C.'"
  - "LRU cache on map_section() with 1000 entry limit for performance"
  - "Separate CodeMappingStatus enum from VerificationStatus for clarity"

patterns-established:
  - "JSON mapping files with _metadata key for versioning"
  - "Pydantic models with Field descriptions following api.py conventions"
  - "src/ package structure for new code modules"

# Metrics
duration: 10min
completed: 2026-01-22
---

# Phase 01 Plan 01: Legal Code Mapping Foundation Summary

**LegalCodeMapper with 283 section mappings across IPC->BNS, CrPC->BNSS, and Evidence Act->BSA for 2024 criminal law reform**

## Performance

- **Duration:** 10 min
- **Started:** 2026-01-22T16:02:35Z
- **Completed:** 2026-01-22T16:12:49Z
- **Tasks:** 3
- **Files modified:** 8

## Accomplishments

- Created src/verification module with Pydantic models for verification results
- Built comprehensive JSON mapping files with 92 IPC, 87 CrPC, and 104 Evidence Act section mappings
- Implemented LegalCodeMapper class with code name normalization and LRU caching
- All three criminal law code pairs (IPC->BNS, CrPC->BNSS, IEA->BSA) now mappable

## Task Commits

Each task was committed atomically:

1. **Task 1: Create module structure and Pydantic models** - `3498f2b` (feat)
2. **Task 2: Create legal code mapping JSON files** - `3f8f1be` (feat)
3. **Task 3: Implement LegalCodeMapper class** - `2f482e3` (feat)

## Files Created/Modified

- `src/__init__.py` - Package marker for src module
- `src/data/__init__.py` - Package marker for data resources
- `src/verification/__init__.py` - Exports all verification classes and models
- `src/verification/models.py` - VerificationStatus, CodeMappingStatus enums; VerificationResult, CodeMappingResult, CaseCitationInput, StatuteCitationInput Pydantic models
- `src/verification/code_mapper.py` - LegalCodeMapper class with map_section(), get_new_equivalent(), is_section_valid() methods
- `src/data/mappings/ipc_to_bns.json` - 92 IPC to BNS section mappings
- `src/data/mappings/crpc_to_bnss.json` - 87 CrPC to BNSS section mappings
- `src/data/mappings/iea_to_bsa.json` - 104 Evidence Act to BSA section mappings
- `requirements.txt` - Added pydantic>=2.0, structlog, cachetools

## Decisions Made

1. **Code normalization via aliases**: Handle variations in code names (IPC, Indian Penal Code, I.P.C., etc.) through a lookup dictionary rather than fuzzy matching for deterministic behavior.

2. **LRU caching**: Used functools.lru_cache(maxsize=1000) on map_section() for performance since the same sections will be looked up repeatedly during document analysis.

3. **Separate enums**: Created CodeMappingStatus distinct from VerificationStatus because mapping operations have different possible outcomes (MAPPED, NO_MAPPING, UNKNOWN_CODE) than citation verification (VERIFIED, UNVERIFIED, BLOCKED, OUTDATED).

4. **JSON metadata convention**: Each mapping file includes `_metadata` key with source, target, effective_date, and version for tracking data provenance.

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

**Ready for:**
- Plan 02: Case citation verification against database
- Plan 03: Audit logging with structlog
- Plan 04: Section number validation
- Plan 05: Outdated code detection and flagging

**Dependencies provided:**
- LegalCodeMapper class ready for integration
- Pydantic models available for API response types
- Code mapping data covers high-frequency sections (302, 438, 65B)

**No blockers.**

---
*Phase: 01-trust-foundation*
*Completed: 2026-01-22*
