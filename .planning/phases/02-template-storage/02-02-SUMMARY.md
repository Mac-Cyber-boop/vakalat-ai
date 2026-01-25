---
phase: 02-template-storage
plan: 02
subsystem: templates
tags: [json, pydantic, fastapi, legal-templates, court-formatting]

# Dependency graph
requires:
  - phase: 02-01
    provides: LegalTemplate schema, TemplateRepository, DocumentType/CourtLevel enums
provides:
  - 12 default template JSON files (4 doc types x 3 court levels)
  - POST /templates/list API endpoint
  - POST /templates/get API endpoint
affects: [03-core-drafting, template-rendering, document-generation]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Court-specific formatting (SC: 14pt/4cm, HC: 14pt/3.5cm, DC: 12pt/3cm)"
    - "Template JSON storage with Pydantic validation"
    - "API endpoint patterns for template CRUD"

key-files:
  created:
    - src/templates/data/bail_application_supreme_court.json
    - src/templates/data/bail_application_high_court.json
    - src/templates/data/bail_application_district_court.json
    - src/templates/data/legal_notice_supreme_court.json
    - src/templates/data/legal_notice_high_court.json
    - src/templates/data/legal_notice_district_court.json
    - src/templates/data/affidavit_supreme_court.json
    - src/templates/data/affidavit_high_court.json
    - src/templates/data/affidavit_district_court.json
    - src/templates/data/petition_supreme_court.json
    - src/templates/data/petition_high_court.json
    - src/templates/data/petition_district_court.json
  modified:
    - api.py

key-decisions:
  - "Required field counts: bail (8), legal notice (7), affidavit (5), petition (7)"
  - "Template summaries returned for list endpoint (not full templates)"
  - "Enum validation with helpful error messages listing valid options"

patterns-established:
  - "Template filename convention: {doc_type}_{court_level}.json"
  - "Court-specific formatting variations based on Supreme Court Rules 2013"
  - "API returns 400 for invalid enums, 404 for non-existent templates"

# Metrics
duration: 8min
completed: 2026-01-25
---

# Phase 02 Plan 02: Template Data and API Summary

**12 default legal document templates with court-specific formatting plus /templates/list and /templates/get API endpoints**

## Performance

- **Duration:** 8 min
- **Started:** 2026-01-25T07:14:03Z
- **Completed:** 2026-01-25T07:22:01Z
- **Tasks:** 2
- **Files modified:** 13 (12 created + 1 modified)

## Accomplishments
- Created 12 template JSON files covering bail applications, legal notices, affidavits, and petitions
- Each template validated against LegalTemplate Pydantic schema
- Implemented court-specific formatting (Supreme Court 14pt/4cm, High Court 14pt/3.5cm, District Court 12pt/3cm)
- Added POST /templates/list endpoint with optional doc_type filtering
- Added POST /templates/get endpoint for full template retrieval

## Task Commits

Each task was committed atomically:

1. **Task 1: Create default template JSON files** - `bc9aa0f` (feat)
2. **Task 2: Add template API endpoints to api.py** - `a2433aa` (feat)

## Files Created/Modified

### Templates Created (src/templates/data/)
- `bail_application_supreme_court.json` - Bail app for Supreme Court with 8 required fields
- `bail_application_high_court.json` - Bail app for High Court
- `bail_application_district_court.json` - Bail app for District Court (12pt font)
- `legal_notice_supreme_court.json` - Legal notice with 7 required fields
- `legal_notice_high_court.json` - Legal notice for High Court
- `legal_notice_district_court.json` - Legal notice for District Court
- `affidavit_supreme_court.json` - Affidavit with 5 required fields
- `affidavit_high_court.json` - Affidavit for High Court
- `affidavit_district_court.json` - Affidavit for District Court
- `petition_supreme_court.json` - Petition with 7 required fields
- `petition_high_court.json` - Writ petition for High Court
- `petition_district_court.json` - Civil petition for District Court

### API Modified
- `api.py` - Added template imports, template_repo initialization, request models, and two POST endpoints

## Decisions Made

1. **Field counts by document type:**
   - Bail applications: 8 required fields (applicant details, FIR info, grounds, relief)
   - Legal notices: 7 required fields (sender/recipient, subject, demand, compliance period)
   - Affidavits: 5 required fields (deponent details, statement, verification place)
   - Petitions: 7 required fields (parties, cause of action, relief, jurisdiction)

2. **Court-level formatting variations:**
   - Supreme Court: 14pt font, 4cm left/right margins (per SC Rules 2013)
   - High Court: 14pt font, 3.5cm margins, 2.5cm top/bottom
   - District Court: 12pt font, 3cm margins for accessibility

3. **API design decisions:**
   - List endpoint returns summaries (not full templates) for efficiency
   - Get endpoint validates enums and returns helpful error messages
   - Invalid doc_type/court_level returns 400 with list of valid options
   - Non-existent template returns 404 with descriptive message

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None - dependencies already in place from Plan 02-01.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- Template storage layer complete with 12 default templates
- API endpoints ready for frontend integration
- Ready for Phase 03: Core Drafting integration
- Templates can be retrieved and used for document generation

---
*Phase: 02-template-storage*
*Completed: 2026-01-25*
