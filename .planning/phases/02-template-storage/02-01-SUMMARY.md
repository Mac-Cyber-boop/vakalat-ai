---
phase: 02-template-storage
plan: 01
subsystem: templates
tags: [pydantic, enum, schema, validation, repository-pattern]

# Dependency graph
requires:
  - phase: 01-trust-foundation
    provides: Pydantic model patterns from verification module
provides:
  - DocumentType and CourtLevel enums for type-safe template classification
  - FormattingRequirements with Supreme Court Rules 2013 defaults
  - TemplateField with validation for field_type and snake_case
  - LegalTemplate model combining all nested schemas
  - TemplateRepository for JSON-based template CRUD
affects: [02-template-storage-plan-02, 03-core-drafting]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Pydantic enum validation for type safety"
    - "Repository pattern for template storage"
    - "JSON file storage for version-control friendly templates"

key-files:
  created:
    - src/templates/__init__.py
    - src/templates/schemas.py
    - src/templates/storage.py
  modified: []

key-decisions:
  - "Use Pydantic field_validator for snake_case and field_type validation"
  - "Store templates as JSON files in data/ directory for version control transparency"
  - "Default FormattingRequirements to Supreme Court Rules 2013 standards"

patterns-established:
  - "Repository pattern: TemplateRepository for storage abstraction"
  - "Enum-based type safety: DocumentType and CourtLevel enums"
  - "Nested Pydantic models: LegalTemplate contains TemplateMetadata, FormattingRequirements, TemplateField"

# Metrics
duration: 4min
completed: 2026-01-25
---

# Phase 2 Plan 01: Template Schemas Summary

**Pydantic models for legal document templates with court-specific formatting and repository pattern for JSON file storage**

## Performance

- **Duration:** 4 min
- **Started:** 2026-01-25T07:03:13Z
- **Completed:** 2026-01-25T07:07:20Z
- **Tasks:** 2
- **Files modified:** 3

## Accomplishments
- Created DocumentType enum with 4 document types (bail_application, legal_notice, affidavit, petition)
- Created CourtLevel enum with 3 court levels (supreme_court, high_court, district_court)
- Implemented FormattingRequirements with Supreme Court Rules 2013 defaults (A4, Times New Roman 14pt, 1.5 spacing)
- Created TemplateField with field_validator for snake_case field_name and allowed field_types
- Built LegalTemplate combining all nested models with proper defaults
- Implemented TemplateRepository with get_template, list_templates, save_template, template_exists methods

## Task Commits

Each task was committed atomically:

1. **Task 1: Create template schemas with Pydantic models** - `fe12aba` (feat)
2. **Task 2: Create TemplateRepository with CRUD operations** - `3c6ccb9` (feat)

**Plan metadata:** (to be added)

## Files Created/Modified
- `src/templates/__init__.py` - Public exports for templates module
- `src/templates/schemas.py` - Pydantic models for template validation (DocumentType, CourtLevel, TemplateMetadata, FormattingRequirements, TemplateField, LegalTemplate)
- `src/templates/storage.py` - TemplateRepository class for JSON file CRUD operations

## Decisions Made
- **field_validator over model_validator:** Used Pydantic v2 field_validator for individual field validation (snake_case, field_type) as it's more precise than model-level validation
- **JSON file storage pattern:** Templates stored as `{doc_type}_{court_level}.json` files for version control transparency and human readability
- **Default factory for FormattingRequirements:** Using default_factory=FormattingRequirements in LegalTemplate to ensure each template gets fresh defaults
- **Regex pattern for snake_case:** Used `^[a-z][a-z0-9]*(_[a-z0-9]+)*$` pattern to validate field_name format

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness
- Template schemas ready for use by Plan 02 (template JSON files)
- TemplateRepository ready for API integration in future phases
- Data directory created at src/templates/data/ awaiting template files

---
*Phase: 02-template-storage*
*Completed: 2026-01-25*
