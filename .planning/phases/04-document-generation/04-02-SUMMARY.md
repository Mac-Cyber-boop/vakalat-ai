---
phase: 04-document-generation
plan: 02
subsystem: api
tags: [prompts, legal-language, llm, document-generation]

# Dependency graph
requires:
  - phase: 02-template-storage
    provides: CourtLevel enum for court-specific prompt variants
provides:
  - System prompts for formal Indian legal language generation
  - Court-specific prompt variants (Supreme/High/District courts)
  - Field-specific generation guidance (grounds_for_bail, relief_sought, facts_summary, etc.)
  - get_generation_prompt builder function with citation integration
affects: [04-03-document-filler, 04-04-draft-api]

# Tech tracking
tech-stack:
  added: []
  patterns: [System prompt composition, Court-level differentiation, Citation integration in prompts]

key-files:
  created:
    - src/generation/prompts.py
  modified:
    - src/generation/__init__.py

key-decisions:
  - "BASE_LEGAL_TONE_PROMPT enforces Hon'ble Court, passive voice, formal submissions per DOC-02"
  - "COURT_SPECIFIC_PROMPTS differentiate by court level (Article 136/142 for SC, 226/227 for HC)"
  - "FIELD_GENERATION_PROMPTS provide task-specific guidance for 6 common legal fields"
  - "get_generation_prompt combines base + court + field + citations for complete system prompt"

patterns-established:
  - "System prompts use triple-quoted strings for multi-line formatting"
  - "Court-specific prompts use CourtLevel enum keys for type safety"
  - "Citation integration adds INTEGRATION RULES section when citations provided"

# Metrics
duration: 7min
completed: 2026-01-26
---

# Phase 4 Plan 2: Legal Tone System Prompts Summary

**Reusable prompt templates enforcing formal Indian legal register with Hon'ble Court address, passive voice submissions, and court-specific requirements**

## Performance

- **Duration:** 7 min
- **Started:** 2026-01-26T20:34:45Z
- **Completed:** 2026-01-26T20:41:18Z
- **Tasks:** 2
- **Files modified:** 2

## Accomplishments
- Created BASE_LEGAL_TONE_PROMPT with formal legal register requirements
- Built COURT_SPECIFIC_PROMPTS for Supreme Court, High Court, and District Court variants
- Implemented FIELD_GENERATION_PROMPTS for 6 common fields (grounds_for_bail, relief_sought, facts_summary, legal_grounds, cause_of_action, prayer)
- Developed get_generation_prompt builder function that combines base + court + field + citations
- Exported all prompt utilities through src/generation module

## Task Commits

Each task was committed atomically:

1. **Task 1: Create legal tone system prompts** - `dd87bdf` (feat)
2. **Task 2: Add prompts to module exports** - `8195fc4` (feat)

## Files Created/Modified
- `src/generation/prompts.py` - System prompts module with BASE_LEGAL_TONE_PROMPT, COURT_SPECIFIC_PROMPTS, FIELD_GENERATION_PROMPTS, and get_generation_prompt function (224 lines)
- `src/generation/__init__.py` - Added prompt exports to existing fact model exports

## Decisions Made

**Prompt composition architecture:**
The get_generation_prompt function uses a builder pattern that combines multiple prompt components:
1. BASE_LEGAL_TONE_PROMPT (always included)
2. Court-specific prompt (if court_level matches)
3. Field-specific prompt (if field_name matches, else generic guidance)
4. Citation integration instructions (if citations provided)

This allows flexible composition while maintaining consistency.

**Court-specific differentiation:**
- **Supreme Court:** References Article 136/142 constitutional powers, "most respectfully submitted", Latin maxims
- **High Court:** References Article 226/227 powers, "respectfully submitted", jurisdictional facts emphasis
- **District Court:** Plain legal English, minimal Latin, "Your Honour" address, focus on statutory provisions

This matches actual Indian legal practice where Supreme Court submissions are most formal.

**Field-specific guidance:**
Implemented detailed task descriptions for 6 common fields:
- grounds_for_bail: 6-point structure (flight risk, cooperation, nature of accusation, health, conviction likelihood, custody duration)
- relief_sought: WHEREFORE construction with lettered sub-clauses
- facts_summary: Numbered chronological paragraphs
- legal_grounds: Legal principle → Cite authority → Apply to facts
- cause_of_action: Define, date, place, liable party (for jurisdiction establishment)
- prayer: Formal prayer section with humble supplication tone

These prompts guide LLM to produce lawyer-quality output.

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

Minor syntax error during file creation (escaped newline characters in string literals). Fixed by editing lines 211 and 225-227 to use proper \n escape sequences.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

Ready for Phase 4 Plan 3 (DocumentFiller):
- System prompts enforce formal legal language per DOC-02
- Court-specific variants available for all three court levels
- Field-specific guidance ready for content generation
- get_generation_prompt function can be called by DocumentFiller to build LLM prompts
- Citation integration instructions support verified citation injection

**Blocker:** None. Prompts are static strings, no external dependencies.

---
*Phase: 04-document-generation*
*Completed: 2026-01-26*
