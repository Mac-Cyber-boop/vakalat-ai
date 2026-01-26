# Roadmap: Vakalat AI v1.0

**Milestone:** Professional Document Drafting
**Created:** 2026-01-22
**Phases:** 6
**Total Requirements:** 22

## Overview

This roadmap delivers professional legal document drafting capabilities for Vakalat AI. The architecture follows a trust-first approach: citation verification before generation, templates before content, grounded outputs before user-facing features. Each phase delivers a complete, verifiable capability that builds toward the end goal of trustworthy legal documents that save lawyers hours of work.

Research confirmed the critical insight: "current AI tools hallucinate fake cases" - one hallucinated citation permanently destroys lawyer trust. This roadmap addresses that concern by establishing citation verification as the foundation (Phase 1) before any document generation occurs.

---

## Phase 1: Trust Foundation

**Goal:** System can verify legal citations and map legal codes before any document generation

**Dependencies:** None (foundation phase)

**Requirements:**
- TRUST-01: System verifies every case citation against legal database before including in document
- TRUST-02: Unverified citations are blocked from output (never generate unverified)
- TRUST-03: System maps old legal codes to new (IPC->BNS, CrPC->BNSS, Evidence Act->BSA)
- TRUST-04: System verifies section numbers exist and aren't repealed before citing
- TRUST-05: System flags outdated legal code references and suggests current equivalents

**Success Criteria:**
1. Given a case citation, system queries Pinecone and returns verification status (verified/unverified) within 2 seconds
2. Given an IPC section number, system returns the equivalent BNS section with confidence score
3. Given a statute citation (e.g., "Section 302 IPC"), system confirms section exists and is not repealed
4. System blocks any citation that cannot be verified against the legal database (fail-safe behavior)
5. System logs all verification attempts with results for audit trail

**Research Flags:** Integration with external legal databases (IndianKanoon, SCC Online) may need API investigation during planning.

**Plans:** 5 plans

Plans:
- [x] 01-01-PLAN.md — Legal code mapping foundation (LegalCodeMapper with IPC->BNS, CrPC->BNSS, IEA->BSA)
- [x] 01-02-PLAN.md — Citation verifier and audit logging (CitationVerifier + structlog infrastructure)
- [x] 01-03-PLAN.md — Section validator (verify sections exist in actual law, repeal status)
- [x] 01-04-PLAN.md — Citation gate and API integration (block unverified citations in draft endpoint)
- [x] 01-05-PLAN.md — Outdated code detection and full integration (complete trust foundation)

---

## Phase 2: Template Storage

**Goal:** System stores and retrieves court-specific document templates with proper formatting

**Dependencies:** None (can run parallel with Phase 1)

**Requirements:**
- TMPL-01: System provides default templates for common documents (bail application, legal notice, affidavit, petition)
- TMPL-02: Templates define court-specific formatting requirements

**Success Criteria:**
1. User can list available templates by document type (bail application, legal notice, affidavit, petition)
2. User can retrieve a template for a specific court (Supreme Court, High Court, District Court)
3. Each template defines required input fields (parties, dates, case number, relief sought)
4. Templates include court-specific formatting metadata (margins, font, spacing requirements)

**Research Flags:** None - standard file-based storage pattern.

**Plans:** 2 plans

Plans:
- [x] 02-01-PLAN.md — Template schemas and repository (Pydantic models + TemplateRepository class)
- [x] 02-02-PLAN.md — Template data and API integration (12 default templates + FastAPI endpoints)

---

## Phase 3: Citation Engine

**Goal:** System retrieves relevant precedents and formats citations in Indian legal standard

**Dependencies:** Phase 1 (Trust Foundation must exist for citation verification)

**Requirements:**
- CITE-01: System suggests relevant case law from database based on legal issues
- CITE-02: System cites specific statutory provisions with proper format (Section X, Act Name, Year)
- CITE-03: Case citations use Indian legal standard format (Party vs Party (Year) Volume Reporter Page)
- CITE-04: UI shows visual verification indicators (green checkmark for verified citations)
- CITE-05: System suggests precedents intelligently based on legal issue, not just keywords
- CITE-06: System prioritizes precedents from the filing court jurisdiction

**Success Criteria:**
1. Given a legal issue (e.g., "anticipatory bail for economic offenses"), system returns 3-5 relevant Supreme Court/High Court precedents
2. Statute citations render as "Section 438, Code of Criminal Procedure, 1973" (proper format)
3. Case citations render as "Arnesh Kumar vs State of Bihar (2014) 8 SCC 273" (proper format)
4. Each citation displays verification status (verified checkmark or unverified warning)
5. For Delhi High Court filings, Delhi HC precedents appear before other High Courts

**Research Flags:** None - builds on existing Pinecone retrieval.

**Plans:** 5 plans

Plans:
- [x] 03-01-PLAN.md — Citation data models and formatter (SCC, AIR, statute formats)
- [x] 03-02-PLAN.md — Precedent retriever with jurisdiction ranking
- [x] 03-03-PLAN.md — Citation recommender orchestrator (retrieval + verification + formatting)
- [x] 03-04-PLAN.md — API endpoints for citation recommendation
- [x] 03-05-PLAN.md — Streamlit UI with verification badges

---

## Phase 4: Document Generation

**Goal:** System generates legal documents with proper structure, language, and fact integration

**Dependencies:** Phase 2 (templates), Phase 3 (citation engine)

**Requirements:**
- DOC-01: User can generate documents with court-specific formatting (Supreme Court, High Courts, District Courts)
- DOC-02: Generated documents use proper legal language and terminology
- DOC-03: System provides structured fact input interface (parties, dates, allegations, relief sought)
- DOC-04: User can edit and revise generated documents iteratively
- DOC-07: Generated content uses formal legal tone ("it is most respectfully submitted", "Hon'ble Court")

**Success Criteria:**
1. User can input case facts via structured form (party names, dates, allegations, relief sought) and receive a generated document
2. Generated bail application contains proper court header, prayer section, and grounds section in correct order
3. Generated content uses formal legal phrases ("It is most respectfully submitted that...", "The Hon'ble Court may be pleased to...")
4. User can request revision of a generated section and receive updated content without regenerating entire document
5. Document structure matches court-specific template (Supreme Court format differs from District Court)

**Research Flags:** LLM prompt engineering for legal tone may need iteration during implementation.

**Plans:** 5 plans

Plans:
- [x] 04-01-PLAN.md — Fact collection models (Pydantic models for structured input - DOC-03)
- [x] 04-02-PLAN.md — Legal tone prompts (System prompts for formal language - DOC-02, DOC-07)
- [x] 04-03-PLAN.md — DocumentGenerator (Template filling + citation integration - DOC-01)
- [x] 04-04-PLAN.md — DocumentReviser (Iterative editing with edit trick - DOC-04)
- [x] 04-05-PLAN.md — API integration (/generate and /revise endpoints)

---

## Phase 5: Template Management

**Goal:** Users can upload custom templates with proper validation, versioning, and lifecycle management

**Dependencies:** Phase 2 (template storage foundation)

**Requirements:**
- TMPL-03: User can upload custom templates
- TMPL-04: System validates uploaded templates against schema
- TMPL-05: Templates support versioning with changelog
- TMPL-06: System tracks template lifecycle (active/deprecated/archived)
- TMPL-07: User can preview template showing required inputs

**Success Criteria:**
1. User can upload a custom template JSON file and receive validation results (pass/fail with specific errors)
2. Invalid template (missing required fields, wrong schema) is rejected with actionable error message
3. User can view template version history with changelog entries
4. User can see template status (active/deprecated/archived) and receives warning when using deprecated template
5. User can preview any template to see required input fields before starting document generation

**Research Flags:** None - standard CRUD and validation patterns.

**Plans:** (created by /gsd:plan-phase)

---

## Phase 6: Production Integration

**Goal:** Professional drafting is production-ready with API endpoint, auth, exports, and error handling

**Dependencies:** Phase 4 (document generation complete), Phase 5 (template management complete)

**Requirements:**
- DOC-05: User can export documents to PDF with court-standard formatting
- DOC-06: User can export documents to DOCX for editing in Word
- PROD-01: Professional drafting available via /draft-pro API endpoint
- PROD-02: Endpoint integrates with existing auth middleware
- PROD-03: Proper error handling and logging for production use

**Success Criteria:**
1. User can export generated document to PDF with correct margins, fonts, and page numbers for court filing
2. User can export generated document to DOCX that opens correctly in Microsoft Word with editable formatting
3. API endpoint /draft-pro accepts document generation requests and returns generated content
4. Requests without valid auth token receive 401 Unauthorized (integrates with existing verify_access middleware)
5. Invalid requests receive structured error responses with actionable messages; all errors are logged

**Research Flags:** WeasyPrint Windows compatibility needs testing during implementation.

**Plans:** (created by /gsd:plan-phase)

---

## Progress

| Phase | Name | Requirements | Status |
|-------|------|--------------|--------|
| 1 | Trust Foundation | TRUST-01, TRUST-02, TRUST-03, TRUST-04, TRUST-05 | Complete (5/5 plans) |
| 2 | Template Storage | TMPL-01, TMPL-02 | Complete (2/2 plans) |
| 3 | Citation Engine | CITE-01, CITE-02, CITE-03, CITE-04, CITE-05, CITE-06 | Complete (5/5 plans) |
| 4 | Document Generation | DOC-01, DOC-02, DOC-03, DOC-04, DOC-07 | Complete (5/5 plans) |
| 5 | Template Management | TMPL-03, TMPL-04, TMPL-05, TMPL-06, TMPL-07 | Pending |
| 6 | Production Integration | DOC-05, DOC-06, PROD-01, PROD-02, PROD-03 | Pending |

## Coverage Summary

**Total v1 Requirements:** 22
**Mapped to Phases:** 22
**Coverage:** 100%

| Category | Requirements | Phase(s) |
|----------|--------------|----------|
| TRUST | 5 | Phase 1 |
| DOC | 7 | Phase 4 (5), Phase 6 (2) |
| CITE | 6 | Phase 3 |
| TMPL | 7 | Phase 2 (2), Phase 5 (5) |
| PROD | 3 | Phase 6 |

## Dependency Graph

```
Phase 1 (Trust Foundation) -----> Phase 3 (Citation Engine) -----> Phase 4 (Document Generation) -----> Phase 6 (Production)
                                                                          ^
Phase 2 (Template Storage) -----------------------------------------------+-----> Phase 5 (Template Management) -----> Phase 6 (Production)
```

**Parallelization opportunities:**
- Phase 1 and Phase 2 can run in parallel (no dependencies)
- Phase 5 can start after Phase 2, potentially overlapping with Phase 3/4

**Critical path:** Phase 1 -> Phase 3 -> Phase 4 -> Phase 6

---
*Roadmap created: 2026-01-22*
*Updated: 2026-01-26 - Phase 4 complete (5 plans executed)*
