# Requirements: Vakalat AI

**Defined:** 2026-01-22
**Core Value:** Trustworthy legal document drafting that saves lawyers hours of work

## v1.0 Requirements

Requirements for professional document drafting milestone. Each maps to roadmap phases.

### Trust Foundation

- [ ] **TRUST-01**: System verifies every case citation against legal database before including in document
- [ ] **TRUST-02**: Unverified citations are blocked from output (never generate unverified)
- [ ] **TRUST-03**: System maps old legal codes to new (IPC→BNS, CrPC→BNSS, Evidence Act→BSA)
- [ ] **TRUST-04**: System verifies section numbers exist and aren't repealed before citing
- [ ] **TRUST-05**: System flags outdated legal code references and suggests current equivalents

### Document Generation

- [ ] **DOC-01**: User can generate documents with court-specific formatting (Supreme Court, High Courts, District Courts)
- [ ] **DOC-02**: Generated documents use proper legal language and terminology
- [ ] **DOC-03**: System provides structured fact input interface (parties, dates, allegations, relief sought)
- [ ] **DOC-04**: User can edit and revise generated documents iteratively
- [ ] **DOC-05**: User can export documents to PDF with court-standard formatting
- [ ] **DOC-06**: User can export documents to DOCX for editing in Word
- [ ] **DOC-07**: Generated content uses formal legal tone ("it is most respectfully submitted", "Hon'ble Court")

### Citation & Precedent

- [ ] **CITE-01**: System suggests relevant case law from database based on legal issues
- [ ] **CITE-02**: System cites specific statutory provisions with proper format (Section X, Act Name, Year)
- [ ] **CITE-03**: Case citations use Indian legal standard format (Party vs Party (Year) Volume Reporter Page)
- [ ] **CITE-04**: UI shows visual verification indicators (green checkmark for verified citations)
- [ ] **CITE-05**: System suggests precedents intelligently based on legal issue, not just keywords
- [ ] **CITE-06**: System prioritizes precedents from the filing court jurisdiction

### Template Management

- [ ] **TMPL-01**: System provides default templates for common documents (bail application, legal notice, affidavit, petition)
- [ ] **TMPL-02**: Templates define court-specific formatting requirements
- [ ] **TMPL-03**: User can upload custom templates
- [ ] **TMPL-04**: System validates uploaded templates against schema
- [ ] **TMPL-05**: Templates support versioning with changelog
- [ ] **TMPL-06**: System tracks template lifecycle (active/deprecated/archived)
- [ ] **TMPL-07**: User can preview template showing required inputs

### Production Integration

- [ ] **PROD-01**: Professional drafting available via /draft-pro API endpoint
- [ ] **PROD-02**: Endpoint integrates with existing auth middleware
- [ ] **PROD-03**: Proper error handling and logging for production use

## v2 Requirements

Deferred to future release. Tracked but not in current roadmap.

### Multi-Language

- **LANG-01**: Hindi language support for document generation
- **LANG-02**: Regional language support (Marathi, Tamil, etc.)

### Collaboration

- **COLLAB-01**: Junior/senior lawyer document review workflow
- **COLLAB-02**: Comment and annotation on drafts

### Advanced Features

- **ADV-01**: Plain language summaries for clients
- **ADV-02**: Deadline calculator for limitation periods
- **ADV-03**: Opposing counsel argument preview

## Out of Scope

Explicitly excluded. Documented to prevent scope creep.

| Feature | Reason |
|---------|--------|
| Fully automated filing | Creates liability, loses trust — lawyers must review |
| Generic "AI will handle it" messaging | Black-box AI is untrustworthy; need transparency |
| Chat-only interface | Hides structure, makes editing difficult |
| Legal advice mode | Unauthorized practice of law concerns |
| Mobile app | Web-first, mobile later |
| Real-time collaboration | Single-user focus for v1 |
| Payment/billing features | Not part of core legal AI |

## Traceability

Which phases cover which requirements. Updated during roadmap creation.

| Requirement | Phase | Status |
|-------------|-------|--------|
| (To be filled by roadmapper) | | |

**Coverage:**
- v1 requirements: 22 total
- Mapped to phases: 0
- Unmapped: 22 (pending roadmap creation)

---
*Requirements defined: 2026-01-22*
*Last updated: 2026-01-22 after initial definition*
