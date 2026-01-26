# 04-03 Summary: DocumentGenerator

## What Was Built
DocumentGenerator class that orchestrates template-first legal document generation, combining templates (Phase 2), citations (Phase 3), and verification (Phase 1) into a complete pipeline.

## Files Changed
| File | Change |
|------|--------|
| src/generation/generator.py | Added DocumentGenerator class with generate_document() and _generate_field_content() methods |
| src/generation/__init__.py | Added exports for DocumentGenerator and GeneratedDocument |

## Key Implementation Details

### GeneratedDocument Model
Output model with fields:
- `content`: Complete rendered document
- `doc_type`, `court_level`: Document classification
- `citations_used`: List of verified citations integrated
- `verification_status`: "verified" or "sanitized"
- `blocked_citations`: Count of citations removed
- `formatting`: Court-specific formatting metadata

### DocumentGenerator Class
Constructor accepts:
- `template_repo` (required): TemplateRepository for template loading
- `citation_recommender` (optional): For retrieving relevant precedents
- `citation_gate` (optional): For blocking unverified citations
- `llm_client` (optional): OpenAI client (creates default if None)

### Pipeline (generate_document)
1. Load template from TemplateRepository
2. Retrieve citations via CitationRecommender (if available)
3. Generate content for text/relief fields via LLM
4. Render template with Jinja2
5. Verify citations via CitationGate (if available)
6. Sanitize output if any citations blocked
7. Return GeneratedDocument with metadata

### Citation Integration Wiring
`_generate_field_content` properly wires citations:
```python
citation_strings = [c.formatted_citation for c in citations] if citations else None
system_prompt = get_generation_prompt(
    court_level=court_level,
    field_name=field_name,
    citations=citation_strings  # Citations passed to prompt builder
)
```

### Field Type Handling
- **LLM fields** (`text`, `relief`): Expanded via LLM with formal legal language
- **Direct fields** (`party`, `date`, `case_number`, `court`): Used as-is

## Commits
- `57b321f`: feat(04-03): add DocumentGenerator with template-first pipeline

## Verification Results
- [x] DocumentGenerator imports without error
- [x] GeneratedDocument model validates correctly
- [x] Citation integration wiring verified via inspection
- [x] Uses gpt-4o-2024-08-06 (not deprecated model)
- [x] Module exports work: `from src.generation import DocumentGenerator, GeneratedDocument`

## Requirements Addressed
- DOC-01: Court-specific formatting via templates and formatting metadata
- DOC-02: Formal legal language via get_generation_prompt integration
- DOC-03: Structured fact input validated by template required_fields
