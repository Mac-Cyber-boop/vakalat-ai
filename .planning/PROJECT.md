# Vakalat AI

## What This Is

An AI legal assistant for Indian lawyers, judges, and legal consultants that provides trusted legal research and document drafting. Unlike generic AI tools that hallucinate fake cases, Vakalat AI only speaks from a verified database of Indian laws and Supreme Court judgments — it says "I don't know" when information isn't in the database.

## Core Value

**Trustworthy legal document drafting that saves lawyers hours of work** — lawyers can rely on the output because it's grounded in real Indian legal data with proper legal language and court formats.

## Requirements

### Validated

- Research queries against Indian law database — existing
- Multi-source retrieval (statutes, case law, precedents) — existing
- Hallucination prevention via verifier agent — existing
- Basic document drafting endpoint — existing
- Document review/analysis endpoint — existing
- PDF upload and analysis — existing
- Streamlit UI for development — existing
- FastAPI backend with token auth — existing
- Data ingestion from HuggingFace datasets — existing
- New criminal law ingestion (BNS/BNSS/BSA 2023) — existing

### Active

- [ ] Professional legal document drafting with proper court formats
- [ ] Custom template upload (petitions, notices, etc.)
- [ ] Default court-standard templates for common document types
- [ ] Improved UI/UX for production use
- [ ] Comprehensive legal language output matching lawyer expectations

### Out of Scope

- Mobile app — web-first, mobile later
- Real-time collaboration — single-user focus for v1
- Payment/billing features — not part of core legal AI
- Multi-language support — English/legal terminology only for v1

## Context

**User research insight:** Lawyers spend hours drafting documents in proper legal language. Current AI tools are generic, hallucinate fake cases, have bad UI, and produce output that can't be trusted.

**Differentiation:**
- India-specific legal database (not generic)
- No hallucination — grounded in Pinecone, explicit "I don't know" behavior
- Proper legal language — court formats, correct terminology
- Trustworthy — lawyers can actually rely on it

**Technical foundation:**
- Python 3.13.5 with FastAPI + Streamlit
- Pinecone (production) / Chroma (development) vector stores
- LangChain for LLM orchestration
- GPT-4o for reasoning
- Agentic pipeline: Planner → Jurist → Verifier

**Data sources:**
- 54+ Indian laws including Constitution, BNS, BNSS, BSA
- Supreme Court judgments 1950-2025
- Critical precedents (Arnesh Kumar, D.K. Basu)

## Constraints

- **LLM Provider**: OpenAI GPT-4o — already integrated, proven quality
- **Vector Store**: Pinecone — production database, switching would require re-ingestion
- **Accuracy requirement**: Zero tolerance for hallucinated cases — must cite real sources only
- **Legal compliance**: Output must use proper Indian court formats and terminology

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Pinecone over local vector DB | Cloud scalability, production reliability | — Pending |
| Verifier agent for hallucination check | Lawyers need to trust output completely | — Pending |
| Focus on drafting over research | User research showed drafting is the bigger pain point | — Pending |

---
*Last updated: 2026-01-21 after initialization*
