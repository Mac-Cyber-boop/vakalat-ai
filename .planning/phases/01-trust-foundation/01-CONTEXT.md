# Phase 1: Trust Foundation - Context

**Gathered:** 2026-01-22
**Status:** Ready for planning

<domain>
## Phase Boundary

System can verify legal citations and map legal codes before any document generation. This is the foundation layer that ensures no hallucinated citations reach users. Builds verification infrastructure that Citation Engine (Phase 3) and Document Generation (Phase 4) will rely on.

</domain>

<decisions>
## Implementation Decisions

### Verification Strictness
- **Block entirely** — Never include unverified citations in output (fail-safe behavior)
- **Exact match required** — Case name + year + citation must all match exactly (no fuzzy matching)
- **Inline verification** — Each citation verified during generation, not as post-processing audit
- **Omit and continue** — When LLM suggests a citation that fails verification, skip it and proceed without (don't fail the entire section)

### Claude's Discretion
- Data source strategy (Pinecone only vs external APIs)
- User feedback mechanism for verification status
- Code mapping behavior (IPC→BNS auto-suggestion approach)
- Verification response time targets
- Audit logging implementation details

</decisions>

<specifics>
## Specific Ideas

- Trust is non-negotiable — "one hallucinated citation permanently destroys lawyer trust" (from research)
- Verification must be invisible to the user flow but absolute in enforcement
- Existing Verifier agent in api.py can be extended for citation-specific verification

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 01-trust-foundation*
*Context gathered: 2026-01-22*
