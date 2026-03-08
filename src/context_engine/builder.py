"""
Context Engineering for Vakalat AI.

Transforms raw retrieved documents into a structured, ordered context window
that improves LLM reasoning accuracy on legal queries.

The key insight: LLMs reason better over structured context than over a raw
wall of text. This module organizes retrieved chunks into labeled sections
ordered by legal authority (statutes → SC precedents → HC precedents → procedure).

Usage:
    builder = ContextBuilder()
    ctx = builder.build(docs, query="anticipatory bail rejection")
    # ctx.formatted is ready to pass to the LLM prompt
"""

from dataclasses import dataclass, field
from typing import Optional


# Max characters per section to avoid context bloat.
# Legal LLM prompts work best under ~6000 chars of context.
MAX_CHARS_PER_SECTION = 1500
MAX_SNIPPET_CHARS = 400


@dataclass
class ContextSection:
    label: str
    items: list[str] = field(default_factory=list)

    def is_empty(self) -> bool:
        return len(self.items) == 0

    def render(self) -> str:
        if self.is_empty():
            return ""
        body = "\n\n".join(self.items)
        return f"## {self.label}\n{body}"


@dataclass
class StructuredContext:
    """
    The structured context window ready to pass to the LLM.

    Attributes:
        formatted: The full structured string for the prompt.
        doc_count: Total documents included.
        section_counts: How many items per section type.
    """
    formatted: str
    doc_count: int
    section_counts: dict[str, int]


class ContextBuilder:
    """
    Builds a structured context window from raw Pinecone documents.

    Categorizes documents by source_type metadata into sections,
    then renders them in authority order with length caps.

    Section order (high authority → low authority):
        1. Relevant Statutes       — direct law text, highest weight for LLM
        2. Supreme Court Precedents — binding on all courts
        3. High Court Precedents   — persuasive/binding in jurisdiction
        4. Procedural Rules        — rules of court, forms, procedures

    Example:
        builder = ContextBuilder()
        docs = vector_db.similarity_search(query, k=15)
        ctx = builder.build(docs, query)
        prompt.invoke({"context": ctx.formatted, ...})
    """

    # Metadata source_type values → section labels
    _TYPE_MAP = {
        "statute": "Relevant Statutes",
        "act": "Relevant Statutes",
        "legislation": "Relevant Statutes",
        "procedure": "Procedural Rules",
        "rules": "Procedural Rules",
    }

    # Court identifiers that map to Supreme Court section
    _SC_IDENTIFIERS = {"supreme_court", "sc", "sci", "s.c."}

    def _classify_doc(self, doc) -> str:
        """Return the section label for a document based on its metadata."""
        metadata = doc.metadata
        source_type = metadata.get("source_type", "").lower()

        # Explicit source_type match
        if source_type in self._TYPE_MAP:
            return self._TYPE_MAP[source_type]

        if source_type == "case_law":
            court = metadata.get("court", "").lower().replace(" ", "_")
            if court in self._SC_IDENTIFIERS or "supreme" in court:
                return "Supreme Court Precedents"
            return "High Court Precedents"

        # Fallback: infer from source filename
        source = metadata.get("source", "").lower()
        if any(kw in source for kw in ["act", "code", "constitution", "statute", "rules", "order"]):
            return "Relevant Statutes"
        if "supreme" in source or "_sc_" in source or source.endswith("sc.pdf"):
            return "Supreme Court Precedents"
        if "high court" in source or "_hc_" in source:
            return "High Court Precedents"

        # Generic fallback — treat as statute (better than losing it)
        return "Relevant Statutes"

    def _format_doc(self, doc, section_label: str) -> str:
        """Format a single document into a labeled snippet."""
        metadata = doc.metadata
        content = doc.page_content

        # Truncate content to avoid bloating any single item
        snippet = content[:MAX_SNIPPET_CHARS]
        if len(content) > MAX_SNIPPET_CHARS:
            snippet += "..."

        # Build the source label
        source_parts = []
        if source := metadata.get("source_book") or metadata.get("source"):
            source_parts.append(source)
        if case_name := metadata.get("case_name"):
            source_parts.append(case_name)
        if section := metadata.get("section"):
            source_parts.append(f"§{section}")

        source_label = " | ".join(source_parts) if source_parts else "Unknown Source"

        return f"[{source_label}]\n{snippet}"

    def build(
        self,
        docs: list,
        query: Optional[str] = None,
        dedup: bool = True,
    ) -> StructuredContext:
        """
        Build a structured context window from retrieved documents.

        Args:
            docs: List of Langchain Document objects from Pinecone retrieval.
            query: The original user query (reserved for future reranking).
            dedup: Whether to deduplicate by content fingerprint.

        Returns:
            StructuredContext with formatted string and metadata.
        """
        # Section order defines LLM attention priority
        section_order = [
            "Relevant Statutes",
            "Supreme Court Precedents",
            "High Court Precedents",
            "Procedural Rules",
        ]

        sections: dict[str, ContextSection] = {
            label: ContextSection(label=label) for label in section_order
        }

        # Dedup by content fingerprint (first 80 chars)
        seen: set[str] = set()
        char_counts: dict[str, int] = {label: 0 for label in section_order}

        for doc in docs:
            # Deduplicate
            fingerprint = doc.page_content[:80].strip()
            if dedup and fingerprint in seen:
                continue
            seen.add(fingerprint)

            section_label = self._classify_doc(doc)

            # Skip if this section is already full
            if char_counts[section_label] >= MAX_CHARS_PER_SECTION:
                continue

            formatted = self._format_doc(doc, section_label)
            sections[section_label].items.append(formatted)
            char_counts[section_label] += len(formatted)

        # Render only non-empty sections in authority order
        rendered_sections = [
            s.render() for s in sections.values() if not s.is_empty()
        ]

        formatted = "\n\n---\n\n".join(rendered_sections)

        section_counts = {
            label: len(sections[label].items) for label in section_order
        }
        total_docs = sum(section_counts.values())

        return StructuredContext(
            formatted=formatted,
            doc_count=total_docs,
            section_counts=section_counts,
        )
