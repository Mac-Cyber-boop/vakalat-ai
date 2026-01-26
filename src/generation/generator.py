"""
Document generator orchestrating template-first legal document generation.

Combines:
- Template loading (Phase 2 TemplateRepository)
- Citation recommendation (Phase 3 CitationRecommender)
- Citation verification (Phase 1 CitationGate)
- LLM content generation with formal legal language

Pipeline:
1. Load template from repository
2. Retrieve relevant citations (if recommender available)
3. Generate formal content for each text/relief field via LLM
4. Render template with filled fields (Jinja2)
5. Verify citations in output (if gate available)
6. Sanitize blocked citations if needed
7. Return GeneratedDocument with metadata
"""

from typing import Literal, Optional

from jinja2 import Template
from openai import OpenAI
from pydantic import BaseModel, Field

from src.citations.recommender import CitationRecommendation, CitationRecommender
from src.generation.prompts import get_generation_prompt
from src.templates.schemas import CourtLevel, DocumentType, LegalTemplate
from src.templates.storage import TemplateRepository
from src.verification.citation_gate import CitationGate, FilteredCitations


class GeneratedDocument(BaseModel):
    """Output model for generated legal documents."""

    content: str = Field(
        description="Complete rendered document content"
    )
    doc_type: str = Field(
        description="Document type that was generated"
    )
    court_level: str = Field(
        description="Court level for formatting"
    )
    citations_used: list[str] = Field(
        default_factory=list,
        description="Citations integrated into content"
    )
    verification_status: Literal["verified", "sanitized"] = Field(
        description="Citation verification outcome"
    )
    blocked_citations: int = Field(
        default=0,
        description="Count of citations blocked by verification"
    )
    formatting: dict = Field(
        default_factory=dict,
        description="Court-specific formatting metadata"
    )
