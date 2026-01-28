"""
Document generation module for Vakalat AI.

This module provides fact collection models and document generation capabilities
for Indian legal documents (bail applications, legal notices, affidavits, petitions).
"""

from src.generation.models import (
    BailApplicationFacts,
    LegalNoticeFacts,
    AffidavitFacts,
    PetitionFacts,
)
from src.generation.prompts import (
    BASE_LEGAL_TONE_PROMPT,
    COURT_SPECIFIC_PROMPTS,
    FIELD_GENERATION_PROMPTS,
    get_generation_prompt,
)
from src.generation.generator import (
    DocumentGenerator,
    GeneratedDocument,
)
from src.generation.reviser import (
    DocumentReviser,
    DocumentEdit,
    RevisionResult,
)

__all__ = [
    # Fact collection models
    "BailApplicationFacts",
    "LegalNoticeFacts",
    "AffidavitFacts",
    "PetitionFacts",
    # System prompts
    "BASE_LEGAL_TONE_PROMPT",
    "COURT_SPECIFIC_PROMPTS",
    "FIELD_GENERATION_PROMPTS",
    "get_generation_prompt",
    # Document generator
    "DocumentGenerator",
    "GeneratedDocument",
    # Document reviser
    "DocumentReviser",
    "DocumentEdit",
    "RevisionResult",
]
