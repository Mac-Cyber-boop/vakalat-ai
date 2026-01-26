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

__all__ = [
    "BailApplicationFacts",
    "LegalNoticeFacts",
    "AffidavitFacts",
    "PetitionFacts",
]
