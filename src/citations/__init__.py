"""
Citation engine for Vakalat AI.

Provides citation formatting, retrieval, and verification badge support
for Indian legal citation standards (SCC, AIR, statute formats).
"""

from src.citations.models import (
    CitationBase,
    CaseCitation,
    StatuteCitation,
    VerificationBadge,
    FormattedCitation,
)
from src.citations.formatter import CitationFormatter
from src.citations.retriever import (
    PrecedentRetriever,
    RetrievedPrecedent,
    JurisdictionWeight,
)

__all__ = [
    # Models
    "CitationBase",
    "CaseCitation",
    "StatuteCitation",
    "VerificationBadge",
    "FormattedCitation",
    # Formatter
    "CitationFormatter",
    # Retriever
    "PrecedentRetriever",
    "RetrievedPrecedent",
    "JurisdictionWeight",
]
