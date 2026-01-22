"""
Verification module for Vakalat AI.

Provides legal citation verification and code mapping functionality.
"""

from src.verification.models import (
    VerificationStatus,
    CodeMappingStatus,
    VerificationResult,
    CodeMappingResult,
    CaseCitationInput,
    StatuteCitationInput,
)
from src.verification.code_mapper import LegalCodeMapper

__all__ = [
    # Status enums
    "VerificationStatus",
    "CodeMappingStatus",
    # Result models
    "VerificationResult",
    "CodeMappingResult",
    # Input models
    "CaseCitationInput",
    "StatuteCitationInput",
    # Mapper class
    "LegalCodeMapper",
]
