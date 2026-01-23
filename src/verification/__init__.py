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
from src.verification.citation_verifier import CitationVerifier
from src.verification.section_validator import SectionValidator, SectionValidationResult
from src.verification.audit import (
    get_audit_logger,
    configure_audit_logging,
    AuditEvent,
    log_verification_attempt,
)

__all__ = [
    # Status enums
    "VerificationStatus",
    "CodeMappingStatus",
    # Result models
    "VerificationResult",
    "CodeMappingResult",
    "SectionValidationResult",
    # Input models
    "CaseCitationInput",
    "StatuteCitationInput",
    # Mapper class
    "LegalCodeMapper",
    # Citation verifier
    "CitationVerifier",
    # Section validator
    "SectionValidator",
    # Audit logging
    "get_audit_logger",
    "configure_audit_logging",
    "AuditEvent",
    "log_verification_attempt",
]
