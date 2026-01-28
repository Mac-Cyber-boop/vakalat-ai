"""
Pydantic models for the verification system.

Provides data structures for:
- Verification results (case citations, statute citations)
- Legal code mapping results (IPC->BNS, CrPC->BNSS, IEA->BSA)
- Input models for citation verification requests
"""

from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


class VerificationStatus(str, Enum):
    """Status codes for verification operations."""
    VERIFIED = "VERIFIED"       # Citation confirmed in database
    UNVERIFIED = "UNVERIFIED"   # Citation not found in database
    BLOCKED = "BLOCKED"         # Citation blocked due to policy
    OUTDATED = "OUTDATED"       # Citation uses repealed/old law
    MAPPED = "MAPPED"           # Old code mapped to new equivalent


class CodeMappingStatus(str, Enum):
    """Status codes for legal code mapping operations."""
    MAPPED = "MAPPED"           # Old section successfully mapped to new
    NO_MAPPING = "NO_MAPPING"   # Section exists but has no direct mapping
    UNKNOWN_CODE = "UNKNOWN_CODE"  # Code type not recognized


class VerificationResult(BaseModel):
    """Result of a citation verification operation."""
    status: VerificationStatus = Field(
        description="Verification status code"
    )
    source_id: Optional[str] = Field(
        default=None,
        description="ID of the source document in the database"
    )
    confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Confidence score from 0.0 to 1.0"
    )
    reason: Optional[str] = Field(
        default=None,
        description="Explanation of the verification result"
    )
    suggested_update: Optional[str] = Field(
        default=None,
        description="Suggested replacement for outdated citations"
    )
    verification_time_ms: int = Field(
        default=0,
        ge=0,
        description="Time taken for verification in milliseconds"
    )


class CodeMappingResult(BaseModel):
    """Result of a legal code mapping operation."""
    status: CodeMappingStatus = Field(
        description="Mapping status code"
    )
    old_code: str = Field(
        description="Original code name (e.g., 'IPC', 'CrPC', 'Evidence Act')"
    )
    old_section: str = Field(
        description="Original section number"
    )
    new_code: Optional[str] = Field(
        default=None,
        description="New code name (e.g., 'BNS', 'BNSS', 'BSA')"
    )
    new_section: Optional[str] = Field(
        default=None,
        description="Equivalent section in new code"
    )
    confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Confidence in mapping accuracy (1.0 = exact equivalent)"
    )
    notes: Optional[str] = Field(
        default=None,
        description="Additional context about the mapping"
    )


class CaseCitationInput(BaseModel):
    """Input model for case citation verification requests."""
    case_name: str = Field(
        description="Name of the case (e.g., 'Arnesh Kumar vs State of Bihar')"
    )
    year: int = Field(
        description="Year of the judgment"
    )
    citation: Optional[str] = Field(
        default=None,
        description="Formal citation (e.g., '(2014) 8 SCC 273')"
    )


class StatuteCitationInput(BaseModel):
    """Input model for statute citation verification requests."""
    act_name: str = Field(
        description="Name of the act (e.g., 'Indian Penal Code', 'BNS')"
    )
    section: str = Field(
        description="Section number (e.g., '302', '65B')"
    )
