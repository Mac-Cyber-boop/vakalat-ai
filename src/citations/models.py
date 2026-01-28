"""
Pydantic models for citation handling in Vakalat AI.

Provides data structures for:
- Case citations (SCC, AIR formats)
- Statute citations (Section N, Act Name, Year)
- Verification badges for UI display
- Formatted citation output with HTML support
"""

from typing import Literal, Optional
from pydantic import BaseModel, Field, computed_field


class CitationBase(BaseModel):
    """Base model for all citation types."""

    raw_text: str = Field(
        description="Original citation text as it appears in the document"
    )
    citation_type: Literal["case", "statute"] = Field(
        description="Type of citation: case law or statute"
    )


class CaseCitation(CitationBase):
    """
    Model for case law citations.

    Supports both SCC format "(Year) Volume SCC Page" and AIR format
    "AIR Year Court Page" commonly used in Indian legal practice.
    """

    citation_type: Literal["case", "statute"] = Field(
        default="case",
        description="Type of citation: always 'case' for case citations"
    )
    party_a: str = Field(
        description="First party name (e.g., 'Arnesh Kumar', 'K.S. Puttaswamy')"
    )
    party_b: str = Field(
        description="Second party name (e.g., 'State of Bihar', 'Union of India')"
    )
    year: int = Field(
        description="Year of the judgment"
    )
    reporter: Optional[Literal["SCC", "AIR", "OTHER"]] = Field(
        default=None,
        description="Reporter name: SCC (Supreme Court Cases), AIR (All India Reporter), or OTHER"
    )
    volume: Optional[int] = Field(
        default=None,
        description="Volume number (primarily used for SCC format)"
    )
    page: Optional[int] = Field(
        default=None,
        description="Page number in the reporter"
    )
    court: Optional[str] = Field(
        default=None,
        description="Court code (SC, Del, Bom, Cal, Mad, Kar, All, etc.)"
    )


class StatuteCitation(CitationBase):
    """
    Model for statute citations.

    Follows Indian legal citation format: "Section N, Act Name, Year"
    """

    citation_type: Literal["case", "statute"] = Field(
        default="statute",
        description="Type of citation: always 'statute' for statute citations"
    )
    section: str = Field(
        description="Section number (may include letters like '302A', '65B')"
    )
    act_name: str = Field(
        description="Full name of the act (e.g., 'Indian Penal Code', 'Bharatiya Nyaya Sanhita')"
    )
    year: Optional[int] = Field(
        default=None,
        description="Year of enactment (e.g., 1860 for IPC, 1973 for CrPC)"
    )


class VerificationBadge(BaseModel):
    """
    Verification badge for UI display.

    Provides visual indicator of citation verification status with
    appropriate icon, tooltip, and styling.
    """

    status: Literal["verified", "unverified", "outdated"] = Field(
        description="Verification status: verified (found in database), unverified (not found), or outdated (uses repealed law)"
    )
    icon: str = Field(
        description="Unicode icon for display (checkmark, warning, or refresh)"
    )
    tooltip: str = Field(
        description="Hover text explaining the verification status"
    )
    css_class: str = Field(
        description="CSS class for styling the badge"
    )

    @classmethod
    def verified(cls, tooltip: str = "Citation verified in database") -> "VerificationBadge":
        """Create a verified badge."""
        return cls(
            status="verified",
            icon="\u2713",  # Checkmark
            tooltip=tooltip,
            css_class="citation-verified"
        )

    @classmethod
    def unverified(cls, tooltip: str = "Citation could not be verified") -> "VerificationBadge":
        """Create an unverified badge."""
        return cls(
            status="unverified",
            icon="\u26A0",  # Warning
            tooltip=tooltip,
            css_class="citation-unverified"
        )

    @classmethod
    def outdated(cls, tooltip: str = "Citation uses outdated/repealed law") -> "VerificationBadge":
        """Create an outdated badge."""
        return cls(
            status="outdated",
            icon="\u21BB",  # Refresh/cycle
            tooltip=tooltip,
            css_class="citation-outdated"
        )


class FormattedCitation(BaseModel):
    """
    Complete formatted citation with verification badge.

    Combines the properly formatted citation text with verification
    status and generates HTML output for UI display.
    """

    formatted_text: str = Field(
        description="Properly formatted citation string"
    )
    badge: VerificationBadge = Field(
        description="Verification status badge"
    )
    source_id: Optional[str] = Field(
        default=None,
        description="Database ID of the source document if verified"
    )

    @computed_field
    @property
    def html_output(self) -> str:
        """
        Generate HTML output combining formatted text and badge.

        Returns a span containing the citation text followed by the
        verification badge with appropriate styling.
        """
        badge_html = (
            f'<span class="citation-badge {self.badge.css_class}" '
            f'title="{self.badge.tooltip}">{self.badge.icon}</span>'
        )
        return f'<span class="citation">{self.formatted_text} {badge_html}</span>'
