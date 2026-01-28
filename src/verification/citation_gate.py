"""
Citation Gate for filtering and blocking unverified citations.

Provides CitationGate class that:
- Extracts case and statute citations from text using regex
- Verifies each citation against the legal database
- Blocks unverified citations (omit and continue behavior)
- Sanitizes output to remove blocked citations
- Logs all filtering operations for audit

Per TRUST-02: Unverified citations must be blocked from output.
Per CONTEXT.md: "omit and continue" when LLM suggests citations that fail verification.
"""

import re
from datetime import datetime, timezone
from typing import List, Tuple, Optional

from pydantic import BaseModel, Field

from src.verification.citation_verifier import CitationVerifier
from src.verification.section_validator import SectionValidator
from src.verification.models import VerificationStatus
from src.verification.audit import get_audit_logger, AuditEvent, log_verification_attempt


class FilteredCitations(BaseModel):
    """Result of citation filtering operation."""

    verified: List[str] = Field(
        default_factory=list,
        description="Citations that passed verification"
    )
    blocked: List[str] = Field(
        default_factory=list,
        description="Citations that failed verification"
    )
    blocked_reasons: dict = Field(
        default_factory=dict,
        description="Citation -> reason mapping for blocked citations"
    )


class CitationGate:
    """
    Gates citations through verification before allowing them in output.

    Extracts citations from text, verifies each against the legal database,
    and provides methods to sanitize output by removing unverified citations.

    Usage:
        gate = CitationGate(citation_verifier, section_validator)
        filtered = gate.filter_citations(text)
        if filtered.blocked:
            sanitized_text = gate.sanitize_output(text, filtered)
    """

    # Regex patterns for case citation extraction
    # Pattern 1: "Name vs/v./versus Name" format
    CASE_NAME_PATTERN = re.compile(
        r'([A-Z][a-zA-Z\.\s]+\s+(?:vs?\.?|versus)\s+[A-Z][a-zA-Z\.\s&]+)',
        re.IGNORECASE
    )

    # Pattern 2: SCC citation format "(2014) 8 SCC 273"
    SCC_PATTERN = re.compile(
        r'\((\d{4})\)\s*(\d+)\s*SCC\s*(\d+)',
        re.IGNORECASE
    )

    # Pattern 3: AIR citation format "AIR 2014 SC 123"
    AIR_PATTERN = re.compile(
        r'AIR\s*(\d{4})\s*SC\s*(\d+)',
        re.IGNORECASE
    )

    # Regex patterns for statute citation extraction
    # Pattern 1: "Section 302, IPC" or "Section 302 of IPC"
    SECTION_ACT_PATTERN = re.compile(
        r'Section\s+(\d+[A-Za-z]?)\s*(?:,|of|under)\s+([A-Za-z\.\s]+)',
        re.IGNORECASE
    )

    # Pattern 2: "IPC Section 420" or "BNS Section 103"
    ACT_SECTION_PATTERN = re.compile(
        r'(IPC|BNS|CrPC|BNSS|IEA|BSA|Indian Penal Code|Bharatiya Nyaya Sanhita)\s+Section\s+(\d+[A-Za-z]?)',
        re.IGNORECASE
    )

    # Pattern 3: "S. 302 IPC" or "Sec. 420 BNS"
    SHORT_SECTION_PATTERN = re.compile(
        r'(?:S\.|Sec\.)\s*(\d+[A-Za-z]?)\s+([A-Za-z\.]+)',
        re.IGNORECASE
    )

    def __init__(
        self,
        citation_verifier: CitationVerifier,
        section_validator: Optional[SectionValidator] = None
    ):
        """
        Initialize the CitationGate.

        Args:
            citation_verifier: CitationVerifier instance for case citation verification
            section_validator: Optional SectionValidator for statute citation verification
        """
        self.citation_verifier = citation_verifier
        self.section_validator = section_validator
        self.logger = get_audit_logger("citation_gate")

    def _extract_case_citations(self, text: str) -> List[str]:
        """
        Extract case citations from text using regex patterns.

        Args:
            text: Text to extract citations from

        Returns:
            List of extracted case citation strings
        """
        if not text:
            return []

        citations = set()

        # Extract "Name vs Name" format
        for match in self.CASE_NAME_PATTERN.finditer(text):
            citation = match.group(1).strip()
            # Clean up extra spaces
            citation = ' '.join(citation.split())
            if len(citation) > 10:  # Minimum reasonable length
                citations.add(citation)

        # Extract SCC format
        for match in self.SCC_PATTERN.finditer(text):
            citation = match.group(0)
            citations.add(citation)

        # Extract AIR format
        for match in self.AIR_PATTERN.finditer(text):
            citation = match.group(0)
            citations.add(citation)

        return list(citations)

    def _extract_statute_citations(self, text: str) -> List[Tuple[str, str]]:
        """
        Extract statute citations from text.

        Args:
            text: Text to extract citations from

        Returns:
            List of (act_name, section) tuples
        """
        if not text:
            return []

        citations = set()

        # Pattern 1: "Section 302, IPC" or "Section 302 of IPC"
        for match in self.SECTION_ACT_PATTERN.finditer(text):
            section = match.group(1).strip()
            act = match.group(2).strip()
            # Clean up act name (remove trailing punctuation)
            act = re.sub(r'[,\.\s]+$', '', act)
            if act and section:
                citations.add((act, section))

        # Pattern 2: "IPC Section 420"
        for match in self.ACT_SECTION_PATTERN.finditer(text):
            act = match.group(1).strip()
            section = match.group(2).strip()
            if act and section:
                citations.add((act, section))

        # Pattern 3: "S. 302 IPC"
        for match in self.SHORT_SECTION_PATTERN.finditer(text):
            section = match.group(1).strip()
            act = match.group(2).strip()
            if act and section:
                citations.add((act, section))

        return list(citations)

    def _parse_case_citation(self, citation: str) -> Tuple[str, int, Optional[str]]:
        """
        Parse a case citation into components for verification.

        Args:
            citation: Raw citation string

        Returns:
            Tuple of (case_name, year, formal_citation)
            Year defaults to 2000 if not found
        """
        case_name = citation
        year = 2000  # Default if year not found
        formal_citation = None

        # Try to extract year from SCC format
        scc_match = self.SCC_PATTERN.search(citation)
        if scc_match:
            year = int(scc_match.group(1))
            formal_citation = scc_match.group(0)
            # Remove SCC citation from case name if present together
            case_name = self.SCC_PATTERN.sub('', citation).strip()

        # Try to extract year from AIR format
        air_match = self.AIR_PATTERN.search(citation)
        if air_match:
            year = int(air_match.group(1))
            formal_citation = air_match.group(0)
            case_name = self.AIR_PATTERN.sub('', citation).strip()

        # Try to extract year from parenthetical (2014)
        year_match = re.search(r'\((\d{4})\)', citation)
        if year_match and year == 2000:
            year = int(year_match.group(1))

        # Clean case name
        case_name = case_name.strip()
        if not case_name:
            case_name = citation

        return (case_name, year, formal_citation)

    def filter_citations(self, text: str) -> FilteredCitations:
        """
        Filter case citations in text through verification.

        Extracts all case citations, verifies each against the database,
        and categorizes them as verified or blocked.

        Args:
            text: Text containing citations to filter

        Returns:
            FilteredCitations with verified and blocked lists
        """
        result = FilteredCitations()

        if not text:
            return result

        citations = self._extract_case_citations(text)

        for citation in citations:
            try:
                case_name, year, formal_citation = self._parse_case_citation(citation)

                # Verify against database
                verification = self.citation_verifier.verify_case_citation(
                    case_name=case_name,
                    year=year,
                    citation=formal_citation
                )

                if verification.status == VerificationStatus.VERIFIED:
                    result.verified.append(citation)
                else:
                    result.blocked.append(citation)
                    result.blocked_reasons[citation] = verification.reason or "Not found in database"

                    # Log blocked citation
                    self._log_blocked_citation(
                        citation=citation,
                        reason=verification.reason or "Not found in database",
                        citation_type="case"
                    )

            except Exception as e:
                # Malformed citations logged but don't crash
                result.blocked.append(citation)
                result.blocked_reasons[citation] = f"Parse error: {str(e)}"
                self._log_blocked_citation(
                    citation=citation,
                    reason=f"Parse error: {str(e)}",
                    citation_type="case"
                )

        return result

    def filter_statute_citations(self, text: str) -> FilteredCitations:
        """
        Filter statute citations in text through verification.

        Extracts all statute citations, verifies each against the database,
        and categorizes them as verified or blocked.

        OUTDATED citations are considered verified (still valid for pre-2024 cases).

        Args:
            text: Text containing citations to filter

        Returns:
            FilteredCitations with verified and blocked lists
        """
        result = FilteredCitations()

        if not text or not self.section_validator:
            return result

        citations = self._extract_statute_citations(text)

        for act_name, section in citations:
            citation_str = f"Section {section}, {act_name}"

            try:
                validation = self.section_validator.validate_section(act_name, section)

                # VERIFIED or OUTDATED are both acceptable (outdated still valid for old cases)
                if validation.status in [VerificationStatus.VERIFIED, VerificationStatus.OUTDATED]:
                    result.verified.append(citation_str)
                else:
                    result.blocked.append(citation_str)
                    result.blocked_reasons[citation_str] = "Section not found in database"

                    self._log_blocked_citation(
                        citation=citation_str,
                        reason="Section not found in database",
                        citation_type="statute"
                    )

            except Exception as e:
                result.blocked.append(citation_str)
                result.blocked_reasons[citation_str] = f"Validation error: {str(e)}"
                self._log_blocked_citation(
                    citation=citation_str,
                    reason=f"Validation error: {str(e)}",
                    citation_type="statute"
                )

        return result

    def sanitize_output(self, text: str, filtered: FilteredCitations) -> str:
        """
        Remove blocked citations from text.

        Per CONTEXT.md: "omit and continue" - don't fail, just remove unverified citations.
        Adds a footnote if any citations were removed.

        Args:
            text: Original text with citations
            filtered: FilteredCitations result from filter_citations

        Returns:
            Sanitized text with blocked citations removed
        """
        if not filtered.blocked:
            return text

        sanitized = text

        for blocked_citation in filtered.blocked:
            # Escape special regex characters in citation
            pattern = re.escape(blocked_citation)
            # Remove the citation and any surrounding quotes/parentheses
            sanitized = re.sub(
                rf'["\'\(]?\s*{pattern}\s*["\'\)]?\s*,?\s*',
                '',
                sanitized,
                flags=re.IGNORECASE
            )

        # Clean up any resulting double spaces or orphaned punctuation
        sanitized = re.sub(r'\s+', ' ', sanitized)
        sanitized = re.sub(r'\s*,\s*,', ',', sanitized)
        sanitized = re.sub(r'\(\s*\)', '', sanitized)

        # Add footnote about removed citations
        footnote = "\n\n<p style='font-size: 0.9em; color: #666;'>Note: Some citations could not be verified against our database and have been omitted.</p>"
        sanitized = sanitized.strip() + footnote

        return sanitized

    def _log_blocked_citation(
        self,
        citation: str,
        reason: str,
        citation_type: str
    ) -> None:
        """
        Log a blocked citation for audit purposes.

        Args:
            citation: The citation that was blocked
            reason: Reason for blocking
            citation_type: Type of citation (case or statute)
        """
        event = AuditEvent(
            event_type="citation_blocked",
            timestamp=datetime.now(timezone.utc),
            input_data={
                "citation": citation,
                "citation_type": citation_type
            },
            result="BLOCKED",
            duration_ms=0,
            reason=reason
        )
        log_verification_attempt(self.logger, event)

    def filter_all_citations(self, text: str) -> FilteredCitations:
        """
        Filter both case and statute citations in text.

        Convenience method that combines filter_citations and filter_statute_citations.

        Args:
            text: Text containing citations to filter

        Returns:
            Combined FilteredCitations result
        """
        case_result = self.filter_citations(text)
        statute_result = self.filter_statute_citations(text)

        return FilteredCitations(
            verified=case_result.verified + statute_result.verified,
            blocked=case_result.blocked + statute_result.blocked,
            blocked_reasons={**case_result.blocked_reasons, **statute_result.blocked_reasons}
        )
