"""
Outdated Code Detector for Indian legal references.

Detects references to old criminal codes (IPC, CrPC, IEA) in text
and suggests current equivalents (BNS, BNSS, BSA).

Used to flag outdated legal references in user queries, documents,
and generated drafts, providing suggestions for modern equivalents.
"""

import re
from datetime import datetime, timezone
from typing import List, Optional, Tuple

from pydantic import BaseModel, Field

from src.verification.code_mapper import LegalCodeMapper, CodeMappingStatus
from src.verification.audit import get_audit_logger


class OutdatedReference(BaseModel):
    """A single outdated legal code reference found in text."""

    original_text: str = Field(
        description="The matched text (e.g., 'Section 302, IPC')"
    )
    old_code: str = Field(
        description="The old code name (e.g., 'IPC')"
    )
    old_section: str = Field(
        description="The section number (e.g., '302')"
    )
    new_code: Optional[str] = Field(
        default=None,
        description="The new code name (e.g., 'BNS')"
    )
    new_section: Optional[str] = Field(
        default=None,
        description="The new section number (e.g., '103')"
    )
    suggestion: str = Field(
        description="Formatted message for user (e.g., 'Section 302, IPC corresponds to Section 103, BNS')"
    )
    position: Tuple[int, int] = Field(
        description="Start and end position in original text"
    )


class DetectionResult(BaseModel):
    """Result of outdated code detection on a text."""

    outdated_references: List[OutdatedReference] = Field(
        default_factory=list,
        description="List of outdated references found"
    )
    has_outdated: bool = Field(
        default=False,
        description="True if any outdated references were found"
    )
    summary: str = Field(
        default="",
        description="Summary message (e.g., 'Found 3 references to old legal codes')"
    )


class OutdatedCodeDetector:
    """
    Detects outdated legal code references in text.

    Uses regex patterns to find references to old Indian criminal codes
    (IPC, CrPC, IEA) and maps them to new equivalents using LegalCodeMapper.

    Usage:
        detector = OutdatedCodeDetector()
        result = detector.detect_outdated("Section 302, IPC applies here")
        for ref in result.outdated_references:
            print(ref.suggestion)
    """

    # Regex patterns for detecting old code references
    # Each pattern captures (section_number, code_name) or (code_name, section_number)
    OLD_CODE_PATTERNS = [
        # Section 302, IPC / Section 302 of IPC / Section 302 IPC
        (
            r'(?:Section|S\.?|Sec\.?)\s*(\d+[A-Za-z]?)\s*(?:,|\s+of\s+|\s+)?\s*(IPC|Indian Penal Code)',
            False  # False = section first, code second
        ),
        # Section 438 CrPC / Section 438 of CrPC
        (
            r'(?:Section|S\.?|Sec\.?)\s*(\d+[A-Za-z]?)\s*(?:,|\s+of\s+|\s+)?\s*(CrPC|Cr\.P\.C\.?|Code of Criminal Procedure)',
            False
        ),
        # Section 65B IEA / Section 65B of Evidence Act
        (
            r'(?:Section|S\.?|Sec\.?)\s*(\d+[A-Za-z]?)\s*(?:,|\s+of\s+|\s+)?\s*(IEA|Indian Evidence Act|Evidence Act)',
            False
        ),
        # IPC Section 302 / IPC S. 302
        (
            r'(IPC|Indian Penal Code)\s+(?:Section|S\.?|Sec\.?)\s*(\d+[A-Za-z]?)',
            True  # True = code first, section second
        ),
        # CrPC Section 438
        (
            r'(CrPC|Cr\.P\.C\.?|Code of Criminal Procedure)\s+(?:Section|S\.?|Sec\.?)\s*(\d+[A-Za-z]?)',
            True
        ),
        # IEA Section 65B
        (
            r'(IEA|Indian Evidence Act|Evidence Act)\s+(?:Section|S\.?|Sec\.?)\s*(\d+[A-Za-z]?)',
            True
        ),
    ]

    # Mapping from pattern code names to canonical names
    CODE_CANONICAL = {
        "IPC": "IPC",
        "Indian Penal Code": "IPC",
        "CrPC": "CrPC",
        "Cr.P.C.": "CrPC",
        "Cr.P.C": "CrPC",
        "Code of Criminal Procedure": "CrPC",
        "IEA": "IEA",
        "Indian Evidence Act": "IEA",
        "Evidence Act": "IEA",
    }

    # New code names
    CODE_NEW_NAMES = {
        "IPC": "BNS",
        "CrPC": "BNSS",
        "IEA": "BSA",
    }

    def __init__(self, code_mapper: Optional[LegalCodeMapper] = None):
        """
        Initialize the detector.

        Args:
            code_mapper: LegalCodeMapper instance for section mapping.
                        Creates one if not provided.
        """
        self.code_mapper = code_mapper or LegalCodeMapper()
        self.logger = get_audit_logger("outdated_detector")

    def detect_outdated(self, text: str) -> DetectionResult:
        """
        Detect outdated legal code references in text.

        Args:
            text: Text to scan for outdated references

        Returns:
            DetectionResult with all findings
        """
        if not text:
            return DetectionResult(
                outdated_references=[],
                has_outdated=False,
                summary="No text provided"
            )

        references: List[OutdatedReference] = []
        seen_positions: set = set()  # Avoid duplicate matches

        for pattern, code_first in self.OLD_CODE_PATTERNS:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                start, end = match.span()

                # Skip if we've already captured this position
                if start in seen_positions:
                    continue
                seen_positions.add(start)

                # Extract section and code based on pattern order
                if code_first:
                    code_name = match.group(1)
                    section = match.group(2)
                else:
                    section = match.group(1)
                    code_name = match.group(2)

                # Normalize code name
                canonical_code = self.CODE_CANONICAL.get(code_name, code_name)

                # Get mapping from LegalCodeMapper
                mapping_result = self.code_mapper.map_section(canonical_code, section)

                new_code = None
                new_section = None

                if mapping_result.status == CodeMappingStatus.MAPPED:
                    new_code = mapping_result.new_code
                    new_section = mapping_result.new_section
                else:
                    # Use default new code name even if no specific mapping
                    new_code = self.CODE_NEW_NAMES.get(canonical_code)

                # Format suggestion
                suggestion = self._format_suggestion(
                    canonical_code, section, new_code, new_section
                )

                references.append(OutdatedReference(
                    original_text=match.group(0),
                    old_code=canonical_code,
                    old_section=section,
                    new_code=new_code,
                    new_section=new_section,
                    suggestion=suggestion,
                    position=(start, end)
                ))

        # Sort by position
        references.sort(key=lambda r: r.position[0])

        # Build result
        has_outdated = len(references) > 0
        if has_outdated:
            summary = f"Found {len(references)} reference(s) to old legal codes"
        else:
            summary = "No outdated legal code references found"

        # Log detection attempt
        self.logger.info(
            "outdated_code_detection",
            text_length=len(text),
            references_found=len(references),
            has_outdated=has_outdated
        )

        return DetectionResult(
            outdated_references=references,
            has_outdated=has_outdated,
            summary=summary
        )

    def _format_suggestion(
        self,
        old_code: str,
        old_section: str,
        new_code: Optional[str],
        new_section: Optional[str]
    ) -> str:
        """
        Format a user-friendly suggestion message.

        Args:
            old_code: Original code name (IPC, CrPC, IEA)
            old_section: Original section number
            new_code: New code name (BNS, BNSS, BSA) if available
            new_section: New section number if available

        Returns:
            Formatted suggestion string
        """
        if new_code and new_section:
            return (
                f"Section {old_section}, {old_code} corresponds to "
                f"Section {new_section}, {new_code} (effective July 1, 2024)"
            )
        elif new_code:
            return (
                f"Section {old_section}, {old_code} - please verify "
                f"current equivalent in {new_code}"
            )
        else:
            return (
                f"Section {old_section}, {old_code} - please verify "
                f"current equivalent"
            )

    def annotate_text(self, text: str, result: DetectionResult) -> str:
        """
        Insert inline annotations for outdated references.

        Args:
            text: Original text
            result: DetectionResult from detect_outdated()

        Returns:
            Text with inline annotations added
        """
        if not result.has_outdated:
            return text

        # Process references in reverse order to preserve positions
        annotated = text
        for ref in reversed(result.outdated_references):
            start, end = ref.position
            original = annotated[start:end]

            if ref.new_section and ref.new_code:
                annotation = f"{original} [Note: Now Section {ref.new_section}, {ref.new_code}]"
            elif ref.new_code:
                annotation = f"{original} [Note: Verify equivalent in {ref.new_code}]"
            else:
                annotation = f"{original} [Note: Verify current equivalent]"

            annotated = annotated[:start] + annotation + annotated[end:]

        return annotated

    def get_suggestions_html(self, result: DetectionResult) -> str:
        """
        Generate HTML snippet with code update suggestions.

        Args:
            result: DetectionResult from detect_outdated()

        Returns:
            HTML string for display in response
        """
        if not result.has_outdated:
            return ""

        suggestions = []
        for ref in result.outdated_references:
            suggestions.append(f"<li>{ref.suggestion}</li>")

        html = f"""
<div style="background-color: #fff3cd; border: 1px solid #ffc107; border-radius: 4px; padding: 12px; margin: 10px 0;">
    <strong style="color: #856404;">Legal Code Update Notice</strong>
    <p style="margin: 8px 0 4px 0; color: #856404;">
        The following references use old legal codes that have been replaced effective July 1, 2024:
    </p>
    <ul style="margin: 4px 0; padding-left: 20px; color: #856404;">
        {''.join(suggestions)}
    </ul>
</div>
"""
        return html.strip()
