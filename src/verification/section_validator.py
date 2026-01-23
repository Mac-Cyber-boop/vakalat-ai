"""
Section Validator for Indian Legal Codes.

Validates section numbers exist in actual law documents stored in Pinecone
and detects repealed/outdated sections with suggested replacements.

Key functionality:
- Query Pinecone to verify section text exists
- Detect old codes (IPC, CrPC, IEA) vs new codes (BNS, BNSS, BSA)
- Mark old code sections as OUTDATED with suggested new equivalents
- Track validation time for performance monitoring
- Log all validation attempts via audit logger
"""

import re
import time
from datetime import datetime
from typing import Optional

from langchain_pinecone import PineconeVectorStore
from pydantic import BaseModel, Field

from src.verification.models import VerificationStatus, VerificationResult
from src.verification.code_mapper import LegalCodeMapper, CodeMappingStatus
from src.verification.audit import get_audit_logger, AuditEvent, log_verification_attempt


class SectionValidationResult(BaseModel):
    """Result of a section validation operation."""

    status: VerificationStatus = Field(
        description="Verification status: VERIFIED, UNVERIFIED, or OUTDATED"
    )
    act_name: str = Field(
        description="Name of the act being validated"
    )
    section: str = Field(
        description="Section number being validated"
    )
    section_text_preview: Optional[str] = Field(
        default=None,
        description="First 200 characters of section text if found"
    )
    is_repealed: bool = Field(
        default=False,
        description="Whether the section has been repealed"
    )
    suggested_replacement: Optional[str] = Field(
        default=None,
        description="Suggested replacement for outdated sections (e.g., 'Section 103, BNS')"
    )
    validation_time_ms: int = Field(
        default=0,
        ge=0,
        description="Time taken for validation in milliseconds"
    )


# Module-level cached mapper instance for quick_check
_quick_check_mapper: Optional[LegalCodeMapper] = None


def _get_quick_check_mapper() -> LegalCodeMapper:
    """Get or create the cached module-level LegalCodeMapper instance."""
    global _quick_check_mapper
    if _quick_check_mapper is None:
        _quick_check_mapper = LegalCodeMapper()
    return _quick_check_mapper


def quick_check(act_name: str, section: str) -> bool:
    """
    Fast pre-filtering check against known section lists.

    Does NOT query Pinecone - just checks if section is in known mappings.
    Uses a cached module-level LegalCodeMapper instance.

    Args:
        act_name: Name of the act (e.g., "IPC", "BNS")
        section: Section number to check

    Returns:
        True if section is in known mappings (IPC, CrPC, IEA, BNS, BNSS, BSA).
    """
    mapper = _get_quick_check_mapper()
    return mapper.is_section_valid(act_name, section)


class SectionValidator:
    """
    Validates section numbers exist in actual law documents.

    Queries Pinecone vector store to verify section text exists,
    detects outdated codes, and suggests current equivalents.

    Usage:
        validator = SectionValidator(vector_store)
        result = validator.validate_section("IPC", "302")
        # Returns SectionValidationResult with OUTDATED status and BNS equivalent
    """

    # Old Indian criminal codes (pre-July 1, 2024)
    OLD_CODES = {
        "IPC", "INDIAN PENAL CODE", "INDIAN PENAL CODE, 1860", "INDIAN PENAL CODE 1860",
        "I.P.C.", "I.P.C",
        "CRPC", "CR.P.C.", "CR.P.C", "CODE OF CRIMINAL PROCEDURE",
        "CODE OF CRIMINAL PROCEDURE, 1973", "CODE OF CRIMINAL PROCEDURE 1973",
        "CRIMINAL PROCEDURE CODE",
        "EVIDENCE ACT", "IEA", "INDIAN EVIDENCE ACT",
        "INDIAN EVIDENCE ACT, 1872", "INDIAN EVIDENCE ACT 1872",
        "I.E.A.", "I.E.A"
    }

    # New Indian criminal codes (effective July 1, 2024)
    NEW_CODES = {
        "BNS", "BHARATIYA NYAYA SANHITA", "BHARATIYA NYAYA SANHITA, 2023",
        "BNSS", "BHARATIYA NAGARIK SURAKSHA SANHITA", "BHARATIYA NAGARIK SURAKSHA SANHITA, 2023",
        "BSA", "BHARATIYA SAKSHYA ADHINIYAM", "BHARATIYA SAKSHYA ADHINIYAM, 2023"
    }

    # Canonical name mappings for normalization
    CANONICAL_NAMES = {
        # IPC variations
        "IPC": "IPC",
        "INDIAN PENAL CODE": "IPC",
        "INDIAN PENAL CODE, 1860": "IPC",
        "INDIAN PENAL CODE 1860": "IPC",
        "I.P.C.": "IPC",
        "I.P.C": "IPC",
        # CrPC variations
        "CRPC": "CrPC",
        "CR.P.C.": "CrPC",
        "CR.P.C": "CrPC",
        "CODE OF CRIMINAL PROCEDURE": "CrPC",
        "CODE OF CRIMINAL PROCEDURE, 1973": "CrPC",
        "CODE OF CRIMINAL PROCEDURE 1973": "CrPC",
        "CRIMINAL PROCEDURE CODE": "CrPC",
        # IEA variations
        "EVIDENCE ACT": "IEA",
        "IEA": "IEA",
        "INDIAN EVIDENCE ACT": "IEA",
        "INDIAN EVIDENCE ACT, 1872": "IEA",
        "INDIAN EVIDENCE ACT 1872": "IEA",
        "I.E.A.": "IEA",
        "I.E.A": "IEA",
        # BNS variations
        "BNS": "BNS",
        "BHARATIYA NYAYA SANHITA": "BNS",
        "BHARATIYA NYAYA SANHITA, 2023": "BNS",
        # BNSS variations
        "BNSS": "BNSS",
        "BHARATIYA NAGARIK SURAKSHA SANHITA": "BNSS",
        "BHARATIYA NAGARIK SURAKSHA SANHITA, 2023": "BNSS",
        # BSA variations
        "BSA": "BSA",
        "BHARATIYA SAKSHYA ADHINIYAM": "BSA",
        "BHARATIYA SAKSHYA ADHINIYAM, 2023": "BSA",
    }

    # Map canonical old code names to their Pinecone source_book values
    # Based on ingest_master.py and existing database structure
    CODE_TO_SOURCE_BOOK = {
        "IPC": ["Indian Penal Code", "IPC"],
        "CrPC": ["Code of Criminal Procedure", "CrPC", "Criminal Procedure Code"],
        "IEA": ["Indian Evidence Act", "Evidence Act", "IEA"],
        "BNS": ["Bharatiya Nyaya Sanhita", "BNS"],
        "BNSS": ["Bharatiya Nagarik Suraksha Sanhita", "BNSS"],
        "BSA": ["Bharatiya Sakshya Adhiniyam", "BSA"],
    }

    # Map canonical old code to its new equivalent
    OLD_TO_NEW_CODE = {
        "IPC": "BNS",
        "CrPC": "BNSS",
        "IEA": "BSA",
    }

    def __init__(
        self,
        vector_store: PineconeVectorStore,
        code_mapper: Optional[LegalCodeMapper] = None
    ):
        """
        Initialize the SectionValidator.

        Args:
            vector_store: PineconeVectorStore instance for querying law documents
            code_mapper: Optional LegalCodeMapper for old->new section mapping.
                        If not provided, a new instance will be created.
        """
        self.vector_store = vector_store
        self.code_mapper = code_mapper if code_mapper is not None else LegalCodeMapper()
        self.logger = get_audit_logger("section_validator")

    def _normalize_act_name(self, act_name: str) -> str:
        """
        Normalize an act name to its canonical form.

        Args:
            act_name: User-provided act name

        Returns:
            Canonical act name (IPC, CrPC, IEA, BNS, BNSS, BSA) or original if unknown
        """
        normalized = act_name.upper().strip()
        return self.CANONICAL_NAMES.get(normalized, act_name)

    def _is_old_code(self, act_name: str) -> bool:
        """
        Check if an act name refers to an old (pre-2024) code.

        Args:
            act_name: Act name (can be user input or canonical)

        Returns:
            True if act is IPC, CrPC, or IEA (or their variations)
        """
        normalized = act_name.upper().strip()
        return normalized in self.OLD_CODES or self._normalize_act_name(act_name) in ["IPC", "CrPC", "IEA"]

    def _is_new_code(self, act_name: str) -> bool:
        """
        Check if an act name refers to a new (post-2024) code.

        Args:
            act_name: Act name (can be user input or canonical)

        Returns:
            True if act is BNS, BNSS, or BSA (or their variations)
        """
        normalized = act_name.upper().strip()
        return normalized in self.NEW_CODES or self._normalize_act_name(act_name) in ["BNS", "BNSS", "BSA"]

    def _extract_section_from_text(self, text: str, section: str) -> bool:
        """
        Check if the given section number appears in the text.

        Args:
            text: Document text content
            section: Section number to search for

        Returns:
            True if section reference is found in text
        """
        # Normalize section for matching
        section_clean = section.strip().upper()

        # Build regex patterns to match section references
        # Match: "Section 302", "S. 302", "Sec. 302", "302."
        patterns = [
            rf'\bSection\s+{re.escape(section_clean)}\b',
            rf'\bS\.\s*{re.escape(section_clean)}\b',
            rf'\bSec\.\s*{re.escape(section_clean)}\b',
            rf'\b{re.escape(section_clean)}\.',  # Section number followed by period
        ]

        text_upper = text.upper()
        for pattern in patterns:
            if re.search(pattern, text_upper, re.IGNORECASE):
                return True

        return False

    def _get_suggested_replacement(self, canonical_code: str, section: str) -> Optional[str]:
        """
        Get suggested replacement for an old code section.

        Args:
            canonical_code: Canonical code name (IPC, CrPC, IEA)
            section: Section number

        Returns:
            Suggestion string like "For matters after July 1, 2024, use Section 103, BNS"
            or None if no mapping exists
        """
        if canonical_code not in self.OLD_TO_NEW_CODE:
            return None

        # Use code_mapper to get the new equivalent
        mapping_result = self.code_mapper.map_section(canonical_code, section)

        if mapping_result.status == CodeMappingStatus.MAPPED and mapping_result.new_section:
            new_code = mapping_result.new_code or self.OLD_TO_NEW_CODE.get(canonical_code)
            return f"For matters after July 1, 2024, use Section {mapping_result.new_section}, {new_code}"
        elif mapping_result.status == CodeMappingStatus.MAPPED and mapping_result.notes:
            # Section was repealed or has special handling
            new_code = self.OLD_TO_NEW_CODE.get(canonical_code)
            return f"For matters after July 1, 2024: {mapping_result.notes} ({new_code})"

        return None

    def validate_section(self, act_name: str, section: str) -> SectionValidationResult:
        """
        Validate that a section exists in the actual law.

        Args:
            act_name: Name of the act (e.g., "IPC", "BNS", "Indian Penal Code")
            section: Section number to validate (e.g., "302", "65B")

        Returns:
            SectionValidationResult with:
            - VERIFIED: Section found in new code (BNS/BNSS/BSA)
            - OUTDATED: Section found in old code (IPC/CrPC/IEA) with suggested replacement
            - UNVERIFIED: Section not found in database
        """
        start_time = time.time()

        # Normalize the act name
        canonical_code = self._normalize_act_name(act_name)

        # Get source_book values for Pinecone filter
        source_books = self.CODE_TO_SOURCE_BOOK.get(canonical_code, [act_name])

        # Build filter for Pinecone query
        # Filter by source_type=statute and source_book matching the act
        filter_dict = {
            "$and": [
                {"source_type": {"$eq": "statute"}},
                {"source_book": {"$in": source_books}}
            ]
        }

        # Query Pinecone
        query_text = f"Section {section}"
        try:
            results = self.vector_store.similarity_search(
                query=query_text,
                k=3,
                filter=filter_dict
            )
        except Exception as e:
            # Handle Pinecone connection errors gracefully
            validation_time_ms = int((time.time() - start_time) * 1000)

            # Log the error
            self._log_validation(
                act_name=act_name,
                section=section,
                result_status="UNVERIFIED",
                validation_time_ms=validation_time_ms,
                error_message=str(e)
            )

            return SectionValidationResult(
                status=VerificationStatus.UNVERIFIED,
                act_name=act_name,
                section=section,
                section_text_preview=None,
                is_repealed=False,
                suggested_replacement=None,
                validation_time_ms=validation_time_ms
            )

        # Check if section is found in results
        section_found = False
        text_preview = None

        for doc in results:
            content = doc.page_content
            if self._extract_section_from_text(content, section):
                section_found = True
                text_preview = content[:200] if len(content) > 200 else content
                break

        validation_time_ms = int((time.time() - start_time) * 1000)

        if section_found:
            if self._is_old_code(canonical_code):
                # Old code section - VERIFIED but OUTDATED
                suggested = self._get_suggested_replacement(canonical_code, section)

                self._log_validation(
                    act_name=act_name,
                    section=section,
                    result_status="OUTDATED",
                    validation_time_ms=validation_time_ms,
                    suggested_replacement=suggested
                )

                return SectionValidationResult(
                    status=VerificationStatus.OUTDATED,
                    act_name=act_name,
                    section=section,
                    section_text_preview=text_preview,
                    is_repealed=False,  # Old codes still valid for pre-2024 cases
                    suggested_replacement=suggested,
                    validation_time_ms=validation_time_ms
                )
            else:
                # New code section - fully VERIFIED
                self._log_validation(
                    act_name=act_name,
                    section=section,
                    result_status="VERIFIED",
                    validation_time_ms=validation_time_ms
                )

                return SectionValidationResult(
                    status=VerificationStatus.VERIFIED,
                    act_name=act_name,
                    section=section,
                    section_text_preview=text_preview,
                    is_repealed=False,
                    suggested_replacement=None,
                    validation_time_ms=validation_time_ms
                )
        else:
            # Section not found
            self._log_validation(
                act_name=act_name,
                section=section,
                result_status="UNVERIFIED",
                validation_time_ms=validation_time_ms
            )

            return SectionValidationResult(
                status=VerificationStatus.UNVERIFIED,
                act_name=act_name,
                section=section,
                section_text_preview=None,
                is_repealed=False,
                suggested_replacement=None,
                validation_time_ms=validation_time_ms
            )

    def _log_validation(
        self,
        act_name: str,
        section: str,
        result_status: str,
        validation_time_ms: int,
        suggested_replacement: Optional[str] = None,
        error_message: Optional[str] = None
    ) -> None:
        """
        Log a validation attempt using the audit logger.

        Args:
            act_name: Act name being validated
            section: Section number being validated
            result_status: Validation result (VERIFIED, UNVERIFIED, OUTDATED)
            validation_time_ms: Time taken for validation
            suggested_replacement: Suggested replacement if outdated
            error_message: Error message if validation failed
        """
        event = AuditEvent(
            event_type="section_validation",
            input_data={
                "act_name": act_name,
                "section": section
            },
            result=result_status,
            duration_ms=validation_time_ms,
            reason=suggested_replacement or error_message
        )
        log_verification_attempt(self.logger, event)

    def validate_statute_citation(self, act_name: str, section: str) -> VerificationResult:
        """
        Validate a statute citation and return a VerificationResult.

        Wrapper method for consistency with CitationVerifier interface.

        Args:
            act_name: Name of the act
            section: Section number

        Returns:
            VerificationResult with appropriate status and details
        """
        result = self.validate_section(act_name, section)

        return VerificationResult(
            status=result.status,
            source_id=None,  # Section validation doesn't have a single source_id
            confidence=1.0 if result.status == VerificationStatus.VERIFIED else 0.0,
            reason=result.suggested_replacement if result.status == VerificationStatus.OUTDATED else (
                f"Section {section} not found in {act_name}" if result.status == VerificationStatus.UNVERIFIED else None
            ),
            suggested_update=result.suggested_replacement,
            verification_time_ms=result.validation_time_ms
        )

    def batch_validate(
        self,
        citations: list[tuple[str, str]]
    ) -> list[SectionValidationResult]:
        """
        Validate multiple (act_name, section) pairs.

        Useful for validating all citations in a document at once.

        Args:
            citations: List of (act_name, section) tuples

        Returns:
            List of SectionValidationResult in same order as input
        """
        results = []
        for act_name, section in citations:
            result = self.validate_section(act_name, section)
            results.append(result)
        return results

    @staticmethod
    def quick_check(act_name: str, section: str) -> bool:
        """
        Fast pre-filtering check against known section lists.

        Delegates to module-level quick_check function.
        Does NOT query Pinecone - just checks if section is in known mappings.

        Args:
            act_name: Name of the act (e.g., "IPC", "BNS")
            section: Section number to check

        Returns:
            True if section is in known mappings.
        """
        return quick_check(act_name, section)
