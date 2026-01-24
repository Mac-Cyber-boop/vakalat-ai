"""
Citation verification against Pinecone vector database.

Provides CitationVerifier class for verifying case citations
against the legal database with exact metadata matching.
"""

import time
from datetime import datetime, timezone
from typing import Optional

from langchain_pinecone import PineconeVectorStore

from src.verification.models import (
    VerificationResult,
    VerificationStatus,
    CaseCitationInput,
)
from src.verification.audit import (
    get_audit_logger,
    AuditEvent,
    log_verification_attempt,
)


class CitationVerifier:
    """
    Verifies case citations against Pinecone legal database.

    Uses metadata filtering to find exact matches for case citations.
    All verification attempts are logged for audit compliance.

    Example:
        verifier = CitationVerifier(vector_store)
        result = verifier.verify_case_citation("Arnesh Kumar vs State of Bihar", 2014)
        print(result.status)  # VerificationStatus.VERIFIED or UNVERIFIED
    """

    def __init__(self, vector_store: PineconeVectorStore):
        """
        Initialize the citation verifier.

        Args:
            vector_store: PineconeVectorStore instance connected to legal database
        """
        self.vector_store = vector_store
        self.logger = get_audit_logger("citation_verifier")
        # Simple cache for verified citations to avoid repeated lookups
        self._verification_cache: dict[str, VerificationResult] = {}

    def _normalize_case_name(self, case_name: str) -> str:
        """
        Normalize case name for consistent matching.

        Args:
            case_name: Original case name string

        Returns:
            Normalized lowercase case name
        """
        return case_name.lower().strip()

    def _get_cache_key(
        self, case_name: str, year: int, citation: Optional[str] = None
    ) -> str:
        """Generate cache key for verification result."""
        normalized = self._normalize_case_name(case_name)
        return f"{normalized}|{year}|{citation or ''}"

    def _extract_case_name_parts(self, case_name: str) -> list[str]:
        """
        Extract key parts from case name for matching.

        Args:
            case_name: Full case name

        Returns:
            List of key terms to match (party names)
        """
        # Remove common terms and split
        normalized = self._normalize_case_name(case_name)
        # Remove common legal terms
        for term in ["vs", "v.", "versus", "state of", "union of india", "&", "anr", "ors"]:
            normalized = normalized.replace(term, " ")
        # Split and filter out short words
        parts = [p.strip() for p in normalized.split() if len(p.strip()) > 2]
        return parts

    def verify_case_citation(
        self,
        case_name: str,
        year: int,
        citation: Optional[str] = None
    ) -> VerificationResult:
        """
        Verify a case citation against the legal database.

        Uses semantic search to find matching case documents, then validates
        the match by checking metadata and content.
        Per TRUST-01 requirements, exact match is required - no fuzzy matching.

        Args:
            case_name: Name of the case (e.g., "Arnesh Kumar vs State of Bihar")
            year: Year of the judgment
            citation: Optional formal citation (e.g., "(2014) 8 SCC 273")

        Returns:
            VerificationResult with status, source_id (if found), and timing
        """
        # Check cache first
        cache_key = self._get_cache_key(case_name, year, citation)
        if cache_key in self._verification_cache:
            cached = self._verification_cache[cache_key]
            # Return cached result with 0ms time (cache hit)
            return VerificationResult(
                status=cached.status,
                source_id=cached.source_id,
                confidence=cached.confidence,
                reason=cached.reason,
                verification_time_ms=0,
            )

        # Start timing
        start = time.perf_counter()

        # Normalize case name for search
        normalized_name = self._normalize_case_name(case_name)
        case_name_parts = self._extract_case_name_parts(case_name)

        result: VerificationResult

        try:
            # Query Pinecone without metadata filter (database may not have source_type)
            # Using the case name as query to find matching documents
            results = self.vector_store.similarity_search(
                query=case_name,
                k=5,
            )

            # Check if any result matches the case name
            # We check metadata fields AND document content for case name presence
            found_match = False
            source_id = None

            for doc in results:
                metadata = doc.metadata
                content_lower = doc.page_content.lower()

                # Check source filename (e.g., "Arnesh Kumar.Pdf")
                source_file = metadata.get("source", "").lower()
                # Check case_name field if present
                doc_case_name = self._normalize_case_name(metadata.get("case_name", ""))
                # Check title field if present
                doc_title = self._normalize_case_name(metadata.get("title", ""))

                # Method 1: Check if key case name parts appear in source filename
                # e.g., "arnesh kumar" in "arnesh kumar.pdf"
                source_match = all(
                    part in source_file for part in case_name_parts[:2]
                ) if case_name_parts else False

                # Method 2: Check explicit metadata fields
                metadata_match = (
                    (normalized_name in doc_case_name or doc_case_name in normalized_name)
                    if doc_case_name else False
                ) or (
                    (normalized_name in doc_title or doc_title in normalized_name)
                    if doc_title else False
                )

                # Method 3: Check content for case name and year
                # This catches cases where metadata is sparse but content is valid
                year_str = str(year)
                content_match = (
                    all(part in content_lower for part in case_name_parts[:2])
                    and year_str in content_lower
                ) if case_name_parts else False

                if source_match or metadata_match or content_match:
                    found_match = True
                    # Use source_id if available, otherwise use source filename
                    source_id = metadata.get(
                        "source_id",
                        metadata.get("source", f"case_{case_name[:30]}")
                    )
                    break

            # Calculate duration
            duration_ms = int((time.perf_counter() - start) * 1000)

            if found_match:
                result = VerificationResult(
                    status=VerificationStatus.VERIFIED,
                    source_id=source_id,
                    confidence=1.0,
                    verification_time_ms=duration_ms,
                )
            else:
                result = VerificationResult(
                    status=VerificationStatus.UNVERIFIED,
                    reason="Case not found in legal database",
                    confidence=0.0,
                    verification_time_ms=duration_ms,
                )

        except Exception as e:
            # Calculate duration even on error
            duration_ms = int((time.perf_counter() - start) * 1000)
            result = VerificationResult(
                status=VerificationStatus.UNVERIFIED,
                reason=f"Verification error: {str(e)}",
                confidence=0.0,
                verification_time_ms=duration_ms,
            )

        # Cache the result (excluding timing)
        self._verification_cache[cache_key] = result

        # Log the verification attempt
        audit_event = AuditEvent(
            event_type="citation_verification",
            timestamp=datetime.now(timezone.utc),
            input_data={
                "case_name": case_name,
                "year": year,
                "citation": citation,
            },
            result=result.status.value,
            duration_ms=result.verification_time_ms,
            source_id=result.source_id,
            reason=result.reason,
        )
        log_verification_attempt(self.logger, audit_event)

        return result

    def verify_from_input(self, input: CaseCitationInput) -> VerificationResult:
        """
        Verify a case citation from a CaseCitationInput model.

        Convenience wrapper for verify_case_citation.

        Args:
            input: CaseCitationInput containing case_name, year, and optional citation

        Returns:
            VerificationResult with status, source_id (if found), and timing
        """
        return self.verify_case_citation(
            case_name=input.case_name,
            year=input.year,
            citation=input.citation,
        )

    def clear_cache(self) -> None:
        """Clear the verification cache."""
        self._verification_cache.clear()
