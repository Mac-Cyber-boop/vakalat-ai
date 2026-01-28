"""
Citation recommender orchestrating retrieval, verification, and formatting.

Provides the main interface for intelligent precedent suggestion (CITE-05):
- Retrieves relevant precedents via PrecedentRetriever
- Verifies citations via CitationVerifier
- Formats output using CitationFormatter

Key components:
- CitationRecommendation: Output model with formatted citation and verification badge
- CitationRecommender: Orchestrator class combining retrieval + verification + formatting
"""

import re
from typing import Literal, Optional

from pydantic import BaseModel, Field
from langchain_pinecone import PineconeVectorStore

from src.citations.retriever import PrecedentRetriever, RetrievedPrecedent
from src.citations.formatter import CitationFormatter
from src.verification.citation_verifier import CitationVerifier
from src.verification.models import VerificationStatus


def _create_badge_html(status: str) -> str:
    """
    Generate HTML for a verification badge.

    Args:
        status: Verification status ('verified' or 'unverified')

    Returns:
        HTML span with appropriate icon and styling
    """
    if status == "verified":
        return '<span class="citation-badge verified" title="Verified in database">&#10003;</span>'
    else:
        return '<span class="citation-badge unverified" title="Could not verify">&#9888;</span>'


class CitationRecommendation(BaseModel):
    """
    A recommended precedent with verification status and formatted citation.

    Combines retrieval intelligence, verification status, and proper
    legal citation formatting into a single recommendation output.
    """

    case_name: str = Field(
        description="Full case name (e.g., 'Arnesh Kumar vs State of Bihar')"
    )
    formatted_citation: str = Field(
        description="Properly formatted citation (SCC or AIR style)"
    )
    year: int = Field(
        description="Year of judgment"
    )
    court: str = Field(
        description="Court name (normalized identifier)"
    )
    relevance_score: float = Field(
        ge=0.0,
        le=1.0,
        description="Combined ranking score from retrieval (0.0-1.0)"
    )
    verification_status: Literal["verified", "unverified"] = Field(
        description="Verification status from CitationVerifier"
    )
    badge_html: str = Field(
        description="HTML span with verification indicator"
    )
    snippet: str = Field(
        description="Relevant excerpt from the case judgment"
    )
    source_id: Optional[str] = Field(
        default=None,
        description="Database reference if verified"
    )


class CitationRecommender:
    """
    Orchestrates retrieval, verification, and formatting for precedent suggestions.

    Combines:
    - PrecedentRetriever: Jurisdiction-aware case retrieval
    - CitationVerifier: Database verification (optional)
    - CitationFormatter: Standard Indian legal citation formatting

    Example:
        recommender = CitationRecommender(vector_store, citation_verifier)
        recommendations = recommender.recommend_precedents(
            legal_issue="anticipatory bail in economic offences",
            filing_court="delhi",
            top_k=5
        )
        for rec in recommendations:
            print(f"{rec.formatted_citation} [{rec.verification_status}]")
    """

    def __init__(
        self,
        vector_store: PineconeVectorStore,
        citation_verifier: Optional[CitationVerifier] = None
    ):
        """
        Initialize the CitationRecommender.

        Args:
            vector_store: PineconeVectorStore instance connected to legal database
            citation_verifier: Optional CitationVerifier for verification status
        """
        self.retriever = PrecedentRetriever(vector_store)
        self.citation_verifier = citation_verifier
        self.formatter = CitationFormatter

    def _parse_case_parties(self, case_name: str) -> tuple[str, str]:
        """
        Extract party names from case name.

        Splits on common separators: 'vs', 'v.', 'versus' (case-insensitive).

        Args:
            case_name: Full case name string

        Returns:
            Tuple of (party_a, party_b). If parsing fails, returns
            (case_name, "").
        """
        # Try different separators in order of preference
        separators = [
            r'\s+vs\.?\s+',      # "vs" or "vs."
            r'\s+versus\s+',     # "versus"
            r'\s+v\.\s+',        # "v."
        ]

        for sep_pattern in separators:
            match = re.split(sep_pattern, case_name, maxsplit=1, flags=re.IGNORECASE)
            if len(match) == 2:
                party_a = match[0].strip()
                party_b = match[1].strip()
                return (party_a, party_b)

        # Fallback: return full name as party_a
        return (case_name.strip(), "")

    def _format_precedent(self, precedent: RetrievedPrecedent) -> str:
        """
        Format a retrieved precedent as a citation string.

        Attempts to use SCC or AIR format if metadata available,
        otherwise falls back to basic format with year.

        Args:
            precedent: Retrieved precedent from PrecedentRetriever

        Returns:
            Formatted citation string
        """
        party_a, party_b = self._parse_case_parties(precedent.case_name)
        metadata = precedent.metadata

        # Try to extract SCC citation info
        scc_volume = metadata.get("scc_volume") or metadata.get("volume")
        scc_page = metadata.get("scc_page") or metadata.get("page")

        if scc_volume and scc_page:
            try:
                return self.formatter.format_scc(
                    party_a=party_a,
                    party_b=party_b or "State",
                    year=precedent.year,
                    volume=int(scc_volume),
                    page=int(scc_page)
                )
            except (ValueError, TypeError):
                pass  # Fall through to AIR or basic format

        # Try AIR format if court info available
        air_page = metadata.get("air_page")
        if air_page and precedent.court:
            try:
                # Get court abbreviation
                court_abbr = self.formatter.HIGH_COURT_ABBREVIATIONS.get(
                    precedent.court.lower().replace(" ", "_"),
                    precedent.court.upper()
                )
                return self.formatter.format_air(
                    party_a=party_a,
                    party_b=party_b or "State",
                    year=precedent.year,
                    court=court_abbr,
                    page=int(air_page)
                )
            except (ValueError, TypeError):
                pass  # Fall through to basic format

        # Basic format: "Case Name (Year)"
        if party_b:
            return f"{party_a} vs {party_b} ({precedent.year})"
        else:
            return f"{precedent.case_name} ({precedent.year})"

    def _verify_precedent(
        self, precedent: RetrievedPrecedent
    ) -> tuple[str, Optional[str]]:
        """
        Verify a precedent against the database.

        Args:
            precedent: Retrieved precedent to verify

        Returns:
            Tuple of (status, source_id) where status is 'verified' or 'unverified'
        """
        if self.citation_verifier is None:
            return ("unverified", None)

        try:
            result = self.citation_verifier.verify_case_citation(
                case_name=precedent.case_name,
                year=precedent.year
            )

            if result.status == VerificationStatus.VERIFIED:
                return ("verified", result.source_id)
            else:
                return ("unverified", None)

        except Exception:
            # On any error, return unverified
            return ("unverified", None)

    def recommend_precedents(
        self,
        legal_issue: str,
        filing_court: str = "supreme_court",
        top_k: int = 5
    ) -> list[CitationRecommendation]:
        """
        Recommend relevant precedents for a legal issue.

        Orchestrates the full pipeline:
        1. Retrieve relevant precedents via PrecedentRetriever
        2. Verify each precedent via CitationVerifier (if available)
        3. Format citations using CitationFormatter
        4. Return sorted list of recommendations

        Args:
            legal_issue: Legal issue or question to find precedents for
            filing_court: Court where case is being filed (for jurisdiction weighting)
            top_k: Number of recommendations to return (default 5)

        Returns:
            List of CitationRecommendation sorted by relevance_score descending
        """
        # Step 1: Retrieve precedents
        precedents = self.retriever.retrieve_precedents(
            legal_issue=legal_issue,
            filing_court=filing_court,
            top_k=top_k
        )

        if not precedents:
            return []

        # Step 2 & 3: Verify and format each precedent
        recommendations = []

        for precedent in precedents:
            # Verify
            status, source_id = self._verify_precedent(precedent)

            # Format citation
            formatted_citation = self._format_precedent(precedent)

            # Create badge HTML
            badge_html = _create_badge_html(status)

            # Create recommendation
            recommendation = CitationRecommendation(
                case_name=precedent.case_name,
                formatted_citation=formatted_citation,
                year=precedent.year,
                court=precedent.court,
                relevance_score=precedent.final_score,
                verification_status=status,
                badge_html=badge_html,
                snippet=precedent.snippet,
                source_id=source_id
            )
            recommendations.append(recommendation)

        # Step 4: Sort by relevance score (should already be sorted, but ensure)
        recommendations.sort(key=lambda r: r.relevance_score, reverse=True)

        return recommendations
