"""
Citation formatter for Indian legal citation standards.

Provides formatting methods for:
- SCC format: "Party vs Party (Year) Volume SCC Page"
- AIR format: "Party vs Party, AIR Year Court Page"
- Statute format: "Section N, Act Name, Year"
"""

from typing import Union

from src.citations.models import (
    CaseCitation,
    StatuteCitation,
    VerificationBadge,
    FormattedCitation,
)


class CitationFormatter:
    """
    Formatter for Indian legal citations.

    Provides static methods for formatting case and statute citations
    according to Indian legal practice conventions.
    """

    # Standard abbreviations for Indian courts
    HIGH_COURT_ABBREVIATIONS: dict[str, str] = {
        # Supreme Court
        "supreme_court": "SC",
        "sc": "SC",
        # Delhi High Court
        "delhi": "Del",
        "delhi_high_court": "Del",
        "del": "Del",
        # Bombay High Court
        "bombay": "Bom",
        "bombay_high_court": "Bom",
        "bom": "Bom",
        # Calcutta High Court
        "calcutta": "Cal",
        "calcutta_high_court": "Cal",
        "cal": "Cal",
        # Madras High Court
        "madras": "Mad",
        "madras_high_court": "Mad",
        "mad": "Mad",
        # Karnataka High Court
        "karnataka": "Kar",
        "karnataka_high_court": "Kar",
        "kar": "Kar",
        # Allahabad High Court
        "allahabad": "All",
        "allahabad_high_court": "All",
        "all": "All",
        # Gujarat High Court
        "gujarat": "Guj",
        "gujarat_high_court": "Guj",
        "guj": "Guj",
        # Kerala High Court
        "kerala": "Ker",
        "kerala_high_court": "Ker",
        "ker": "Ker",
        # Punjab and Haryana High Court
        "punjab": "P&H",
        "punjab_haryana": "P&H",
        "punjab_and_haryana_high_court": "P&H",
        # Rajasthan High Court
        "rajasthan": "Raj",
        "rajasthan_high_court": "Raj",
        "raj": "Raj",
        # Andhra Pradesh High Court
        "andhra_pradesh": "AP",
        "andhra_pradesh_high_court": "AP",
        "ap": "AP",
        # Telangana High Court
        "telangana": "Tel",
        "telangana_high_court": "Tel",
        "tel": "Tel",
        # Orissa High Court
        "orissa": "Ori",
        "orissa_high_court": "Ori",
        "odisha": "Ori",
        "odisha_high_court": "Ori",
        "ori": "Ori",
        # Patna High Court
        "patna": "Pat",
        "patna_high_court": "Pat",
        "pat": "Pat",
        # Jharkhand High Court
        "jharkhand": "Jhar",
        "jharkhand_high_court": "Jhar",
        "jhar": "Jhar",
        # Chhattisgarh High Court
        "chhattisgarh": "CG",
        "chhattisgarh_high_court": "CG",
        "cg": "CG",
        # Madhya Pradesh High Court
        "madhya_pradesh": "MP",
        "madhya_pradesh_high_court": "MP",
        "mp": "MP",
        # Uttarakhand High Court
        "uttarakhand": "Utt",
        "uttarakhand_high_court": "Utt",
        "utt": "Utt",
        # Himachal Pradesh High Court
        "himachal_pradesh": "HP",
        "himachal_pradesh_high_court": "HP",
        "hp": "HP",
        # Gauhati High Court
        "gauhati": "Gau",
        "gauhati_high_court": "Gau",
        "gau": "Gau",
        # Jammu and Kashmir High Court
        "jammu_kashmir": "J&K",
        "jammu_and_kashmir_high_court": "J&K",
        "jk": "J&K",
    }

    @staticmethod
    def format_scc(
        party_a: str,
        party_b: str,
        year: int,
        volume: int,
        page: int
    ) -> str:
        """
        Format a citation in SCC (Supreme Court Cases) format.

        Args:
            party_a: First party name
            party_b: Second party name
            year: Year of judgment
            volume: Volume number
            page: Page number

        Returns:
            Formatted citation string: "Party A vs Party B (Year) Volume SCC Page"

        Example:
            >>> CitationFormatter.format_scc("Arnesh Kumar", "State of Bihar", 2014, 8, 273)
            "Arnesh Kumar vs State of Bihar (2014) 8 SCC 273"
        """
        return f"{party_a} vs {party_b} ({year}) {volume} SCC {page}"

    @staticmethod
    def format_air(
        party_a: str,
        party_b: str,
        year: int,
        court: str,
        page: int
    ) -> str:
        """
        Format a citation in AIR (All India Reporter) format.

        Args:
            party_a: First party name
            party_b: Second party name
            year: Year of judgment
            court: Court abbreviation (SC, Del, Bom, etc.)
            page: Page number

        Returns:
            Formatted citation string: "Party A vs Party B, AIR Year Court Page"

        Example:
            >>> CitationFormatter.format_air("K.S. Puttaswamy", "Union of India", 2017, "SC", 4161)
            "K.S. Puttaswamy vs Union of India, AIR 2017 SC 4161"
        """
        # Normalize court abbreviation if needed
        court_abbr = CitationFormatter.HIGH_COURT_ABBREVIATIONS.get(
            court.lower().replace(" ", "_"),
            court  # Use as-is if not found
        )
        return f"{party_a} vs {party_b}, AIR {year} {court_abbr} {page}"

    @staticmethod
    def format_statute(
        section: str,
        act_name: str,
        year: int | None = None
    ) -> str:
        """
        Format a statute citation.

        Args:
            section: Section number (may include letters like "302A")
            act_name: Full name of the act
            year: Year of enactment (optional)

        Returns:
            Formatted citation string: "Section N, Act Name, Year" or "Section N, Act Name"

        Example:
            >>> CitationFormatter.format_statute("438", "Code of Criminal Procedure", 1973)
            "Section 438, Code of Criminal Procedure, 1973"
        """
        if year is not None:
            return f"Section {section}, {act_name}, {year}"
        return f"Section {section}, {act_name}"

    @staticmethod
    def get_badge_html(badge: VerificationBadge) -> str:
        """
        Generate HTML for a verification badge.

        Args:
            badge: VerificationBadge model instance

        Returns:
            HTML string: '<span class="citation-badge {css_class}" title="{tooltip}">{icon}</span>'
        """
        return (
            f'<span class="citation-badge {badge.css_class}" '
            f'title="{badge.tooltip}">{badge.icon}</span>'
        )

    @classmethod
    def format_with_badge(
        cls,
        citation: Union[CaseCitation, StatuteCitation],
        badge: VerificationBadge,
        source_id: str | None = None
    ) -> FormattedCitation:
        """
        Format a citation and attach a verification badge.

        Args:
            citation: CaseCitation or StatuteCitation model instance
            badge: VerificationBadge to attach
            source_id: Optional database ID if verified

        Returns:
            FormattedCitation with formatted text, badge, and HTML output
        """
        if isinstance(citation, CaseCitation):
            # Determine format based on available data
            if citation.reporter == "SCC" and citation.volume and citation.page:
                formatted_text = cls.format_scc(
                    citation.party_a,
                    citation.party_b,
                    citation.year,
                    citation.volume,
                    citation.page
                )
            elif citation.reporter == "AIR" and citation.court and citation.page:
                formatted_text = cls.format_air(
                    citation.party_a,
                    citation.party_b,
                    citation.year,
                    citation.court,
                    citation.page
                )
            else:
                # Fallback: basic format without reporter details
                formatted_text = f"{citation.party_a} vs {citation.party_b} ({citation.year})"
        elif isinstance(citation, StatuteCitation):
            formatted_text = cls.format_statute(
                citation.section,
                citation.act_name,
                citation.year
            )
        else:
            # Should not happen with proper type hints
            formatted_text = citation.raw_text

        return FormattedCitation(
            formatted_text=formatted_text,
            badge=badge,
            source_id=source_id
        )
