"""
Precedent retriever with jurisdiction-aware ranking.

Provides intelligent case law suggestions by:
- Retrieving relevant precedents from Pinecone vector database
- Ranking by court authority (Supreme Court > Same HC > Other HC > Tribunal)
- Boosting recent cases (last 5 years)
- Combining semantic similarity with jurisdiction and recency weights

Key components:
- JurisdictionWeight: Authority weights for different court levels
- RetrievedPrecedent: Pydantic model for ranked case results
- PrecedentRetriever: Main retriever class with ranking logic
"""

import re
from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field
from langchain_pinecone import PineconeVectorStore


class JurisdictionWeight:
    """
    Authority weights for different court levels.

    Supreme Court judgments are binding on all courts.
    High Court judgments are binding within their jurisdiction only.
    Tribunal decisions have limited precedential value.
    """
    SUPREME_COURT = 1.0      # Binding on all courts
    SAME_HIGH_COURT = 0.9    # Binding in jurisdiction
    OTHER_HIGH_COURT = 0.6   # Persuasive only
    TRIBUNAL = 0.4           # Limited authority
    UNKNOWN = 0.5            # Default for unidentified courts


class RetrievedPrecedent(BaseModel):
    """
    A case law precedent retrieved and ranked for relevance.

    Combines semantic similarity with jurisdiction authority and
    temporal recency for intelligent case suggestions.
    """
    case_name: str = Field(
        description="Full case name (e.g., 'Arnesh Kumar vs State of Bihar')"
    )
    year: int = Field(
        description="Year of judgment"
    )
    court: str = Field(
        description="Court identifier (e.g., 'supreme_court', 'delhi')"
    )
    semantic_score: float = Field(
        ge=0.0,
        le=1.0,
        description="Raw similarity score from Pinecone (0-1)"
    )
    jurisdiction_weight: float = Field(
        ge=0.0,
        le=1.0,
        description="Authority weight based on court level"
    )
    recency_weight: float = Field(
        ge=0.0,
        le=1.0,
        description="Temporal relevance weight (recent = higher)"
    )
    final_score: float = Field(
        ge=0.0,
        le=1.0,
        description="Computed ranking score (50% semantic + 30% jurisdiction + 20% recency)"
    )
    source_id: str = Field(
        description="Pinecone document ID for retrieval"
    )
    snippet: str = Field(
        description="Relevant excerpt from the judgment (first 500 chars)"
    )
    metadata: dict = Field(
        default_factory=dict,
        description="Raw metadata from Pinecone document"
    )


def _calculate_recency(year: int, current_year: Optional[int] = None) -> float:
    """
    Calculate recency weight based on judgment age.

    Recent cases (last 5 years) get full weight.
    Older cases get progressively lower weights.

    Args:
        year: Year of judgment
        current_year: Reference year (defaults to current year)

    Returns:
        Recency weight from 0.4 to 1.0
    """
    if current_year is None:
        current_year = datetime.now().year

    age = current_year - year

    if age <= 5:
        return 1.0
    if age <= 10:
        return 0.8
    if age <= 20:
        return 0.6
    return 0.4


# Court name normalization map
# Maps various court name formats to canonical identifiers
COURT_NORMALIZATIONS = {
    # Supreme Court
    "supreme court": "supreme_court",
    "supreme court of india": "supreme_court",
    "sci": "supreme_court",
    "s.c.": "supreme_court",
    "sc": "supreme_court",
    "hon'ble supreme court": "supreme_court",

    # Delhi High Court
    "delhi high court": "delhi",
    "high court of delhi": "delhi",
    "delhi hc": "delhi",
    "dhc": "delhi",

    # Bombay High Court
    "bombay high court": "bombay",
    "high court of bombay": "bombay",
    "bombay hc": "bombay",
    "bhc": "bombay",

    # Madras High Court
    "madras high court": "madras",
    "high court of madras": "madras",
    "madras hc": "madras",
    "mhc": "madras",

    # Calcutta High Court
    "calcutta high court": "calcutta",
    "high court of calcutta": "calcutta",
    "calcutta hc": "calcutta",
    "chc": "calcutta",

    # Karnataka High Court
    "karnataka high court": "karnataka",
    "high court of karnataka": "karnataka",
    "karnataka hc": "karnataka",

    # Gujarat High Court
    "gujarat high court": "gujarat",
    "high court of gujarat": "gujarat",
    "gujarat hc": "gujarat",

    # Allahabad High Court
    "allahabad high court": "allahabad",
    "high court of allahabad": "allahabad",
    "allahabad hc": "allahabad",

    # Punjab and Haryana High Court
    "punjab and haryana high court": "punjab_haryana",
    "punjab & haryana high court": "punjab_haryana",
    "p&h high court": "punjab_haryana",

    # Kerala High Court
    "kerala high court": "kerala",
    "high court of kerala": "kerala",
    "kerala hc": "kerala",

    # Telangana High Court
    "telangana high court": "telangana",
    "high court of telangana": "telangana",

    # Andhra Pradesh High Court
    "andhra pradesh high court": "andhra_pradesh",
    "high court of andhra pradesh": "andhra_pradesh",
    "ap high court": "andhra_pradesh",

    # Rajasthan High Court
    "rajasthan high court": "rajasthan",
    "high court of rajasthan": "rajasthan",

    # Madhya Pradesh High Court
    "madhya pradesh high court": "madhya_pradesh",
    "high court of madhya pradesh": "madhya_pradesh",
    "mp high court": "madhya_pradesh",

    # Orissa High Court
    "orissa high court": "orissa",
    "high court of orissa": "orissa",
    "odisha high court": "orissa",

    # Patna High Court
    "patna high court": "patna",
    "high court of patna": "patna",

    # Jharkhand High Court
    "jharkhand high court": "jharkhand",
    "high court of jharkhand": "jharkhand",

    # Chhattisgarh High Court
    "chhattisgarh high court": "chhattisgarh",
    "high court of chhattisgarh": "chhattisgarh",

    # Uttarakhand High Court
    "uttarakhand high court": "uttarakhand",
    "high court of uttarakhand": "uttarakhand",

    # Himachal Pradesh High Court
    "himachal pradesh high court": "himachal_pradesh",
    "high court of himachal pradesh": "himachal_pradesh",
    "hp high court": "himachal_pradesh",

    # Jammu & Kashmir High Court
    "jammu and kashmir high court": "jammu_kashmir",
    "j&k high court": "jammu_kashmir",

    # Gauhati High Court
    "gauhati high court": "gauhati",
    "guwahati high court": "gauhati",
    "high court of gauhati": "gauhati",

    # Tripura High Court
    "tripura high court": "tripura",
    "high court of tripura": "tripura",

    # Meghalaya High Court
    "meghalaya high court": "meghalaya",
    "high court of meghalaya": "meghalaya",

    # Manipur High Court
    "manipur high court": "manipur",
    "high court of manipur": "manipur",

    # Sikkim High Court
    "sikkim high court": "sikkim",
    "high court of sikkim": "sikkim",

    # Tribunals
    "nclat": "tribunal",
    "nclt": "tribunal",
    "itat": "tribunal",
    "cestat": "tribunal",
    "cat": "tribunal",
    "ncdrc": "tribunal",
    "armed forces tribunal": "tribunal",
    "aft": "tribunal",
    "drt": "tribunal",
    "debt recovery tribunal": "tribunal",
}

# List of known High Court identifiers for jurisdiction matching
HIGH_COURTS = {
    "delhi", "bombay", "madras", "calcutta", "karnataka", "gujarat",
    "allahabad", "punjab_haryana", "kerala", "telangana", "andhra_pradesh",
    "rajasthan", "madhya_pradesh", "orissa", "patna", "jharkhand",
    "chhattisgarh", "uttarakhand", "himachal_pradesh", "jammu_kashmir",
    "gauhati", "tripura", "meghalaya", "manipur", "sikkim"
}


class PrecedentRetriever:
    """
    Retrieves and ranks case law precedents from Pinecone.

    Implements jurisdiction-aware ranking that:
    1. Fetches semantically similar cases
    2. Weights by court authority (Supreme Court > Same HC > Other HC)
    3. Boosts recent cases (last 5 years)
    4. Returns ranked RetrievedPrecedent objects

    Example:
        retriever = PrecedentRetriever(vector_store)
        precedents = retriever.retrieve_precedents(
            legal_issue="anticipatory bail in economic offences",
            filing_court="delhi",
            top_k=5
        )
        for p in precedents:
            print(f"{p.case_name} ({p.year}) - Score: {p.final_score:.2f}")
    """

    def __init__(self, vector_store: PineconeVectorStore):
        """
        Initialize the PrecedentRetriever.

        Args:
            vector_store: PineconeVectorStore instance connected to legal database
        """
        self.vector_store = vector_store

    def _normalize_court(self, court_name: str) -> str:
        """
        Normalize court name to canonical identifier.

        Args:
            court_name: Raw court name from metadata

        Returns:
            Normalized court identifier (e.g., 'supreme_court', 'delhi')
        """
        if not court_name:
            return "unknown"

        normalized = court_name.lower().strip()

        # Direct lookup
        if normalized in COURT_NORMALIZATIONS:
            return COURT_NORMALIZATIONS[normalized]

        # Check if any key is contained in the court name
        for key, value in COURT_NORMALIZATIONS.items():
            if key in normalized:
                return value

        # Check for "high court" pattern
        if "high court" in normalized:
            # Try to extract state name
            for hc in HIGH_COURTS:
                if hc.replace("_", " ") in normalized:
                    return hc
            return "other_high_court"

        return "unknown"

    def _extract_court(self, metadata: dict) -> str:
        """
        Extract court identifier from document metadata.

        Tries multiple field names defensively.

        Args:
            metadata: Pinecone document metadata

        Returns:
            Normalized court identifier
        """
        # Try various metadata field names
        court_fields = ["court", "Court_Name", "court_name", "Court", "forum"]

        for field in court_fields:
            if field in metadata and metadata[field]:
                return self._normalize_court(str(metadata[field]))

        # Try to extract from source filename
        source = metadata.get("source", "")
        if source:
            source_lower = source.lower()
            if "supreme" in source_lower or "sc_" in source_lower:
                return "supreme_court"
            for hc in HIGH_COURTS:
                if hc.replace("_", " ") in source_lower or f"{hc}_" in source_lower:
                    return hc

        return "unknown"

    def _extract_year(self, metadata: dict, content: str) -> int:
        """
        Extract judgment year from metadata or content.

        Args:
            metadata: Pinecone document metadata
            content: Document text content

        Returns:
            Year as integer (defaults to 2000 if not found)
        """
        # Try metadata fields
        year_fields = ["year", "Year", "judgment_year", "date", "judgment_date"]

        for field in year_fields:
            if field in metadata and metadata[field]:
                value = metadata[field]
                # Handle string year
                if isinstance(value, str):
                    # Extract 4-digit year from string
                    match = re.search(r'\b(19|20)\d{2}\b', value)
                    if match:
                        return int(match.group())
                elif isinstance(value, int):
                    if 1900 <= value <= 2100:
                        return value

        # Try to extract from content - look for year patterns
        # Common formats: (2014), [2014], - 2014 -, AIR 2014 SC
        year_patterns = [
            r'\((\d{4})\)',  # (2014)
            r'\[(\d{4})\]',  # [2014]
            r'AIR\s+(\d{4})',  # AIR 2014
            r'SCC\s+\(?(\d{4})',  # SCC (2014) or SCC 2014
            r'dated\s+\d{1,2}[./\-]\d{1,2}[./\-](\d{4})',  # dated 01/01/2014
        ]

        for pattern in year_patterns:
            match = re.search(pattern, content)
            if match:
                year = int(match.group(1))
                if 1900 <= year <= 2100:
                    return year

        # Default to 2000 if nothing found
        return 2000

    def _extract_case_name(self, metadata: dict, content: str) -> str:
        """
        Extract case name from metadata or content.

        Args:
            metadata: Pinecone document metadata
            content: Document text content

        Returns:
            Case name string
        """
        # Try metadata fields
        name_fields = ["case_name", "title", "Case_Name", "Title", "name"]

        for field in name_fields:
            if field in metadata and metadata[field]:
                return str(metadata[field])

        # Try to extract from source filename
        source = metadata.get("source", "")
        if source:
            # Remove extension and path
            name = source.split("/")[-1].split("\\")[-1]
            name = re.sub(r'\.(pdf|txt|docx?)$', '', name, flags=re.IGNORECASE)
            if name:
                return name

        # Extract from content - look for "vs" or "v." pattern
        vs_pattern = r'([A-Z][A-Za-z\s.]+)\s+(?:vs?\.?|versus)\s+([A-Z][A-Za-z\s.]+)'
        match = re.search(vs_pattern, content[:500])
        if match:
            return f"{match.group(1).strip()} vs {match.group(2).strip()}"

        return "Unknown Case"

    def _calculate_jurisdiction_weight(self, court: str, filing_court: str) -> float:
        """
        Calculate jurisdiction weight based on court authority.

        Args:
            court: Court that decided the precedent
            filing_court: Court where current case is being filed

        Returns:
            Jurisdiction weight (0.4 to 1.0)
        """
        # Supreme Court is binding on all
        if court == "supreme_court":
            return JurisdictionWeight.SUPREME_COURT

        # Tribunal has limited authority
        if court == "tribunal":
            return JurisdictionWeight.TRIBUNAL

        # Unknown court gets default weight
        if court == "unknown":
            return JurisdictionWeight.UNKNOWN

        # Same High Court is strongly relevant
        normalized_filing = self._normalize_court(filing_court)
        if court == normalized_filing:
            return JurisdictionWeight.SAME_HIGH_COURT

        # Other High Court is persuasive only
        if court in HIGH_COURTS:
            return JurisdictionWeight.OTHER_HIGH_COURT

        return JurisdictionWeight.UNKNOWN

    def _rank_precedents(
        self,
        results: list,
        filing_court: str,
        current_year: Optional[int] = None
    ) -> list[RetrievedPrecedent]:
        """
        Rank retrieved results by combined score.

        Applies 50/30/20 weighting:
        - 50% semantic similarity
        - 30% jurisdiction authority
        - 20% temporal recency

        Args:
            results: Pinecone search results (Document objects)
            filing_court: Court where case is being filed
            current_year: Reference year for recency calculation

        Returns:
            List of RetrievedPrecedent sorted by final_score descending
        """
        if current_year is None:
            current_year = datetime.now().year

        ranked = []

        for i, doc in enumerate(results):
            metadata = doc.metadata
            content = doc.page_content

            # Extract fields
            court = self._extract_court(metadata)
            year = self._extract_year(metadata, content)
            case_name = self._extract_case_name(metadata, content)

            # Calculate component scores
            # Semantic score: normalize rank to 0-1 (first result = 1.0)
            # Pinecone returns by similarity, so we use inverse rank
            semantic_score = 1.0 - (i * 0.05)  # Slight decrease per rank
            semantic_score = max(0.5, min(1.0, semantic_score))  # Clamp to [0.5, 1.0]

            jurisdiction_weight = self._calculate_jurisdiction_weight(court, filing_court)
            recency_weight = _calculate_recency(year, current_year)

            # Combined score: 50% semantic, 30% jurisdiction, 20% recency
            final_score = (
                0.5 * semantic_score +
                0.3 * jurisdiction_weight +
                0.2 * recency_weight
            )

            # Create snippet (first 500 chars)
            snippet = content[:500] if len(content) > 500 else content

            # Get source ID
            source_id = metadata.get(
                "source_id",
                metadata.get("id", metadata.get("source", f"doc_{i}"))
            )

            precedent = RetrievedPrecedent(
                case_name=case_name,
                year=year,
                court=court,
                semantic_score=semantic_score,
                jurisdiction_weight=jurisdiction_weight,
                recency_weight=recency_weight,
                final_score=final_score,
                source_id=str(source_id),
                snippet=snippet,
                metadata=metadata
            )
            ranked.append(precedent)

        # Sort by final score descending
        ranked.sort(key=lambda p: p.final_score, reverse=True)

        return ranked

    def retrieve_precedents(
        self,
        legal_issue: str,
        filing_court: str,
        top_k: int = 5
    ) -> list[RetrievedPrecedent]:
        """
        Retrieve and rank relevant case law precedents.

        Args:
            legal_issue: Legal issue or question to find precedents for
            filing_court: Court where case is being filed (for jurisdiction weighting)
            top_k: Number of precedents to return (default 5)

        Returns:
            List of RetrievedPrecedent sorted by combined relevance score
        """
        # Over-fetch to allow for re-ranking
        fetch_k = top_k * 2

        try:
            # Try with source_type filter first
            results = self.vector_store.similarity_search(
                query=legal_issue,
                k=fetch_k,
                filter={"source_type": {"$eq": "case_law"}}
            )
        except Exception:
            # Fall back to unfiltered search if metadata filter fails
            try:
                results = self.vector_store.similarity_search(
                    query=legal_issue,
                    k=fetch_k
                )
                # Post-filter to case law only
                results = [
                    doc for doc in results
                    if doc.metadata.get("source_type") == "case_law"
                    or "judgment" in doc.metadata.get("source", "").lower()
                    or "vs" in doc.page_content[:200].lower()
                    or "v." in doc.page_content[:200].lower()
                ]
            except Exception:
                # If all fails, return empty list
                return []

        if not results:
            return []

        # Rank the results
        ranked = self._rank_precedents(results, filing_court)

        # Return top_k
        return ranked[:top_k]
