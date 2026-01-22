"""
Legal Code Mapper for Indian criminal law code conversions.

Maps sections between old and new criminal laws:
- IPC (1860) -> BNS (2023)
- CrPC (1973) -> BNSS (2023)
- Indian Evidence Act (1872) -> BSA (2023)

Effective date: July 1, 2024
"""

import json
from functools import lru_cache
from pathlib import Path
from typing import Optional

from src.verification.models import CodeMappingResult, CodeMappingStatus


class LegalCodeMapper:
    """
    Maps sections between old Indian criminal codes and their 2023 replacements.

    Usage:
        mapper = LegalCodeMapper()
        result = mapper.map_section("IPC", "302")
        # Returns CodeMappingResult with BNS section 103
    """

    # Aliases for code names to handle variations in user input
    CODE_ALIASES = {
        # IPC aliases
        "IPC": "IPC",
        "INDIAN PENAL CODE": "IPC",
        "INDIAN PENAL CODE, 1860": "IPC",
        "INDIAN PENAL CODE 1860": "IPC",
        "I.P.C.": "IPC",
        "I.P.C": "IPC",

        # CrPC aliases
        "CRPC": "CrPC",
        "CR.P.C.": "CrPC",
        "CR.P.C": "CrPC",
        "CODE OF CRIMINAL PROCEDURE": "CrPC",
        "CODE OF CRIMINAL PROCEDURE, 1973": "CrPC",
        "CODE OF CRIMINAL PROCEDURE 1973": "CrPC",
        "CRIMINAL PROCEDURE CODE": "CrPC",

        # Evidence Act aliases
        "EVIDENCE ACT": "IEA",
        "IEA": "IEA",
        "INDIAN EVIDENCE ACT": "IEA",
        "INDIAN EVIDENCE ACT, 1872": "IEA",
        "INDIAN EVIDENCE ACT 1872": "IEA",
        "I.E.A.": "IEA",
        "I.E.A": "IEA",
    }

    # Map canonical code names to JSON file names
    CODE_TO_FILE = {
        "IPC": "ipc_to_bns.json",
        "CrPC": "crpc_to_bnss.json",
        "IEA": "iea_to_bsa.json",
    }

    def __init__(self, mappings_dir: Optional[Path] = None):
        """
        Initialize the mapper with mapping data.

        Args:
            mappings_dir: Directory containing mapping JSON files.
                         Defaults to src/data/mappings/ relative to this file.
        """
        if mappings_dir is None:
            # Default to src/data/mappings/ relative to this file's location
            mappings_dir = Path(__file__).parent.parent / "data" / "mappings"

        self.mappings_dir = Path(mappings_dir)
        self._mappings: dict[str, dict] = {}

        # Load all mappings at initialization
        for code, filename in self.CODE_TO_FILE.items():
            filepath = self.mappings_dir / filename
            self._mappings[code] = self._load_mapping(filepath)

    def _load_mapping(self, path: Path) -> dict:
        """
        Load a single mapping JSON file.

        Args:
            path: Path to the JSON file

        Returns:
            Dictionary of section mappings
        """
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                # Remove metadata entry if present
                return {k: v for k, v in data.items() if not k.startswith("_")}
        except FileNotFoundError:
            return {}
        except json.JSONDecodeError:
            return {}

    def _normalize_code(self, code: str) -> Optional[str]:
        """
        Normalize a code name to its canonical form.

        Args:
            code: User-provided code name (e.g., "Indian Penal Code", "IPC")

        Returns:
            Canonical code name (IPC, CrPC, or IEA) or None if unknown
        """
        normalized = code.upper().strip()
        return self.CODE_ALIASES.get(normalized)

    def _normalize_section(self, section: str) -> str:
        """
        Normalize a section number for lookup.

        Args:
            section: Section number (may include sub-clauses)

        Returns:
            Normalized section string
        """
        # Strip whitespace and standardize format
        return section.strip().upper()

    @lru_cache(maxsize=1000)
    def map_section(self, old_code: str, section: str) -> CodeMappingResult:
        """
        Map a section from an old code to its new equivalent.

        Args:
            old_code: Name of the old code (e.g., "IPC", "CrPC", "Evidence Act")
            section: Section number to map (e.g., "302", "438", "65B")

        Returns:
            CodeMappingResult with mapping details
        """
        canonical_code = self._normalize_code(old_code)

        if canonical_code is None:
            return CodeMappingResult(
                status=CodeMappingStatus.UNKNOWN_CODE,
                old_code=old_code,
                old_section=section,
                confidence=0.0,
                notes=f"Unknown code: '{old_code}'. Supported codes: IPC, CrPC, Evidence Act"
            )

        mapping_data = self._mappings.get(canonical_code, {})
        normalized_section = self._normalize_section(section)

        # Try exact match first
        entry = mapping_data.get(normalized_section)

        # If not found, try lowercase version (for sections like "65B" vs "65b")
        if entry is None:
            entry = mapping_data.get(section.strip())

        if entry is None:
            return CodeMappingResult(
                status=CodeMappingStatus.NO_MAPPING,
                old_code=old_code,
                old_section=section,
                confidence=0.0,
                notes=f"No mapping found for {canonical_code} Section {section}"
            )

        # Handle repealed sections
        if entry.get("status") == "repealed":
            return CodeMappingResult(
                status=CodeMappingStatus.MAPPED,
                old_code=old_code,
                old_section=section,
                new_code=entry.get("new_code"),
                new_section=None,
                confidence=entry.get("confidence", 1.0),
                notes=entry.get("notes", "Section repealed")
            )

        return CodeMappingResult(
            status=CodeMappingStatus.MAPPED,
            old_code=old_code,
            old_section=section,
            new_code=entry.get("new_code"),
            new_section=entry.get("new_section"),
            confidence=entry.get("confidence", 1.0),
            notes=entry.get("notes")
        )

    def get_new_equivalent(self, old_code: str, section: str) -> Optional[str]:
        """
        Convenience method to get just the new section number.

        Args:
            old_code: Name of the old code
            section: Section number to map

        Returns:
            New section number as string, or None if no mapping exists
        """
        result = self.map_section(old_code, section)
        if result.status == CodeMappingStatus.MAPPED and result.new_section:
            return result.new_section
        return None

    def is_section_valid(self, code: str, section: str) -> bool:
        """
        Check if a section exists in the mapping database.

        Note: This only checks if the section is in our mapping database.
        A section might exist in the actual law but not be mapped yet.

        Args:
            code: Code name to check
            section: Section number to verify

        Returns:
            True if section is found in mappings
        """
        canonical_code = self._normalize_code(code)
        if canonical_code is None:
            return False

        mapping_data = self._mappings.get(canonical_code, {})
        normalized_section = self._normalize_section(section)

        return normalized_section in mapping_data or section.strip() in mapping_data

    def get_all_sections(self, code: str) -> list[str]:
        """
        Get all mapped sections for a given code.

        Args:
            code: Code name (e.g., "IPC", "CrPC", "Evidence Act")

        Returns:
            List of section numbers that have mappings
        """
        canonical_code = self._normalize_code(code)
        if canonical_code is None:
            return []

        return list(self._mappings.get(canonical_code, {}).keys())
