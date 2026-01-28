"""
Template storage layer for Vakalat AI.

Provides repository pattern for template CRUD operations:
- Load templates from JSON files
- List templates with optional filtering
- Save templates to JSON files
- Check template existence

Templates are stored as JSON files in the data/ directory.
"""

import json
import logging
from pathlib import Path
from typing import List, Optional

from pydantic import ValidationError

from src.templates.schemas import (
    DocumentType,
    CourtLevel,
    LegalTemplate,
)


# Configure logging for the storage module
logger = logging.getLogger(__name__)


class TemplateRepository:
    """
    Repository for legal document templates.

    Provides CRUD operations for templates stored as JSON files.
    Templates are loaded and validated as LegalTemplate Pydantic models.

    Usage:
        repo = TemplateRepository()  # Uses default data directory
        template = repo.get_template(DocumentType.BAIL_APPLICATION, CourtLevel.SUPREME_COURT)
        templates = repo.list_templates(DocumentType.LEGAL_NOTICE)
    """

    def __init__(self, data_dir: Optional[Path] = None):
        """
        Initialize the template repository.

        Args:
            data_dir: Path to directory containing template JSON files.
                     Defaults to src/templates/data/ relative to this file.
        """
        if data_dir is None:
            # Default to data/ directory relative to this file
            self.data_dir = Path(__file__).parent / "data"
        else:
            self.data_dir = Path(data_dir)

        # Create data directory if it doesn't exist
        self.data_dir.mkdir(parents=True, exist_ok=True)

        logger.debug(f"TemplateRepository initialized with data_dir: {self.data_dir}")

    def _get_filename(self, doc_type: DocumentType, court_level: CourtLevel) -> str:
        """
        Generate filename for a template based on its type and court level.

        Args:
            doc_type: Document type enum value
            court_level: Court level enum value

        Returns:
            Filename string (e.g., "bail_application_supreme_court.json")
        """
        return f"{doc_type.value}_{court_level.value}.json"

    def _get_filepath(self, doc_type: DocumentType, court_level: CourtLevel) -> Path:
        """
        Get full path to a template file.

        Args:
            doc_type: Document type enum value
            court_level: Court level enum value

        Returns:
            Full Path to the template file
        """
        return self.data_dir / self._get_filename(doc_type, court_level)

    def get_template(
        self, doc_type: DocumentType, court_level: CourtLevel
    ) -> Optional[LegalTemplate]:
        """
        Load a template by document type and court level.

        Args:
            doc_type: Document type to retrieve
            court_level: Court level to retrieve

        Returns:
            LegalTemplate if found and valid, None otherwise
        """
        filepath = self._get_filepath(doc_type, court_level)

        if not filepath.exists():
            logger.debug(f"Template not found: {filepath}")
            return None

        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
            template = LegalTemplate(**data)
            logger.debug(f"Loaded template: {filepath}")
            return template
        except json.JSONDecodeError as e:
            logger.warning(f"Invalid JSON in template file {filepath}: {e}")
            return None
        except ValidationError as e:
            logger.warning(f"Template validation failed for {filepath}: {e}")
            return None

    def list_templates(
        self, doc_type: Optional[DocumentType] = None
    ) -> List[LegalTemplate]:
        """
        List all templates, optionally filtered by document type.

        Args:
            doc_type: Optional document type filter. If None, returns all templates.

        Returns:
            List of LegalTemplate objects, sorted by (doc_type, court_level)
        """
        templates: List[LegalTemplate] = []

        # Glob all JSON files in data directory
        for filepath in self.data_dir.glob("*.json"):
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                template = LegalTemplate(**data)

                # Apply filter if specified
                if doc_type is None or template.metadata.doc_type == doc_type:
                    templates.append(template)
            except json.JSONDecodeError as e:
                logger.warning(f"Skipping invalid JSON file {filepath}: {e}")
            except ValidationError as e:
                logger.warning(f"Skipping invalid template {filepath}: {e}")

        # Sort by (doc_type, court_level) for consistent ordering
        templates.sort(
            key=lambda t: (t.metadata.doc_type.value, t.metadata.court_level.value)
        )

        logger.debug(f"Listed {len(templates)} templates (filter: {doc_type})")
        return templates

    def save_template(self, template: LegalTemplate) -> Path:
        """
        Save a template to the data directory.

        Args:
            template: LegalTemplate to save

        Returns:
            Path to the saved file
        """
        filepath = self._get_filepath(
            template.metadata.doc_type, template.metadata.court_level
        )

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(template.model_dump(), f, indent=2, ensure_ascii=False)

        logger.info(f"Saved template: {filepath}")
        return filepath

    def template_exists(
        self, doc_type: DocumentType, court_level: CourtLevel
    ) -> bool:
        """
        Check if a template exists without loading it.

        Args:
            doc_type: Document type to check
            court_level: Court level to check

        Returns:
            True if template file exists, False otherwise
        """
        filepath = self._get_filepath(doc_type, court_level)
        return filepath.exists()
