"""
Template preview generation for Vakalat AI.

Provides lightweight preview models that show template field metadata
without including the full template_content. This enables quick template
selection by showing users what inputs are required before starting
document generation.
"""

import logging
from typing import List, Optional

from pydantic import BaseModel, Field

from src.templates.schemas import (
    DocumentType,
    CourtLevel,
    TemplateStatus,
    TemplateField,
    LegalTemplate,
)
from src.templates.storage import TemplateRepository


# Configure logging for the preview module
logger = logging.getLogger(__name__)


class FieldPreview(BaseModel):
    """
    Preview of a template field showing metadata for user display.

    Contains all information needed for the user to understand
    what input is expected for a given field.
    """
    field_name: str = Field(
        description="Unique field identifier (snake_case)"
    )
    label: str = Field(
        description="Human-readable label for the field"
    )
    field_type: str = Field(
        description="Field data type: text|date|party|case_number|relief|court"
    )
    required: bool = Field(
        description="Whether this field must be provided"
    )
    description: str = Field(
        description="Explanation of what this field contains"
    )
    example: Optional[str] = Field(
        default=None,
        description="Example value for this field"
    )
    validation_regex: Optional[str] = Field(
        default=None,
        description="Optional regex pattern for validation"
    )


class TemplatePreview(BaseModel):
    """
    Lightweight preview of a template showing metadata and field information.

    Does NOT include template_content - only metadata needed for template
    selection and understanding what inputs are required.
    """
    doc_type: DocumentType = Field(
        description="Type of legal document"
    )
    court_level: CourtLevel = Field(
        description="Court level this template is designed for"
    )
    name: str = Field(
        description="Human-readable template name"
    )
    description: str = Field(
        description="Brief description of the template's purpose"
    )
    version: str = Field(
        description="Semantic version e.g. 1.0.0"
    )
    status: TemplateStatus = Field(
        description="Template lifecycle status (active/deprecated/archived)"
    )

    # Field counts for quick overview
    required_field_count: int = Field(
        description="Number of required fields"
    )
    optional_field_count: int = Field(
        description="Number of optional fields"
    )

    # Full field lists for detailed view
    required_fields: List[FieldPreview] = Field(
        description="List of required field previews"
    )
    optional_fields: List[FieldPreview] = Field(
        description="List of optional field previews"
    )

    # Formatting info (just font for quick reference)
    font: str = Field(
        description="Font family for document text"
    )
    font_size: int = Field(
        description="Font size in points"
    )


def _field_to_preview(field: TemplateField) -> FieldPreview:
    """
    Convert a TemplateField to a FieldPreview.

    Args:
        field: TemplateField from a LegalTemplate

    Returns:
        FieldPreview with the same field metadata
    """
    return FieldPreview(
        field_name=field.field_name,
        label=field.label,
        field_type=field.field_type,
        required=field.required,
        description=field.description,
        example=field.example,
        validation_regex=field.validation_regex,
    )


def generate_preview(template: LegalTemplate) -> TemplatePreview:
    """
    Generate a lightweight preview from a LegalTemplate.

    Extracts field metadata and template identification without
    including the full template_content.

    Args:
        template: Full LegalTemplate object

    Returns:
        TemplatePreview with field metadata and counts
    """
    # Convert required fields to previews
    required_previews = [
        _field_to_preview(field) for field in template.required_fields
    ]

    # Convert optional fields to previews
    optional_previews = [
        _field_to_preview(field) for field in template.optional_fields
    ]

    return TemplatePreview(
        doc_type=template.metadata.doc_type,
        court_level=template.metadata.court_level,
        name=template.metadata.name,
        description=template.metadata.description,
        version=template.metadata.version,
        status=template.status,
        required_field_count=len(required_previews),
        optional_field_count=len(optional_previews),
        required_fields=required_previews,
        optional_fields=optional_previews,
        font=template.formatting.font,
        font_size=template.formatting.font_size,
    )


def get_template_preview(
    doc_type: DocumentType,
    court_level: CourtLevel,
    repository: Optional[TemplateRepository] = None,
) -> Optional[TemplatePreview]:
    """
    Get a preview for a specific template by type and court level.

    Args:
        doc_type: Document type to retrieve
        court_level: Court level to retrieve
        repository: Optional TemplateRepository instance.
                   If None, creates a new repository with default settings.

    Returns:
        TemplatePreview if template found, None otherwise
    """
    if repository is None:
        repository = TemplateRepository()

    template = repository.get_template(doc_type, court_level)
    if template is None:
        logger.debug(f"Template not found for preview: {doc_type.value}/{court_level.value}")
        return None

    preview = generate_preview(template)
    logger.debug(f"Generated preview for: {doc_type.value}/{court_level.value}")
    return preview


def list_template_previews(
    doc_type: Optional[DocumentType] = None,
    status: Optional[TemplateStatus] = None,
    repository: Optional[TemplateRepository] = None,
) -> List[TemplatePreview]:
    """
    List previews for all templates with optional filtering.

    Args:
        doc_type: Optional document type filter. If None, includes all types.
        status: Optional status filter. If None, includes all statuses.
        repository: Optional TemplateRepository instance.
                   If None, creates a new repository with default settings.

    Returns:
        List of TemplatePreview objects, sorted by (doc_type, court_level)
    """
    if repository is None:
        repository = TemplateRepository()

    # Get all templates (optionally filtered by doc_type)
    templates = repository.list_templates(doc_type)

    # Filter by status if specified
    if status is not None:
        templates = [t for t in templates if t.status == status]

    # Generate previews
    previews = [generate_preview(t) for t in templates]

    logger.debug(
        f"Listed {len(previews)} template previews "
        f"(doc_type filter: {doc_type}, status filter: {status})"
    )

    return previews
