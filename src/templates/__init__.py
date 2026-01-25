"""
Templates module for Vakalat AI.

Provides legal document template storage and validation functionality.
"""

from src.templates.schemas import (
    DocumentType,
    CourtLevel,
    TemplateMetadata,
    FormattingRequirements,
    TemplateField,
    LegalTemplate,
)

__all__ = [
    # Enums
    "DocumentType",
    "CourtLevel",
    # Metadata models
    "TemplateMetadata",
    "FormattingRequirements",
    # Field model
    "TemplateField",
    # Main template model
    "LegalTemplate",
]
