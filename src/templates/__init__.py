"""
Templates module for Vakalat AI.

Provides legal document template storage and validation functionality.
"""

from src.templates.schemas import (
    DocumentType,
    CourtLevel,
    TemplateStatus,
    TemplateMetadata,
    FormattingRequirements,
    TemplateField,
    ChangelogEntry,
    ALLOWED_TRANSITIONS,
    LegalTemplate,
)
from src.templates.storage import TemplateRepository

__all__ = [
    # Enums
    "DocumentType",
    "CourtLevel",
    "TemplateStatus",
    # Metadata models
    "TemplateMetadata",
    "FormattingRequirements",
    # Field model
    "TemplateField",
    # Version tracking
    "ChangelogEntry",
    "ALLOWED_TRANSITIONS",
    # Main template model
    "LegalTemplate",
    # Repository
    "TemplateRepository",
]
