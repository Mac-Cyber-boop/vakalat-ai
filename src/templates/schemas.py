"""
Pydantic models for legal document template validation.

Provides data structures for:
- Document type enumeration (bail_application, legal_notice, affidavit, petition)
- Court level enumeration (supreme_court, high_court, district_court)
- Template metadata and formatting requirements
- Template field definitions with validation
- Complete legal template schema

Formatting defaults are based on Supreme Court Rules 2013.
"""

from enum import Enum
from typing import List, Optional, Set, Dict
from datetime import datetime
from pydantic import BaseModel, Field, field_validator
import re
import semver


class DocumentType(str, Enum):
    """Legal document types supported by the template system."""
    BAIL_APPLICATION = "bail_application"
    LEGAL_NOTICE = "legal_notice"
    AFFIDAVIT = "affidavit"
    PETITION = "petition"


class CourtLevel(str, Enum):
    """Indian court hierarchy levels."""
    SUPREME_COURT = "supreme_court"
    HIGH_COURT = "high_court"
    DISTRICT_COURT = "district_court"


class TemplateStatus(str, Enum):
    """Template lifecycle status."""
    ACTIVE = "active"
    DEPRECATED = "deprecated"
    ARCHIVED = "archived"


# Valid lifecycle transitions
# active -> deprecated -> archived (cannot go backwards)
# active -> archived directly allowed
ALLOWED_TRANSITIONS: Dict[TemplateStatus, Set[TemplateStatus]] = {
    TemplateStatus.ACTIVE: {TemplateStatus.DEPRECATED, TemplateStatus.ARCHIVED},
    TemplateStatus.DEPRECATED: {TemplateStatus.ARCHIVED},
    TemplateStatus.ARCHIVED: set(),  # Terminal state
}


class TemplateMetadata(BaseModel):
    """Metadata for a legal document template."""
    doc_type: DocumentType = Field(
        description="Type of legal document"
    )
    court_level: CourtLevel = Field(
        description="Court level this template is designed for"
    )
    version: str = Field(
        description="Semantic version e.g. 1.0.0"
    )
    name: str = Field(
        description="Human-readable template name"
    )
    description: str = Field(
        description="Brief description of the template's purpose"
    )


class FormattingRequirements(BaseModel):
    """
    Court-mandated formatting standards for legal documents.

    Defaults are based on Supreme Court Rules 2013:
    - A4 paper (29.7cm X 21cm)
    - Times New Roman font, 14pt
    - 1.5 line spacing
    - 4cm left/right margins, 2cm top/bottom
    - 75 GSM paper quality
    - Double-sided printing
    """
    paper_size: str = Field(
        default="A4",
        description="29.7cm X 21cm"
    )
    font: str = Field(
        default="Times New Roman",
        description="Font family for document text"
    )
    font_size: int = Field(
        default=14,
        ge=12,
        le=16,
        description="Font size in points (12-16 allowed)"
    )
    line_spacing: float = Field(
        default=1.5,
        description="Line spacing multiplier"
    )
    margin_left_right: str = Field(
        default="4cm",
        description="Left and right margin width"
    )
    margin_top_bottom: str = Field(
        default="2cm",
        description="Top and bottom margin height"
    )
    paper_quality: str = Field(
        default="75 GSM",
        description="Paper weight specification"
    )
    double_sided: bool = Field(
        default=True,
        description="Whether to print on both sides"
    )


# Valid field types for template fields
ALLOWED_FIELD_TYPES = {"text", "date", "party", "case_number", "relief", "court"}

# Pattern for snake_case validation
SNAKE_CASE_PATTERN = re.compile(r'^[a-z][a-z0-9]*(_[a-z0-9]+)*$')


class TemplateField(BaseModel):
    """Definition of a field in a legal document template."""
    field_name: str = Field(
        min_length=1,
        max_length=100,
        description="Unique field identifier (snake_case)"
    )
    field_type: str = Field(
        description="Field data type: text|date|party|case_number|relief|court"
    )
    required: bool = Field(
        description="Whether this field must be provided"
    )
    label: str = Field(
        description="Human-readable label for the field"
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

    @field_validator('field_type')
    @classmethod
    def validate_field_type(cls, v: str) -> str:
        """Ensure field_type is one of the allowed types."""
        if v not in ALLOWED_FIELD_TYPES:
            raise ValueError(
                f"field_type must be one of {sorted(ALLOWED_FIELD_TYPES)}, got '{v}'"
            )
        return v

    @field_validator('field_name')
    @classmethod
    def validate_snake_case(cls, v: str) -> str:
        """Ensure field_name is in snake_case format."""
        if not SNAKE_CASE_PATTERN.match(v):
            raise ValueError(
                f"field_name must be lowercase snake_case (e.g., 'applicant_name'), got '{v}'"
            )
        return v


class ChangelogEntry(BaseModel):
    """Entry in template version changelog."""
    version: str = Field(description="Semantic version for this entry")
    date: str = Field(description="ISO timestamp when change was made")
    changes: List[str] = Field(description="List of changes in this version")
    author: str = Field(default="system", description="Who made the change")

    @field_validator('version')
    @classmethod
    def validate_semver(cls, v: str) -> str:
        """Ensure version is valid semantic version."""
        try:
            semver.Version.parse(v)
        except ValueError:
            raise ValueError(f"Invalid semantic version: {v}. Use format like '1.0.0'")
        return v


class LegalTemplate(BaseModel):
    """
    Complete legal document template with metadata, formatting, and fields.

    A template defines:
    - What type of document it is (metadata)
    - How the document should be formatted (formatting)
    - What fields need to be filled (required_fields, optional_fields)
    - The template structure for future Jinja2 rendering (template_content)
    """
    metadata: TemplateMetadata = Field(
        description="Template identification and classification"
    )
    formatting: FormattingRequirements = Field(
        default_factory=FormattingRequirements,
        description="Court-mandated formatting standards"
    )
    required_fields: List[TemplateField] = Field(
        description="Fields that must be provided to generate document"
    )
    optional_fields: List[TemplateField] = Field(
        default_factory=list,
        description="Fields that may optionally be provided"
    )
    status: TemplateStatus = Field(
        default=TemplateStatus.ACTIVE,
        description="Template lifecycle status (active/deprecated/archived)"
    )
    changelog: List[ChangelogEntry] = Field(
        default_factory=list,
        description="Version history with changelog entries"
    )
    template_content: str = Field(
        default="",
        description="Template structure for future Jinja2 use"
    )
    created_at: str = Field(
        default_factory=lambda: datetime.now().isoformat(),
        description="ISO timestamp when template was created"
    )
    updated_at: str = Field(
        default_factory=lambda: datetime.now().isoformat(),
        description="ISO timestamp when template was last updated"
    )
