"""
Templates module for Vakalat AI.

Provides legal document template storage, validation, and versioning functionality.
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
from src.templates.versioning import (
    compare_versions,
    is_version_higher,
    bump_version,
    create_changelog_entry,
    validate_version_upgrade,
    get_version_parts,
    format_version_diff,
)
from src.templates.upload import (
    validate_file_size,
    validate_json_syntax,
    validate_template_schema,
    validate_template_upload,
    validate_template_json,
    process_template_upload,
    UploadValidationResult,
    UploadProcessResult,
    MAX_TEMPLATE_SIZE_BYTES,
)
from src.templates.lifecycle import (
    validate_status_transition,
    change_template_status,
    get_template_status,
    is_template_usable,
    get_templates_by_status,
    StatusChangeResult,
    UsabilityResult,
    UsabilityStatus,
)
from src.templates.preview import (
    FieldPreview,
    TemplatePreview,
    generate_preview,
    get_template_preview,
    list_template_previews,
)

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
    # Versioning utilities
    "compare_versions",
    "is_version_higher",
    "bump_version",
    "create_changelog_entry",
    "validate_version_upgrade",
    "get_version_parts",
    "format_version_diff",
    # Upload validation
    "validate_file_size",
    "validate_json_syntax",
    "validate_template_schema",
    "validate_template_upload",
    "validate_template_json",
    "process_template_upload",
    "UploadValidationResult",
    "UploadProcessResult",
    "MAX_TEMPLATE_SIZE_BYTES",
    # Lifecycle management
    "validate_status_transition",
    "change_template_status",
    "get_template_status",
    "is_template_usable",
    "get_templates_by_status",
    "StatusChangeResult",
    "UsabilityResult",
    "UsabilityStatus",
    # Preview generation
    "FieldPreview",
    "TemplatePreview",
    "generate_preview",
    "get_template_preview",
    "list_template_previews",
]
