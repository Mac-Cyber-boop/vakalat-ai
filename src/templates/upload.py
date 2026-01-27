"""
Template upload validation and processing for Vakalat AI.

Provides:
- JSON syntax validation
- Schema validation against LegalTemplate model
- Version validation for updates
- Complete upload processing pipeline
"""

import json
from typing import Optional, Tuple, Union
from datetime import datetime

from pydantic import ValidationError

from src.templates.schemas import LegalTemplate, TemplateStatus, DocumentType, CourtLevel
from src.templates.storage import TemplateRepository
from src.templates.versioning import (
    validate_version_upgrade,
    create_changelog_entry,
    is_version_higher
)


# Maximum template file size (500KB should be plenty for JSON templates)
MAX_TEMPLATE_SIZE_BYTES = 500 * 1024


class UploadValidationResult:
    """Result of template upload validation."""

    def __init__(
        self,
        is_valid: bool,
        template: Optional[LegalTemplate] = None,
        errors: Optional[list] = None
    ):
        self.is_valid = is_valid
        self.template = template
        self.errors = errors or []

    def __repr__(self) -> str:
        if self.is_valid:
            return f"UploadValidationResult(valid=True, template={self.template.metadata.name if self.template else None})"
        return f"UploadValidationResult(valid=False, errors={self.errors})"


class UploadProcessResult:
    """Result of template upload processing."""

    def __init__(
        self,
        success: bool,
        template: Optional[LegalTemplate] = None,
        is_update: bool = False,
        old_version: Optional[str] = None,
        new_version: Optional[str] = None,
        errors: Optional[list] = None
    ):
        self.success = success
        self.template = template
        self.is_update = is_update
        self.old_version = old_version
        self.new_version = new_version
        self.errors = errors or []

    def __repr__(self) -> str:
        if self.success:
            action = "updated" if self.is_update else "created"
            return f"UploadProcessResult(success=True, {action}, version={self.new_version})"
        return f"UploadProcessResult(success=False, errors={self.errors})"


def validate_file_size(content: Union[str, bytes]) -> Tuple[bool, Optional[str]]:
    """
    Validate that content does not exceed maximum size.

    Args:
        content: String or bytes content to check

    Returns:
        Tuple of (is_valid, error_message)
    """
    if isinstance(content, str):
        size = len(content.encode('utf-8'))
    else:
        size = len(content)

    if size > MAX_TEMPLATE_SIZE_BYTES:
        return (
            False,
            f"Template file too large: {size:,} bytes (max: {MAX_TEMPLATE_SIZE_BYTES:,} bytes)"
        )
    return (True, None)


def validate_json_syntax(content: str) -> Tuple[bool, Optional[dict], Optional[str]]:
    """
    Validate JSON syntax and parse content.

    Args:
        content: JSON string to parse

    Returns:
        Tuple of (is_valid, parsed_dict, error_message)
        - If valid: (True, parsed_dict, None)
        - If invalid: (False, None, error_message)
    """
    try:
        data = json.loads(content)
        if not isinstance(data, dict):
            return (False, None, "JSON content must be an object, not an array or primitive")
        return (True, data, None)
    except json.JSONDecodeError as e:
        # Provide actionable error message with line/column
        return (
            False,
            None,
            f"Invalid JSON syntax at line {e.lineno}, column {e.colno}: {e.msg}"
        )


def validate_template_schema(data: dict) -> Tuple[bool, Optional[LegalTemplate], list]:
    """
    Validate parsed JSON against LegalTemplate schema.

    Args:
        data: Parsed JSON dictionary

    Returns:
        Tuple of (is_valid, template, errors)
        - If valid: (True, LegalTemplate, [])
        - If invalid: (False, None, [error messages])
    """
    try:
        template = LegalTemplate.model_validate(data)
        return (True, template, [])
    except ValidationError as e:
        errors = []
        for error in e.errors():
            # Build field path from error location
            field_path = '.'.join(str(loc) for loc in error['loc'])
            message = error['msg']
            errors.append(f"{field_path}: {message}")
        return (False, None, errors)


def validate_template_upload(
    content: str,
    existing_version: Optional[str] = None
) -> UploadValidationResult:
    """
    Complete validation pipeline for template upload.

    Validates:
    1. File size within limits
    2. JSON syntax
    3. Schema against LegalTemplate model
    4. Version higher than existing (if updating)

    Args:
        content: Raw JSON string content
        existing_version: Current template version if updating

    Returns:
        UploadValidationResult with validation outcome
    """
    errors = []

    # Step 1: File size check
    size_valid, size_error = validate_file_size(content)
    if not size_valid:
        return UploadValidationResult(is_valid=False, errors=[size_error])

    # Step 2: JSON syntax check
    json_valid, data, json_error = validate_json_syntax(content)
    if not json_valid:
        return UploadValidationResult(is_valid=False, errors=[json_error])

    # Step 3: Schema validation
    schema_valid, template, schema_errors = validate_template_schema(data)
    if not schema_valid:
        return UploadValidationResult(is_valid=False, errors=schema_errors)

    # Step 4: Version check (if updating existing template)
    if existing_version is not None:
        new_version = template.metadata.version
        version_valid, version_error = validate_version_upgrade(
            new_version=new_version,
            existing_version=existing_version,
            require_higher=True
        )
        if not version_valid:
            return UploadValidationResult(is_valid=False, errors=[version_error])

    return UploadValidationResult(is_valid=True, template=template)


def process_template_upload(
    content: str,
    repository: TemplateRepository,
    change_description: Optional[str] = None,
    author: str = "system"
) -> UploadProcessResult:
    """
    Process a template upload: validate, handle versioning, and save.

    This is the main entry point for template uploads. It:
    1. Validates the content
    2. Checks if template exists (update vs create)
    3. For updates: validates version and creates changelog entry
    4. Saves the template

    Args:
        content: Raw JSON string content
        repository: TemplateRepository for storage operations
        change_description: Description for changelog (required for updates)
        author: Who is making the change

    Returns:
        UploadProcessResult with processing outcome
    """
    # Check if template exists
    # First parse just enough to get doc_type and court_level
    json_valid, data, json_error = validate_json_syntax(content)
    if not json_valid:
        return UploadProcessResult(success=False, errors=[json_error])

    # Extract identifiers from metadata
    metadata = data.get('metadata', {})
    doc_type = metadata.get('doc_type')
    court_level = metadata.get('court_level')

    if not doc_type or not court_level:
        return UploadProcessResult(
            success=False,
            errors=["Template must have metadata.doc_type and metadata.court_level"]
        )

    # Convert to enums for repository lookup
    try:
        doc_type_enum = DocumentType(doc_type)
        court_level_enum = CourtLevel(court_level)
    except ValueError as e:
        return UploadProcessResult(
            success=False,
            errors=[f"Invalid metadata value: {e}"]
        )

    # Check for existing template
    existing_template = repository.get_template(doc_type_enum, court_level_enum)
    is_update = existing_template is not None

    # Validate with existing version if updating
    existing_version = existing_template.metadata.version if existing_template else None
    validation_result = validate_template_upload(content, existing_version)

    if not validation_result.is_valid:
        return UploadProcessResult(
            success=False,
            errors=validation_result.errors
        )

    template = validation_result.template
    new_version = template.metadata.version

    # For updates, require change description and add changelog entry
    if is_update:
        if not change_description:
            return UploadProcessResult(
                success=False,
                errors=["change_description is required when updating an existing template"]
            )

        # Create changelog entry
        changelog_entry = create_changelog_entry(
            version=new_version,
            changes=[change_description],
            author=author
        )

        # Preserve existing changelog and append new entry
        existing_changelog = existing_template.changelog if existing_template else []
        template.changelog = existing_changelog + [changelog_entry]

    # Update timestamp
    template.updated_at = datetime.now().isoformat()

    # Save template
    try:
        repository.save_template(template)
    except Exception as e:
        return UploadProcessResult(
            success=False,
            errors=[f"Failed to save template: {str(e)}"]
        )

    return UploadProcessResult(
        success=True,
        template=template,
        is_update=is_update,
        old_version=existing_version,
        new_version=new_version
    )


def validate_template_json(json_content: str) -> UploadValidationResult:
    """
    Validate template JSON without saving (dry run).

    Useful for pre-flight validation before upload.

    Args:
        json_content: Raw JSON string

    Returns:
        UploadValidationResult with validation outcome
    """
    return validate_template_upload(json_content, existing_version=None)
