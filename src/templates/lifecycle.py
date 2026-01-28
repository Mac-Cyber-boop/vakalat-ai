"""
Template lifecycle management for Vakalat AI.

Provides status transition logic for templates:
- validate_status_transition: Check if transition is allowed
- change_template_status: Change status with changelog recording
- get_template_status: Get current status of a template
- is_template_usable: Check if template can be used (with warnings)

Lifecycle states:
- ACTIVE: Template is in production use
- DEPRECATED: Template shows warning but remains usable
- ARCHIVED: Template is blocked from use

Valid transitions:
- active -> deprecated (show warning, still usable)
- active -> archived (direct decommission)
- deprecated -> archived (final retirement)
- archived -> (none) - terminal state
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Optional, List

from src.templates.schemas import (
    DocumentType,
    CourtLevel,
    TemplateStatus,
    ChangelogEntry,
    LegalTemplate,
    ALLOWED_TRANSITIONS,
)
from src.templates.storage import TemplateRepository


# Configure logging for lifecycle module
logger = logging.getLogger(__name__)


class UsabilityStatus(str, Enum):
    """Result of template usability check."""
    USABLE = "usable"
    WARNING = "warning"
    BLOCKED = "blocked"


@dataclass
class StatusChangeResult:
    """Result of a status change operation."""
    success: bool
    previous_status: Optional[TemplateStatus] = None
    new_status: Optional[TemplateStatus] = None
    error: Optional[str] = None
    changelog_entry: Optional[ChangelogEntry] = None


@dataclass
class UsabilityResult:
    """Result of template usability check."""
    status: UsabilityStatus
    template_status: Optional[TemplateStatus] = None
    message: Optional[str] = None
    can_use: bool = True


def validate_status_transition(
    current_status: TemplateStatus,
    target_status: TemplateStatus,
) -> tuple[bool, Optional[str]]:
    """
    Validate if a status transition is allowed.

    Args:
        current_status: Current template status
        target_status: Desired target status

    Returns:
        Tuple of (is_valid, error_message)
        - (True, None) if transition is allowed
        - (False, error_message) if transition is not allowed
    """
    if current_status == target_status:
        return False, f"Template is already {current_status.value}"

    allowed = ALLOWED_TRANSITIONS.get(current_status, set())

    if target_status not in allowed:
        if current_status == TemplateStatus.ARCHIVED:
            return False, "Cannot transition from archived status - it is a terminal state"

        allowed_list = [s.value for s in allowed] if allowed else ["none"]
        return False, (
            f"Invalid transition: {current_status.value} -> {target_status.value}. "
            f"Allowed transitions from {current_status.value}: {allowed_list}"
        )

    return True, None


def change_template_status(
    doc_type: DocumentType,
    court_level: CourtLevel,
    target_status: TemplateStatus,
    reason: str,
    author: str = "system",
    repository: Optional[TemplateRepository] = None,
) -> StatusChangeResult:
    """
    Change the status of a template with changelog recording.

    Args:
        doc_type: Document type of template
        court_level: Court level of template
        target_status: New status to set
        reason: Reason for the status change (recorded in changelog)
        author: Who is making the change
        repository: Optional repository instance (creates default if not provided)

    Returns:
        StatusChangeResult with success/failure and details
    """
    if repository is None:
        repository = TemplateRepository()

    # Load template
    template = repository.get_template(doc_type, court_level)
    if template is None:
        return StatusChangeResult(
            success=False,
            error=f"Template not found: {doc_type.value}/{court_level.value}"
        )

    current_status = template.status

    # Validate transition
    is_valid, error = validate_status_transition(current_status, target_status)
    if not is_valid:
        return StatusChangeResult(
            success=False,
            previous_status=current_status,
            error=error
        )

    # Create changelog entry for status change
    changelog_entry = ChangelogEntry(
        version=template.metadata.version,
        date=datetime.now().isoformat(),
        changes=[f"Status changed: {current_status.value} -> {target_status.value}. Reason: {reason}"],
        author=author
    )

    # Update template
    template.status = target_status
    template.changelog.append(changelog_entry)
    template.updated_at = datetime.now().isoformat()

    # Save template
    repository.save_template(template)

    logger.info(
        f"Template status changed: {doc_type.value}/{court_level.value} "
        f"{current_status.value} -> {target_status.value} by {author}"
    )

    return StatusChangeResult(
        success=True,
        previous_status=current_status,
        new_status=target_status,
        changelog_entry=changelog_entry
    )


def get_template_status(
    doc_type: DocumentType,
    court_level: CourtLevel,
    repository: Optional[TemplateRepository] = None,
) -> Optional[TemplateStatus]:
    """
    Get the current status of a template.

    Args:
        doc_type: Document type of template
        court_level: Court level of template
        repository: Optional repository instance

    Returns:
        TemplateStatus if template exists, None otherwise
    """
    if repository is None:
        repository = TemplateRepository()

    template = repository.get_template(doc_type, court_level)
    if template is None:
        return None

    return template.status


def is_template_usable(
    doc_type: DocumentType,
    court_level: CourtLevel,
    repository: Optional[TemplateRepository] = None,
) -> UsabilityResult:
    """
    Check if a template can be used.

    Returns:
        - USABLE (can_use=True, no message) for ACTIVE templates
        - WARNING (can_use=True, with message) for DEPRECATED templates
        - BLOCKED (can_use=False, with error) for ARCHIVED templates
        - BLOCKED (can_use=False) if template not found

    Args:
        doc_type: Document type of template
        court_level: Court level of template
        repository: Optional repository instance

    Returns:
        UsabilityResult with status and details
    """
    if repository is None:
        repository = TemplateRepository()

    template = repository.get_template(doc_type, court_level)
    if template is None:
        return UsabilityResult(
            status=UsabilityStatus.BLOCKED,
            template_status=None,
            message=f"Template not found: {doc_type.value}/{court_level.value}",
            can_use=False
        )

    if template.status == TemplateStatus.ACTIVE:
        return UsabilityResult(
            status=UsabilityStatus.USABLE,
            template_status=TemplateStatus.ACTIVE,
            message=None,
            can_use=True
        )

    if template.status == TemplateStatus.DEPRECATED:
        return UsabilityResult(
            status=UsabilityStatus.WARNING,
            template_status=TemplateStatus.DEPRECATED,
            message=(
                f"Template '{template.metadata.name}' is deprecated. "
                "Consider using a newer template version. "
                "This template may be archived in the future."
            ),
            can_use=True
        )

    if template.status == TemplateStatus.ARCHIVED:
        return UsabilityResult(
            status=UsabilityStatus.BLOCKED,
            template_status=TemplateStatus.ARCHIVED,
            message=(
                f"Template '{template.metadata.name}' is archived and cannot be used. "
                "Please select an active template."
            ),
            can_use=False
        )

    # Fallback for any unknown status
    return UsabilityResult(
        status=UsabilityStatus.BLOCKED,
        template_status=template.status,
        message=f"Unknown template status: {template.status}",
        can_use=False
    )


def get_templates_by_status(
    status: TemplateStatus,
    doc_type: Optional[DocumentType] = None,
    repository: Optional[TemplateRepository] = None,
) -> List[LegalTemplate]:
    """
    Get all templates with a specific status.

    Args:
        status: Status to filter by
        doc_type: Optional document type filter
        repository: Optional repository instance

    Returns:
        List of templates with the specified status
    """
    if repository is None:
        repository = TemplateRepository()

    templates = repository.list_templates(doc_type)
    return [t for t in templates if t.status == status]
