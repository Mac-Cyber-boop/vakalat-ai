"""
Audit logging infrastructure for the verification system.

Provides structured logging with structlog for:
- Citation verification attempts
- Verification results (VERIFIED, UNVERIFIED, BLOCKED)
- Performance tracking (duration_ms)
"""

import logging
import structlog
from datetime import datetime, timezone
from typing import Optional
from pydantic import BaseModel, Field


def configure_audit_logging() -> None:
    """
    Configure structlog with JSON output for production audit logging.

    Uses stdout for container compatibility (no file configuration).
    Follows api.py conventions for cloud deployment.
    """
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_audit_logger(name: str) -> structlog.BoundLogger:
    """
    Get a bound logger with the specified module name.

    Args:
        name: Module name for log attribution (e.g., "citation_verifier")

    Returns:
        BoundLogger instance with module context
    """
    return structlog.get_logger(module=name)


class AuditEvent(BaseModel):
    """Audit event model for verification logging."""

    event_type: str = Field(
        description="Type of audit event (e.g., 'citation_verification', 'citation_blocked')"
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="UTC timestamp of the event"
    )
    input_data: dict = Field(
        description="Input data being verified (e.g., case citation details)"
    )
    result: str = Field(
        description="Verification result (VERIFIED, UNVERIFIED, BLOCKED)"
    )
    duration_ms: int = Field(
        ge=0,
        description="Time taken for verification in milliseconds"
    )
    source_id: Optional[str] = Field(
        default=None,
        description="Source document ID if found in database"
    )
    reason: Optional[str] = Field(
        default=None,
        description="Reason for UNVERIFIED or BLOCKED status"
    )


def log_verification_attempt(
    logger: structlog.BoundLogger,
    event: AuditEvent
) -> None:
    """
    Log a verification attempt with appropriate log level.

    Args:
        logger: The structlog bound logger
        event: The audit event to log

    Logs at INFO level for VERIFIED, WARNING for BLOCKED/UNVERIFIED.
    """
    event_dict = event.model_dump()
    # Convert datetime to ISO string for JSON serialization
    event_dict["timestamp"] = event.timestamp.isoformat()

    if event.result == "VERIFIED":
        logger.info("verification_attempt", **event_dict)
    elif event.result == "BLOCKED":
        logger.warning("verification_attempt", **event_dict)
    else:
        logger.info("verification_attempt", **event_dict)
