"""
Export models for legal document export functionality.

Provides data structures for:
- Export format enumeration (PDF, DOCX)
- Export result with bytes, filename, and metadata
"""

from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


class ExportFormat(str, Enum):
    """Supported document export formats."""
    PDF = "pdf"
    DOCX = "docx"


class ExportResult(BaseModel):
    """Result of a document export operation."""
    
    format: ExportFormat = Field(
        description="Export format used"
    )
    content_bytes: bytes = Field(
        description="Raw bytes of the exported document"
    )
    filename: str = Field(
        description="Suggested filename for the exported document"
    )
    page_count: Optional[int] = Field(
        default=None,
        description="Number of pages in the document (PDF only)"
    )
    
    class Config:
        # Allow bytes field to be serialized
        arbitrary_types_allowed = True
    
    def to_base64(self) -> str:
        """Convert content bytes to base64 string for API response."""
        import base64
        return base64.b64encode(self.content_bytes).decode('utf-8')
