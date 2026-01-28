# Phase 6: Production Integration - Research

**Researched:** 2026-01-28
**Domain:** PDF/DOCX export, FastAPI production patterns, error handling
**Confidence:** HIGH

## Summary

Phase 6 adds production-ready export capabilities (PDF and DOCX) and a unified `/draft-pro` API endpoint with proper error handling. The research confirms that the existing stack (fpdf2 2.8.5 already installed, python-docx 1.2.0 already installed) is well-suited for this phase.

**Key findings:**
1. **fpdf2** is the right choice for PDF generation - already installed, handles court-standard formatting, supports Times New Roman, works on Windows without dependencies
2. **python-docx** is already installed and handles all DOCX requirements - margins, fonts, page setup
3. **WeasyPrint should be avoided** due to significant Windows compatibility issues with GTK dependencies
4. The existing structlog audit infrastructure can be extended for production error logging
5. Binary file responses should use FastAPI's StreamingResponse with BytesIO for in-memory PDF/DOCX generation

**Primary recommendation:** Use fpdf2 (already installed) for PDF export and python-docx (already installed) for DOCX export. Create separate `/export/pdf` and `/export/docx` endpoints that accept generated document content.

## Standard Stack

The established libraries/tools for this domain:

### Core (Already Installed)
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| fpdf2 | 2.8.5 | PDF generation | Pure Python, no external deps, court-standard formatting support |
| python-docx | 1.2.0 | DOCX generation | Full Word document control, margins/fonts/styles |
| structlog | (installed) | Audit logging | Already configured in verification system |

### Supporting (Already Installed)
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| FastAPI | (installed) | API framework | Core API infrastructure |
| pydantic | 2.x | Request/response models | Validation and serialization |

### Alternatives Rejected
| Instead of | Rejected | Reason |
|------------|----------|--------|
| fpdf2 | WeasyPrint | Windows GTK dependency issues, installation failures on Win11 |
| fpdf2 | ReportLab | More complex API, overkill for document export use case |
| python-docx | docxtpl | python-docx is simpler, docxtpl adds Jinja complexity |

**Installation:**
```bash
# Already installed - no new dependencies needed
pip show fpdf2 python-docx structlog
```

## Architecture Patterns

### Recommended Project Structure
```
src/
├── export/                    # NEW: Export module
│   ├── __init__.py           # Module exports
│   ├── pdf_exporter.py       # PDF generation with fpdf2
│   ├── docx_exporter.py      # DOCX generation with python-docx
│   └── formatters.py         # FormattingRequirements to library format converters
├── generation/               # EXISTING: Document generation
│   ├── generator.py          # DocumentGenerator returns GeneratedDocument
│   └── ...
└── templates/                # EXISTING: Template storage
    ├── schemas.py            # FormattingRequirements model
    └── ...
```

### Pattern 1: Exporter Interface
**What:** Common interface for PDF and DOCX exporters with formatting application
**When to use:** When converting GeneratedDocument content to binary formats
**Example:**
```python
# Source: Derived from FormattingRequirements schema and fpdf2/python-docx docs
from abc import ABC, abstractmethod
from io import BytesIO
from src.templates.schemas import FormattingRequirements

class DocumentExporter(ABC):
    """Abstract base for document exporters."""

    @abstractmethod
    def export(
        self,
        content: str,
        formatting: FormattingRequirements,
        filename: str
    ) -> BytesIO:
        """Export content to binary format with formatting applied."""
        pass
```

### Pattern 2: fpdf2 with Subclassed Header/Footer
**What:** Subclass FPDF to add automatic page numbers
**When to use:** All PDF exports need court-standard page numbers
**Example:**
```python
# Source: https://py-pdf.github.io/fpdf2/Tutorial.html
from fpdf import FPDF

class LegalPDF(FPDF):
    def header(self):
        # Court case header (optional, can be empty)
        pass

    def footer(self):
        self.set_y(-15)  # 1.5cm from bottom
        self.set_font("times", "I", 10)
        self.cell(0, 10, f"Page {self.page_no()}/{{nb}}", align="C")

def create_pdf(content: str, formatting: FormattingRequirements) -> BytesIO:
    pdf = LegalPDF()

    # Apply court-specific margins (convert cm to mm)
    margin_mm = float(formatting.margin_left_right.replace("cm", "")) * 10
    pdf.set_margins(margin_mm, margin_mm, margin_mm)

    # Set font (Times New Roman is built-in as "times")
    pdf.add_page()
    pdf.set_font("times", size=formatting.font_size)

    # Line spacing applied via cell height
    line_height = formatting.font_size * formatting.line_spacing * 0.353  # pt to mm

    # Write content
    pdf.multi_cell(0, line_height, content)

    # Output to BytesIO
    buffer = BytesIO()
    pdf.output(buffer)
    buffer.seek(0)
    return buffer
```

### Pattern 3: python-docx with Section-Based Formatting
**What:** Use document sections to control page layout
**When to use:** All DOCX exports need court-standard margins/fonts
**Example:**
```python
# Source: https://python-docx.readthedocs.io/en/latest/user/sections.html
from docx import Document
from docx.shared import Inches, Pt, Cm
from docx.enum.text import WD_ALIGN_PARAGRAPH

def create_docx(content: str, formatting: FormattingRequirements) -> BytesIO:
    doc = Document()
    section = doc.sections[0]

    # A4 paper size
    section.page_height = Inches(11.69)
    section.page_width = Inches(8.27)

    # Court-specific margins (convert cm to Inches)
    margin_cm = float(formatting.margin_left_right.replace("cm", ""))
    section.left_margin = Cm(margin_cm)
    section.right_margin = Cm(margin_cm)

    margin_tb = float(formatting.margin_top_bottom.replace("cm", ""))
    section.top_margin = Cm(margin_tb)
    section.bottom_margin = Cm(margin_tb)

    # Set default paragraph style
    style = doc.styles['Normal']
    font = style.font
    font.name = formatting.font  # "Times New Roman"
    font.size = Pt(formatting.font_size)

    # Add content
    paragraph = doc.add_paragraph(content)
    paragraph.style = style

    # Output to BytesIO
    buffer = BytesIO()
    doc.save(buffer)
    buffer.seek(0)
    return buffer
```

### Pattern 4: FastAPI File Response with BytesIO
**What:** Return in-memory file as download response
**When to use:** `/export/pdf` and `/export/docx` endpoints
**Example:**
```python
# Source: https://fastapi.tiangolo.com/advanced/custom-response/
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from io import BytesIO

@app.post("/export/pdf")
async def export_pdf(req: ExportRequest):
    # Generate PDF in memory
    pdf_buffer: BytesIO = pdf_exporter.export(req.content, req.formatting)

    # Return as downloadable file
    return StreamingResponse(
        pdf_buffer,
        media_type="application/pdf",
        headers={
            "Content-Disposition": f"attachment; filename={req.filename}.pdf"
        }
    )
```

### Pattern 5: Structured Error Response
**What:** Consistent error format with error codes
**When to use:** All API error responses
**Example:**
```python
# Source: https://fastapi.tiangolo.com/tutorial/handling-errors/
from pydantic import BaseModel
from fastapi import HTTPException
from fastapi.responses import JSONResponse

class ErrorResponse(BaseModel):
    error_code: str
    message: str
    details: dict = {}

@app.exception_handler(ValueError)
async def value_error_handler(request, exc):
    return JSONResponse(
        status_code=400,
        content=ErrorResponse(
            error_code="VALIDATION_ERROR",
            message=str(exc),
            details={}
        ).model_dump()
    )
```

### Anti-Patterns to Avoid
- **WeasyPrint on Windows:** GTK dependencies cause installation failures on Windows 10/11
- **Base64 encoding for large files:** Adds 33% overhead, use binary StreamingResponse instead
- **Writing temp files:** Use BytesIO for in-memory generation, no disk I/O needed
- **Catching bare exceptions:** Always catch specific exception types

## Don't Hand-Roll

Problems that look simple but have existing solutions:

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| PDF generation | Custom PDF bytes | fpdf2 | PDF spec is complex, font embedding, page breaks |
| Word documents | Custom XML manipulation | python-docx | OOXML format is complex, schema validation |
| Margin unit conversion | Manual math | Library converters | Cm(), Inches(), Pt() handle conversions |
| Page numbering | Manual tracking | fpdf2 footer() override | Automatic on every page, handles total pages |
| Font embedding | Manual TTF parsing | fpdf2.add_font() | Font subsetting handled automatically |

**Key insight:** Both PDF and DOCX are complex binary formats with extensive specifications. The installed libraries (fpdf2, python-docx) abstract this complexity entirely.

## Common Pitfalls

### Pitfall 1: Font Name Mismatch in fpdf2
**What goes wrong:** Using "Times New Roman" instead of "times" causes font not found error
**Why it happens:** fpdf2 uses short names for built-in fonts
**How to avoid:** Use "times" for Times New Roman, "helvetica" for Arial
**Warning signs:** Error "Font not found" or garbled text

### Pitfall 2: Margin Unit Confusion
**What goes wrong:** Mixing cm, mm, inches, points without conversion
**Why it happens:** FormattingRequirements stores as "4cm" string, libraries expect numbers
**How to avoid:** Parse string, convert to library's unit type (fpdf2: mm, python-docx: Cm/Inches)
**Warning signs:** Margins too large or too small

### Pitfall 3: BytesIO Position Not Reset
**What goes wrong:** StreamingResponse returns empty file
**Why it happens:** After writing, BytesIO position is at end of buffer
**How to avoid:** Always call `buffer.seek(0)` before returning
**Warning signs:** 0-byte downloads

### Pitfall 4: Line Spacing Calculation
**What goes wrong:** Line spacing too tight or too wide
**Why it happens:** fpdf2 uses cell height in mm, not line spacing multiplier
**How to avoid:** Convert: `line_height_mm = font_size_pt * spacing * 0.353`
**Warning signs:** Text overlapping or excessive gaps

### Pitfall 5: Missing Content-Disposition Header
**What goes wrong:** Browser displays binary instead of downloading
**Why it happens:** Missing or incorrect Content-Disposition header
**How to avoid:** Set `"attachment; filename=document.pdf"` for download, `"inline"` for preview
**Warning signs:** PDF opens in browser as garbled text

### Pitfall 6: Exception Handler Import
**What goes wrong:** Starlette exceptions not caught by FastAPI handler
**Why it happens:** FastAPI HTTPException is subclass of Starlette's, but need to handle both
**How to avoid:** Register handlers for `starlette.exceptions.HTTPException`
**Warning signs:** Unhandled exceptions returning raw error pages

## Code Examples

Verified patterns from official sources:

### PDF Export with Court Formatting
```python
# Source: fpdf2 tutorial + FormattingRequirements integration
from fpdf import FPDF
from io import BytesIO
from src.templates.schemas import FormattingRequirements

class CourtPDF(FPDF):
    """PDF with automatic page numbers for court filing."""

    def footer(self):
        self.set_y(-15)
        self.set_font("times", "I", 10)
        self.cell(0, 10, f"Page {self.page_no()}/{{nb}}", align="C")

def export_to_pdf(
    content: str,
    formatting: FormattingRequirements,
    title: str = ""
) -> BytesIO:
    """
    Export document content to PDF with court-standard formatting.

    Args:
        content: Document text content
        formatting: FormattingRequirements from template
        title: Optional document title for header

    Returns:
        BytesIO buffer containing PDF bytes
    """
    pdf = CourtPDF()

    # Parse margins (stored as "4cm" string)
    margin_lr = float(formatting.margin_left_right.replace("cm", "")) * 10  # cm to mm
    margin_tb = float(formatting.margin_top_bottom.replace("cm", "")) * 10

    pdf.set_margins(margin_lr, margin_tb, margin_lr)
    pdf.set_auto_page_break(auto=True, margin=margin_tb)

    pdf.add_page()

    # Set font (Times New Roman = "times" in fpdf2)
    font_name = "times" if "Times" in formatting.font else "helvetica"
    pdf.set_font(font_name, size=formatting.font_size)

    # Calculate line height from spacing multiplier
    # 1 pt = 0.353 mm, line height = font_size * spacing * conversion
    line_height = formatting.font_size * formatting.line_spacing * 0.353

    # Write content with multi_cell for automatic wrapping
    pdf.multi_cell(0, line_height, content)

    # Output to buffer
    buffer = BytesIO()
    pdf.output(buffer)
    buffer.seek(0)
    return buffer
```

### DOCX Export with Court Formatting
```python
# Source: python-docx docs + FormattingRequirements integration
from docx import Document
from docx.shared import Inches, Pt, Cm
from docx.enum.section import WD_ORIENT
from io import BytesIO
from src.templates.schemas import FormattingRequirements

def export_to_docx(
    content: str,
    formatting: FormattingRequirements
) -> BytesIO:
    """
    Export document content to DOCX with court-standard formatting.

    Args:
        content: Document text content
        formatting: FormattingRequirements from template

    Returns:
        BytesIO buffer containing DOCX bytes
    """
    doc = Document()
    section = doc.sections[0]

    # A4 paper size (portrait)
    section.page_height = Inches(11.69)
    section.page_width = Inches(8.27)
    section.orientation = WD_ORIENT.PORTRAIT

    # Parse and apply margins
    margin_lr = float(formatting.margin_left_right.replace("cm", ""))
    margin_tb = float(formatting.margin_top_bottom.replace("cm", ""))

    section.left_margin = Cm(margin_lr)
    section.right_margin = Cm(margin_lr)
    section.top_margin = Cm(margin_tb)
    section.bottom_margin = Cm(margin_tb)

    # Configure default style
    style = doc.styles['Normal']
    font = style.font
    font.name = formatting.font  # "Times New Roman"
    font.size = Pt(formatting.font_size)

    # Line spacing (1.5 = 150%)
    paragraph_format = style.paragraph_format
    paragraph_format.line_spacing = formatting.line_spacing

    # Add content as paragraphs
    for para_text in content.split('\n\n'):
        if para_text.strip():
            para = doc.add_paragraph(para_text.strip())
            para.style = style

    # Output to buffer
    buffer = BytesIO()
    doc.save(buffer)
    buffer.seek(0)
    return buffer
```

### FastAPI Export Endpoints
```python
# Source: FastAPI custom-response docs
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import Literal

class ExportRequest(BaseModel):
    content: str = Field(description="Document content to export")
    doc_type: str = Field(description="Document type for filename")
    court_level: str = Field(description="Court level for formatting lookup")
    format: Literal["pdf", "docx"] = Field(description="Export format")

@app.post("/export")
async def export_document(req: ExportRequest):
    """
    Export generated document to PDF or DOCX.

    Returns binary file as attachment download.
    """
    # Get formatting from template
    template = template_repo.get_template(
        DocumentType(req.doc_type),
        CourtLevel(req.court_level)
    )
    if not template:
        raise HTTPException(404, "Template not found")

    formatting = template.formatting
    filename = f"{req.doc_type}_{req.court_level}"

    if req.format == "pdf":
        buffer = export_to_pdf(req.content, formatting)
        media_type = "application/pdf"
        extension = "pdf"
    else:
        buffer = export_to_docx(req.content, formatting)
        media_type = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        extension = "docx"

    return StreamingResponse(
        buffer,
        media_type=media_type,
        headers={
            "Content-Disposition": f"attachment; filename={filename}.{extension}"
        }
    )
```

### Structured Error Handler
```python
# Source: FastAPI error handling tutorial + structlog integration
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
from pydantic import BaseModel
import structlog

logger = structlog.get_logger(module="api")

class ErrorDetail(BaseModel):
    error_code: str
    message: str
    details: dict = {}

@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    logger.warning(
        "http_error",
        status_code=exc.status_code,
        detail=str(exc.detail),
        path=str(request.url.path)
    )
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorDetail(
            error_code=f"HTTP_{exc.status_code}",
            message=str(exc.detail)
        ).model_dump()
    )

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    logger.warning(
        "validation_error",
        errors=exc.errors(),
        path=str(request.url.path)
    )
    return JSONResponse(
        status_code=400,
        content=ErrorDetail(
            error_code="VALIDATION_ERROR",
            message="Request validation failed",
            details={"errors": exc.errors()}
        ).model_dump()
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logger.error(
        "unhandled_error",
        error_type=type(exc).__name__,
        error_message=str(exc),
        path=str(request.url.path)
    )
    return JSONResponse(
        status_code=500,
        content=ErrorDetail(
            error_code="INTERNAL_ERROR",
            message="An unexpected error occurred"
        ).model_dump()
    )
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| WeasyPrint for PDF | fpdf2 for programmatic PDF | 2024+ | Avoids Windows GTK issues |
| PyFPDF (unmaintained) | fpdf2 (active fork) | 2021 | Active maintenance, Unicode support |
| temp files for export | BytesIO in-memory | Standard | No disk I/O, cleaner code |

**Deprecated/outdated:**
- PyFPDF: Unmaintained since 2018, use fpdf2 instead
- WeasyPrint on Windows: Requires MSYS2/GTK installation, unreliable

## Open Questions

Things that couldn't be fully resolved:

1. **Page number placement for double-sided printing**
   - What we know: FormattingRequirements has `double_sided: bool`
   - What's unclear: Should page numbers alternate left/right for double-sided?
   - Recommendation: For v1, use centered page numbers regardless of double_sided setting

2. **Legal document structure preservation**
   - What we know: Content comes as plain text from GeneratedDocument
   - What's unclear: How to preserve numbered paragraph structure, indentation
   - Recommendation: For v1, export as plain paragraphs; structured export in future phase

## Sources

### Primary (HIGH confidence)
- fpdf2 Official Documentation - https://py-pdf.github.io/fpdf2/
  - Tutorial: margins, fonts, page numbers
  - Unicode: font embedding, Windows paths
- python-docx Official Documentation - https://python-docx.readthedocs.io/
  - Sections: margins, page size
  - Styles: fonts, paragraph formatting
- FastAPI Official Documentation - https://fastapi.tiangolo.com/
  - Custom responses: StreamingResponse, FileResponse
  - Error handling: exception handlers

### Secondary (MEDIUM confidence)
- [fpdf2 PyPI](https://pypi.org/project/fpdf2/) - Version 2.8.5 (2025/10/29)
- [WeasyPrint GitHub Issues](https://github.com/Kozea/WeasyPrint/issues/2480) - Windows 11 installation failures
- [FastAPI Error Handling Guide](https://betterstack.com/community/guides/scaling-python/error-handling-fastapi/)

### Tertiary (LOW confidence)
- WebSearch results for content negotiation patterns - verified with FastAPI docs
- WebSearch results for base64 vs binary best practices - general API design principles

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - Libraries already installed, verified versions
- Architecture: HIGH - Based on official documentation patterns
- Pitfalls: HIGH - Common issues documented in official sources
- Export patterns: MEDIUM - Derived from docs, not production-tested

**Research date:** 2026-01-28
**Valid until:** 2026-03-28 (60 days - stable libraries, mature patterns)
