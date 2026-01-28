"""
PDF exporter for legal documents.

Uses fpdf2 to generate court-standard formatted PDF documents
with proper margins, fonts, and page numbers.
"""

import io
from datetime import datetime
from typing import Optional

from fpdf import FPDF

from src.export.models import ExportFormat, ExportResult
from src.generation.generator import GeneratedDocument
from src.templates.schemas import FormattingRequirements


class LegalDocumentPDF(FPDF):
    """
    Custom FPDF class with page numbering footer.
    
    Adds "Page X of Y" footer to each page.
    """
    
    def footer(self):
        """Add page number footer."""
        self.set_y(-15)  # Position 15mm from bottom
        self.set_font("Times", "I", 10)
        self.set_text_color(128, 128, 128)
        self.cell(0, 10, f"Page {self.page_no()} of {{nb}}", align="C")


class PDFExporter:
    """
    Exports legal documents to PDF with court-standard formatting.
    
    Applies formatting from template:
    - Paper size (A4)
    - Font (Times New Roman)
    - Margins (4cm left/right, 2cm top/bottom)
    - Font size (14pt)
    - Line spacing (1.5)
    - Page numbers
    
    Example:
        exporter = PDFExporter()
        result = exporter.export(generated_doc, formatting)
        pdf_bytes = result.content_bytes
    """
    
    # Conversion: 1cm = 10mm
    CM_TO_MM = 10
    
    def __init__(self):
        """Initialize the PDF exporter."""
        pass
    
    def export(
        self,
        document: GeneratedDocument,
        formatting: Optional[FormattingRequirements] = None
    ) -> ExportResult:
        """
        Export a generated document to PDF.
        
        Args:
            document: The generated legal document
            formatting: Optional formatting requirements. If None, uses defaults.
            
        Returns:
            ExportResult with PDF bytes and metadata
        """
        # Use default formatting if not provided
        if formatting is None:
            formatting = FormattingRequirements()
        
        # Parse margins (e.g., "4cm" -> 40mm)
        margin_lr = self._parse_margin(formatting.margin_left_right)
        margin_tb = self._parse_margin(formatting.margin_top_bottom)
        
        # Create PDF
        pdf = LegalDocumentPDF(orientation="P", unit="mm", format="A4")
        pdf.alias_nb_pages()  # Enable total page count in footer
        
        # Set margins
        pdf.set_left_margin(margin_lr)
        pdf.set_right_margin(margin_lr)
        pdf.set_top_margin(margin_tb)
        pdf.set_auto_page_break(auto=True, margin=margin_tb + 10)
        
        # Add first page
        pdf.add_page()
        
        # Set font - use Times (built-in), closest to Times New Roman
        # Font size from formatting (default 14pt)
        pdf.set_font("Times", size=formatting.font_size)
        
        # Calculate line height for 1.5 spacing
        # Standard line height is font_size * 1.2, then multiply by line_spacing
        line_height = formatting.font_size * 0.352778 * formatting.line_spacing  # pt to mm
        
        # Write content
        self._write_content(pdf, document.content, line_height)
        
        # Generate PDF bytes
        pdf_bytes = pdf.output()
        
        # Generate filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{document.doc_type}_{document.court_level}_{timestamp}.pdf"
        
        return ExportResult(
            format=ExportFormat.PDF,
            content_bytes=pdf_bytes,
            filename=filename,
            page_count=pdf.page_no()
        )
    
    def _parse_margin(self, margin_str: str) -> float:
        """
        Parse margin string to millimeters.
        
        Args:
            margin_str: Margin string like "4cm" or "2cm"
            
        Returns:
            Margin in millimeters
        """
        # Remove whitespace
        margin_str = margin_str.strip().lower()
        
        if margin_str.endswith("cm"):
            value = float(margin_str[:-2])
            return value * self.CM_TO_MM
        elif margin_str.endswith("mm"):
            return float(margin_str[:-2])
        elif margin_str.endswith("in"):
            value = float(margin_str[:-2])
            return value * 25.4  # inches to mm
        else:
            # Assume mm if no unit
            try:
                return float(margin_str)
            except ValueError:
                return 40  # Default 4cm
    
    def _write_content(self, pdf: FPDF, content: str, line_height: float):
        """
        Write document content to PDF with proper formatting.
        
        Handles:
        - Paragraphs separated by blank lines
        - Basic text wrapping
        - Preserves line breaks within content
        
        Args:
            pdf: FPDF instance
            content: Document content string
            line_height: Line height in mm
        """
        # Split content into paragraphs
        paragraphs = content.split('\n\n')
        
        for i, paragraph in enumerate(paragraphs):
            if not paragraph.strip():
                continue
                
            # Handle lines within paragraph
            lines = paragraph.split('\n')
            for line in lines:
                if line.strip():
                    # Use multi_cell for automatic text wrapping
                    pdf.multi_cell(0, line_height, line.strip())
                else:
                    # Empty line within paragraph
                    pdf.ln(line_height / 2)
            
            # Add spacing between paragraphs
            if i < len(paragraphs) - 1:
                pdf.ln(line_height)
