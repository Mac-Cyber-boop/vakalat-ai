"""
Vision RAG Ingestion Pipeline for Vakalat AI.

Problem with standard text extraction (PyMuPDF text mode):
    Legal PDFs contain tables, section headers, marginal notes, and formatted
    schedules that plain text extraction destroys. A bail application table
    becomes a jumbled string; a court fee schedule loses its columns entirely.

This Solution — Vision-Assisted Extraction:
    1. Render each PDF page as a high-resolution image (PyMuPDF pixmap)
    2. Send the image to GPT-4o Vision with a structured extraction prompt
    3. GPT-4o returns structured JSON: document type, sections, tables as text,
       key clauses — preserving layout context that raw text loses
    4. Create LangChain Documents with rich metadata
    5. Embed and upsert to Pinecone

Why GPT-4o Vision instead of ColPali:
    ColPali produces image embeddings but requires a local GPU and a separate
    vector index that supports multi-vector retrieval. GPT-4o Vision extracts
    structured text FROM the image, which we then embed normally — this slots
    into the existing Pinecone + OpenAIEmbeddings pipeline without changes.

Usage:
    ingestor = VisionIngestor(vector_store)
    result = ingestor.ingest_pdf("judgments/arnesh_kumar.PDF", source_type="case_law")
    print(f"Ingested {result.pages_processed} pages, {result.chunks_added} chunks")

    # Or batch ingest a directory:
    results = ingestor.ingest_directory("judgments/", source_type="case_law")
"""

import base64
import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import fitz  # PyMuPDF
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from openai import OpenAI


# Resolution for page rendering. 2.0 = 144 DPI (good for legal text/tables).
# Higher = better quality but larger images and more API cost.
RENDER_SCALE = 2.0

# GPT-4o Vision has a 20MB image limit. At 2x scale, most legal PDF pages
# are well under this. Pages exceeding the limit are skipped with a warning.
MAX_IMAGE_BYTES = 18 * 1024 * 1024  # 18MB safety margin

# Skip pages with fewer than this many characters of extracted content.
# Catches blank pages, cover pages, and purely decorative pages.
MIN_CONTENT_CHARS = 80

# Prompt sent to GPT-4o Vision for each page
_EXTRACTION_PROMPT = """You are a legal document parser. Analyze this page from an Indian legal document.

Extract and return a JSON object with EXACTLY these fields:
{
  "doc_type": "statute" | "case_law" | "procedural" | "unknown",
  "source_book": "Full name of the act/judgment/document (e.g. 'Bharatiya Nyaya Sanhita, 2023')",
  "case_name": "Case name if this is a judgment (e.g. 'Arnesh Kumar vs State of Bihar'), else null",
  "court": "Court name if case_law (e.g. 'Supreme Court of India'), else null",
  "year": "Year as integer (e.g. 2014), or null if not found",
  "sections": ["List of section/article numbers visible on this page, e.g. ['302', '304', '304A']"],
  "content": "Full structured text of this page. For tables: preserve them as markdown tables. For sections: preserve section numbers and headings. For judgments: preserve paragraph structure. Do NOT summarize — extract verbatim."
}

Rules:
- Return ONLY the JSON object. No markdown, no explanation.
- For tables, use markdown format: | Col1 | Col2 | ... |
- Preserve section numbers exactly as written (e.g., '438', '439', 'Schedule II')
- If a field cannot be determined, use null
- content must be the actual text, not a summary"""


@dataclass
class PageExtraction:
    """Result of extracting a single PDF page via Vision."""
    page_num: int
    doc_type: str
    source_book: str
    case_name: Optional[str]
    court: Optional[str]
    year: Optional[int]
    sections: list[str]
    content: str
    skipped: bool = False
    skip_reason: str = ""


@dataclass
class VisionIngestResult:
    """Summary of a completed Vision RAG ingestion."""
    file_path: str
    pages_processed: int
    pages_skipped: int
    chunks_added: int
    errors: list[str] = field(default_factory=list)


class VisionIngestor:
    """
    Ingests legal PDF files into Pinecone using GPT-4o Vision extraction.

    Produces richer chunks than plain text extraction by:
    - Preserving tables as markdown
    - Tagging each chunk with section numbers from that page
    - Inferring document type, court, and year from visual layout
    - Attaching page numbers for source verification

    Args:
        vector_store: Connected PineconeVectorStore to upsert chunks into.
        verbose: If True, prints progress per page.
    """

    def __init__(self, vector_store: PineconeVectorStore, verbose: bool = True):
        self.vector_store = vector_store
        self.verbose = verbose
        self._client = OpenAI()  # Uses OPENAI_API_KEY from env

    def _render_page_to_base64(self, page: fitz.Page) -> Optional[str]:
        """
        Render a PDF page to a base64-encoded PNG image.

        Args:
            page: PyMuPDF page object.

        Returns:
            Base64-encoded PNG string, or None if the image is too large.
        """
        matrix = fitz.Matrix(RENDER_SCALE, RENDER_SCALE)
        pixmap = page.get_pixmap(matrix=matrix, alpha=False)
        img_bytes = pixmap.tobytes("png")

        if len(img_bytes) > MAX_IMAGE_BYTES:
            return None

        return base64.b64encode(img_bytes).decode("utf-8")

    def _extract_page_via_vision(
        self, page_b64: str, page_num: int
    ) -> Optional[dict]:
        """
        Send a rendered page image to GPT-4o Vision and parse the JSON response.

        Args:
            page_b64: Base64-encoded PNG of the page.
            page_num: Page number (for error messages).

        Returns:
            Parsed dict from GPT-4o, or None on failure.
        """
        try:
            response = self._client.chat.completions.create(
                model="gpt-4o",
                max_tokens=2000,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{page_b64}",
                                    "detail": "high",
                                },
                            },
                            {
                                "type": "text",
                                "text": _EXTRACTION_PROMPT,
                            },
                        ],
                    }
                ],
            )

            raw = response.choices[0].message.content.strip()

            # Strip markdown code fences if the model wraps in ```json
            raw = re.sub(r"^```(?:json)?\s*", "", raw)
            raw = re.sub(r"\s*```$", "", raw)

            return json.loads(raw)

        except json.JSONDecodeError as e:
            if self.verbose:
                print(f"  [Page {page_num}] JSON parse error: {e}")
            return None
        except Exception as e:
            if self.verbose:
                print(f"  [Page {page_num}] Vision API error: {e}")
            return None

    def _parse_extraction(self, raw: dict, page_num: int) -> PageExtraction:
        """Convert the raw GPT-4o JSON dict into a typed PageExtraction."""
        content = raw.get("content", "").strip()

        return PageExtraction(
            page_num=page_num,
            doc_type=raw.get("doc_type") or "unknown",
            source_book=raw.get("source_book") or "Unknown Document",
            case_name=raw.get("case_name"),
            court=raw.get("court"),
            year=raw.get("year"),
            sections=raw.get("sections") or [],
            content=content,
        )

    def _extraction_to_document(
        self,
        extraction: PageExtraction,
        file_name: str,
        source_type_override: Optional[str],
    ) -> Document:
        """
        Convert a PageExtraction into a LangChain Document for Pinecone.

        Each document gets rich metadata:
        - source_type: for metadata filtering in retrieval
        - source_book, case_name, court, year: for citation display
        - sections: comma-separated section numbers on this page
        - page_num: for "Source: Page X" attribution in answers
        - extraction_method: "vision" to distinguish from plain-text chunks
        """
        source_type = source_type_override or extraction.doc_type

        metadata = {
            "source_type": source_type,
            "source_book": extraction.source_book,
            "page_num": extraction.page_num,
            "sections": ", ".join(extraction.sections) if extraction.sections else "",
            "extraction_method": "vision",
            "source": file_name,
        }

        if extraction.case_name:
            metadata["case_name"] = extraction.case_name
        if extraction.court:
            metadata["court"] = extraction.court
        if extraction.year:
            metadata["year"] = extraction.year

        return Document(page_content=extraction.content, metadata=metadata)

    def ingest_pdf(
        self,
        pdf_path: str,
        source_type: Optional[str] = None,
        delay_between_pages: float = 0.5,
    ) -> VisionIngestResult:
        """
        Ingest a single PDF file using Vision extraction.

        Args:
            pdf_path: Path to the PDF file.
            source_type: Override inferred doc_type. Use "statute", "case_law",
                         or "procedural". If None, GPT-4o infers from the page.
            delay_between_pages: Seconds to wait between Vision API calls
                                  to avoid rate limiting (default 0.5s).

        Returns:
            VisionIngestResult with counts and any errors.
        """
        path = Path(pdf_path)
        file_name = path.name
        result = VisionIngestResult(file_path=str(path), pages_processed=0, pages_skipped=0, chunks_added=0)

        if self.verbose:
            print(f"\nIngesting: {file_name}")

        try:
            pdf = fitz.open(str(path))
        except Exception as e:
            result.errors.append(f"Cannot open PDF: {e}")
            return result

        documents_to_add = []

        for page_num in range(len(pdf)):
            display_num = page_num + 1

            # Render page to image
            page = pdf[page_num]
            page_b64 = self._render_page_to_base64(page)

            if page_b64 is None:
                result.pages_skipped += 1
                result.errors.append(f"Page {display_num}: image too large, skipped")
                if self.verbose:
                    print(f"  [Page {display_num}] Skipped — image too large")
                continue

            if self.verbose:
                print(f"  [Page {display_num}/{len(pdf)}] Extracting via Vision...", end=" ")

            # Extract via GPT-4o Vision
            raw = self._extract_page_via_vision(page_b64, display_num)

            if raw is None:
                result.pages_skipped += 1
                result.errors.append(f"Page {display_num}: Vision extraction failed")
                if self.verbose:
                    print("FAILED")
                continue

            extraction = self._parse_extraction(raw, display_num)

            # Skip pages with insufficient content (blank, cover, etc.)
            if len(extraction.content) < MIN_CONTENT_CHARS:
                result.pages_skipped += 1
                if self.verbose:
                    print(f"Skipped (thin content: {len(extraction.content)} chars)")
                continue

            doc = self._extraction_to_document(extraction, file_name, source_type)
            documents_to_add.append(doc)
            result.pages_processed += 1

            if self.verbose:
                sections_str = f"§{', §'.join(extraction.sections[:3])}" if extraction.sections else "no sections"
                print(f"OK ({len(extraction.content)} chars, {sections_str})")

            # Rate limit protection
            if delay_between_pages > 0:
                time.sleep(delay_between_pages)

        pdf.close()

        # Batch upsert to Pinecone
        if documents_to_add:
            if self.verbose:
                print(f"\nUpserting {len(documents_to_add)} chunks to Pinecone...")
            self.vector_store.add_documents(documents_to_add)
            result.chunks_added = len(documents_to_add)
            if self.verbose:
                print("Done.")

        return result

    def ingest_directory(
        self,
        directory: str,
        source_type: Optional[str] = None,
        delay_between_pages: float = 0.5,
    ) -> list[VisionIngestResult]:
        """
        Ingest all PDF files in a directory.

        Args:
            directory: Path to directory containing PDF files.
            source_type: Override source type for all files (optional).
            delay_between_pages: Rate limit delay between Vision API calls.

        Returns:
            List of VisionIngestResult, one per PDF file.
        """
        dir_path = Path(directory)
        pdf_files = list(dir_path.glob("*.pdf")) + list(dir_path.glob("*.PDF"))

        if not pdf_files:
            print(f"No PDF files found in {directory}")
            return []

        print(f"Found {len(pdf_files)} PDF files in {directory}")
        results = []

        for pdf_path in pdf_files:
            result = self.ingest_pdf(
                str(pdf_path),
                source_type=source_type,
                delay_between_pages=delay_between_pages,
            )
            results.append(result)

        # Summary
        total_pages = sum(r.pages_processed for r in results)
        total_chunks = sum(r.chunks_added for r in results)
        total_skipped = sum(r.pages_skipped for r in results)
        print(f"\nBatch complete: {len(results)} files | {total_pages} pages | {total_chunks} chunks | {total_skipped} skipped")

        return results
