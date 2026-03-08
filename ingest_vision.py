"""
Vision RAG Ingestion Script

Re-ingests legal PDFs using GPT-4o Vision instead of plain text extraction.
Preserves tables, section headers, and legal formatting that PyMuPDF text
extraction destroys.

Usage:
    # Ingest a single PDF
    python ingest_vision.py --file judgments/arnesh_kumar.PDF --type case_law

    # Ingest all PDFs in a directory
    python ingest_vision.py --dir judgments/ --type case_law

    # Ingest statutes
    python ingest_vision.py --dir acts/ --type statute

Source type options: statute, case_law, procedural
If --type is omitted, GPT-4o will infer the type from each page.
"""

import argparse
import os

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone

from src.ingestion import VisionIngestor

load_dotenv()


def main():
    parser = argparse.ArgumentParser(description="Vision RAG PDF Ingestion")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--file", help="Path to a single PDF file")
    group.add_argument("--dir", help="Path to a directory of PDF files")
    parser.add_argument(
        "--type",
        choices=["statute", "case_law", "procedural"],
        default=None,
        help="Source type override (optional — GPT-4o infers if omitted)",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.5,
        help="Seconds between Vision API calls (default 0.5, increase if rate-limited)",
    )
    args = parser.parse_args()

    # Validate environment
    required = ["OPENAI_API_KEY", "PINECONE_API_KEY", "PINECONE_INDEX_NAME"]
    missing = [k for k in required if not os.getenv(k)]
    if missing:
        print(f"ERROR: Missing environment variables: {missing}")
        print("Make sure your .env file is configured.")
        return

    # Connect to Pinecone
    print("Connecting to Pinecone...")
    embeddings = OpenAIEmbeddings()
    vector_store = PineconeVectorStore(
        index_name=os.getenv("PINECONE_INDEX_NAME"),
        embedding=embeddings,
    )

    ingestor = VisionIngestor(vector_store, verbose=True)

    if args.file:
        result = ingestor.ingest_pdf(
            args.file,
            source_type=args.type,
            delay_between_pages=args.delay,
        )
        print(f"\nResult: {result.pages_processed} pages ingested, {result.chunks_added} chunks added")
        if result.errors:
            print("Errors:")
            for err in result.errors:
                print(f"  - {err}")

    elif args.dir:
        results = ingestor.ingest_directory(
            args.dir,
            source_type=args.type,
            delay_between_pages=args.delay,
        )


if __name__ == "__main__":
    main()
