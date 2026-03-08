# ingest_sft_data.py
# Ingests Prarabdha/indian-legal-supervised-fine-tuning-data into Pinecone.
#
# Dataset: 6.06M legal Q&A triples {context, question, response}
# What this adds:
#   - Pre-answered legal questions with verified context
#   - Multilingual coverage (Hindi, Tamil, Marathi, Bengali, Kannada, Telugu)
#   - SC + HC judgments + statutory provisions in Q&A form
#
# WHY NOT INGEST ALL 6M ROWS:
#   - Pinecone free tier: 100k vectors max
#   - OpenAI embeddings cost: 6M * ~400 tokens = ~$240 — too expensive
#   - Many rows are duplicates or low-quality short snippets
#   - Smart filtering + dedup gives you the best 50k rows (default)
#
# WHAT WE INGEST:
#   - The `context` field as the document (actual legal text)
#   - `question` and `response` stored in metadata (for display in UI)
#   - Deduplication by context fingerprint (first 120 chars)
#   - Minimum context length filter (300 chars)
#
# Usage:
#   python ingest_sft_data.py                    # Default: 50k rows, all languages
#   python ingest_sft_data.py --limit 20000      # Cap at 20k rows
#   python ingest_sft_data.py --min-length 500   # Only longer contexts
#   python ingest_sft_data.py --lang en           # English only (faster, focused)

import os
import argparse
from dotenv import load_dotenv
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from datasets import load_dataset
from tqdm import tqdm

load_dotenv()

DATASET_ID = "Prarabdha/indian-legal-supervised-fine-tuning-data"
BATCH_SIZE = 50
DEFAULT_LIMIT = 50_000

# Languages present in the dataset
KNOWN_LANGUAGES = {"en", "hi", "ta", "mr", "bn", "kn", "te", "or"}

# Approximate language detection by script character range.
# Not perfect but fast — avoids a full langdetect dependency.
def _detect_lang_approx(text: str) -> str:
    """Approximate language detection based on Unicode script ranges."""
    if not text:
        return "unknown"
    sample = text[:200]
    devanagari = sum(1 for c in sample if '\u0900' <= c <= '\u097F')  # Hindi/Marathi
    tamil = sum(1 for c in sample if '\u0B80' <= c <= '\u0BFF')
    bengali = sum(1 for c in sample if '\u0980' <= c <= '\u09FF')
    kannada = sum(1 for c in sample if '\u0C80' <= c <= '\u0CFF')
    telugu = sum(1 for c in sample if '\u0C00' <= c <= '\u0C7F')

    threshold = len(sample) * 0.15  # 15% non-ASCII chars = non-English

    if devanagari > threshold:
        return "hi"  # Hindi or Marathi — close enough for filtering
    if tamil > threshold:
        return "ta"
    if bengali > threshold:
        return "bn"
    if kannada > threshold:
        return "kn"
    if telugu > threshold:
        return "te"
    return "en"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Ingest Indian Legal SFT Dataset into Pinecone"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_LIMIT,
        help=f"Max rows to ingest after filtering. Default: {DEFAULT_LIMIT:,}"
    )
    parser.add_argument(
        "--min-length",
        type=int,
        default=300,
        help="Minimum context length in characters. Default: 300. Increase to get denser content."
    )
    parser.add_argument(
        "--lang",
        default=None,
        help="Filter by language code: en, hi, ta, mr, bn, kn, te. Default: all languages."
    )
    parser.add_argument(
        "--no-dedup",
        action="store_true",
        help="Disable deduplication (not recommended for 6M dataset)."
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print("------------------------------------------------")
    print("VAKALAT AI: Legal SFT Dataset Ingestion")
    print(f"Dataset   : {DATASET_ID}")
    print(f"Limit     : {args.limit:,} rows after filtering")
    print(f"Min length: {args.min_length} chars")
    print(f"Language  : {args.lang or 'all'}")
    print(f"Dedup     : {'off' if args.no_dedup else 'on'}")
    print("------------------------------------------------")
    print()
    print("NOTE: This dataset is 6M rows. Streaming from HuggingFace.")
    print("      Will stop automatically once limit is reached.")
    print("      Ctrl+C at any time to stop and keep what's ingested.")
    print()

    if args.lang and args.lang not in KNOWN_LANGUAGES:
        print(f"ERROR: Unknown language '{args.lang}'. Choose from: {KNOWN_LANGUAGES}")
        return

    # Validate env
    required = ["OPENAI_API_KEY", "PINECONE_API_KEY", "PINECONE_INDEX_NAME"]
    missing = [k for k in required if not os.getenv(k)]
    if missing:
        print(f"ERROR: Missing env vars: {missing}")
        return

    # Connect Pinecone
    embeddings = OpenAIEmbeddings()
    vector_db = PineconeVectorStore(
        index_name=os.getenv("PINECONE_INDEX_NAME"),
        embedding=embeddings
    )

    print(f"Connecting to HuggingFace (streaming)...")
    try:
        # Stream the dataset — never loads 6M rows into RAM
        dataset = load_dataset(
            DATASET_ID,
            split="train",
            streaming=True,
            trust_remote_code=True
        )
    except Exception as e:
        print(f"ERROR: Failed to load dataset: {e}")
        return

    print("Stream connected. Starting ingestion...\n")

    seen_fingerprints: set[str] = set()
    batch: list[Document] = []
    total_ingested = 0
    total_skipped = 0
    total_deduped = 0
    total_scanned = 0

    progress = tqdm(total=args.limit, desc="Ingesting", unit="rows")

    try:
        for row in dataset:
            total_scanned += 1

            context = row.get("context", "")
            question = row.get("question", "")
            response = row.get("response", "")

            # --- FILTER 1: Context must be a non-empty string ---
            if not context or not isinstance(context, str):
                total_skipped += 1
                continue

            context = context.strip()

            # --- FILTER 2: Minimum length ---
            if len(context) < args.min_length:
                total_skipped += 1
                continue

            # --- FILTER 3: Language ---
            if args.lang:
                detected = _detect_lang_approx(context)
                if detected != args.lang:
                    total_skipped += 1
                    continue

            # --- FILTER 4: Deduplication by fingerprint ---
            if not args.no_dedup:
                fingerprint = context[:120].lower().strip()
                if fingerprint in seen_fingerprints:
                    total_deduped += 1
                    continue
                seen_fingerprints.add(fingerprint)

            # Detect language for metadata
            lang_code = _detect_lang_approx(context)

            # Build document
            # - page_content = context (the legal text — what gets embedded)
            # - metadata stores question + response for display in UI
            doc = Document(
                page_content=context,
                metadata={
                    "source_type": "case_law",  # SFT data is primarily judgment-derived
                    "source_dataset": DATASET_ID,
                    "source_id": f"sft_{total_ingested}",
                    "language": lang_code,
                    # Truncate question/response in metadata (Pinecone metadata limit: 40KB)
                    "question": question[:500] if question else "",
                    "verified_answer": response[:1000] if response else "",
                }
            )

            batch.append(doc)
            total_ingested += 1
            progress.update(1)

            # Flush batch to Pinecone
            if len(batch) >= BATCH_SIZE:
                vector_db.add_documents(batch)
                batch = []

            # Stop once limit reached
            if total_ingested >= args.limit:
                break

    except KeyboardInterrupt:
        print(f"\nStopped by user.")

    finally:
        # Flush remaining
        if batch:
            vector_db.add_documents(batch)

        progress.close()

    print()
    print("------------------------------------------------")
    print(f"DONE")
    print(f"  Scanned  : {total_scanned:,} rows")
    print(f"  Ingested : {total_ingested:,} chunks")
    print(f"  Skipped  : {total_skipped:,} (too short / wrong language)")
    print(f"  Deduped  : {total_deduped:,} (duplicate contexts)")
    print("------------------------------------------------")


if __name__ == "__main__":
    main()
