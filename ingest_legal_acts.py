# ingest_legal_acts.py
# Ingests vikashkuamrevolve/indian-legal-acts into Pinecone.
#
# Dataset: 2,380 Indian acts (Central + State jurisdictions)
# Fields: Short Title, Act Number, Enactment Date, Entity, Markdown (full text)
# What this adds: State-level legislation missing from the existing 54 statutes.
#
# Usage:
#   python ingest_legal_acts.py              # All jurisdictions, all rows
#   python ingest_legal_acts.py --jurisdiction central   # Central acts only
#   python ingest_legal_acts.py --limit 500              # Cap at 500 rows

import os
import argparse
from dotenv import load_dotenv
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from datasets import load_dataset
from tqdm import tqdm

load_dotenv()

DATASET_ID = "vikashkuamrevolve/indian-legal-acts"
BATCH_SIZE = 50
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
MIN_TEXT_LENGTH = 100


def parse_args():
    parser = argparse.ArgumentParser(description="Ingest Indian Legal Acts into Pinecone")
    parser.add_argument(
        "--jurisdiction",
        default=None,
        help="Filter by jurisdiction split name (e.g. 'central', 'andhra_pradesh'). Default: all."
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max rows to ingest per jurisdiction split. Default: all."
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print("------------------------------------------------")
    print("VAKALAT AI: Indian Legal Acts Ingestion")
    print(f"Dataset : {DATASET_ID}")
    print(f"Filter  : {args.jurisdiction or 'all jurisdictions'}")
    print(f"Limit   : {args.limit or 'no limit'}")
    print("------------------------------------------------")

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
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )

    print(f"\nLoading dataset from HuggingFace: {DATASET_ID}...")
    try:
        # Load all splits as a DatasetDict
        dataset_dict = load_dataset(DATASET_ID)
    except Exception as e:
        print(f"ERROR: Failed to load dataset: {e}")
        return

    available_splits = list(dataset_dict.keys())
    print(f"Available splits (jurisdictions): {available_splits}")

    # Filter to requested jurisdiction if specified
    if args.jurisdiction:
        if args.jurisdiction not in available_splits:
            print(f"ERROR: '{args.jurisdiction}' not in {available_splits}")
            return
        splits_to_ingest = [args.jurisdiction]
    else:
        splits_to_ingest = available_splits

    total_chunks = 0
    total_rows = 0
    skipped = 0

    for split_name in splits_to_ingest:
        split_data = dataset_dict[split_name]
        rows_in_split = len(split_data)
        limit = min(args.limit, rows_in_split) if args.limit else rows_in_split

        print(f"\n[{split_name}] {rows_in_split} rows — ingesting {limit}")

        batch = []

        for i, row in enumerate(tqdm(split_data, total=limit, desc=split_name)):
            if i >= limit:
                break

            markdown_text = row.get("Markdown", "")

            # Skip thin/empty entries
            if not markdown_text or not isinstance(markdown_text, str):
                skipped += 1
                continue
            if len(markdown_text.strip()) < MIN_TEXT_LENGTH:
                skipped += 1
                continue

            short_title = row.get("Short Title", "Unknown Act")
            act_number = row.get("Act Number", "")
            enactment_date = row.get("Enactment Date", "")
            entity = row.get("Entity", split_name)
            view_url = row.get("View", "")

            # Determine jurisdiction type
            jurisdiction_type = "central" if entity.lower() == "central" else "state"

            # Chunk the full act text
            chunks = text_splitter.create_documents([markdown_text.strip()])

            for chunk_idx, chunk in enumerate(chunks):
                chunk.metadata = {
                    "source_type": "statute",
                    "source_book": short_title,
                    "act_number": str(act_number),
                    "enactment_date": str(enactment_date),
                    "jurisdiction": entity,
                    "jurisdiction_type": jurisdiction_type,
                    "source_dataset": DATASET_ID,
                    "source_id": f"{DATASET_ID}_{split_name}_{i}",
                    "chunk_index": chunk_idx,
                    # View URL links back to official indiacode.nic.in PDF
                    "official_source": str(view_url),
                }
                batch.append(chunk)

            total_rows += 1

            # Flush batch to Pinecone
            if len(batch) >= BATCH_SIZE:
                vector_db.add_documents(batch)
                total_chunks += len(batch)
                batch = []

        # Flush remaining
        if batch:
            vector_db.add_documents(batch)
            total_chunks += len(batch)
            batch = []

    print("\n------------------------------------------------")
    print(f"DONE: {total_rows} acts ingested | {total_chunks} chunks added | {skipped} skipped")
    print("------------------------------------------------")


if __name__ == "__main__":
    main()
