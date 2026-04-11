#!/usr/bin/env python3
"""
Corpus ingestion script — downloads and ingests 3 real SEC filings.
All sources are public (SEC EDGAR) — no API key required.
Run: python scripts/ingest_corpus.py
"""
import os
import time
import requests
import subprocess
import sys

CORPUS = [
    {
        "url": "https://www.sec.gov/Archives/edgar/data/320193/000032019323000106/aapl-20230930.htm",
        "filename": "Apple_10K_FY2023.html",
        "ticker": "AAPL",
        "doc_type": "10-K",
        "fiscal_year": "FY2023",
        "description": "Apple Inc. Annual Report FY2023"
    },
    {
        "url": "https://www.sec.gov/Archives/edgar/data/320193/000032019322000108/aapl-20220924.htm",
        "filename": "Apple_10K_FY2022.html",
        "ticker": "AAPL",
        "doc_type": "10-K",
        "fiscal_year": "FY2022",
        "description": "Apple Inc. Annual Report FY2022"
    },
    {
        "url": "https://www.sec.gov/Archives/edgar/data/789019/000095017023035122/msft-20230630.htm",
        "filename": "Microsoft_10K_FY2023.html",
        "ticker": "MSFT",
        "doc_type": "10-K",
        "fiscal_year": "FY2023",
        "description": "Microsoft Corporation Annual Report FY2023"
    }
]

HEADERS = {"User-Agent": "Research selfrag@example.com"}
DOWNLOAD_DIR = "data/raw"

def download_filing(entry: dict) -> str:
    os.makedirs(DOWNLOAD_DIR, exist_ok=True)
    filepath = os.path.join(DOWNLOAD_DIR, entry["filename"])

    if os.path.exists(filepath):
        print(f"  Already downloaded: {entry['filename']}")
        return filepath

    print(f"  Downloading: {entry['description']}...")
    try:
        r = requests.get(entry["url"], headers=HEADERS, timeout=60)
        r.raise_for_status()
        with open(filepath, "wb") as f:
            f.write(r.content)
        print(f"  Saved: {filepath} ({len(r.content):,} bytes)")
        time.sleep(1)  # SEC EDGAR rate limit courtesy pause
        return filepath
    except Exception as e:
        print(f"  ERROR downloading {entry['filename']}: {e}")
        return None

def ingest_filing(entry: dict, filepath: str) -> bool:
    print(f"  Ingesting: {entry['description']}...")
    cmd = [
        sys.executable, "ingest.py",
        "--file", filepath,
        "--ticker", entry["ticker"],
        "--doc-type", entry["doc_type"],
        "--fiscal-year", entry["fiscal_year"]
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
        print(result.stdout)
        return True
    else:
        print(f"  ERROR: {result.stderr}")
        return False

if __name__ == "__main__":
    print("=== Self-RAG Financial Corpus Ingestion ===\n")
    success_count = 0
    for entry in CORPUS:
        print(f"[{entry['ticker']} {entry['fiscal_year']}]")
        filepath = download_filing(entry)
        if filepath and ingest_filing(entry, filepath):
            success_count += 1
        print()

    print(f"=== Done: {success_count}/{len(CORPUS)} documents ingested ===")

    # Final ChromaDB count
    import sys
    import os
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from app.rag.chroma_store import ChromaStore
    store = ChromaStore()
    print(f"Total chunks in ChromaDB: {store.get_doc_count()}")
