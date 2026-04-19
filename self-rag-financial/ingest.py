"""Main Ingestion Script"""
import argparse
import os

from app.rag.ingest_pipeline import run_ingestion

def main():
    parser = argparse.ArgumentParser(description="Ingest Financial Documents into Vector DB")
    parser.add_argument("--file", required=True, help="Path to PDF/HTML or URL of document")
    parser.add_argument("--ticker", help="Stock ticker (e.g. AAPL)")
    parser.add_argument("--doc-type", dest="doc_type", help="Document type (e.g. 10-K, 10-Q)")
    parser.add_argument("--fiscal-year", dest="fiscal_year", help="Fiscal year (e.g. FY2023)")
    parser.add_argument("--force", action="store_true", help="Force re-ingestion if already exists")
    args = parser.parse_args()

    try:
        result = run_ingestion(
            file_path=args.file,
            ticker=args.ticker,
            doc_type=args.doc_type,
            fiscal_year=args.fiscal_year,
            force=args.force
        )
        if result.get("skipped"):
            print(result["message"])
        else:
            print(f"? {result['message']}")
            print(f"  Chunks: {result['chunks_created']} | "
                  f"Tables: {result['tables']} | Prose: {result['prose']}")
            print(f"  Words: {result['total_words']:,} | Time: {result['elapsed_seconds']}s")
    except Exception as e:
        print(f"Error during ingestion: {e}")

if __name__ == "__main__":
    main()
