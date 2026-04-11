"""Main Ingestion Script"""
import argparse
import os
import time
from datetime import datetime

from app.rag.chunker import ChunkingStrategy
from app.rag.extractor import FinancialExtractor
from app.rag.loaders import PDFLoader, HTMLLoader
from app.rag.embedder import Embedder
from app.rag.chroma_store import ChromaStore

def main():
    parser = argparse.ArgumentParser(description="Ingest Financial Documents into Vector DB")
    parser.add_argument("--file", required=True, help="Path to PDF/HTML or URL of document")
    parser.add_argument("--ticker", help="Stock ticker (e.g. AAPL)")
    parser.add_argument("--doc-type", dest="doc_type", help="Document type (e.g. 10-K, 10-Q)")
    parser.add_argument("--fiscal-year", dest="fiscal_year", help="Fiscal year (e.g. FY2023)")
    parser.add_argument("--force", action="store_true", help="Force re-ingestion if already exists")
    args = parser.parse_args()

    start_time = time.time()
    
    if not os.path.exists(args.file) and not args.file.startswith("http"):
        print(f"Error: File {args.file} not found.")
        return

    # Extract metadata
    extractor = FinancialExtractor()
    meta = extractor.extract_metadata_from_filename(args.file)
    
    ticker = args.ticker or meta["ticker"] or "UNKNOWN"
    doc_type = args.doc_type or meta["doc_type"] or "UNKNOWN"
    fiscal_year = args.fiscal_year or meta["fiscal_year"] or "UNKNOWN"
    filing_date = meta["filing_date"]

    print(f"Ingesting: {ticker} {doc_type} {fiscal_year}")

    chroma_store = ChromaStore()
    
    if chroma_store.document_exists(ticker, doc_type, fiscal_year) and not args.force:
        print("Document already ingested. Use --force to re-ingest.")
        return

    embedder = Embedder()
    chunker = ChunkingStrategy()

    print(f"Loading {args.file}...")
    if args.file.endswith(".pdf"):
        pages = PDFLoader().load(args.file)
    elif args.file.lower().endswith(('.html', '.htm', '.txt')) or args.file.startswith("http"):
        pages = HTMLLoader().load(args.file)
    else:
        print("Unsupported file extension. Proceeding with HTML Loader as fallback.")
        pages = HTMLLoader().load(args.file)

    chunks = []
    chunk_index = 0
    total_words = 0
    table_count = 0
    prose_count = 0

    print("Chunking and extracting...")
    for idx, page in enumerate(pages):
        text = extractor.clean_text(page.get("text", ""))
        total_words += page.get("word_count", 0)
        
        # Process tables
        for table in page.get("tables", []):
            for t_chunk in chunker.chunk_table(table):
                chunks.append({
                    "id": f"{ticker}_{doc_type}_{fiscal_year}_chunk_{chunk_index:04d}",
                    "text": t_chunk,
                    "metadata": {
                        "ticker": ticker,
                        "doc_type": doc_type,
                        "fiscal_year": fiscal_year,
                        "filing_date": filing_date or "",
                        "source_file": os.path.basename(args.file),
                        "chunk_type": "table",
                        "chunk_index": chunk_index,
                        "section": page.get("section_title", ""),
                        "word_count": len(t_chunk.split()),
                        "ingested_at": datetime.utcnow().isoformat()
                    }
                })
                chunk_index += 1
                table_count += 1
                
        # Process prose
        for p_chunk in chunker.chunk_prose(text):
            ctype = chunker.detect_chunk_type(p_chunk)
            chunks.append({
                "id": f"{ticker}_{doc_type}_{fiscal_year}_chunk_{chunk_index:04d}",
                "text": p_chunk,
                "metadata": {
                    "ticker": ticker,
                    "doc_type": doc_type,
                    "fiscal_year": fiscal_year,
                    "filing_date": filing_date or "",
                    "source_file": os.path.basename(args.file),
                    "chunk_type": ctype,
                    "chunk_index": chunk_index,
                    "section": page.get("section_title", ""),
                    "word_count": len(p_chunk.split()),
                    "ingested_at": datetime.utcnow().isoformat()
                }
            })
            chunk_index += 1
            prose_count += 1

    if not chunks:
        print("No chunks created. Aborting.")
        return

    print(f"Embedding and storing {len(chunks)} chunks...")
    embeddings = embedder.embed([c["text"] for c in chunks])
    added = chroma_store.add_chunks(chunks, embeddings)

    if total_words > 5000:
        print("Document is large. Generating hierarchical summaries...")
        summary_chunk = chunker.chunk_hierarchical(" ".join([c["text"] for c in chunks]), f"{ticker} {doc_type}")
        s_id = f"{ticker}_{doc_type}_{fiscal_year}_summary"
        s_meta = {
            "ticker": ticker, 
            "doc_type": doc_type, 
            "fiscal_year": fiscal_year, 
            "ingested_at": datetime.utcnow().isoformat()
        }
        s_emb = embedder.embed_single(summary_chunk["summary"])
        chroma_store.add_summaries([{"id": s_id, "text": summary_chunk["summary"], "metadata": s_meta}], [s_emb])

    elapsed = time.time() - start_time
    print(f"\n✓ Ingested {args.file}")
    print(f"  Chunks created: {added}")
    print(f"  Tables: {table_count} | Prose: {prose_count}")
    print(f"  Total words: {total_words}")
    print(f"  Time: {elapsed:.1f}s")

if __name__ == "__main__":
    main()