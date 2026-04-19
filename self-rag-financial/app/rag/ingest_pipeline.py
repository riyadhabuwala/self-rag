"""
Ingestion pipeline  callable from both CLI (ingest.py) and API (/ingest).
"""
import os
import logging
import time
from datetime import datetime
from app.rag.loaders import PDFLoader, HTMLLoader
from app.rag.chunker import ChunkingStrategy
from app.rag.extractor import FinancialExtractor
from app.rag.embedder import Embedder
from app.rag.chroma_store import ChromaStore

logger = logging.getLogger(__name__)

# Module-level singletons  initialized once on first call
_embedder = None
_store = None
_extractor = None
_chunker = None

def _get_singletons():
    global _embedder, _store, _extractor, _chunker
    if _embedder is None:
        _embedder = Embedder()
        _store = ChromaStore()
        _extractor = FinancialExtractor()
        _chunker = ChunkingStrategy()
    return _embedder, _store, _extractor, _chunker

def run_ingestion(
    file_path: str,
    ticker: str = None,
    doc_type: str = None,
    fiscal_year: str = None,
    force: bool = False
) -> dict:
    """
    Ingest a document into ChromaDB.
    Returns dict: {chunks_created, tables, prose, total_words,
                   elapsed_seconds, document_id, skipped}
    Raises ValueError on invalid input.
    Raises FileNotFoundError if file does not exist.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    ext = os.path.splitext(file_path)[1].lower()
    if ext not in ['.pdf', '.html', '.htm']:
        raise ValueError(f"Unsupported file type: {ext}. Use .pdf or .html")

    embedder, store, extractor, chunker = _get_singletons()
    start = time.time()

    # Metadata resolution (filename ? CLI/API args)
    filename_meta = extractor.extract_metadata_from_filename(
        os.path.basename(file_path)
    )
    ticker = ticker or filename_meta.get("ticker") or "UNKNOWN"
    doc_type = doc_type or filename_meta.get("doc_type") or "unknown"
    fiscal_year = fiscal_year or filename_meta.get("fiscal_year") or "unknown"

    # Duplicate check
    if not force and store.document_exists(ticker, doc_type, fiscal_year):
        return {
            "skipped": True,
            "message": f"{ticker} {doc_type} {fiscal_year} already ingested. "
                       "Use force=True to re-ingest.",
            "chunks_created": 0
        }

    # Load document
    if ext == '.pdf':
        pages = PDFLoader().load(file_path)
    else:
        pages = HTMLLoader().load(file_path)

    # Extract text-level metadata from first page
    first_text = pages[0]["text"] if pages else ""
    text_meta = extractor.extract_metadata_from_text(first_text)
    filing_date = text_meta.get("filing_date", "")

    # Build chunks
    all_chunks = []
    table_count = 0
    prose_count = 0
    total_words = 0
    chunk_index = 0

    for page in pages:
        page_num = page.get("page_number", 0)
        section = page.get("section_title", "")

        # Process tables
        for table_text in page.get("tables", []):
            clean = extractor.clean_text(table_text)
            if not clean.strip():
                continue
            for chunk_text in chunker.chunk_table(clean):
                all_chunks.append(_build_chunk(
                    chunk_text, chunk_index, "table",
                    ticker, doc_type, fiscal_year, filing_date,
                    file_path, page_num, section
                ))
                chunk_index += 1
                table_count += 1
                total_words += len(chunk_text.split())

        # Process prose
        clean_prose = extractor.clean_text(page.get("text", ""))
        if not clean_prose.strip():
            continue
        for chunk_text in chunker.chunk_prose(clean_prose):
            all_chunks.append(_build_chunk(
                chunk_text, chunk_index, "prose",
                ticker, doc_type, fiscal_year, filing_date,
                file_path, page_num, section
            ))
            chunk_index += 1
            prose_count += 1
            total_words += len(chunk_text.split())

    if not all_chunks:
        raise ValueError("No chunks produced  document may be empty or unreadable")

    # Embed and store
    logger.info(f"Embedding {len(all_chunks)} chunks...")
    texts = [c["text"] for c in all_chunks]
    embeddings = embedder.embed(texts)
    added = store.add_chunks(all_chunks, embeddings)

    elapsed = time.time() - start
    document_id = f"{ticker}_{doc_type}_{fiscal_year}"

    return {
        "skipped": False,
        "chunks_created": added,
        "tables": table_count,
        "prose": prose_count,
        "total_words": total_words,
        "elapsed_seconds": round(elapsed, 1),
        "document_id": document_id,
        "message": f"Successfully ingested {document_id}"
    }

def _build_chunk(text, index, chunk_type, ticker, doc_type,
                 fiscal_year, filing_date, file_path, page_num, section):
    safe_ticker = (ticker or "unknown").replace("/", "_")
    safe_doc = (doc_type or "unknown").replace("/", "_")
    safe_year = (fiscal_year or "unknown").replace("/", "_")
    
    metadata = {
        "ticker": ticker or "unknown",
        "company_name": "unknown",
        "doc_type": doc_type or "unknown",
        "fiscal_year": fiscal_year or "unknown",
        "filing_date": filing_date or "unknown",
        "source_file": os.path.basename(file_path) if file_path else "unknown",
        "page_number": page_num or 0,
        "chunk_type": chunk_type or "unknown",
        "chunk_index": index or 0,
        "section": section or "unknown",
        "word_count": len(text.split()),
        "ingested_at": datetime.utcnow().isoformat()
    }
    
    # ChromaDB rejects None values in metadata. Filter them out explicitly just in case.
    metadata = {k: v for k, v in metadata.items() if v is not None}
    
    return {
        "id": f"{safe_ticker}_{safe_doc}_{safe_year}_chunk_{index:04d}",
        "text": text,
        "metadata": metadata
    }
