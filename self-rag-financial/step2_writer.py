import os
import time

files = {
    'app/rag/chunker.py': '''"""Adaptive Chunking Module."""
import re
from typing import List, Dict

class ChunkingStrategy:
    def chunk_prose(self, text: str, chunk_size: int = 400, chunk_overlap: int = 50) -> List[str]:
        sentences = re.split(r'(?<=[.?!])\s+', text)
        chunks, current_chunk, current_word_count = [], [], 0
        for sentence in sentences:
            words = sentence.split()
            word_count = len(words)
            if current_word_count + word_count > chunk_size and current_chunk:
                chunk_str = " ".join(current_chunk)
                if len(chunk_str.split()) >= 30:
                    chunks.append(chunk_str)
                overlap_words, overlap_count = [], 0
                for s in reversed(current_chunk):
                    s_words = s.split()
                    if overlap_count + len(s_words) <= chunk_overlap:
                        overlap_words.insert(0, s)
                        overlap_count += len(s_words)
                    else:
                        break
                current_chunk = overlap_words + [sentence]
                current_word_count = sum(len(s.split()) for s in current_chunk)
            else:
                current_chunk.append(sentence)
                current_word_count += word_count
        if current_chunk:
            chunk_str = " ".join(current_chunk)
            if len(chunk_str.split()) >= 30:
                chunks.append(chunk_str)
        return chunks

    def chunk_table(self, table_text: str) -> List[str]:
        words = table_text.split()
        if len(words) <= 800:
            return ["TABLE: " + table_text]
        lines = table_text.split('\n')
        if not lines:
            return []
        header = lines[0]
        chunks, current_chunk, current_word_count = [], [header], len(header.split())
        for line in lines[1:]:
            line_word_count = len(line.split())
            if current_word_count + line_word_count > 800:
                chunks.append("TABLE: " + "\n".join(current_chunk))
                current_chunk = [header, line]
                current_word_count = len(header.split()) + line_word_count
            else:
                current_chunk.append(line)
                current_word_count += line_word_count
        if len(current_chunk) > 1:
            chunks.append("TABLE: " + "\n".join(current_chunk))
        return chunks

    def chunk_hierarchical(self, text: str, doc_title: str) -> dict:
        words = text.split()
        return {"summary": " ".join(words[:200]), "detail_chunks": self.chunk_prose(text)}

    def detect_chunk_type(self, text: str) -> str:
        lines = text.strip().split('\n')
        if "|" in text or any(re.match(r'^\s*\d', line) for line in lines):
            return "table"
        if all(line.startswith(' ') or line.startswith('\t') for line in lines if line.strip()):
            return "code"
        if len(text.split()) < 15 and not re.search(r'[.?!]$', text.strip()):
            return "header"
        return "prose"
''',

    'app/rag/extractor.py': '''"""Financial Entity and Metadata Extractor."""
import re

class FinancialExtractor:
    def extract_metadata_from_filename(self, filename: str) -> dict:
        ticker = re.search(r'([A-Z]{1,5})', filename)
        doc_type = re.search(r'(10-K|10-Q|earnings-transcript|annual-report)', filename, re.IGNORECASE)
        fiscal_year = re.search(r'(FY20\d{2}|Q[1-4]-20\d{2})', filename, re.IGNORECASE)
        filing_date = re.search(r'\d{4}-\d{2}-\d{2}', filename)
        return {
            "ticker": ticker.group(1) if ticker else None,
            "doc_type": doc_type.group(1).upper() if doc_type else None,
            "fiscal_year": fiscal_year.group(1).upper() if fiscal_year else None,
            "filing_date": filing_date.group(0) if filing_date else None
        }

    def extract_financial_entities(self, text: str) -> dict:
        return {
            "tickers": re.findall(r'\$[A-Z]{1,5}\b', text),
            "amounts": re.findall(r'\$\d+(?:\.\d+)?(?:B|M| billion| million|,\d{3})?', text),
            "dates": re.findall(r'\b(?:Q[1-4] 20\d{2}|FY20\d{2}|fiscal year 20\d{2}|H[12] 20\d{2})\b', text, re.IGNORECASE),
            "metrics": [m for m in ["revenue", "net income", "EPS", "EBITDA", "gross margin", "operating income", "free cash flow", "ROE", "ROA", "P/E ratio", "market cap", "dividend yield", "Tier 1 Capital", "Basel III", "leverage ratio", "liquidity ratio"] if m.lower() in text.lower()],
            "terms": [t for t in ["SEC", "GAAP", "IFRS", "10-K", "10-Q", "8-K", "proxy statement", "material weakness", "going concern", "restatement", "insider trading"] if t.lower() in text.lower()]
        }

    def clean_text(self, text: str) -> str:
        text = re.sub(r' {3,}', ' ', text)
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = text.replace('\x00', '')
        lines = text.split('\n')
        cleaned = [l for l in lines if not re.match(r'^(Page \d+ of \d+|EDGAR Filing|.*Copyright.*|https?://\S+)$', l.strip(), re.IGNORECASE)]
        return '\n'.join(cleaned)
''',

    'app/rag/loaders.py': '''"""Document Loaders"""
import pdfplumber
from bs4 import BeautifulSoup
import requests
from typing import List, Dict

class PDFLoader:
    def load(self, file_path: str) -> List[Dict]:
        pages = []
        try:
            with pdfplumber.open(file_path) as pdf:
                for i, page in enumerate(pdf.pages):
                    text = page.extract_text()
                    if text is None: continue
                    tables = ["\n".join(" | ".join(str(c) for c in row if c is not None) for row in table) for table in (page.extract_tables() or [])]
                    word_count = len(text.split())
                    pages.append({
                        "page_number": i + 1, "text": text, "tables": tables,
                        "is_table_heavy": (len(" ".join(tables)) > 0.4 * len(text)), "word_count": word_count
                    })
        except Exception as e:
            print(f"Warning: Error loading PDF {file_path}: {e}")
        return pages

class HTMLLoader:
    def load(self, file_path_or_url: str) -> List[Dict]:
        html = ""
        try:
            if file_path_or_url.startswith("http"):
                html = requests.get(file_path_or_url, headers={'User-Agent': 'Mozilla/5.0'}).text
            else:
                with open(file_path_or_url, "r", encoding="utf-8") as f:
                    html = f.read()
        except Exception as e:
            print(f"Error loading HTML {file_path_or_url}: {e}"); return []
        soup = BeautifulSoup(html, 'lxml')
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()
        for elem in soup.find_all(class_=lambda c: c and any(x in c for x in ["header", "footer", "nav", "sidebar", "breadcrumb"])):
            elem.decompose()
        main_content = soup.find(id="filing-content") or soup.find(class_="body") or soup.find("body")
        if not main_content: main_content = soup
        sections = []
        for tag in main_content.find_all(['h1', 'h2', 'h3']):
            content = []
            for nxt in tag.find_all_next():
                if nxt.name in ['h1', 'h2', 'h3']: break
                content.append(nxt.get_text(separator=" ", strip=True))
            text = " ".join(content)
            sections.append({
                "section_title": tag.get_text(strip=True), "text": text,
                "page_number": None, "tables": [], "is_table_heavy": False, "word_count": len(text.split())
            })
        return sections
''',

    'app/rag/embedder.py': '''"""Embedding Generator"""
from sentence_transformers import SentenceTransformer
from app.config import settings
from typing import List

class Embedder:
    def __init__(self, model_name: str = settings.EMBEDDING_MODEL):
        print(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)

    def embed(self, texts: List[str]) -> List[List[float]]:
        return self.model.encode(texts, batch_size=32, show_progress_bar=False).tolist()

    def embed_single(self, text: str) -> List[float]:
        return self.embed([text])[0]
''',

    'app/rag/chroma_store.py': '''"""ChromaDB Storage"""
import chromadb
from app.config import settings
from typing import List, Dict

class ChromaStore:
    def __init__(self):
        self.client = chromadb.PersistentClient(path=settings.CHROMA_PERSIST_DIR)
        self.main_collection = self.client.get_or_create_collection(settings.CHROMA_COLLECTION_NAME)
        self.summary_collection = self.client.get_or_create_collection(settings.CHROMA_SUMMARY_COLLECTION)
        print(f"ChromaDB initialized — {self.main_collection.count()} documents in main collection")

    def add_chunks(self, chunks: List[Dict], embeddings: List[List[float]]) -> int:
        if not chunks: return 0
        existing = set(self.main_collection.get(ids=[c["id"] for c in chunks])['ids'])
        new_chunks = []
        new_embeddings = []
        for i, c in enumerate(chunks):
            if c["id"] not in existing:
                new_chunks.append(c)
                new_embeddings.append(embeddings[i])
        
        if not new_chunks: return 0
        self.main_collection.add(
            ids=[c["id"] for c in new_chunks],
            embeddings=new_embeddings,
            documents=[c["text"] for c in new_chunks],
            metadatas=[c["metadata"] for c in new_chunks]
        )
        return len(new_chunks)

    def add_summaries(self, summaries: List[Dict], embeddings: List[List[float]]) -> int:
        if not summaries: return 0
        existing = set(self.summary_collection.get(ids=[s["id"] for s in summaries])['ids'])
        new_sums = []
        new_embs = []
        for i, s in enumerate(summaries):
            if s["id"] not in existing:
                new_sums.append(s)
                new_embs.append(embeddings[i])
                
        if not new_sums: return 0
        self.summary_collection.add(
            ids=[s["id"] for s in new_sums],
            embeddings=new_embs,
            documents=[s["text"] for s in new_sums],
            metadatas=[s["metadata"] for s in new_sums]
        )
        return len(new_sums)

    def get_doc_count(self) -> int:
        return self.main_collection.count()

    def document_exists(self, ticker: str, doc_type: str, fiscal_year: str) -> bool:
        if not ticker or not doc_type or not fiscal_year: return False
        try:
            res = self.main_collection.get(where={"$and": [{"ticker": ticker}, {"doc_type": doc_type}, {"fiscal_year": fiscal_year}]}, limit=1)
            return len(res['ids']) > 0
        except Exception:
            return False
''',

    'ingest.py': '''"""Main Ingestion Script"""
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", required=True)
    parser.add_argument("--ticker")
    parser.add_argument("--doc-type", dest="doc_type")
    parser.add_argument("--fiscal-year", dest="fiscal_year")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    start_time = time.time()
    if not os.path.exists(args.file) and not args.file.startswith("http"):
        print(f"Error: File {args.file} not found."); return

    extractor = FinancialExtractor()
    meta = extractor.extract_metadata_from_filename(args.file)
    ticker = args.ticker or meta["ticker"] or "UNKNOWN"
    doc_type = args.doc_type or meta["doc_type"] or "UNKNOWN"
    fiscal_year = args.fiscal_year or meta["fiscal_year"] or "UNKNOWN"
    filing_date = meta["filing_date"]

    chroma_store = ChromaStore()
    if chroma_store.document_exists(ticker, doc_type, fiscal_year) and not args.force:
        print("Already ingested. Use --force to re-ingest."); return

    embedder = Embedder()
    chunker = ChunkingStrategy()

    print(f"Loading {args.file}...")
    if args.file.endswith(".pdf"):
        pages = PDFLoader().load(args.file)
    elif args.file.lower().endswith(('.html', '.htm', '.txt')) or args.file.startswith("http"):
        pages = HTMLLoader().load(args.file)
    else:
        raise ValueError("Unsupported file extension")

    chunks = []
    chunk_index = 0
    total_words = 0
    table_count = 0
    prose_count = 0

    for idx, page in enumerate(pages):
        text = extractor.clean_text(page["text"])
        total_words += page.get("word_count", 0)
        
        for table in page.get("tables", []):
            for t_chunk in chunker.chunk_table(table):
                chunks.append({"id": f"{ticker}_{doc_type}_{fiscal_year}_chunk_{chunk_index:04d}", "text": t_chunk,
                               "metadata": {"ticker": ticker, "company_name": "", "doc_type": doc_type,
                                              "fiscal_year": fiscal_year, "filing_date": filing_date or "",
                                              "source_file": os.path.basename(args.file), "page_number": page.get("page_number") or 0,
                                              "chunk_type": "table", "chunk_index": chunk_index,
                                              "section": page.get("section_title", ""), "word_count": len(t_chunk.split()),
                                              "ingested_at": datetime.utcnow().isoformat()}})
                chunk_index += 1
                table_count += 1
                
        for p_chunk in chunker.chunk_prose(text):
            ctype = chunker.detect_chunk_type(p_chunk)
            chunks.append({"id": f"{ticker}_{doc_type}_{fiscal_year}_chunk_{chunk_index:04d}", "text": p_chunk,
                           "metadata": {"ticker": ticker, "company_name": "", "doc_type": doc_type,
                                          "fiscal_year": fiscal_year, "filing_date": filing_date or "",
                                          "source_file": os.path.basename(args.file), "page_number": page.get("page_number") or 0,
                                          "chunk_type": ctype, "chunk_index": chunk_index,
                                          "section": page.get("section_title", ""), "word_count": len(p_chunk.split()),
                                          "ingested_at": datetime.utcnow().isoformat()}})
            chunk_index += 1
            prose_count += 1

    if not chunks:
        print("No chunks created."); return

    print(f"Embedding {len(chunks)} chunks...")
    embeddings = embedder.embed([c["text"] for c in chunks])
    added = chroma_store.add_chunks(chunks, embeddings)

    if total_words > 5000:
        print("Document is large. Generating hierarchical summaries...")
        summary_chunk = chunker.chunk_hierarchical(" ".join([c["text"] for c in chunks]), f"{ticker} {doc_type}")
        s_id = f"{ticker}_{doc_type}_{fiscal_year}_summary"
        s_meta = {"ticker": ticker, "doc_type": doc_type, "fiscal_year": fiscal_year, "ingested_at": datetime.utcnow().isoformat()}
        s_emb = embedder.embed_single(summary_chunk["summary"])
        chroma_store.add_summaries([{"id": s_id, "text": summary_chunk["summary"], "metadata": s_meta}], [s_emb])


    elapsed = time.time() - start_time
    print(f"\\n✓ Ingested {args.file}")
    print(f"  Chunks created: {added}")
    print(f"  Tables: {table_count} | Prose: {prose_count}")
    print(f"  Total words: {total_words}")
    print(f"  Time: {elapsed:.1f}s")

if __name__ == "__main__":
    main()
'''
}

os.chdir(r'D:\c drive\self-rag\self-rag-financial')
for path, content in files.items():
    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)
print('Files written successfully.')