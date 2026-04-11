"""ChromaDB Storage"""
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
        if not chunks:
            return 0
            
        # Check against existing IDs
        existing = set(self.main_collection.get(ids=[c["id"] for c in chunks])['ids'])
        
        new_chunks = []
        new_embeddings = []
        for i, c in enumerate(chunks):
            if c["id"] not in existing:
                new_chunks.append(c)
                new_embeddings.append(embeddings[i])
        
        if not new_chunks:
            return 0
            
        self.main_collection.add(
            ids=[c["id"] for c in new_chunks],
            embeddings=new_embeddings,
            documents=[c["text"] for c in new_chunks],
            metadatas=[c["metadata"] for c in new_chunks]
        )
        return len(new_chunks)

    def add_summaries(self, summaries: List[Dict], embeddings: List[List[float]]) -> int:
        if not summaries:
            return 0
            
        existing = set(self.summary_collection.get(ids=[s["id"] for s in summaries])['ids'])
        
        new_sums = []
        new_embs = []
        for i, s in enumerate(summaries):
            if s["id"] not in existing:
                new_sums.append(s)
                new_embs.append(embeddings[i])
                
        if not new_sums:
            return 0
            
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
        if not ticker or not doc_type or not fiscal_year:
            return False
            
        try:
            res = self.main_collection.get(
                where={"$and": [{"ticker": ticker}, {"doc_type": doc_type}, {"fiscal_year": fiscal_year}]}, 
                limit=1
            )
            return len(res['ids']) > 0
        except Exception:
            return False