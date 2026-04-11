"""BM25 Sparse Index"""
import re
from typing import List
from rank_bm25 import BM25Okapi
from app.rag.chroma_store import ChromaStore

class BM25Index:
    def __init__(self):
        self.index = None
        self.chunk_ids = []
        self.corpus_texts = []

    def build(self, chroma_store: ChromaStore) -> None:
        results = chroma_store.main_collection.get(include=["documents", "metadatas"])
        self.chunk_ids = results["ids"]
        self.corpus_texts = results["documents"]
        
        tokenized_corpus = []
        for text in self.corpus_texts:
            tokenized_corpus.append(re.findall(r'\b\w+\b', text.lower()))
            
        self.index = BM25Okapi(tokenized_corpus)
        print(f"BM25 index built — {len(self.chunk_ids)} documents indexed")

    def search(self, query: str, top_k: int = 10) -> List[dict]:
        if not self.is_built():
            return []
            
        tokenized_query = re.findall(r'\b\w+\b', query.lower())
        scores = self.index.get_scores(tokenized_query)
        
        # Track items with their score and index
        scored_indices = [(i, float(score)) for i, score in enumerate(scores) if score > 0]
        scored_indices.sort(key=lambda x: x[1], reverse=True)
        
        results = []
        for rank, (i, score) in enumerate(scored_indices[:top_k], start=1):
            results.append({
                "chunk_id": self.chunk_ids[i],
                "bm25_score": score,
                "rank": rank
            })
            
        return results

    def is_built(self) -> bool:
        return self.index is not None