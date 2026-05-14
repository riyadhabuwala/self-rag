"""ChromaDB & Pinecone Storage Wrapper"""
import chromadb
from backend.app.config import settings
from typing import List, Dict

class PineconeCollectionWrapper:
    def __init__(self, index, namespace: str):
        self.index = index
        self.namespace = namespace

    def add(self, ids: list, embeddings: list, documents: list, metadatas: list):
        vectors = []
        for i in range(len(ids)):
            meta = metadatas[i] or {}
            # Pinecone metadata values must be str, num, bool, or list of str
            meta["text"] = documents[i]
            vectors.append({
                "id": ids[i],
                "values": embeddings[i],
                "metadata": meta
            })
        
        # Upsert in batches of 100
        batch_size = 100
        for i in range(0, len(vectors), batch_size):
            self.index.upsert(vectors=vectors[i:i+batch_size], namespace=self.namespace)

    def count(self) -> int:
        stats = self.index.describe_index_stats()
        ns_stats = stats.get("namespaces", {}).get(self.namespace, {})
        return ns_stats.get("vector_count", 0)

    def get(self, ids: list = None, include: list = None, where: dict = None, limit: int = None):
        # Pinecone doesn't support fetching by 'where' without a vector query natively easily except via query
        # But 'fetch' supports IDs.
        if ids:
            res = self.index.fetch(ids=ids, namespace=self.namespace)
            out_ids = []
            out_docs = []
            out_metas = []
            for k, v in res.get("vectors", {}).items():
                out_ids.append(k)
                out_metas.append(v.get("metadata", {}))
                out_docs.append(v.get("metadata", {}).get("text", ""))
            return {"ids": out_ids, "documents": out_docs, "metadatas": out_metas}
            
        if where:
            # Fake query with dummy vector to get matching docs
            dummy_vector = [0.0] * settings.EMBEDDING_DIMENSION
            res = self.index.query(vector=dummy_vector, filter=where, top_k=limit or 10000, include_metadata=True, namespace=self.namespace)
            out_ids = [m["id"] for m in res.get("matches", [])]
            return {"ids": out_ids}
            
        # If neither ids nor where, returning all is not possible in Pinecone (no direct scroll).
        # We will just return empty for safety or attempt a dummy query.
        return {"ids": [], "documents": [], "metadatas": []}

    def query(self, query_embeddings: list, n_results: int, include: list = None, where: dict = None):
        res_ids = []
        res_docs = []
        res_metas = []
        res_dists = []
        
        for q_emb in query_embeddings:
            res = self.index.query(
                vector=q_emb,
                top_k=n_results,
                filter=where,
                include_metadata=True,
                namespace=self.namespace
            )
            
            ids = []
            docs = []
            metas = []
            dists = []
            
            for match in res.get("matches", []):
                ids.append(match["id"])
                # Pinecone returns similarity score (1-distance)
                # Chroma returns distance. So distance = 1 - score
                dists.append(1.0 - match.get("score", 0.0))
                meta = match.get("metadata", {})
                docs.append(meta.get("text", ""))
                
                # Remove text from metadata so it matches chroma
                meta_copy = dict(meta)
                if "text" in meta_copy:
                    del meta_copy["text"]
                metas.append(meta_copy)
                
            res_ids.append(ids)
            res_docs.append(docs)
            res_metas.append(metas)
            res_dists.append(dists)
            
        return {
            "ids": res_ids,
            "documents": res_docs,
            "metadatas": res_metas,
            "distances": res_dists
        }

class ChromaStore:
    def __init__(self):
        self.use_pinecone = bool(settings.PINECONE_API_KEY)
        
        if self.use_pinecone:
            from pinecone import Pinecone
            pc = Pinecone(api_key=settings.PINECONE_API_KEY)
            self.index = pc.Index(settings.PINECONE_INDEX_NAME)
            self.main_collection = PineconeCollectionWrapper(self.index, namespace=settings.CHROMA_COLLECTION_NAME)
            self.summary_collection = PineconeCollectionWrapper(self.index, namespace=settings.CHROMA_SUMMARY_COLLECTION)
            print(f"Pinecone initialized — {self.main_collection.count()} documents in main namespace")
        else:
            self.client = chromadb.PersistentClient(path=settings.CHROMA_PERSIST_DIR)
            self.main_collection = self.client.get_or_create_collection(settings.CHROMA_COLLECTION_NAME)
            self.summary_collection = self.client.get_or_create_collection(settings.CHROMA_SUMMARY_COLLECTION)
            print(f"ChromaDB initialized — {self.main_collection.count()} documents in main collection")

    def add_chunks(self, chunks: List[Dict], embeddings: List[List[float]]) -> int:
        if not chunks:
            return 0
            
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
            where_clause = {"$and": [{"ticker": ticker}, {"doc_type": doc_type}, {"fiscal_year": fiscal_year}]}
            if self.use_pinecone:
                where_clause = {"ticker": ticker, "doc_type": doc_type, "fiscal_year": fiscal_year}
                
            res = self.main_collection.get(where=where_clause, limit=1)
            return len(res['ids']) > 0
        except Exception:
            return False

    def fetch_by_ids(self, chunk_ids: List[str]) -> List[dict]:
        if not chunk_ids:
            return []
            
        res = self.main_collection.get(ids=chunk_ids, include=["documents", "metadatas"])
        
        data_map = {}
        for i, cid in enumerate(res['ids']):
            data_map[cid] = {
                "chunk_id": cid,
                "text": res['documents'][i],
                "metadata": res['metadatas'][i]
            }
            
        final_results = []
        for cid in chunk_ids:
            if cid in data_map:
                final_results.append(data_map[cid])
                
        return final_results