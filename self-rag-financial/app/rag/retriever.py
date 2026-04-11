"""Hybrid Retriever"""
import json
import logging
from typing import List
from langchain_groq import ChatGroq
from app.config import settings
from app.rag.embedder import Embedder
from app.rag.chroma_store import ChromaStore
from app.rag.bm25_index import BM25Index
from app.rag.retriever_utils import build_chroma_filter, deduplicate_by_chunk_id

logger = logging.getLogger(__name__)

class HybridRetriever:
    def __init__(self, embedder: Embedder, chroma_store: ChromaStore, bm25_index: BM25Index):
        self.embedder = embedder
        self.chroma_store = chroma_store
        self.bm25_index = bm25_index
        self.groq_client = ChatGroq(
            model=settings.GROQ_GRADING_MODEL,
            temperature=0.3,
            api_key=settings.GROQ_API_KEY
        )

    def _dense_search(self, query: str, top_k: int, filters: dict = None) -> List[dict]:
        query_vector = self.embedder.embed_single(query)
        where_clause = build_chroma_filter(filters)
        
        kwargs = {
            "query_embeddings": [query_vector],
            "n_results": top_k,
            "include": ["documents", "metadatas", "distances"]
        }
        if where_clause:
            kwargs["where"] = where_clause
            
        res = self.chroma_store.main_collection.query(**kwargs)
        
        results = []
        if not res.get("ids", []) or not res["ids"][0]:
            return results
            
        ids = res["ids"][0]
        docs = res["documents"][0]
        metas = res["metadatas"][0]
        dists = res["distances"][0]
        
        for rank, (cid, text, meta, dist) in enumerate(zip(ids, docs, metas, dists), start=1):
            results.append({
                "chunk_id": cid,
                "text": text,
                "metadata": meta,
                "distance": float(dist),
                "rank": rank
            })
            
        return results

    def _bm25_search(self, query: str, top_k: int, filters: dict = None) -> List[dict]:
        raw_results = self.bm25_index.search(query, top_k=top_k * 3)
        if not raw_results:
            return []
            
        if filters:
            where_clause = build_chroma_filter(filters)
            if where_clause:
                valid_ids_res = self.chroma_store.main_collection.get(
                    ids=[r["chunk_id"] for r in raw_results],
                    where=where_clause,
                    include=[]
                )
                valid_set = set(valid_ids_res.get("ids", []))
                raw_results = [r for r in raw_results if r["chunk_id"] in valid_set]
                
        # Re-rank strictly sequentially after filtered extraction
        results = []
        for rank, r in enumerate(raw_results[:top_k], start=1):
            results.append({
                "chunk_id": r["chunk_id"],
                "bm25_score": r["bm25_score"],
                "rank": rank
            })
            
        return results

    def _reciprocal_rank_fusion(self, dense_results: List[dict], bm25_results: List[dict]) -> List[dict]:
        k = settings.RRF_K
        
        dense_map = {r["chunk_id"]: r for r in dense_results}
        bm25_map = {r["chunk_id"]: r for r in bm25_results}
        
        all_ids = set(dense_map.keys()).union(set(bm25_map.keys()))
        
        fused = []
        bm25_only_ids = []
        
        for cid in all_ids:
            dense_rank = dense_map[cid]["rank"] if cid in dense_map else len(dense_results) + 1
            bm25_rank = bm25_map[cid]["rank"] if cid in bm25_map else len(bm25_results) + 1
            
            rrf_score = float(1.0 / (dense_rank + k)) + float(1.0 / (bm25_rank + k))
            
            item = {
                "chunk_id": cid,
                "rrf_score": rrf_score,
                "dense_rank": dense_rank,
                "bm25_rank": bm25_rank
            }
            if cid in dense_map:
                item["text"] = dense_map[cid]["text"]
                item["metadata"] = dense_map[cid]["metadata"]
            else:
                bm25_only_ids.append(cid)
            
            fused.append(item)
            
        if bm25_only_ids:
            fetched = self.chroma_store.fetch_by_ids(bm25_only_ids)
            fetched_map = {f["chunk_id"]: f for f in fetched}
            for item in fused:
                if item["chunk_id"] in bm25_only_ids:
                    f_data = fetched_map.get(item["chunk_id"], {})
                    item["text"] = f_data.get("text", "")
                    item["metadata"] = f_data.get("metadata", {})
                    
        fused.sort(key=lambda x: x["rrf_score"], reverse=True)
        return fused

    def _expand_query(self, query: str) -> List[str]:
        system_prompt = "You are a financial document search assistant. Generate query variants to improve document retrieval. Return ONLY a JSON array of strings. No explanation, no markdown, no preamble."
        user_prompt = f"""Generate {settings.MULTI_QUERY_COUNT} semantically diverse reformulations of this financial query. Each variant should use different terminology but seek the same information. Focus on financial document language that would appear in SEC filings.

Original query: {query}

Return ONLY a JSON array. Example format:
["variant 1", "variant 2", "variant 3"]"""
        
        messages = [
            ("system", system_prompt),
            ("human", user_prompt)
        ]
        
        try:
            response = self.groq_client.invoke(messages)
            content = response.content.strip()
            
            if content.startswith("```json"):
                content = content[7:]
            if content.startswith("```"):
                content = content[3:]
            if content.endswith("```"):
                content = content[:-3]
                
            content = content.strip()
            
            variants = json.loads(content)
            if not isinstance(variants, list):
                variants = []
                
        except Exception as e:
            logger.warning(f"Query expansion failed: {e}")
            variants = []
            
        final_queries = [query]
        for v in variants:
            if v and isinstance(v, str) and v != query and v not in final_queries:
                final_queries.append(v)
                
        return final_queries

    def retrieve(self, query: str, top_k: int = None, filters: dict = None, use_multi_query: bool = True) -> List[dict]:
        top_k = top_k or settings.TOP_K_FINAL
        queries = self._expand_query(query) if use_multi_query else [query]
        
        all_dense = []
        all_bm25 = []
        
        for q in queries:
            d_res = self._dense_search(q, top_k=settings.TOP_K_DENSE, filters=filters)
            b_res = self._bm25_search(q, top_k=settings.TOP_K_BM25, filters=filters)
            
            for r in d_res:
                r["retrieval_query"] = q
            for r in b_res:
                r["retrieval_query"] = q
                
            all_dense.extend(d_res)
            all_bm25.extend(b_res)
            
        all_dense = deduplicate_by_chunk_id(all_dense, keep="best_rank")
        all_bm25 = deduplicate_by_chunk_id(all_bm25, keep="best_rank")
        
        fused = self._reciprocal_rank_fusion(all_dense, all_bm25)
        
        dense_q_map = {r["chunk_id"]: r["retrieval_query"] for r in all_dense}
        bm25_q_map = {r["chunk_id"]: r["retrieval_query"] for r in all_bm25}
        
        results = []
        for r in fused[:top_k]:
            r["retrieval_query"] = dense_q_map.get(r["chunk_id"]) or bm25_q_map.get(r["chunk_id"]) or query
            results.append(r)
            
        return results

    def retrieve_summaries(self, query: str, top_k: int = 3, filters: dict = None) -> List[dict]:
        query_vector = self.embedder.embed_single(query)
        where_clause = build_chroma_filter(filters)
        
        kwargs = {
            "query_embeddings": [query_vector],
            "n_results": top_k,
            "include": ["documents", "metadatas", "distances"]
        }
        if where_clause:
            kwargs["where"] = where_clause
            
        res = self.chroma_store.summary_collection.query(**kwargs)
        
        results = []
        if not res.get("ids", []) or not res["ids"][0]:
            return results
            
        ids = res["ids"][0]
        docs = res["documents"][0]
        metas = res["metadatas"][0]
        dists = res["distances"][0]
        
        for rank, (cid, text, meta, dist) in enumerate(zip(ids, docs, metas, dists), start=1):
            results.append({
                "chunk_id": cid,
                "text": text,
                "metadata": meta,
                "distance": float(dist),
                "rank": rank,
                "retrieval_query": query
            })
            
        return results