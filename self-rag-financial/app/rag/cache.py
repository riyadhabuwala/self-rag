import json
import logging
import time
import hashlib
from typing import Optional, List
import numpy as np
from upstash_redis import Redis
from app.config import settings
from app.rag.embedder import Embedder
from datetime import datetime

logger = logging.getLogger(__name__)

class SemanticCache:
    def __init__(self, embedder: Embedder, backend: str = None):
        self.embedder = embedder
        self.backend = backend or settings.CACHE_BACKEND
        self.threshold = settings.CACHE_SIMILARITY_THRESHOLD
        self._memory_store: dict = {}
        self._redis: Redis | None = None
        self._redis_available: bool = False

        if self.backend == "upstash":
            try:
                self._redis = Redis(
                    url=settings.UPSTASH_REDIS_REST_URL,
                    token=settings.UPSTASH_REDIS_REST_TOKEN
                )
                self._redis.ping()
                self._redis_available = True
                logger.info("Upstash Redis connected successfully")
            except Exception as e:
                logger.warning(f"Upstash Redis unavailable: {e}. Falling back to in-memory cache.")
                self._redis_available = False
                
        logger.info(f"SemanticCache initialized — backend={self.backend}, redis_available={self._redis_available}")

    def _hash_key(self, text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]

    def _cosine_similarity(self, vec_a: List[float], vec_b: List[float]) -> float:
        a = np.array(vec_a)
        b = np.array(vec_b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    def _determine_ttl(self, response: dict) -> int:
        if response.get("confidence") == "low" or response.get("groundedness") == "no":
            return settings.CACHE_TTL_FALLBACK
        if response.get("cache_hit") is True:
            return settings.CACHE_TTL_STABLE
        
        current_year = str(datetime.utcnow().year)
        for source in response.get("sources", []):
            if current_year in str(source.get("fiscal_year", "")):
                return settings.CACHE_TTL_RECENT
                
        return settings.CACHE_TTL_STABLE

    def _redis_get_all_keys(self) -> List[str]:
        if not self._redis_available:
            return []
        try:
            raw = self._redis.get("cache:index")
            if raw is None:
                return []
            return json.loads(raw)
        except Exception as e:
            logger.warning(f"Redis get keys failed: {e}")
            return []

    def _redis_add_to_index(self, hash_key: str) -> None:
        if not self._redis_available:
            return
        try:
            keys = self._redis_get_all_keys()
            if hash_key not in keys:
                keys.append(hash_key)
                self._redis.set("cache:index", json.dumps(keys))
        except Exception as e:
            logger.warning(f"Redis add to index failed: {e}")

    def get(self, query: str) -> Optional[dict]:
        try:
            query_embedding = self.embedder.embed_single(query)

            if self.backend == "upstash" and self._redis_available:
                keys = self._redis_get_all_keys()
                for key in keys:
                    raw_embedding = self._redis.get(f"cache:query:{key}")
                    if raw_embedding is None:
                        continue
                    
                    stored_embedding = json.loads(raw_embedding)
                    similarity = self._cosine_similarity(query_embedding, stored_embedding)
                    
                    if similarity >= self.threshold:
                        raw_response = self._redis.get(f"cache:response:{key}")
                        if raw_response is None:
                            continue
                        
                        response = json.loads(raw_response)
                        self._redis.incr(f"cache:meta:{key}:hit_count")
                        logger.info(f"[CACHE HIT] Redis similarity={similarity:.4f} key={key}")
                        return response
                return None

            for key, data in self._memory_store.items():
                similarity = self._cosine_similarity(query_embedding, data["embedding"])
                if similarity >= self.threshold:
                    logger.info(f"[CACHE HIT] Memory similarity={similarity:.4f}")
                    return data["response"]
            
            return None
        except Exception as e:
            logger.error(f"Cache get error: {e}")
            return None

    def set(self, query: str, response: dict) -> None:
        try:
            query_embedding = self.embedder.embed_single(query)
            hash_key = self._hash_key(query)
            ttl = self._determine_ttl(response)

            if self.backend == "upstash" and self._redis_available:
                try:
                    self._redis.setex(f"cache:query:{hash_key}", ttl, json.dumps(query_embedding))
                    self._redis.setex(f"cache:response:{hash_key}", ttl, json.dumps(response))
                    self._redis.setex(f"cache:meta:{hash_key}", ttl, json.dumps({
                        "original_query": query,
                        "created_at": time.time(),
                        "hit_count": 0,
                        "ttl": ttl
                    }))
                    self._redis_add_to_index(hash_key)
                    logger.info(f"[CACHE SET] Redis key={hash_key} ttl={ttl}s")
                    return
                except Exception as e:
                    logger.warning(f"Redis set failed: {e}. Storing in memory.")
                    self._redis_available = False

            self._memory_store[hash_key] = {
                "embedding": query_embedding,
                "response": response,
                "meta": {
                    "original_query": query,
                    "created_at": time.time(),
                    "ttl": ttl
                }
            }
            if len(self._memory_store) > 500:
                oldest_key = next(iter(self._memory_store))
                del self._memory_store[oldest_key]
            logger.info(f"[CACHE SET] Memory key={hash_key}")
        except Exception as e:
            logger.error(f"Cache set error: {e}")

    def invalidate(self, query: str) -> bool:
        try:
            hash_key = self._hash_key(query)
            deleted = False
            
            if self._redis_available:
                try:
                    self._redis.delete(f"cache:query:{hash_key}")
                    self._redis.delete(f"cache:response:{hash_key}")
                    self._redis.delete(f"cache:meta:{hash_key}")
                    
                    keys = self._redis_get_all_keys()
                    if hash_key in keys:
                        keys.remove(hash_key)
                        self._redis.set("cache:index", json.dumps(keys))
                        deleted = True
                except Exception as e:
                    logger.warning(f"Redis invalidate failed: {e}")

            if hash_key in self._memory_store:
                del self._memory_store[hash_key]
                deleted = True
                
            return deleted
        except Exception as e:
            logger.error(f"Cache invalidate error: {e}")
            return False

    def clear(self) -> None:
        try:
            if self._redis_available:
                try:
                    keys = self._redis_get_all_keys()
                    for key in keys:
                        self._redis.delete(f"cache:query:{key}")
                        self._redis.delete(f"cache:response:{key}")
                        self._redis.delete(f"cache:meta:{key}")
                    self._redis.delete("cache:index")
                except Exception as e:
                    logger.warning(f"Redis clear failed: {e}")
                    
            self._memory_store = {}
            logger.info("Cache cleared")
        except Exception as e:
            logger.error(f"Cache clear error: {e}")

    def get_stats(self) -> dict:
        return {
            "backend": self.backend,
            "redis_available": self._redis_available,
            "memory_store_size": len(self._memory_store),
            "threshold": self.threshold,
            "redis_key_count": len(self._redis_get_all_keys()) if self._redis_available else 0
        }
