from backend.app.config import settings
from typing import List
import os
import gc
import asyncio
import logging

_model_cache = {}

class Embedder:
    def __init__(self, model_name: str = settings.EMBEDDING_MODEL):
        self.model_name = model_name

    @property
    def model(self):
        global _model_cache
        if self.model_name not in _model_cache:
            import gc
            # pyrefly: ignore [missing-import]
            from sentence_transformers import SentenceTransformer
            print(f"Loading embedding model: {self.model_name} (Memory-optimized)...")
            # Load model and immediately clear memory
            model = SentenceTransformer(self.model_name, device="cpu")
            _model_cache[self.model_name] = model
            gc.collect() 
        return _model_cache[self.model_name]

    def embed(self, texts: List[str]) -> List[List[float]]:
        # Generate embeddings in batches
        embeddings = self.model.encode(texts, batch_size=32, show_progress_bar=False)
        return embeddings.tolist()

    def embed_single(self, text: str) -> List[float]:
        return self.embed([text])[0]