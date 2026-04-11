"""Embedding Generator"""
from sentence_transformers import SentenceTransformer
from app.config import settings
from typing import List

class Embedder:
    def __init__(self, model_name: str = settings.EMBEDDING_MODEL):
        print(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)

    def embed(self, texts: List[str]) -> List[List[float]]:
        # Generate embeddings in batches
        embeddings = self.model.encode(texts, batch_size=32, show_progress_bar=False)
        return embeddings.tolist()

    def embed_single(self, text: str) -> List[float]:
        return self.embed([text])[0]