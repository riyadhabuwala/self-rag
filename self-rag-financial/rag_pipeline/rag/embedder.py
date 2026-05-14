from backend.app.config import settings
from typing import List

_model_cache = {}

class Embedder:
    def __init__(self, model_name: str = settings.EMBEDDING_MODEL):
        self.model_name = model_name

    @property
    def model(self):
        global _model_cache
        if self.model_name not in _model_cache:
            # pyrefly: ignore [missing-import]
            from sentence_transformers import SentenceTransformer
            print(f"Loading embedding model: {self.model_name} (this may take a moment)...")
            _model_cache[self.model_name] = SentenceTransformer(self.model_name)
        return _model_cache[self.model_name]

    def embed(self, texts: List[str]) -> List[List[float]]:
        # Generate embeddings in batches
        embeddings = self.model.encode(texts, batch_size=32, show_progress_bar=False)
        return embeddings.tolist()

    def embed_single(self, text: str) -> List[float]:
        return self.embed([text])[0]