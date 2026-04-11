"""Configuration module."""
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # Groq Settings
    GROQ_API_KEY: str
    GROQ_GENERATION_MODEL: str = "llama-3.3-70b-versatile"
    GROQ_GRADING_MODEL: str = "llama-3.1-8b-instant"
    GROQ_MAX_TOKENS: int = 1000
    GROQ_TEMPERATURE: float = 0.0

    # ChromaDB Settings
    CHROMA_PERSIST_DIR: str = "./chroma_db"
    CHROMA_COLLECTION_NAME: str = "financial_docs"
    CHROMA_SUMMARY_COLLECTION: str = "doc_summaries"

    # Embedding Settings
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    EMBEDDING_DIMENSION: int = 384

    # Retrieval Settings
    TOP_K_DENSE: int = 10
    TOP_K_BM25: int = 10
    TOP_K_FINAL: int = 5
    RRF_K: int = 60
    MULTI_QUERY_COUNT: int = 3

    # Self-RAG Settings
    MAX_RETRIES: int = 3
    USEFULNESS_THRESHOLD: int = 3
    HALLUCINATION_RETRY: bool = True

    # Cache Settings
    UPSTASH_REDIS_REST_URL: str = ""
    UPSTASH_REDIS_REST_TOKEN: str = ""
    CACHE_SIMILARITY_THRESHOLD: float = 0.92
    CACHE_TTL_STABLE: int = 259200
    CACHE_TTL_RECENT: int = 21600
    CACHE_TTL_FALLBACK: int = 3600
    CACHE_BACKEND: str = "upstash"

    # LangSmith Settings
    LANGCHAIN_TRACING_V2: bool = False
    LANGCHAIN_API_KEY: str = ""
    LANGCHAIN_PROJECT: str = "self-rag-financial"

    # API Settings
    API_KEY: str = "dev-key-change-in-production"
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    # App Settings
    USE_LOCAL_LLM: bool = False
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "llama3.2"
    DEBUG: bool = False

    model_config = {
        "env_file": ".env",
        "case_sensitive": False
    }

settings = Settings()
