import os
import json

files = {
    "app/__init__.py": '"""App module."""\n',
    "app/main.py": '"""Main API module."""\napp = None\n# TODO: implement in StepN\n',
    "app/config.py": '''"""Configuration module."""
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
''',
    "app/schemas.py": '''"""Schemas module."""
from typing import Optional, List
from pydantic import BaseModel

class QueryRequest(BaseModel):
    query: str
    session_id: Optional[str] = None
    filters: Optional[dict] = None
    max_retries: Optional[int] = None

class IngestRequest(BaseModel):
    file_path: str
    metadata: Optional[dict] = None

class CreateSessionRequest(BaseModel):
    title: Optional[str] = None
    document_filter: Optional[dict] = None

class SourceDocument(BaseModel):
    document: str
    page: Optional[int] = None
    ticker: Optional[str] = None
    fiscal_year: Optional[str] = None
    doc_type: Optional[str] = None
    chunk_id: Optional[str] = None

class QueryResponse(BaseModel):
    answer: str
    confidence: str
    groundedness: str
    usefulness_score: int
    sources: List[SourceDocument]
    unsupported_claims: List[str]
    retries: int
    active_query: str
    response_time_ms: int
    cache_hit: bool
    session_id: str

class SessionResponse(BaseModel):
    session_id: str
    title: Optional[str] = None
    created_at: str
    updated_at: str
    message_count: int
    is_active: bool

class MessageResponse(BaseModel):
    message_id: str
    session_id: str
    role: str
    content: str
    created_at: str
    confidence: Optional[str] = None
    groundedness: Optional[str] = None
    usefulness_score: Optional[int] = None
    sources: Optional[List[SourceDocument]] = None
    unsupported_claims: Optional[List[str]] = None
    cache_hit: Optional[bool] = None
    response_time_ms: Optional[int] = None

class HealthResponse(BaseModel):
    status: str
    chroma_doc_count: int
    redis_status: str
    model: str

class IngestResponse(BaseModel):
    status: str
    chunks_created: int
    document_id: str
    message: str
''',
    "app/database.py": '"""Database stub."""\n# TODO: implement in StepN\n',
    "app/graph/__init__.py": '"""Graph package."""\n',
    "app/graph/state.py": '"""State stub."""\n# TODO: implement in StepN\n',
    "app/graph/nodes.py": '"""Nodes stub."""\n# TODO: implement in StepN\n',
    "app/graph/edges.py": '"""Edges stub."""\n# TODO: implement in StepN\n',
    "app/graph/builder.py": '"""Builder stub."""\n# TODO: implement in StepN\n',
    "app/rag/__init__.py": '"""RAG package."""\n',
    "app/rag/retriever.py": '"""Retriever stub."""\n# TODO: implement in StepN\n',
    "app/rag/graders.py": '"""Graders stub."""\n# TODO: implement in StepN\n',
    "app/rag/prompts.py": '"""Prompts stub."""\n# TODO: implement in StepN\n',
    "app/rag/cache.py": '"""Cache stub."""\n# TODO: implement in StepN\n',
    "app/rag/chunker.py": '"""Chunker stub."""\n# TODO: implement in StepN\n',
    "app/rag/extractor.py": '"""Extractor stub."""\n# TODO: implement in StepN\n',
    "tests/__init__.py": '"""Tests package."""\n',
    "tests/test_graders.py": '"""Test graders stub."""\n# TODO: implement in StepN\n',
    "tests/test_retriever.py": '"""Test retriever stub."""\n# TODO: implement in StepN\n',
    "tests/test_pipeline.py": '"""Test pipeline stub."""\n# TODO: implement in StepN\n',
    "tests/eval_dataset.json": '[]\n',
    "data/raw/.gitkeep": '',
    "ingest.py": '"""Ingest stub."""\n# TODO: implement in StepN\n',
    "evaluate.py": '"""Evaluate stub."""\n# TODO: implement in StepN\n',
    ".env.example": '''# Groq API (required) — get free key at console.groq.com
GROQ_API_KEY=your_groq_api_key_here
GROQ_GENERATION_MODEL=llama-3.3-70b-versatile
GROQ_GRADING_MODEL=llama-3.1-8b-instant

# ChromaDB
CHROMA_PERSIST_DIR=./chroma_db
CHROMA_COLLECTION_NAME=financial_docs

# Upstash Redis (optional) — get free instance at upstash.com
UPSTASH_REDIS_REST_URL=your_upstash_rest_url_here
UPSTASH_REDIS_REST_TOKEN=your_upstash_rest_token_here
CACHE_BACKEND=memory

# LangSmith (optional) — get free key at smith.langchain.com
LANGCHAIN_TRACING_V2=false
LANGCHAIN_API_KEY=your_langsmith_api_key_here
LANGCHAIN_PROJECT=self-rag-financial

# API
API_KEY=dev-key-change-in-production

# Local LLM dev mode (optional)
USE_LOCAL_LLM=false
''',
    ".env": '''# Groq API (required) — get free key at console.groq.com
GROQ_API_KEY=placeholder_key
GROQ_GENERATION_MODEL=llama-3.3-70b-versatile
GROQ_GRADING_MODEL=llama-3.1-8b-instant

# ChromaDB
CHROMA_PERSIST_DIR=./chroma_db
CHROMA_COLLECTION_NAME=financial_docs

# Upstash Redis (optional) — get free instance at upstash.com
UPSTASH_REDIS_REST_URL=your_upstash_rest_url_here
UPSTASH_REDIS_REST_TOKEN=your_upstash_rest_token_here
CACHE_BACKEND=memory

# LangSmith (optional) — get free key at smith.langchain.com
LANGCHAIN_TRACING_V2=false
LANGCHAIN_API_KEY=your_langsmith_api_key_here
LANGCHAIN_PROJECT=self-rag-financial

# API
API_KEY=dev-key-change-in-production

# Local LLM dev mode (optional)
USE_LOCAL_LLM=false
''',
    ".gitignore": '''.env
chroma_db/
chat_history.db
data/raw/
''',
    "requirements.txt": '''# LLM & Agent Framework
langchain
langchain-groq
langchain-community
langgraph
langsmith

# API Backend
fastapi
uvicorn[standard]
python-multipart

# Vector Store & Embeddings
chromadb
sentence-transformers

# Sparse Retrieval
rank-bm25

# Document Parsing
pdfplumber
beautifulsoup4
lxml

# Database
# sqlite3 is built into Python — no package needed

# Caching
upstash-redis

# Evaluation
ragas
datasets

# Financial Data
yfinance
requests

# Utilities
pydantic
pydantic-settings
python-dotenv
numpy
pandas
httpx
'''
}

base_dir = "self-rag-financial"
os.makedirs(base_dir, exist_ok=True)
for path, content in files.items():
    full_path = os.path.join(base_dir, path)
    os.makedirs(os.path.dirname(full_path), exist_ok=True)
    with open(full_path, "w", encoding="utf-8") as f:
        f.write(content)
print("Files created successfully")
