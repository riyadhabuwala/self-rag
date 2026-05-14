"""Schemas module."""
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
