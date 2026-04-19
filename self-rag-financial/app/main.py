import logging
import os
import time
import uuid
import json
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Optional, List
import asyncio
from concurrent.futures import ThreadPoolExecutor

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, UploadFile, File, Header, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.config import settings
from app.schemas import (
    QueryRequest, QueryResponse, SourceDocument,
    SessionResponse, MessageResponse, CreateSessionRequest,
    HealthResponse, IngestResponse
)
from app.database import Database
from app.graph.builder import build_graph, run_query
from app.rag.chroma_store import ChromaStore

logging.basicConfig(
    level=logging.DEBUG if settings.DEBUG else logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

# These are initialized in lifespan and shared across all requests
_graph = None
_db = None
_cache = None
_executor = ThreadPoolExecutor(max_workers=4)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # STARTUP
    global _graph, _db, _cache
    logger.info("Starting Self-RAG Financial Intelligence System...")

    logger.info("Initializing database...")
    _db = Database()

    logger.info("Building Self-RAG graph (loading models  may take 30s)...")
    # Run in thread pool so it doesn't block the event loop
    loop = asyncio.get_event_loop()
    _graph = await loop.run_in_executor(_executor, build_graph)
    logger.info("Self-RAG graph ready")

    logger.info("Initializing semantic cache...")
    from app.rag.cache import SemanticCache
    from app.rag.embedder import Embedder
    _cache_embedder = Embedder()
    _cache = SemanticCache(embedder=_cache_embedder)
    logger.info(f"Semantic cache ready — {_cache.get_stats()}")

    yield

    # SHUTDOWN
    logger.info("Shutting down...")
    _executor.shutdown(wait=False)

app = FastAPI(
    title="Self-RAG Financial Intelligence API",
    description="""
    A production-grade self-reflective RAG system for financial document intelligence.

    ## Features
    - **Self-RAG reflection loop**  retrieval routing, document grading,
      hallucination detection, usefulness scoring
    - **Hybrid retrieval**  dense + sparse (BM25) + Reciprocal Rank Fusion
    - **Financial domain specialised**  SEC 10-K/10-Q filings, earnings transcripts
    - **Confidence reporting**  every response includes groundedness verdict
    - **Chat history**  full conversation persistence with session management

    ## Authentication
    All endpoints except /health require an X-API-Key header.
    """,
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    duration_ms = int((time.time() - start) * 1000)
    logger.info(
        f"{request.method} {request.url.path} "
        f"? {response.status_code} [{duration_ms}ms]"
    )
    return response

async def verify_api_key(x_api_key: str = Header(..., alias="X-API-Key")):
    if x_api_key != settings.API_KEY:
        raise HTTPException(
            status_code=401,
            detail="Invalid API key. Provide a valid key in X-API-Key header."
        )
    return x_api_key

def get_graph():
    if _graph is None:
        raise HTTPException(status_code=503,
            detail="Graph not initialized. Server is still starting up.")
    return _graph

def get_db():
    if _db is None:
        raise HTTPException(status_code=503,
            detail="Database not initialized.")
    return _db

def get_cache():
    return _cache

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "detail": "An internal error occurred.",
            "type": type(exc).__name__
        }
    )

@app.post("/query", response_model=QueryResponse, dependencies=[Depends(verify_api_key)])
async def query_endpoint(
    request: QueryRequest,
    background_tasks: BackgroundTasks,
    graph=Depends(get_graph),
    db=Depends(get_db),
    cache=Depends(get_cache)
):
    """Execute a Self-RAG query against ingested financial documents."""
    try:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            _executor,
            lambda: run_query(
                query=request.query,
                session_id=request.session_id,
                filters=request.filters,
                max_retries=request.max_retries,
                graph=graph,
                db=db,
                cache=cache
            )
        )
        
        sources = [
            SourceDocument(
                document=s.get("document", ""),
                page=s.get("page"),
                ticker=s.get("ticker"),
                fiscal_year=s.get("fiscal_year"),
                chunk_id=s.get("chunk_id")
            )
            for s in result.get("sources", [])
        ]
        
        answer = result.get("answer", "")
        if not answer:
            answer = "No answer could be generated. Please try rephrasing your query."
            
        return QueryResponse(
            answer=answer,
            confidence=result.get("confidence", ""),
            groundedness=result.get("groundedness", ""),
            usefulness_score=result.get("usefulness_score", 0),
            sources=sources,
            unsupported_claims=result.get("unsupported_claims", []),
            retries=result.get("retry_count", 0),
            active_query=result.get("active_query", ""),
            response_time_ms=result.get("response_time_ms", 0),
            cache_hit=result.get("cache_hit", False),
            session_id=result.get("session_id", request.session_id or "")
        )
    except Exception as e:
        logger.error(f"Query error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ingest", response_model=IngestResponse, dependencies=[Depends(verify_api_key)])
async def ingest_endpoint(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    ticker: Optional[str] = None,
    doc_type: Optional[str] = None,
    fiscal_year: Optional[str] = None,
    force: bool = False
):
    """Ingest a document (PDF/HTML) into the ChromaDB vector store."""
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ['.pdf', '.html', '.htm']:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {ext}. Use .pdf or .html")

    import tempfile
    suffix = ext
    os.makedirs("data/raw", exist_ok=True)
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix, dir="data/raw") as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    from app.rag.ingest_pipeline import run_ingestion

    def do_ingest():
        try:
            result = run_ingestion(tmp_path, ticker, doc_type, fiscal_year, force)
            logger.info(f"Background ingestion complete: {result}")
        except Exception as e:
            logger.error(f"Background ingestion failed: {e}")
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    background_tasks.add_task(do_ingest)
    
    return IngestResponse(
        status="processing",
        chunks_created=0,
        document_id=f"{ticker or 'unknown'}_{doc_type or 'unknown'}",
        message=f"Ingestion started for {file.filename}. Check /health for updated document count."
    )

@app.get("/health", response_model=HealthResponse)
async def health_endpoint():
    """Check the health and operational status of the system."""
    try:
        store = ChromaStore()
        doc_count = store.get_doc_count()
        chroma_status = "ok"
    except Exception as e:
        doc_count = -1
        chroma_status = f"error: {str(e)}"

    if _cache is not None:
        stats = _cache.get_stats()
        redis_status = "connected" if stats["redis_available"] else "fallback_memory"
    else:
        redis_status = "not_configured"

    return HealthResponse(
        status="ok",
        chroma_doc_count=doc_count,
        redis_status=redis_status,
        model=settings.GROQ_GENERATION_MODEL
    )

@app.get("/docs-info", dependencies=[Depends(verify_api_key)])
async def docs_info_endpoint():
    """Get information about all indexed documents."""
    try:
        store = ChromaStore()
        results = store.main_collection.get(include=["metadatas"])
        
        docs = {}
        total_chunks = len(results.get("metadatas", []))
        
        for meta in results.get("metadatas", []):
            if not meta: continue
            t = meta.get("ticker", "unknown")
            d = meta.get("doc_type", "unknown")
            f = meta.get("fiscal_year", "unknown")
            doc_id = f"{t}_{d}_{f}"
            
            if doc_id not in docs:
                docs[doc_id] = {
                    "ticker": t,
                    "doc_type": d,
                    "fiscal_year": f,
                    "source_file": meta.get("source_file", ""),
                    "chunk_count": 0,
                    "ingested_at": meta.get("ingested_at", "")
                }
            docs[doc_id]["chunk_count"] += 1

        return {
            "total_documents": len(docs),
            "total_chunks": total_chunks,
            "documents": [
                {
                    "document_id": doc_id,
                    "ticker": doc["ticker"],
                    "doc_type": doc["doc_type"],
                    "fiscal_year": doc["fiscal_year"],
                    "source_file": doc["source_file"],
                    "chunk_count": doc["chunk_count"],
                    "ingested_at": doc["ingested_at"]
                }
                for doc_id, doc in docs.items()
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/sessions", dependencies=[Depends(verify_api_key)])
async def list_sessions_endpoint(
    include_archived: bool = False,
    db=Depends(get_db)
):
    """List all chat sessions."""
    sessions = db.list_sessions(include_archived=include_archived)
    return {"sessions": sessions, "total": len(sessions)}

@app.get("/sessions/{session_id}", dependencies=[Depends(verify_api_key)])
async def get_session_endpoint(
    session_id: str,
    db=Depends(get_db)
):
    """Get session details and all associated messages."""
    session = db.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
    messages = db.get_messages(session_id)
    return {"session": session, "messages": messages}

@app.post("/sessions", dependencies=[Depends(verify_api_key)])
async def create_session_endpoint(
    request: CreateSessionRequest,
    db=Depends(get_db)
):
    """Create a new chat session."""
    session = db.create_session(
        title=request.title,
        document_filter=request.document_filter
    )
    return session

@app.delete("/sessions/{session_id}", dependencies=[Depends(verify_api_key)])
async def delete_session_endpoint(
    session_id: str,
    archive: bool = True,
    db=Depends(get_db)
):
    """Archive or fully delete a session from history."""
    session = db.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
    if archive:
        db.archive_session(session_id)
        return {"status": "archived", "session_id": session_id}
    else:
        db.delete_session(session_id)
        return {"status": "deleted", "session_id": session_id}

@app.get("/metrics", dependencies=[Depends(verify_api_key)])
async def metrics_endpoint(db=Depends(get_db)):
    """Get system usage and validation metrics."""
    metrics = db.get_metrics()
    store = ChromaStore()
    metrics["total_chunks_indexed"] = store.get_doc_count()
    metrics["timestamp"] = datetime.utcnow().isoformat()
    return metrics

@app.get("/evaluate", dependencies=[Depends(verify_api_key)])
async def evaluate_endpoint(background_tasks: BackgroundTasks):
    """Triggers the RAGAS evaluation pipeline as a background task."""
    def run_eval():
        try:
            import subprocess, sys
            result = subprocess.run(
                [sys.executable, "evaluate.py"],
                capture_output=True, text=True, timeout=600
            )
            if result.returncode == 0:
                logger.info("RAGAS evaluation complete")
            else:
                logger.error(f"RAGAS evaluation failed: {result.stderr}")
        except Exception as e:
            logger.error(f"Evaluation error: {e}")

    background_tasks.add_task(run_eval)
    return {
        "status": "started",
        "message": "RAGAS evaluation running in background. Results will be written to evaluation_results.csv."
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level="debug" if settings.DEBUG else "info"
    )

