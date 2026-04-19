import sqlite3
import json
import uuid
import logging
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional

from app.config import settings

logger = logging.getLogger(__name__)

class Database:
    def __init__(self, db_path: str = None):
        self.db_path = db_path or settings.SQLITE_DB_PATH
        self._initialize_db()
        logger.info(f"Database initialized at {self.db_path}")

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    def _initialize_db(self) -> None:
        with self._get_connection() as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id      TEXT PRIMARY KEY,
                    created_at      TEXT NOT NULL,
                    updated_at      TEXT NOT NULL,
                    title           TEXT,
                    message_count   INTEGER DEFAULT 0,
                    document_filter TEXT,
                    is_active       INTEGER DEFAULT 1
                )
            ''')
            conn.execute('''
                CREATE TABLE IF NOT EXISTS messages (
                    message_id          TEXT PRIMARY KEY,
                    session_id          TEXT NOT NULL,
                    role                TEXT NOT NULL,
                    content             TEXT NOT NULL,
                    created_at          TEXT NOT NULL,
                    query_type          TEXT,
                    confidence          TEXT,
                    groundedness        TEXT,
                    usefulness_score    INTEGER,
                    retry_count         INTEGER DEFAULT 0,
                    sources             TEXT,
                    unsupported_claims  TEXT,
                    active_query        TEXT,
                    response_time_ms    INTEGER,
                    cache_hit           INTEGER DEFAULT 0,
                    FOREIGN KEY (session_id) REFERENCES sessions(session_id)
                )
            ''')
            conn.execute('''
                CREATE TABLE IF NOT EXISTS retrieved_docs_log (
                    log_id              TEXT PRIMARY KEY,
                    message_id          TEXT NOT NULL,
                    chunk_id            TEXT NOT NULL,
                    relevance_verdict   TEXT NOT NULL,
                    relevance_reason    TEXT,
                    retrieval_score     REAL,
                    was_used            INTEGER DEFAULT 0,
                    created_at          TEXT NOT NULL,
                    FOREIGN KEY (message_id) REFERENCES messages(message_id)
                )
            ''')
            
            conn.execute("CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_messages_created ON messages(created_at)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_docs_log_message ON retrieved_docs_log(message_id)")
            conn.commit()

    def create_session(self, title: str = None, document_filter: dict = None) -> dict:
        session_id = str(uuid.uuid4())
        try:
            now = datetime.now(timezone.utc).isoformat()
        except:
            now = datetime.utcnow().isoformat()

        if title is None:
            title = f"Session {now[:10]}"
            
        doc_filter_str = json.dumps(document_filter) if document_filter else None
        
        with self._get_connection() as conn:
            conn.execute(
                "INSERT INTO sessions (session_id, created_at, updated_at, title, message_count, document_filter, is_active) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (session_id, now, now, title, 0, doc_filter_str, 1)
            )
            conn.commit()
            
        return {
            "session_id": session_id,
            "created_at": now,
            "updated_at": now,
            "title": title,
            "message_count": 0,
            "document_filter": document_filter,
            "is_active": True
        }

    def get_session(self, session_id: str) -> Optional[dict]:
        with self._get_connection() as conn:
            row = conn.execute("SELECT * FROM sessions WHERE session_id = ?", (session_id,)).fetchone()
            if not row:
                return None
            
            res = dict(row)
            if res.get("document_filter"):
                try:
                    res["document_filter"] = json.loads(res["document_filter"])
                except Exception:
                    pass
            return res

    def list_sessions(self, include_archived: bool = False) -> List[dict]:
        with self._get_connection() as conn:
            if include_archived:
                rows = conn.execute("SELECT * FROM sessions ORDER BY updated_at DESC").fetchall()
            else:
                rows = conn.execute("SELECT * FROM sessions WHERE is_active = 1 ORDER BY updated_at DESC").fetchall()
            
            results = []
            for row in rows:
                r = dict(row)
                if r.get("document_filter"):
                    try:
                        r["document_filter"] = json.loads(r["document_filter"])
                    except Exception:
                        pass
                results.append(r)
            return results

    def update_session_title(self, session_id: str, title: str) -> None:
        now = datetime.now(timezone.utc).isoformat()
        with self._get_connection() as conn:
            conn.execute("UPDATE sessions SET title = ?, updated_at = ? WHERE session_id = ?", (title, now, session_id))
            conn.commit()

    def archive_session(self, session_id: str) -> None:
        now = datetime.now(timezone.utc).isoformat()
        with self._get_connection() as conn:
            conn.execute("UPDATE sessions SET is_active = 0, updated_at = ? WHERE session_id = ?", (now, session_id))
            conn.commit()

    def delete_session(self, session_id: str) -> None:
        with self._get_connection() as conn:
            conn.execute(
                "DELETE FROM retrieved_docs_log WHERE message_id IN (SELECT message_id FROM messages WHERE session_id = ?)",
                (session_id,)
            )
            conn.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
            conn.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
            conn.commit()

    def save_message(self, session_id: str, role: str, content: str, metadata: dict = None) -> str:
        message_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()
        metadata = metadata or {}
        
        sources_str = json.dumps(metadata.get("sources", []))
        claims_str = json.dumps(metadata.get("unsupported_claims", []))
        
        with self._get_connection() as conn:
            conn.execute(
                """INSERT INTO messages (
                    message_id, session_id, role, content, created_at, 
                    query_type, confidence, groundedness, usefulness_score, 
                    retry_count, sources, unsupported_claims, active_query, 
                    response_time_ms, cache_hit
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    message_id, session_id, role, content, now,
                    metadata.get("query_type"),
                    metadata.get("confidence"),
                    metadata.get("groundedness"),
                    metadata.get("usefulness_score"),
                    metadata.get("retry_count", 0),
                    sources_str,
                    claims_str,
                    metadata.get("active_query"),
                    metadata.get("response_time_ms"),
                    1 if metadata.get("cache_hit") else 0
                )
            )
            
            conn.execute("UPDATE sessions SET message_count = message_count + 1, updated_at = ? WHERE session_id = ?", (now, session_id))
            
            if role == "user":
                row = conn.execute("SELECT message_count, title FROM sessions WHERE session_id = ?", (session_id,)).fetchone()
                if row and row["message_count"] == 1 and row["title"].startswith("Session "):
                    new_title = content[:60]
                    conn.execute("UPDATE sessions SET title = ?, updated_at = ? WHERE session_id = ?", (new_title, now, session_id))
            
            conn.commit()        
        return message_id

    def get_messages(self, session_id: str) -> List[dict]:
        with self._get_connection() as conn:
            rows = conn.execute("SELECT * FROM messages WHERE session_id = ? ORDER BY created_at ASC", (session_id,)).fetchall()
            results = []
            for row in rows:
                r = dict(row)
                if r.get("sources") and isinstance(r["sources"], str):
                    try: r["sources"] = json.loads(r["sources"])
                    except Exception: r["sources"] = []
                if r.get("unsupported_claims") and isinstance(r["unsupported_claims"], str):
                    try: r["unsupported_claims"] = json.loads(r["unsupported_claims"])
                    except Exception: r["unsupported_claims"] = []
                results.append(r)
            return results

    def get_message(self, message_id: str) -> Optional[dict]:
        with self._get_connection() as conn:
            row = conn.execute("SELECT * FROM messages WHERE message_id = ?", (message_id,)).fetchone()
            if not row:
                return None
            r = dict(row)
            if r.get("sources") and isinstance(r["sources"], str):
                try: r["sources"] = json.loads(r["sources"])
                except Exception: r["sources"] = []
            if r.get("unsupported_claims") and isinstance(r["unsupported_claims"], str):
                try: r["unsupported_claims"] = json.loads(r["unsupported_claims"])
                except Exception: r["unsupported_claims"] = []
            return r

    def log_retrieved_docs(self, message_id: str, retrieved_chunks: List[dict], relevant_chunks: List[dict]) -> None:
        if not retrieved_chunks:
            return
            
        now = datetime.now(timezone.utc).isoformat()
        relevant_ids = {c["chunk_id"] for c in relevant_chunks}
        
        records = []
        for chunk in retrieved_chunks:
            was_used = 1 if chunk["chunk_id"] in relevant_ids else 0
            
            verdict_val = "unknown"
            reason_val = ""
            if "relevance_verdict" in chunk:
                if isinstance(chunk["relevance_verdict"], dict):
                    verdict_val = chunk["relevance_verdict"].get("verdict", "unknown")
                    reason_val = chunk["relevance_verdict"].get("reason", "")
                else:
                    verdict_val = str(chunk["relevance_verdict"])
                    
            score = chunk.get("rrf_score", chunk.get("bm25_score", 0.0))
            if score is None:
                score = 0.0
                
            records.append((
                str(uuid.uuid4()), message_id, chunk["chunk_id"], 
                verdict_val, reason_val, float(score), was_used, now
            ))
            
        with self._get_connection() as conn:
            conn.executemany(
                "INSERT INTO retrieved_docs_log "
                "(log_id, message_id, chunk_id, relevance_verdict, relevance_reason, retrieval_score, was_used, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                records
            )
            conn.commit()

    def get_retrieved_docs_log(self, message_id: str) -> List[dict]:
        with self._get_connection() as conn:
            rows = conn.execute("SELECT * FROM retrieved_docs_log WHERE message_id = ?", (message_id,)).fetchall()
            return [dict(row) for row in rows]

    def get_metrics(self) -> dict:
        metrics = {
            "total_queries": 0,
            "confidence_distribution": {"high": 0, "medium": 0, "low": 0},
            "avg_usefulness_score": 0.0,
            "cache_hit_rate": 0.0,
            "hallucination_catch_rate": 0.0,
            "avg_response_time_ms": 0.0,
            "avg_retry_count": 0.0
        }
        
        with self._get_connection() as conn:
            row = conn.execute("SELECT COUNT(*) as count FROM messages WHERE role = 'assistant'").fetchone()
            if row and row["count"]:
                metrics["total_queries"] = row["count"]
            else:
                return metrics
                
            conf_rows = conn.execute("SELECT confidence, COUNT(*) as count FROM messages WHERE role = 'assistant' GROUP BY confidence").fetchall()
            for r in conf_rows:
                if r["confidence"]:
                    metrics["confidence_distribution"][r["confidence"]] = r["count"]
            
            u_row = conn.execute("SELECT AVG(usefulness_score) as avg FROM messages WHERE role = 'assistant' AND usefulness_score IS NOT NULL").fetchone()
            if u_row and u_row["avg"] is not None:
                metrics["avg_usefulness_score"] = float(u_row["avg"])
                
            c_row = conn.execute("SELECT SUM(cache_hit) as hits, COUNT(*) as total FROM messages WHERE role = 'assistant'").fetchone()
            if c_row and c_row["total"] and c_row["total"] > 0:
                metrics["cache_hit_rate"] = float(c_row["hits"] or 0) / float(c_row["total"])
                
            h_row = conn.execute("SELECT COUNT(*) as count FROM messages WHERE role = 'assistant' AND unsupported_claims != '[]' AND unsupported_claims IS NOT NULL").fetchone()
            if h_row and metrics["total_queries"] > 0:
                metrics["hallucination_catch_rate"] = float(h_row["count"] or 0) / float(metrics["total_queries"])
                
            rt_row = conn.execute("SELECT AVG(response_time_ms) as avg FROM messages WHERE role = 'assistant' AND response_time_ms IS NOT NULL").fetchone()
            if rt_row and rt_row["avg"] is not None:
                metrics["avg_response_time_ms"] = float(rt_row["avg"])
                
            rc_row = conn.execute("SELECT AVG(retry_count) as avg FROM messages WHERE role = 'assistant'").fetchone()
            if rc_row and rc_row["avg"] is not None:
                metrics["avg_retry_count"] = float(rc_row["avg"])
                
        return metrics
