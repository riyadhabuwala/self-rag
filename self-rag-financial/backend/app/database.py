import sqlite3
import json
import uuid
import logging
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional

from backend.app.config import settings

logger = logging.getLogger(__name__)

class Database:
    def __init__(self, db_path: str = None):
        self.is_postgres = bool(settings.DATABASE_URL and settings.DATABASE_URL.startswith("postgres"))
        
        if self.is_postgres:
            import psycopg2
            from psycopg2.extras import RealDictCursor
            self.db_url = settings.DATABASE_URL
            logger.info("Database initialized using PostgreSQL")
        else:
            self.db_path = db_path or settings.SQLITE_DB_PATH
            logger.info(f"Database initialized using SQLite at {self.db_path}")
            
        self._initialize_db()

    def _get_connection(self):
        if self.is_postgres:
            import psycopg2
            from psycopg2.extras import RealDictCursor
            conn = psycopg2.connect(self.db_url, cursor_factory=RealDictCursor)
            return conn
        else:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA foreign_keys = ON")
            return conn

    def _execute(self, conn, query: str, params: tuple = ()):
        if self.is_postgres:
            query = query.replace('?', '%s')
            # Handle SQLite's INTEGER PRIMARY KEY vs Postgres SERIAL/VARCHAR
            cursor = conn.cursor()
            cursor.execute(query, params)
            return cursor
        else:
            return conn.execute(query, params)

    def _executemany(self, conn, query: str, params_list: list):
        if self.is_postgres:
            query = query.replace('?', '%s')
            cursor = conn.cursor()
            cursor.executemany(query, params_list)
        else:
            conn.executemany(query, params_list)

    def _initialize_db(self) -> None:
        try:
            conn = self._get_connection()
            if self.is_postgres:
                # Postgres Schema
                conn.cursor().execute('''
                    CREATE TABLE IF NOT EXISTS sessions (
                        session_id      VARCHAR PRIMARY KEY,
                        created_at      VARCHAR NOT NULL,
                        updated_at      VARCHAR NOT NULL,
                        title           VARCHAR,
                        message_count   INTEGER DEFAULT 0,
                        document_filter VARCHAR,
                        is_active       INTEGER DEFAULT 1
                    )
                ''')
                conn.cursor().execute('''
                    CREATE TABLE IF NOT EXISTS messages (
                        message_id          VARCHAR PRIMARY KEY,
                        session_id          VARCHAR NOT NULL REFERENCES sessions(session_id),
                        role                VARCHAR NOT NULL,
                        content             TEXT NOT NULL,
                        created_at          VARCHAR NOT NULL,
                        query_type          VARCHAR,
                        confidence          VARCHAR,
                        groundedness        VARCHAR,
                        usefulness_score    INTEGER,
                        retry_count         INTEGER DEFAULT 0,
                        sources             TEXT,
                        unsupported_claims  TEXT,
                        active_query        TEXT,
                        response_time_ms    INTEGER,
                        cache_hit           INTEGER DEFAULT 0
                    )
                ''')
                conn.cursor().execute('''
                    CREATE TABLE IF NOT EXISTS retrieved_docs_log (
                        log_id              VARCHAR PRIMARY KEY,
                        message_id          VARCHAR NOT NULL REFERENCES messages(message_id),
                        chunk_id            VARCHAR NOT NULL,
                        relevance_verdict   VARCHAR NOT NULL,
                        relevance_reason    TEXT,
                        retrieval_score     REAL,
                        was_used            INTEGER DEFAULT 0,
                        created_at          VARCHAR NOT NULL
                    )
                ''')
                conn.cursor().execute("CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id)")
                conn.cursor().execute("CREATE INDEX IF NOT EXISTS idx_messages_created ON messages(created_at)")
                conn.cursor().execute("CREATE INDEX IF NOT EXISTS idx_docs_log_message ON retrieved_docs_log(message_id)")
            else:
                self._execute(conn, '''
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
                self._execute(conn, '''
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
                self._execute(conn, '''
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
                
                self._execute(conn, "CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id)")
                self._execute(conn, "CREATE INDEX IF NOT EXISTS idx_messages_created ON messages(created_at)")
                self._execute(conn, "CREATE INDEX IF NOT EXISTS idx_docs_log_message ON retrieved_docs_log(message_id)")
            
            conn.commit()
        except Exception as e:
            logger.error(f"Error initializing DB: {e}")
        finally:
            if 'conn' in locals() and conn:
                conn.close()

    def create_session(self, title: str = None, document_filter: dict = None) -> dict:
        session_id = str(uuid.uuid4())
        try:
            now = datetime.now(timezone.utc).isoformat()
        except:
            now = datetime.utcnow().isoformat()

        if title is None:
            title = f"Session {now[:10]}"
            
        doc_filter_str = json.dumps(document_filter) if document_filter else None
        
        conn = self._get_connection()
        try:
            self._execute(conn,
                "INSERT INTO sessions (session_id, created_at, updated_at, title, message_count, document_filter, is_active) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (session_id, now, now, title, 0, doc_filter_str, 1)
            )
            conn.commit()
        finally:
            conn.close()
            
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
        conn = self._get_connection()
        try:
            cursor = self._execute(conn, "SELECT * FROM sessions WHERE session_id = ?", (session_id,))
            row = cursor.fetchone()
            if not row:
                return None
            
            res = dict(row)
            if res.get("document_filter"):
                try:
                    res["document_filter"] = json.loads(res["document_filter"])
                except Exception:
                    pass
            return res
        finally:
            conn.close()

    def list_sessions(self, include_archived: bool = False) -> List[dict]:
        conn = self._get_connection()
        try:
            if include_archived:
                cursor = self._execute(conn, "SELECT * FROM sessions ORDER BY updated_at DESC")
            else:
                cursor = self._execute(conn, "SELECT * FROM sessions WHERE is_active = 1 ORDER BY updated_at DESC")
            
            rows = cursor.fetchall()
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
        finally:
            conn.close()

    def update_session_title(self, session_id: str, title: str) -> None:
        now = datetime.now(timezone.utc).isoformat()
        conn = self._get_connection()
        try:
            self._execute(conn, "UPDATE sessions SET title = ?, updated_at = ? WHERE session_id = ?", (title, now, session_id))
            conn.commit()
        finally:
            conn.close()

    def archive_session(self, session_id: str) -> None:
        now = datetime.now(timezone.utc).isoformat()
        conn = self._get_connection()
        try:
            self._execute(conn, "UPDATE sessions SET is_active = 0, updated_at = ? WHERE session_id = ?", (now, session_id))
            conn.commit()
        finally:
            conn.close()

    def delete_session(self, session_id: str) -> None:
        conn = self._get_connection()
        try:
            self._execute(conn,
                "DELETE FROM retrieved_docs_log WHERE message_id IN (SELECT message_id FROM messages WHERE session_id = ?)",
                (session_id,)
            )
            self._execute(conn, "DELETE FROM messages WHERE session_id = ?", (session_id,))
            self._execute(conn, "DELETE FROM sessions WHERE session_id = ?", (session_id,))
            conn.commit()
        finally:
            conn.close()

    def save_message(self, session_id: str, role: str, content: str, metadata: dict = None) -> str:
        message_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()
        metadata = metadata or {}
        
        sources_str = json.dumps(metadata.get("sources", []))
        claims_str = json.dumps(metadata.get("unsupported_claims", []))
        
        conn = self._get_connection()
        try:
            self._execute(conn,
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
            
            self._execute(conn, "UPDATE sessions SET message_count = message_count + 1, updated_at = ? WHERE session_id = ?", (now, session_id))
            
            if role == "user":
                cursor = self._execute(conn, "SELECT message_count, title FROM sessions WHERE session_id = ?", (session_id,))
                row = cursor.fetchone()
                if row and row["message_count"] == 1 and row["title"].startswith("Session "):
                    new_title = content[:60]
                    self._execute(conn, "UPDATE sessions SET title = ?, updated_at = ? WHERE session_id = ?", (new_title, now, session_id))
            
            conn.commit()        
        finally:
            conn.close()
            
        return message_id

    def get_messages(self, session_id: str) -> List[dict]:
        conn = self._get_connection()
        try:
            cursor = self._execute(conn, "SELECT * FROM messages WHERE session_id = ? ORDER BY created_at ASC", (session_id,))
            rows = cursor.fetchall()
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
        finally:
            conn.close()

    def get_message(self, message_id: str) -> Optional[dict]:
        conn = self._get_connection()
        try:
            cursor = self._execute(conn, "SELECT * FROM messages WHERE message_id = ?", (message_id,))
            row = cursor.fetchone()
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
        finally:
            conn.close()

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
            
        conn = self._get_connection()
        try:
            self._executemany(conn,
                "INSERT INTO retrieved_docs_log "
                "(log_id, message_id, chunk_id, relevance_verdict, relevance_reason, retrieval_score, was_used, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                records
            )
            conn.commit()
        finally:
            conn.close()

    def get_retrieved_docs_log(self, message_id: str) -> List[dict]:
        conn = self._get_connection()
        try:
            cursor = self._execute(conn, "SELECT * FROM retrieved_docs_log WHERE message_id = ?", (message_id,))
            rows = cursor.fetchall()
            return [dict(row) for row in rows]
        finally:
            conn.close()

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
        
        conn = self._get_connection()
        try:
            cursor = self._execute(conn, "SELECT COUNT(*) as count FROM messages WHERE role = 'assistant'")
            row = cursor.fetchone()
            if row and row["count"]:
                metrics["total_queries"] = row["count"]
            else:
                return metrics
                
            cursor = self._execute(conn, "SELECT confidence, COUNT(*) as count FROM messages WHERE role = 'assistant' GROUP BY confidence")
            conf_rows = cursor.fetchall()
            for r in conf_rows:
                if r["confidence"]:
                    metrics["confidence_distribution"][r["confidence"]] = r["count"]
            
            cursor = self._execute(conn, "SELECT AVG(usefulness_score) as avg FROM messages WHERE role = 'assistant' AND usefulness_score IS NOT NULL")
            u_row = cursor.fetchone()
            if u_row and u_row["avg"] is not None:
                metrics["avg_usefulness_score"] = float(u_row["avg"])
                
            cursor = self._execute(conn, "SELECT SUM(cache_hit) as hits, COUNT(*) as total FROM messages WHERE role = 'assistant'")
            c_row = cursor.fetchone()
            if c_row and c_row["total"] and c_row["total"] > 0:
                metrics["cache_hit_rate"] = float(c_row["hits"] or 0) / float(c_row["total"])
                
            # In PostgreSQL unsupported_claims text needs checking differently, but != '[]' works for string matching
            cursor = self._execute(conn, "SELECT COUNT(*) as count FROM messages WHERE role = 'assistant' AND unsupported_claims != '[]' AND unsupported_claims IS NOT NULL")
            h_row = cursor.fetchone()
            if h_row and metrics["total_queries"] > 0:
                metrics["hallucination_catch_rate"] = float(h_row["count"] or 0) / float(metrics["total_queries"])
                
            cursor = self._execute(conn, "SELECT AVG(response_time_ms) as avg FROM messages WHERE role = 'assistant' AND response_time_ms IS NOT NULL")
            rt_row = cursor.fetchone()
            if rt_row and rt_row["avg"] is not None:
                metrics["avg_response_time_ms"] = float(rt_row["avg"])
                
            cursor = self._execute(conn, "SELECT AVG(retry_count) as avg FROM messages WHERE role = 'assistant'")
            rc_row = cursor.fetchone()
            if rc_row and rc_row["avg"] is not None:
                metrics["avg_retry_count"] = float(rc_row["avg"])
                
        except Exception as e:
            logger.error(f"Error fetching metrics: {e}")
        finally:
            conn.close()
            
        return metrics
