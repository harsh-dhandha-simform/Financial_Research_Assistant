"""
Supabase-backed session store with in-memory cache.

Persists UserSession objects in a Supabase Postgres table so sessions
survive app restarts. Uses an in-memory dict as a read-through cache.

The user_sessions table:
    session_id TEXT PRIMARY KEY
    user_identifier TEXT NOT NULL
    session_data JSONB NOT NULL
    created_at TIMESTAMPTZ
    updated_at TIMESTAMPTZ

Usage:
    from sessions.session_store import session_store

    session = session_store.get("session-id")
    session = session_store.get_or_create("session-id", user_identifier="test")
    session_store.update(session)
"""

import json
import logging
import os
import shutil
from datetime import datetime, timedelta
from typing import Optional

from config import settings
from sessions.session_model import UserSession

logger = logging.getLogger(__name__)


class SessionStore:
    """Supabase (Postgres) backed session store with in-memory caching.

    All sessions are persisted in the user_sessions table via synchronous
    psycopg2 calls. An in-memory dict acts as a fast read-through cache.

    Redis is NOT required — it was previously used but Supabase provides
    persistent storage tied to authenticated users.
    """

    def __init__(self):
        self._cache: dict[str, UserSession] = {}
        self._db_uri: str = ""
        self._conn = None

        # Build a sync postgres URI from the async one in settings
        if settings.supabase_uri:
            # Convert postgresql+asyncpg:// to postgresql://
            self._db_uri = settings.supabase_uri.replace(
                "postgresql+asyncpg://", "postgresql://"
            )
            self._connect()

    def _connect(self):
        """Establish sync connection to Supabase Postgres."""
        if not self._db_uri:
            logger.warning("No supabase_uri configured — sessions will be in-memory only")
            return
        try:
            import psycopg2
            self._conn = psycopg2.connect(self._db_uri)
            self._conn.autocommit = True
            logger.info("SessionStore connected to Supabase Postgres")
            with self._conn.cursor() as cur:
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS user_sessions (
                        session_id TEXT PRIMARY KEY,
                        user_identifier TEXT NOT NULL,
                        session_data JSONB NOT NULL,
                        created_at TIMESTAMPTZ DEFAULT NOW(),
                        updated_at TIMESTAMPTZ DEFAULT NOW()
                    );
                """)
        except Exception as exc:
            logger.warning(
                "Supabase Postgres connection failed: %s — sessions will be in-memory only",
                exc,
            )
            self._conn = None

    def _ensure_conn(self):
        """Re-establish connection if it was lost."""
        if self._conn is None:
            self._connect()
            return
        try:
            with self._conn.cursor() as cur:
                cur.execute("SELECT 1")
        except Exception:
            self._connect()

    # ── Serialization ────────────────────────────────────────────────────

    @staticmethod
    def _serialize(session: UserSession) -> str:
        """Convert a UserSession to a JSON string for storage."""
        data = {
            "session_id": session.session_id,
            "thread_id": session.thread_id,
            "created_at": session.created_at.isoformat(),
            "last_active": session.last_active.isoformat(),
            "ingested_documents": session.ingested_documents,
            "collection_name": session.collection_name,
            "pipeline_status": session.pipeline_status,
            "last_pipeline_result": session.last_pipeline_result,
            "chat_history": session.chat_history,
            "company_name": session.company_name,
            "ticker": session.ticker,
            "user_identifier": session.user_identifier,
        }
        return json.dumps(data, default=str)

    @staticmethod
    def _deserialize(data_json: str) -> UserSession:
        """Reconstruct a UserSession from its JSON representation."""
        data = json.loads(data_json) if isinstance(data_json, str) else data_json
        return UserSession(
            session_id=data["session_id"],
            thread_id=data.get("thread_id", data["session_id"]),
            created_at=datetime.fromisoformat(data["created_at"]),
            last_active=datetime.fromisoformat(data["last_active"]),
            ingested_documents=data.get("ingested_documents", []),
            collection_name=data.get("collection_name", ""),
            pipeline_status=data.get("pipeline_status", "idle"),
            last_pipeline_result=data.get("last_pipeline_result"),
            chat_history=data.get("chat_history", []),
            company_name=data.get("company_name", ""),
            ticker=data.get("ticker", ""),
            user_identifier=data.get("user_identifier", ""),
        )

    # ── CRUD ─────────────────────────────────────────────────────────────

    def create(self, session: UserSession, user_identifier: str = "") -> UserSession:
        """Create a new session. Writes to both cache and Supabase."""
        self._cache[session.session_id] = session
        self._write_to_db(session, user_identifier)
        logger.info(
            "Created session: %s (collection: %s, user: %s)",
            session.session_id, session.collection_name, user_identifier or "anonymous",
        )
        return session

    def get(self, session_id: str) -> Optional[UserSession]:
        """Get a session by ID. Returns None if not found."""
        # Check in-memory cache first
        session = self._cache.get(session_id)
        if session:
            return session

        # Fallback: check Supabase
        return self._read_from_db(session_id)

    def get_by_user(self, user_identifier: str) -> Optional[UserSession]:
        """Get the most recent session for a user. Used on login to restore state."""
        if not self._conn or not user_identifier:
            return None
        try:
            self._ensure_conn()
            with self._conn.cursor() as cur:
                cur.execute(
                    "SELECT session_data FROM user_sessions "
                    "WHERE user_identifier = %s ORDER BY updated_at DESC LIMIT 1",
                    (user_identifier,)
                )
                row = cur.fetchone()
                if row:
                    data = row[0] if isinstance(row[0], dict) else json.loads(row[0])
                    session = self._deserialize(data)
                    self._cache[session.session_id] = session
                    return session
        except Exception as exc:
            logger.warning("DB read for user %s failed: %s", user_identifier, exc)
            if self._conn:
                try:
                    self._conn.rollback()
                except Exception:
                    pass
        return None

    def get_all_by_user(self, user_identifier: str) -> list[UserSession]:
        """Get all sessions for a user, sorted newest first."""
        if not self._conn or not user_identifier:
            return []
        sessions = []
        try:
            self._ensure_conn()
            with self._conn.cursor() as cur:
                cur.execute(
                    "SELECT session_data FROM user_sessions "
                    "WHERE user_identifier = %s ORDER BY updated_at DESC",
                    (user_identifier,)
                )
                rows = cur.fetchall()
                for row in rows:
                    try:
                        data = row[0] if isinstance(row[0], dict) else json.loads(row[0])
                        session = self._deserialize(data)
                        self._cache[session.session_id] = session
                        sessions.append(session)
                    except Exception as e:
                        logger.warning("Failed to deserialize session: %s", e)
        except Exception as exc:
            logger.warning("DB read_all for user %s failed: %s", user_identifier, exc)
            if self._conn:
                try:
                    self._conn.rollback()
                except Exception:
                    pass
        return sessions
    def get_or_create(self, session_id: str, user_identifier: str = "") -> UserSession:
        """Get existing session or create a new one."""
        existing = self.get(session_id)
        if existing:
            if user_identifier and not existing.user_identifier:
                existing.user_identifier = user_identifier
            existing.touch()
            self.update(existing)
            return existing

        session = UserSession(
            session_id=session_id,
            thread_id=session_id,
            collection_name=f"fin_{session_id[:8]}",
            user_identifier=user_identifier,
        )
        return self.create(session, user_identifier)

    def update(self, session: UserSession) -> None:
        """Update an existing session in cache and DB."""
        session.touch()
        self._cache[session.session_id] = session
        self._write_to_db(session, session.user_identifier)

    def delete(self, session_id: str) -> None:
        """Delete a session from cache and DB."""
        self._cache.pop(session_id, None)
        if self._conn:
            try:
                self._ensure_conn()
                with self._conn.cursor() as cur:
                    cur.execute("DELETE FROM user_sessions WHERE session_id = %s", (session_id,))
            except Exception as exc:
                logger.warning("DB DELETE failed for session %s: %s", session_id, exc)
        logger.info("Deleted session: %s", session_id)
        self.cleanup(session_id)

    def cleanup(self, session_id: str) -> None:
        """Cleanup temp files for a session, but keep Qdrant collection."""
        session = self.get(session_id)
        if not session:
            return

        img_dir = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "data", "images", session_id[:8],
        )
        if os.path.exists(img_dir):
            try:
                shutil.rmtree(img_dir)
                logger.info("Cleaned up temp images for session %s", session_id)
            except Exception as exc:
                logger.warning("Failed to clean up images for %s: %s", session_id, exc)

        # We do NOT delete doc_dir (the actual uploaded documents)
        # per the user's request. We only delete the extracted images/crops.

    def list_sessions(self) -> list[UserSession]:
        """Return all active sessions from cache."""
        return list(self._cache.values())

    # ── Internal ─────────────────────────────────────────────────────────

    def _write_to_db(self, session: UserSession, user_identifier: str = ""):
        """Write/upsert a session to Supabase."""
        if not self._conn:
            return
        try:
            self._ensure_conn()
            data_json = self._serialize(session)
            with self._conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO user_sessions (session_id, user_identifier, session_data, updated_at)
                    VALUES (%s, %s, %s::jsonb, NOW())
                    ON CONFLICT (session_id) DO UPDATE SET
                        session_data = EXCLUDED.session_data,
                        updated_at = NOW()
                    """,
                    (session.session_id, user_identifier or "anonymous", data_json)
                )
        except Exception as exc:
            logger.warning(
                "Failed to write session %s to DB: %s", session.session_id, exc,
            )
            if self._conn:
                try:
                    self._conn.rollback()
                except Exception:
                    pass

    def _read_from_db(self, session_id: str) -> Optional[UserSession]:
        """Read a session from Supabase."""
        if not self._conn:
            return None
        try:
            self._ensure_conn()
            with self._conn.cursor() as cur:
                cur.execute(
                    "SELECT session_data FROM user_sessions WHERE session_id = %s",
                    (session_id,)
                )
                row = cur.fetchone()
                if row:
                    data = row[0] if isinstance(row[0], dict) else json.loads(row[0])
                    session = self._deserialize(data)
                    self._cache[session_id] = session
                    return session
        except Exception as exc:
            logger.warning("DB read failed for session %s: %s", session_id, exc)
            if self._conn:
                try:
                    self._conn.rollback()
                except Exception:
                    pass
        return None


# ── Singleton ────────────────────────────────────────────────────────────────
session_store = SessionStore()
