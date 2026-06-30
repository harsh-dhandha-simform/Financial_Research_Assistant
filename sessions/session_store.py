"""
Supabase-backed session store with in-memory cache.

DATA MODEL (new — user-scoped collections):
  - user_sessions:    one row per Chainlit thread / conversation
  - documents:        one row per ingested document, owned by user_id (NOT session_id)
  - user_collections: one row per user — their single Qdrant collection (fin_{user_id})

Qdrant collections are user-scoped: fin_{user_id}.  Every session for the same user
reads from and writes to the SAME collection.  Sessions are conversation threads only.

Usage:
    from sessions.session_store import session_store

    session = session_store.get_by_thread(thread_id, user_id)
    session = session_store.create_new(thread_id, user_id)
    session_store.update(session)
    collection = session_store.get_user_collection(user_id)   # "fin_{user_id}"
    docs = session_store.get_docs_for_user(user_id)
"""

import json
import logging
import os
import shutil
from datetime import datetime, timedelta, timezone
from typing import Optional

from config import settings
from sessions.session_model import UserSession

logger = logging.getLogger(__name__)

# ── UTC helper ────────────────────────────────────────────────────────────────

def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class SessionStore:
    """Supabase (Postgres) backed session store with in-memory caching.

    All sessions are persisted in the user_sessions table via synchronous
    psycopg2 calls.  An in-memory dict acts as a fast read-through cache.
    """

    def __init__(self):
        self._cache: dict[str, UserSession] = {}
        self._db_uri: str = ""
        self._conn = None

        # Build a sync postgres URI from the async one in settings
        if settings.supabase_uri:
            self._db_uri = settings.supabase_uri.replace(
                "postgresql+asyncpg://", "postgresql://"
            )
            self._connect()

    def _connect(self):
        """Establish sync connection to Supabase Postgres and ensure all tables exist."""
        if not self._db_uri:
            logger.warning("No supabase_uri configured — sessions will be in-memory only")
            return
        try:
            import psycopg2
            self._conn = psycopg2.connect(self._db_uri)
            self._conn.autocommit = True
            logger.info("SessionStore connected to Supabase Postgres")
            self._ensure_schema()
        except Exception as exc:
            logger.warning(
                "Supabase Postgres connection failed: %s — sessions will be in-memory only",
                exc,
            )
            self._conn = None

    def _ensure_schema(self):
        """Create all required tables if they don't exist."""
        with self._conn.cursor() as cur:
            # Original sessions table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS user_sessions (
                    session_id       TEXT PRIMARY KEY,
                    user_identifier  TEXT NOT NULL,
                    session_data     JSONB NOT NULL,
                    created_at       TIMESTAMPTZ DEFAULT NOW(),
                    updated_at       TIMESTAMPTZ DEFAULT NOW()
                );
            """)

            # New: one row per user — their single Qdrant collection
            cur.execute("""
                CREATE TABLE IF NOT EXISTS user_collections (
                    user_id           TEXT PRIMARY KEY,
                    collection_name   TEXT NOT NULL,
                    created_at        TIMESTAMPTZ DEFAULT NOW(),
                    chunk_count       INT DEFAULT 0,
                    zero_chunks_since TIMESTAMPTZ,
                    status            TEXT DEFAULT 'active'
                );
            """)

            # New: one row per ingested document (user-scoped, not session-scoped)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    doc_id       TEXT PRIMARY KEY,
                    user_id      TEXT NOT NULL,
                    session_id   TEXT NOT NULL,
                    name         TEXT NOT NULL,
                    type         TEXT NOT NULL,
                    source_url   TEXT DEFAULT '',
                    local_path   TEXT DEFAULT '',
                    ingested_at  TIMESTAMPTZ DEFAULT NOW(),
                    chunk_count  INT DEFAULT 0,
                    image_count  INT DEFAULT 0,
                    images       JSONB DEFAULT '[]',
                    status       TEXT DEFAULT 'active'
                );
            """)

            # Index for fast user lookups on documents
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_documents_user_id
                ON documents (user_id, status);
            """)
        logger.info("SessionStore schema verified (user_sessions, user_collections, documents)")

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

    # ═════════════════════════════════════════════════════════════════════════
    # User Collection — get or create fin_{user_id}
    # ═════════════════════════════════════════════════════════════════════════

    def get_user_collection(self, user_id: str) -> str:
        """Return the Qdrant collection name for this user.

        If no collection row exists yet, insert one and (lazily) create
        the Qdrant collection on first use.  Collection is NEVER duplicated.
        """
        if not user_id or user_id == "anonymous":
            # Fallback for unauthenticated use — use a shared collection
            return "fin_anonymous"

        collection_name = f"fin_{user_id}"

        if self._conn:
            try:
                self._ensure_conn()
                with self._conn.cursor() as cur:
                    cur.execute(
                        "SELECT collection_name FROM user_collections WHERE user_id = %s",
                        (user_id,)
                    )
                    row = cur.fetchone()
                    if row:
                        return row[0]
                    # First time for this user — register the collection
                    cur.execute(
                        """
                        INSERT INTO user_collections (user_id, collection_name, status)
                        VALUES (%s, %s, 'active')
                        ON CONFLICT (user_id) DO NOTHING
                        """,
                        (user_id, collection_name)
                    )
                    logger.info("Registered user collection: %s → %s", user_id, collection_name)
            except Exception as exc:
                logger.warning("get_user_collection DB error for %s: %s", user_id, exc)
                try:
                    self._conn.rollback()
                except Exception:
                    pass

        return collection_name

    def update_collection_chunk_count(self, user_id: str, delta: int):
        """Increment (or decrement) the chunk_count for a user's collection.

        Sets zero_chunks_since when count reaches 0 (triggers pending_deletion lifecycle).
        """
        if not self._conn or not user_id:
            return
        try:
            self._ensure_conn()
            with self._conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE user_collections
                    SET chunk_count = GREATEST(0, chunk_count + %s),
                        zero_chunks_since = CASE
                            WHEN (chunk_count + %s) <= 0 THEN NOW()
                            ELSE NULL
                        END,
                        status = CASE
                            WHEN (chunk_count + %s) <= 0 THEN 'pending_deletion'
                            ELSE 'active'
                        END
                    WHERE user_id = %s
                    """,
                    (delta, delta, delta, user_id)
                )
        except Exception as exc:
            logger.warning("update_collection_chunk_count failed for %s: %s", user_id, exc)
            try:
                self._conn.rollback()
            except Exception:
                pass

    def check_and_cleanup_empty_collections(self):
        """Delete Qdrant collections that have been empty for > 24 hours.

        Call this from on_chat_start as a lazy maintenance check.
        """
        if not self._conn:
            return
        try:
            self._ensure_conn()
            with self._conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT user_id, collection_name FROM user_collections
                    WHERE status = 'pending_deletion'
                    AND zero_chunks_since < NOW() - INTERVAL '24 hours'
                    """
                )
                rows = cur.fetchall()

            for user_id, collection_name in rows:
                try:
                    from qdrant_client import QdrantClient
                    client = QdrantClient(
                        url=settings.qdrant_url,
                        api_key=settings.qdrant_api_key or None,
                    )
                    existing = [c.name for c in client.get_collections().collections]
                    if collection_name in existing:
                        client.delete_collection(collection_name)
                        logger.info(
                            "Deleted empty Qdrant collection %s for user %s (24h threshold)",
                            collection_name, user_id
                        )
                    with self._conn.cursor() as cur:
                        cur.execute(
                            "UPDATE user_collections SET status = 'deleted' WHERE user_id = %s",
                            (user_id,)
                        )
                except Exception as exc:
                    logger.warning("Failed to cleanup collection for %s: %s", user_id, exc)
        except Exception as exc:
            logger.warning("check_and_cleanup_empty_collections error: %s", exc)

    # ═════════════════════════════════════════════════════════════════════════
    # Documents — user-scoped CRUD
    # ═════════════════════════════════════════════════════════════════════════

    def add_document(self, doc_meta: dict, user_id: str, session_id: str) -> bool:
        """Insert a new document row into the documents table.

        doc_meta keys: doc_id, name, type, source_url, local_path,
                       chunk_count, image_count, images (list of dicts)
        """
        if not self._conn:
            return False
        try:
            self._ensure_conn()
            with self._conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO documents
                        (doc_id, user_id, session_id, name, type, source_url,
                         local_path, chunk_count, image_count, images)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb)
                    ON CONFLICT (doc_id) DO UPDATE SET
                        chunk_count = EXCLUDED.chunk_count,
                        image_count = EXCLUDED.image_count,
                        images = EXCLUDED.images
                    """,
                    (
                        doc_meta["doc_id"],
                        user_id,
                        session_id,
                        doc_meta["name"],
                        doc_meta["type"],
                        doc_meta.get("source_url", ""),
                        doc_meta.get("local_path", ""),
                        doc_meta.get("chunk_count", 0),
                        doc_meta.get("image_count", 0),
                        json.dumps(doc_meta.get("images", [])),
                    )
                )
            logger.debug(
                "[SOURCES-WRITE] user_id=%s, doc_id=%s, session_id=%s",
                user_id, doc_meta["doc_id"], session_id
            )
            return True
        except Exception as exc:
            logger.warning("add_document failed for user %s: %s", user_id, exc)
            try:
                self._conn.rollback()
            except Exception:
                pass
            return False

    def get_docs_for_user(self, user_id: str) -> list[dict]:
        """Return all active documents for a user, newest first.

        Each dict matches the doc_meta format used throughout the codebase.
        """
        if not user_id:
            return []

        docs = []
        if self._conn:
            try:
                self._ensure_conn()
                with self._conn.cursor() as cur:
                    cur.execute(
                        """
                        SELECT doc_id, user_id, session_id, name, type, source_url,
                               local_path, chunk_count, image_count, images,
                               ingested_at
                        FROM documents
                        WHERE user_id = %s AND status = 'active'
                        ORDER BY ingested_at DESC
                        """,
                        (user_id,)
                    )
                    rows = cur.fetchall()
                    for row in rows:
                        docs.append({
                            "doc_id": row[0],
                            "user_id": row[1],
                            "session_id": row[2],
                            "name": row[3],
                            "type": row[4],
                            "source_url": row[5] or "",
                            "local_path": row[6] or "",
                            "chunk_count": row[7] or 0,
                            "image_count": row[8] or 0,
                            "images": row[9] if row[9] else [],
                            "ingested_at": row[10].isoformat() if row[10] else "",
                        })
            except Exception as exc:
                logger.warning("get_docs_for_user failed for %s: %s", user_id, exc)
                try:
                    self._conn.rollback()
                except Exception:
                    pass

        logger.debug("[SOURCES-READ] user_id=%s, found=%d docs", user_id, len(docs))
        return docs

    def delete_document(self, doc_id: str, user_id: str) -> dict:
        """Soft-delete a document. Returns metadata needed for Qdrant cleanup.

        Returns dict with {doc_id, collection_name, chunk_count} or empty dict on failure.
        """
        if not self._conn:
            return {}
        try:
            self._ensure_conn()
            with self._conn.cursor() as cur:
                # Ownership check
                cur.execute(
                    "SELECT doc_id, chunk_count FROM documents WHERE doc_id = %s AND user_id = %s AND status = 'active'",
                    (doc_id, user_id)
                )
                row = cur.fetchone()
                if not row:
                    return {}

                chunk_count = row[1] or 0
                cur.execute(
                    "UPDATE documents SET status = 'deleted' WHERE doc_id = %s AND user_id = %s",
                    (doc_id, user_id)
                )

            collection_name = self.get_user_collection(user_id)
            self.update_collection_chunk_count(user_id, -chunk_count)
            return {
                "doc_id": doc_id,
                "collection_name": collection_name,
                "chunk_count": chunk_count,
            }
        except Exception as exc:
            logger.warning("delete_document failed for doc %s user %s: %s", doc_id, user_id, exc)
            try:
                self._conn.rollback()
            except Exception:
                pass
            return {}

    # ═════════════════════════════════════════════════════════════════════════
    # Session CRUD
    # ═════════════════════════════════════════════════════════════════════════

    @staticmethod
    def _serialize(session: UserSession) -> str:
        """Convert a UserSession to a JSON string for storage."""
        data = {
            "session_id": session.session_id,
            "thread_id": session.thread_id,
            "created_at": session.created_at.isoformat(),
            "last_active": session.last_active.isoformat(),
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
    def _deserialize(data_json) -> UserSession:
        """Reconstruct a UserSession from its JSON representation."""
        data = json.loads(data_json) if isinstance(data_json, str) else data_json
        
        # Safely coerce chat_history — it may be double-encoded as a JSON string
        raw_history = data.get("chat_history", [])
        if isinstance(raw_history, str):
            try:
                raw_history = json.loads(raw_history)
            except Exception:
                raw_history = []
        if not isinstance(raw_history, list):
            raw_history = []
        # Drop any entries that are not dicts
        raw_history = [m for m in raw_history if isinstance(m, dict)]
        
        return UserSession(
            session_id=data["session_id"],
            thread_id=data.get("thread_id", data["session_id"]),
            created_at=datetime.fromisoformat(data["created_at"]),
            last_active=datetime.fromisoformat(data["last_active"]),
            collection_name=data.get("collection_name", ""),
            pipeline_status=data.get("pipeline_status", "idle"),
            last_pipeline_result=data.get("last_pipeline_result"),
            chat_history=raw_history,
            company_name=data.get("company_name", ""),
            ticker=data.get("ticker", ""),
            user_identifier=data.get("user_identifier", ""),
        )

    def get_by_thread(self, thread_id: str, user_id: str = "") -> Optional[UserSession]:
        """Look up an existing session by Chainlit thread_id.

        Returns None if not found — caller must create a new session.
        Does NOT create anything.
        """
        # In-memory cache keyed by session_id (== thread_id)
        session = self._cache.get(thread_id)
        if session:
            return session

        return self._read_from_db(thread_id)

    def create_new(self, thread_id: str, user_id: str) -> UserSession:
        """Create a brand-new session for a new Chainlit thread.

        Fetches the user's collection from user_collections (creating it if
        needed) so every session for the same user shares one Qdrant collection.
        """
        collection_name = self.get_user_collection(user_id)
        session = UserSession(
            session_id=thread_id,
            thread_id=thread_id,
            collection_name=collection_name,
            user_identifier=user_id,
        )
        self._cache[thread_id] = session
        self._write_to_db(session, user_id)
        logger.info(
            "Created new session %s for user %s (collection: %s)",
            thread_id[:8], user_id, collection_name
        )
        return session

    def get(self, session_id: str) -> Optional[UserSession]:
        """Get a session by ID. Returns None if not found."""
        session = self._cache.get(session_id)
        if session:
            return session
        return self._read_from_db(session_id)

    def get_by_user(self, user_identifier: str) -> Optional[UserSession]:
        """Get the most recent session for a user."""
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

    # Keep get_or_create as a compatibility shim — new code should use get_by_thread / create_new
    def get_or_create(self, session_id: str, user_identifier: str = "") -> UserSession:
        """Get existing session or create a new one (compatibility shim)."""
        existing = self.get(session_id)
        if existing:
            if user_identifier and not existing.user_identifier:
                existing.user_identifier = user_identifier
                existing.touch()
                self.update(existing)
            return existing
        return self.create_new(session_id, user_identifier)

    def update(self, session: UserSession) -> None:
        """Update an existing session in cache and DB."""
        session.touch()
        self._cache[session.session_id] = session
        self._write_to_db(session, session.user_identifier)

    def delete(self, session_id: str) -> None:
        """Delete a session from cache and DB. Does NOT delete documents or Qdrant chunks."""
        user_id = ""
        session = self.get(session_id)
        if session:
            user_id = session.user_identifier
        self._cache.pop(session_id, None)
        if self._conn:
            try:
                self._ensure_conn()
                with self._conn.cursor() as cur:
                    cur.execute("DELETE FROM user_sessions WHERE session_id = %s", (session_id,))
            except Exception as exc:
                logger.warning("DB DELETE failed for session %s: %s", session_id, exc)
        logger.info(
            "Deleted session %s for user %s. Documents preserved (user-scoped).",
            session_id, user_id
        )
        self._cleanup_images(session_id)

    def _cleanup_images(self, session_id: str):
        """Cleanup extracted/cropped images for a session (temp files only)."""
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

    def cleanup(self, session_id: str) -> None:
        """Alias for _cleanup_images — kept for backward compat."""
        self._cleanup_images(session_id)

    def list_sessions(self) -> list[UserSession]:
        """Return all cached sessions."""
        return list(self._cache.values())

    # ── Internal ─────────────────────────────────────────────────────────

    def _write_to_db(self, session: UserSession, user_identifier: str = ""):
        """Write/upsert a session to Supabase. Does NOT include ingested_documents
        (those live in the documents table now)."""
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
        """Read a session from Supabase by session_id."""
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
