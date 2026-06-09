"""
Redis-backed session store with in-memory cache.

Persists UserSession objects in Redis so sessions survive app restarts.
Uses an in-memory dict as a read-through cache to avoid repeated Redis hits.

Redis connection: configurable via REDIS_URL env var (default: redis://localhost:6379).
Sessions persist for 24 hours minimum (Redis TTL).

Usage:
    from sessions.session_store import session_store

    session = session_store.get("session-id")  # or None
    session = session_store.create(UserSession(...))
    session_store.update(session)
    session_store.delete("session-id")
    session_store.cleanup_expired(max_age_hours=24)
"""

import json
import logging
import os
import shutil
from datetime import datetime, timedelta
from typing import Optional

import redis

from config import settings
from sessions.session_model import UserSession

logger = logging.getLogger(__name__)

# Redis key prefix for session data
SESSION_PREFIX = "finsession:"
# Default TTL: 24 hours
DEFAULT_TTL_SECONDS = 24 * 60 * 60


class SessionStore:
    """Redis-backed session store with in-memory caching.

    All sessions are stored in Redis with a configurable TTL.
    An in-memory dict acts as a fast read-through cache.
    """

    def __init__(self, redis_url: str = ""):
        self._redis_url = redis_url or settings.redis_url
        self._cache: dict[str, UserSession] = {}
        self._redis: Optional[redis.Redis] = None
        self._ttl = DEFAULT_TTL_SECONDS

        self._connect()

    # ── Redis connection ─────────────────────────────────────────────────

    def _connect(self):
        """Establish connection to Redis."""
        try:
            self._redis = redis.from_url(
                self._redis_url,
                decode_responses=True,
                socket_connect_timeout=5,
                retry_on_timeout=True,
            )
            # Verify connection
            self._redis.ping()
            logger.info("SessionStore connected to Redis at %s", self._redis_url)
        except Exception as exc:
            logger.warning(
                "Redis connection failed (%s): %s — sessions will be in-memory only",
                self._redis_url, exc,
            )
            self._redis = None

    @property
    def is_redis_available(self) -> bool:
        """Check if Redis is reachable."""
        if self._redis is None:
            return False
        try:
            self._redis.ping()
            return True
        except Exception:
            return False

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
        }
        return json.dumps(data, default=str)

    @staticmethod
    def _deserialize(data_json: str) -> UserSession:
        """Reconstruct a UserSession from its JSON representation."""
        data = json.loads(data_json)
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
        )

    def _redis_key(self, session_id: str) -> str:
        """Build the Redis key for a session."""
        return f"{SESSION_PREFIX}{session_id}"

    # ── CRUD ─────────────────────────────────────────────────────────────

    def create(self, session: UserSession) -> UserSession:
        """Create a new session. Writes to both cache and Redis."""
        self._cache[session.session_id] = session
        self._write_to_redis(session)
        logger.info(
            "Created session: %s (collection: %s)",
            session.session_id, session.collection_name,
        )
        return session

    def get(self, session_id: str) -> Optional[UserSession]:
        """Get a session by ID. Returns None if not found."""
        # Check in-memory cache first
        session = self._cache.get(session_id)
        if session:
            return session

        # Fallback: check Redis
        if self._redis:
            try:
                data_json = self._redis.get(self._redis_key(session_id))
                if data_json:
                    session = self._deserialize(data_json)
                    self._cache[session_id] = session
                    return session
            except Exception as exc:
                logger.warning("Redis GET failed for session %s: %s", session_id, exc)

        return None

    def get_or_create(self, session_id: str) -> UserSession:
        """Get existing session or create a new one."""
        existing = self.get(session_id)
        if existing:
            existing.touch()
            self.update(existing)
            return existing

        session = UserSession(
            session_id=session_id,
            thread_id=session_id,
            collection_name=f"fin_{session_id[:8]}",
        )
        return self.create(session)

    def update(self, session: UserSession) -> None:
        """Update an existing session in cache and Redis."""
        session.touch()
        self._cache[session.session_id] = session
        self._write_to_redis(session)

    def delete(self, session_id: str) -> None:
        """Delete a session from cache and Redis."""
        self._cache.pop(session_id, None)
        if self._redis:
            try:
                self._redis.delete(self._redis_key(session_id))
            except Exception as exc:
                logger.warning("Redis DELETE failed for session %s: %s", session_id, exc)
        logger.info("Deleted session: %s", session_id)

    def cleanup(self, session_id: str) -> None:
        """Cleanup temp files for a session, but keep Qdrant collection.

        Only cleans up temporary image files. The Qdrant collection
        is preserved so the user can resume.
        """
        session = self.get(session_id)
        if not session:
            return

        # Clean up temp images
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

    def cleanup_expired(self, max_age_hours: int = 24) -> int:
        """Remove expired sessions from the in-memory cache.

        Redis handles TTL expiry automatically. This only cleans
        up the in-memory cache and temp files.
        """
        cutoff = datetime.utcnow() - timedelta(hours=max_age_hours)
        expired_ids = [
            sid for sid, session in self._cache.items()
            if session.last_active < cutoff
        ]

        for sid in expired_ids:
            self.cleanup(sid)
            self._cache.pop(sid, None)

        if expired_ids:
            logger.info(
                "Cleaned up %d expired sessions from cache (> %dh old)",
                len(expired_ids), max_age_hours,
            )
        return len(expired_ids)

    def list_sessions(self) -> list[UserSession]:
        """Return all active sessions from cache."""
        return list(self._cache.values())

    # ── Internal ─────────────────────────────────────────────────────────

    def _write_to_redis(self, session: UserSession):
        """Write a session to Redis with TTL."""
        if not self._redis:
            return
        try:
            data_json = self._serialize(session)
            self._redis.setex(
                self._redis_key(session.session_id),
                self._ttl,
                data_json,
            )
        except Exception as exc:
            logger.warning(
                "Failed to write session %s to Redis: %s",
                session.session_id, exc,
            )


# ── Singleton ────────────────────────────────────────────────────────────────
session_store = SessionStore()
