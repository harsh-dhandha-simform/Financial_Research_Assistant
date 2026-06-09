"""
Session management — persistent user sessions with SQLite backend.

Usage:
    from sessions.session_store import session_store
    from sessions.session_model import UserSession

    session = session_store.get_or_create(session_id)
"""

from sessions.session_model import UserSession
from sessions.session_store import SessionStore
