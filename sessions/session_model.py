"""
UserSession dataclass — in-memory representation of a user session.

Tracks ingested documents, pipeline state, chat history, and per-session
Qdrant collection name.

Usage:
    from sessions.session_model import UserSession

    session = UserSession(
        session_id="abc12345",
        thread_id="abc12345",
        collection_name="fin_abc12345",
    )
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


@dataclass
class UserSession:
    """Persistent user session state.

    Fields:
        session_id: Unique session identifier (== Chainlit thread_id).
        thread_id: LangGraph MemorySaver thread_id (== session_id).
        created_at: Session creation timestamp.
        last_active: Last activity timestamp.
        collection_name: Qdrant collection for this user (fin_{user_id}).
        pipeline_status: idle | running | waiting_hitl | done | error.
        last_pipeline_result: Stores InvestmentMemo + ResearchReport dicts.
        chat_history: Last N turns of chat (dicts with role + content).
        company_name: Current company being analysed.
        ticker: Current ticker symbol.
        user_identifier: Authenticated user identifier.
    """

    session_id: str
    thread_id: str                    # == session_id, used by LangGraph MemorySaver
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_active: datetime = field(default_factory=datetime.utcnow)

    # Per-user Qdrant collection — set by session_store.create_new(), NOT derived from session_id
    collection_name: str = ""

    # Pipeline state
    pipeline_status: str = "idle"     # idle | running | waiting_hitl | done | error
    last_pipeline_result: Optional[dict] = None  # stores InvestmentMemo + ResearchReport

    # Chat state
    chat_history: list[dict] = field(default_factory=list)  # last 6 turns max

    # Active analysis context
    company_name: str = ""
    ticker: str = ""
    user_identifier: str = ""

    def __post_init__(self):
        """Ensure thread_id is set."""
        if not self.thread_id:
            self.thread_id = self.session_id

    def touch(self):
        """Update last_active timestamp."""
        self.last_active = datetime.utcnow()

    @property
    def has_documents(self) -> bool:
        """True when the user has a registered Qdrant collection."""
        return bool(self.collection_name)

    def trim_chat_history(self, max_turns: int = 6):
        """Keep only the last N turns (each turn = user + assistant)."""
        max_messages = max_turns * 2
        if len(self.chat_history) > max_messages:
            self.chat_history = self.chat_history[-max_messages:]
