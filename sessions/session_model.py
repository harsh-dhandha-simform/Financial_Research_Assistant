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
        session_id: Unique session identifier (from Chainlit).
        thread_id: LangGraph MemorySaver thread_id (== session_id).
        created_at: Session creation timestamp.
        last_active: Last activity timestamp.
        ingested_documents: List of ingested doc metadata dicts.
            Each doc: {doc_id, name, type, source_url, ingested_at,
                       chunk_count, image_count, local_path, collection_name}
        collection_name: Per-session Qdrant collection (fin_{session_id[:8]}).
        pipeline_status: idle | running | waiting_hitl | done | error.
        last_pipeline_result: Stores InvestmentMemo + ResearchReport dicts.
        chat_history: Last N turns of chat (dicts with role + content).
        company_name: Current company being analysed.
        ticker: Current ticker symbol.
    """

    session_id: str
    thread_id: str                    # == session_id, used by LangGraph MemorySaver
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_active: datetime = field(default_factory=datetime.utcnow)

    # Document tracking
    ingested_documents: list[dict] = field(default_factory=list)
    # Each doc: {doc_id, name, type, source_url, ingested_at, chunk_count,
    #            image_count, local_path, collection_name, images: [{path, page, ...}]}

    # Per-session Qdrant collection (isolates vector data per user)
    collection_name: str = ""         # set to f"fin_{session_id[:8]}" on init

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
        """Set collection_name from session_id if not provided."""
        if not self.collection_name and self.session_id:
            self.collection_name = f"fin_{self.session_id[:8]}"
        if not self.thread_id:
            self.thread_id = self.session_id

    def touch(self):
        """Update last_active timestamp."""
        self.last_active = datetime.utcnow()

    def add_document(self, doc_meta: dict):
        """Register a newly ingested document."""
        self.ingested_documents.append(doc_meta)
        self.touch()

    def remove_document(self, doc_id: str) -> bool:
        """Remove a document by doc_id. Returns True if found."""
        before = len(self.ingested_documents)
        self.ingested_documents = [
            d for d in self.ingested_documents if d.get("doc_id") != doc_id
        ]
        return len(self.ingested_documents) < before

    @property
    def has_documents(self) -> bool:
        """Check if any documents are ingested."""
        return len(self.ingested_documents) > 0

    @property
    def total_chunks(self) -> int:
        """Total chunk count across all ingested documents."""
        return sum(d.get("chunk_count", 0) for d in self.ingested_documents)

    @property
    def total_images(self) -> int:
        """Total image count across all ingested documents."""
        return sum(d.get("image_count", 0) for d in self.ingested_documents)

    def trim_chat_history(self, max_turns: int = 6):
        """Keep only the last N turns (each turn = user + assistant)."""
        max_messages = max_turns * 2
        if len(self.chat_history) > max_messages:
            self.chat_history = self.chat_history[-max_messages:]
