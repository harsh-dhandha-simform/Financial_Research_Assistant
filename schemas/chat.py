"""
Pydantic models for the Conversational RAG chat layer (Module 19).

The chat graph maintains message history and uses it for:
  - Query rewriting (incorporating conversation context)
  - Query decomposition (splitting complex questions)
  - Answer generation with source attribution
"""

from datetime import datetime, timezone
from enum import Enum

from pydantic import BaseModel, Field

from schemas.citation import Citation


class MessageRole(str, Enum):
    """Chat message roles."""

    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class ChatMessage(BaseModel):
    """A single message in the conversation history."""

    role: MessageRole = Field(..., description="Message role")
    content: str = Field(..., description="Message content")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When the message was created",
    )
    sources: list[Citation] = Field(
        default_factory=list,
        description="Source chunks used to generate this response",
    )
    metadata: dict = Field(default_factory=dict, description="Additional metadata")


class ChatRequest(BaseModel):
    """Incoming request for the conversational RAG endpoint."""

    query: str = Field(..., description="User's question")
    session_id: str = Field(
        default="", description="Session ID to maintain conversation context"
    )
    doc_ids: list[str] = Field(
        default_factory=list,
        description="Restrict retrieval to specific document IDs (empty = search all)",
    )
    top_k: int = Field(
        default=5, ge=1, le=20, description="Number of chunks to retrieve"
    )


class ChatResponse(BaseModel):
    """Response from the conversational RAG endpoint."""

    answer: str = Field(..., description="Generated answer")
    session_id: str = Field(..., description="Session ID for follow-up messages")
    sources: list[Citation] = Field(
        default_factory=list,
        description="Source chunks used to generate the answer",
    )
    relevant: bool = Field(
        default=True,
        description="Whether relevant context was found for the query",
    )
    rewritten_query: str = Field(
        default="",
        description="The rewritten query after context expansion (for debugging)",
    )
