"""
Chat router — conversational RAG Q&A.

POST /chat          → ask a question with RAG retrieval + memory
POST /chat/clear    → clear session history
"""

import logging

from fastapi import APIRouter

from api.models import ChatRequest, ChatResponse
from chat.agent import chat, clear_session

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("", response_model=ChatResponse)
async def chat_endpoint(req: ChatRequest):
    """Ask a question about an analysed company.

    Uses RAG retrieval over ingested documents, conversation memory
    for multi-turn context, and pipeline results if available.
    """
    result = chat(
        question=req.question,
        session_id=req.session_id,
    )
    return result


@router.post("/clear")
async def clear_chat(session_id: str):
    """Clear conversation history for a session."""
    clear_session(session_id)
    return {"status": "cleared", "session_id": session_id}
