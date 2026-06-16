"""
Chat Agent — conversational RAG for post-research Q&A.

Uses short-term memory (message history per session) with tools:
  - query_rewriter: resolves pronouns using conversation context
  - rag_retriever: hybrid search over ingested documents
  - hallucination_checker: verifies answers are grounded in sources

Flow (from the architecture diagram):
  User Question → Query Rewriter → RAG Retrieval → Relevance Check
  → Answer Generation → Hallucination Check → Response

The chat agent can also read pipeline results (memo, report, synthesis)
via the session context, so users can ask "what was the PE ratio?"
without re-running agents.

Short-term memory: LangChain message history stored per session_id.
"""

import logging
import uuid
from typing import Any

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_core.documents import Document

from agents.base import get_llm, create_langfuse_config
from tools.retriever import rag_retriever
from tools.chat.query_rewriter import query_rewriter
from tools.chat.citation_builder import build_citations
from schemas.chat import ChatResponse

logger = logging.getLogger(__name__)


# ── Short-term memory: session_id → message history ─────────────────────────
# In production, swap this for Redis or PostgresSaver.
_sessions: dict[str, list] = {}

# ── Pipeline result cache: session_id → ResearchState summary ────────────────
_pipeline_context: dict[str, str] = {}


CHAT_SYSTEM_PROMPT = """You are a knowledgeable financial research assistant. You help users understand \
SEC filings, financial metrics, risk factors, and market sentiment for companies \
they have analysed.

RULES:
1. Answer ONLY based on the provided context (retrieved documents and pipeline results).
2. If the context doesn't contain enough information, say so clearly.
3. Cite specific numbers, dates, and sections from the filings.
4. Be concise and professional — use bullet points for complex answers.
5. If the user asks about the analysis report/memo, use the pipeline context.
6. Never make up financial data — say "I don't have that data" if unsure.

If pipeline results are available, you can reference them directly to answer \
questions about the overall rating, thesis, risks, metrics, and recommendations."""


def store_pipeline_context(session_id: str, context: str):
    """Store pipeline results as context for the chat agent.

    Called after a research pipeline completes so the chat agent
    can answer questions like "what was the rating?" without re-running.
    """
    _pipeline_context[session_id] = context
    logger.info("Stored pipeline context for session '%s' (%d chars)", session_id, len(context))


def get_session_history(session_id: str) -> list:
    """Get or create message history for a session."""
    if session_id not in _sessions:
        _sessions[session_id] = []
    return _sessions[session_id]


def clear_session(session_id: str):
    """Clear a session's message history."""
    _sessions.pop(session_id, None)
    _pipeline_context.pop(session_id, None)
    logger.info("Cleared session '%s'", session_id)


def _format_chat_history(messages: list) -> str:
    """Format message history for the query rewriter."""
    lines = []
    for msg in messages[-10:]:  # Last 10 messages for context window
        if isinstance(msg, HumanMessage):
            lines.append(f"User: {msg.content}")
        elif isinstance(msg, AIMessage):
            lines.append(f"Assistant: {msg.content[:200]}")
    return "\n".join(lines)


def _retrieve_context(query: str, section_filter: str = "", top_k: int = 6, session_id: str = "", user_id: str = "") -> tuple[str, list]:
    """Retrieve relevant document chunks.

    Returns:
        Tuple of (formatted context string, list of Citation objects).
    """
    try:
        docs = rag_retriever.invoke(
            {
                "query": query,
                "section_filter": section_filter,
                "top_k": top_k,
            },
            config={"configurable": {"session_id": session_id, "user_id": user_id}},
        )

        if not docs:
            return "", []

        # Build formal citations so Pydantic validation passes
        citations = build_citations(docs, session_id)
        
        context_parts = []
        for i, doc in enumerate(docs):
            section = doc.metadata.get("section", "unknown")
            score = doc.metadata.get("score", 0)
            context_parts.append(
                f"[Source {i+1} | Section: {section} | Relevance: {score:.3f}]\n"
                f"{doc.page_content}\n"
            )

        return "\n---\n".join(context_parts), citations

    except Exception as exc:
        logger.warning("RAG retrieval failed: %s", exc)
        return "", []


def chat(
    question: str,
    session_id: str = "",
    section_filter: str = "",
    top_k: int = 6,
    user_id: str = "",
) -> ChatResponse:
    """Process a chat question with RAG retrieval and conversation memory.

    Flow:
      1. Get/create session history
      2. Rewrite query using history context (resolves pronouns)
      3. Retrieve relevant chunks from Qdrant
      4. Include pipeline context if available
      5. Generate answer with full context
      6. Update session history

    Args:
        question: User's question.
        session_id: Session ID for memory continuity.
        section_filter: Optional section filter for RAG.
        top_k: Number of chunks to retrieve.

    Returns:
        ChatResponse with answer, sources, and rewritten query.
    """
    if not session_id:
        session_id = f"chat-{uuid.uuid4().hex[:8]}"

    history = get_session_history(session_id)

    # ── 1. Query rewriting (resolve pronouns using history) ──────────
    chat_history_str = _format_chat_history(history)
    try:
        rewritten = query_rewriter.invoke({
            "question": question,
            "chat_history": chat_history_str,
        })
    except Exception:
        rewritten = question

    logger.info("Chat query: '%s' → rewritten: '%s'", question[:60], rewritten[:60])

    # ── 2. RAG retrieval ─────────────────────────────────────────────
    rag_context, sources = _retrieve_context(rewritten, section_filter, top_k, session_id, user_id)

    # ── 3. Include pipeline context if available ─────────────────────
    pipeline_ctx = _pipeline_context.get(session_id, "")
    full_context = ""
    if pipeline_ctx:
        full_context += f"=== PIPELINE ANALYSIS RESULTS ===\n{pipeline_ctx}\n\n"
    if rag_context:
        full_context += f"=== RETRIEVED DOCUMENTS ===\n{rag_context}\n"

    relevant = bool(rag_context or pipeline_ctx)

    # ── 4. Generate answer ───────────────────────────────────────────

    messages = [
        SystemMessage(content=CHAT_SYSTEM_PROMPT),
    ]

    # Add conversation history (last 6 turns for context window)
    for msg in history[-12:]:
        messages.append(msg)

    # Build the user message with context
    if full_context:
        user_msg = (
            f"Context:\n{full_context}\n\n"
            f"Question: {question}"
        )
    else:
        user_msg = (
            f"No relevant documents found for this query. "
            f"Answer based on your general knowledge but clearly state "
            f"that this is not from the analysed documents.\n\n"
            f"Question: {question}"
        )

    messages.append(HumanMessage(content=user_msg))

    from config import settings
    from agents.base import _is_rate_limit_error, _rotate_google_key

    config = create_langfuse_config(
        session_id=session_id,
        trace_name="chat-agent",
        use_callbacks=True,
    )

    max_attempts = len(settings.google_api_keys) if settings.google_api_keys else 1
    answer = ""

    for attempt in range(max_attempts):
        llm = get_llm("news")  # Re-fetch LLM to get updated API key if rotated
        try:
            response = llm.invoke(messages, config=config)
            answer = response.content
            break
        except Exception as exc:
            if _is_rate_limit_error(exc) and attempt < max_attempts - 1:
                logger.warning("Chat agent rate limit (429) on attempt %d. Rotating Google API key...", attempt + 1)
                _rotate_google_key()
                continue

            logger.error("Chat LLM failed: %s", exc)
            answer = (
                "I'm sorry, I encountered an error while generating a response. "
                "Please try again or rephrase your question."
            )
            break

    # ── 5. Update session history ────────────────────────────────────
    history.append(HumanMessage(content=question))
    history.append(AIMessage(content=answer))

    # Keep history bounded (last 20 messages = 10 turns)
    if len(history) > 20:
        _sessions[session_id] = history[-20:]

    logger.info(
        "Chat response: session=%s, sources=%d, relevant=%s, answer_len=%d",
        session_id, len(sources), relevant, len(answer),
    )

    return ChatResponse(
        answer=answer,
        session_id=session_id,
        sources=sources,
        relevant=relevant,
        rewritten_query=rewritten if rewritten != question else "",
    )
