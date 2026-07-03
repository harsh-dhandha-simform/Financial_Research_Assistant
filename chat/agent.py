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

_sessions: dict[str, list] = {}
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
from config import settings

from tools.retriever import rag_retriever
from tools.chat.query_rewriter import query_rewriter
from tools.chat.citation_builder import build_citations
from schemas.chat import ChatResponse

logger = logging.getLogger(__name__)


# ── Pipeline result cache: session_id → ResearchState summary ────────────────
_pipeline_context: dict[str, str] = {}


CHAT_SYSTEM_PROMPT = """You are a knowledgeable financial research assistant. You help users understand \
SEC filings, financial metrics, risk factors, and market sentiment for companies \
they have analysed.

RULES:
1. Answer based on the provided context (retrieved documents and pipeline results).
2. You also have access to the FULL conversation history above — use it to:
   - Answer meta-questions like "what did I ask before?" or "summarise our chat"
   - Resolve what 'it', 'they', 'the company' refers to
   - Provide continuity across questions
3. If the context doesn’t contain enough information AND chat history doesn’t help, say so.
4. Cite specific numbers, dates, and sections from the filings.
5. Be concise and professional — use bullet points for complex answers.
6. If the user asks about the analysis report/memo, use the pipeline context.
7. Never make up financial data — say "I don’t have that data" if unsure.
8. For questions about what was previously asked/said, look at the conversation history messages above.

If pipeline results are available, you can reference them directly to answer \
questions about the overall rating, thesis, risks, metrics, and recommendations."""


def store_pipeline_context(session_id: str, context: str):
    """Store pipeline results as context for the chat agent.

    Called after a research pipeline completes so the chat agent
    can answer questions like "what was the rating?" without re-running.
    """
    _pipeline_context[session_id] = context
    logger.info("Stored pipeline context for session '%s' (%d chars)", session_id, len(context))


def clear_session(session_id: str):
    """Clear a session's message history."""
    _pipeline_context.pop(session_id, None)
    logger.info("Cleared session '%s'", session_id)


def _format_chat_history(messages: list) -> str:
    """Format message history for the query rewriter."""
    lines = []
    for msg in messages[-20:]:  # Last 20 messages (10 turns) for context window
        if isinstance(msg, HumanMessage):
            lines.append(f"User: {msg.content}")
        elif isinstance(msg, AIMessage):
            lines.append(f"Assistant: {msg.content[:300]}")
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
            if isinstance(doc.metadata, dict):
                section = doc.metadata.get("section", "unknown")
                score = doc.metadata.get("score", 0)
            else:
                section = "unknown"
                score = 0
            context_parts.append(
                f"[Source {i+1} | Section: {section} | Relevance: {score:.3f}]\n"
                f"{doc.page_content}\n"
            )

        return "\n---\n".join(context_parts), citations

    except Exception as exc:
        logger.warning("RAG retrieval failed: %s", exc)
        return "", []
def _generate_answer_with_fallback(messages: list, config: dict) -> tuple[str, Exception | None]:
    """Helper to try models in AGENT_MODELS['news'] with fallbacks and key rotation."""
    from agents.base import (
        AGENT_MODELS, _create_llm, _get_api_key,
        _is_rate_limit_error, _is_overloaded_error,
        _rotate_google_key, _classify_error,
    )
    model_configs = AGENT_MODELS.get("news", [])
    answer = ""
    last_error: Exception | None = None

    # Try up to 2 full cycles of the model fallback chain
    for cycle in range(2):
        if cycle > 0:
            logger.info("Chat agent [cycle %d]: Retrying model chain after Google API key rotation", cycle)

        for i, model_config in enumerate(model_configs):
            label = "primary" if i == 0 else f"fallback-{i}"
            if cycle > 0:
                label += f"-cycle{cycle}"

            # Google: try all available API keys before giving up on this model
            google_key_attempts = (
                len(settings.google_api_keys) if model_config.provider == "google" else 1
            )

            for key_attempt in range(google_key_attempts):
                api_key = _get_api_key(model_config.provider)
                if not api_key:
                    logger.warning(
                        "Chat agent [%s]: no API key for %s — skipping",
                        label, model_config.provider,
                    )
                    break  # No key → skip this model entirely

                try:
                    llm = _create_llm(model_config)
                    response = llm.invoke(messages, config=config)
                    answer = response.content
                    logger.info(
                        "Chat agent [%s]: ✅ answered via %s",
                        label, model_config.model_id,
                    )
                    break  # Success — stop key/model iteration

                except Exception as exc:
                    error_type = _classify_error(exc)
                    last_error = exc

                    # Google 429 (rate limit): rotate key and retry same model.
                    # Google 503 (overloaded): key rotation won't help — skip to next model.
                    if (
                        model_config.provider == "google"
                        and _is_rate_limit_error(exc)
                        and not _is_overloaded_error(exc)
                        and key_attempt < google_key_attempts - 1
                    ):
                        logger.warning(
                            "Chat agent [%s]: ❌ rate limit (429) on key %d/%d for %s — rotating key",
                            label, key_attempt + 1, google_key_attempts, model_config.model_id,
                        )
                        _rotate_google_key()
                        continue  # retry with next Google key

                    # Any other error (503, timeout, connection, etc.): fall through to next model
                    logger.warning(
                        "Chat agent [%s]: ❌ %s from %s — trying next model. Error: %s",
                        label, error_type, model_config.model_id, str(exc)[:200],
                    )
                    break  # Exit key-attempt loop → outer loop tries next model

            if answer:
                break  # Got an answer — exit model loop

        if answer:
            break  # Got an answer — exit cycle loop
        else:
            # If all fallback models failed in the first cycle, rotate Google API key for the next attempt
            if cycle == 0:
                logger.warning("Chat agent: all fallback models failed in cycle 0. Rotating Google key and retrying...")
                _rotate_google_key()

    return answer, last_error


def chat(
    question: str,
    session_id: str = "",
    section_filter: str = "",
    top_k: int = 6,
    user_id: str = "",
    skip_rag: bool = False,
    history: list[dict] = None,
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

    if not isinstance(history, list):
        history = []

    langchain_history = []
    safe_history = [m for m in history[-30:] if isinstance(m, dict)]  # last 15 turns
    for msg in safe_history:
        if isinstance(msg, dict):
            role = msg.get("role")
            content = msg.get("content", "")
            if role == "user":
                langchain_history.append(HumanMessage(content=content))
            elif role == "assistant":
                langchain_history.append(AIMessage(content=content))
        elif hasattr(msg, "type"):
            # It's already a LangChain message (or similar)
            langchain_history.append(msg)

    # ── 1. Query rewriting (resolve pronouns using history) ──────────
    chat_history_str = _format_chat_history(langchain_history)
    try:
        rewritten = query_rewriter.invoke({
            "question": question,
            "chat_history": chat_history_str,
        })
    except Exception:
        rewritten = question

    logger.info("Chat query: '%s' → rewritten: '%s'", question[:60], rewritten[:60])

    # ── 2. RAG retrieval ─────────────────────────────────────────────
    if not skip_rag:
        rag_context, sources = _retrieve_context(rewritten, section_filter, top_k, session_id, user_id)
    else:
        rag_context, sources = "", []

    # ── 3. Include pipeline context if available ─────────────────────
    pipeline_ctx = _pipeline_context.get(session_id, "") if isinstance(_pipeline_context, dict) else ""
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

    # Add conversation history (last 5 turns for context window)
    for msg in langchain_history[-10:]:
        messages.append(msg)

    # Build the user message with context
    if full_context:
        user_msg = (
            f"Context:\n{full_context}\n\n"
            f"Question: {question}"
        )
    elif skip_rag:
        # Conversational / chitchat queries that don't need context
        user_msg = f"Question: {question}"
    else:
        user_msg = (
            f"No relevant documents found for this query. "
            f"Answer based on your general knowledge but clearly state "
            f"that this is not from the analysed documents.\n\n"
            f"Question: {question}"
        )

    messages.append(HumanMessage(content=user_msg))

    config = create_langfuse_config(
        session_id=session_id,
        trace_name="chat-agent",
        use_callbacks=True,
    )

    answer = ""
    last_error: Exception | None = None
    current_messages = list(messages)
    attempts = 3

    for attempt in range(attempts):
        logger.info("Chat agent: generation attempt %d/%d", attempt + 1, attempts)
        answer, last_error = _generate_answer_with_fallback(current_messages, config)
        if not answer:
            break

        # Check for hallucinations only if there is retrieval context to verify against
        if not skip_rag and relevant:
            try:
                from tools.chat.hallucination import hallucination_checker
                check_res = hallucination_checker(answer, full_context)
                is_grounded = check_res.get("is_grounded", True)
                ungrounded = check_res.get("ungrounded_claims", [])

                if is_grounded:
                    logger.info("Chat agent: answer grounded successfully on attempt %d", attempt + 1)
                    break
                else:
                    logger.warning(
                        "Chat agent: hallucination detected on attempt %d: %d ungrounded claims: %s",
                        attempt + 1, len(ungrounded), ungrounded
                    )
                    if attempt < attempts - 1:
                        # Append the hallucinated answer and a correction instruction to the conversation context
                        current_messages.append(AIMessage(content=answer))
                        claims_str = ", ".join(f"'{c}'" for c in ungrounded)
                        current_messages.append(HumanMessage(
                            content=(
                                f"Your previous answer contained claims not supported by the provided context: {claims_str}.\n"
                                f"Please rewrite the answer. Ensure every single statement you make is directly "
                                f"supported by the context. Do not extrapolate, assume, or make up facts."
                            )
                        ))
                        # Continue to next attempt
                        continue
            except Exception as e:
                logger.warning("Chat agent: hallucination check failed: %s — accepting current answer", e)
                break
        else:
            # Conversational/generic chitchat (no source context loaded)
            break

    if not answer:
        logger.error("Chat agent: all attempts/models failed. Last error: %s", last_error)
        answer = (
            "I'm sorry, I encountered an error generating a response. "
            "Please try again in a moment."
        )


    # ── 5. Update session history ────────────────────────────────────
    history.append(HumanMessage(content=question))
    history.append(AIMessage(content=answer))

    # Keep history bounded (last 10 messages = 5 turns)
    if len(history) > 10:
        _sessions[session_id] = history[-10:]

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
