"""
Chat Guardrail — validates user questions in the Chat Q&A profile.

Runs before any RAG retrieval is performed. If the question is off-topic
or inappropriate, it returns a rejection message immediately.
"""

import json
import logging
from typing import tuple

from langchain_openai import ChatOpenAI

from config import settings
from callbacks import get_langfuse_handler

logger = logging.getLogger(__name__)

CHAT_GUARDRAIL_SYSTEM = """You are a strict guardrail for a financial chat assistant.
Your job is to determine if the user's question is relevant to financial research,
a specific company, SEC filings, investing, or market analysis.

If the query is valid, respond with:
{"valid": true, "reason": "valid financial question"}

If the query is completely unrelated (e.g. coding help, writing a poem, recipes,
or general chitchat), respond with:
{"valid": false, "reason": "Brief explanation to the user of why you cannot answer this. Be polite."}

Output MUST be valid JSON only.
"""

def check_chat_guardrail(question: str, session_id: str) -> tuple[bool, str]:
    """Check if a chat question is valid before performing RAG.
    
    Returns:
        (is_valid, reason_or_rejection_message)
    """
    if not settings.groq_api_key:
        logger.warning("GROQ_API_KEY missing — skipping chat guardrail")
        return True, ""

    llm = ChatOpenAI(
        base_url="https://api.groq.com/openai/v1",
        api_key=settings.groq_api_key,
        model="compound-mini",
        temperature=0,
        max_tokens=150,
    )

    handler = get_langfuse_handler(
        session_id=session_id,
        trace_name="chat_guardrail",
    )

    try:
        response = llm.invoke(
            [
                {"role": "system", "content": CHAT_GUARDRAIL_SYSTEM},
                {"role": "user", "content": question},
            ],
            config={"callbacks": [handler]},
        )
        
        try:
            result = json.loads(response.content)
            is_valid = result.get("valid", True)
            reason = result.get("reason", "No reason provided")
        except json.JSONDecodeError:
            text = response.content.lower()
            is_valid = "true" in text or "yes" in text
            reason = response.content[:200]

        if not is_valid:
            logger.info("Chat Guardrail REJECTED: %s", reason)
            return False, f"I can only answer questions related to financial research and analysis. {reason}"
        
        return True, ""

    except Exception as exc:
        logger.warning("Chat Guardrail failed: %s — allowing question", exc)
        return True, ""
