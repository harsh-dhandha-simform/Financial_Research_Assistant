"""
Query Guardrail node — validates the user's research query.

Uses groq/llama-3.1-8b-instant to quickly classify if the query is a valid
financial/investment research request. If not, it routes to END.
"""

import json
import logging
from typing import Any

from langchain_openai import ChatOpenAI

from config import settings
from graph.state import ResearchState
from callbacks import get_langfuse_handler
from langfuse.decorators import observe

logger = logging.getLogger(__name__)

GUARDRAIL_SYSTEM = """You are a strict guardrail for a financial research assistant.
Your job is to determine if the user's query is related to financial research,
investment analysis, company metrics, or stock market news.

If the query is a valid financial research request, respond with:
{"valid": true, "reason": "valid financial query"}

If the query is completely unrelated (e.g. coding help, recipes, creative writing,
or general chitchat), respond with:
{"valid": false, "reason": "Brief explanation of why it was rejected"}

Output MUST be valid JSON only.
"""

@observe()
def query_guardrail_node(state: ResearchState) -> dict[str, Any]:
    """Validate the user query before kicking off the research pipeline."""
    logger.info("Guardrail: checking query: '%s'", state.query[:80])

    if not settings.groq_api_key:
        logger.warning("GROQ_API_KEY missing — skipping query guardrail")
        return {"guardrail_rejected": False, "rejection_message": ""}

    llm = ChatOpenAI(
        base_url="https://api.groq.com/openai/v1",
        api_key=settings.groq_api_key,
        model="llama-3.1-8b-instant",
        temperature=0,
        max_tokens=150,
    )

    handler = get_langfuse_handler(
        session_id=state.session_id,
        trace_name="query_guardrail",
    )

    try:
        response = llm.invoke(
            [
                {"role": "system", "content": GUARDRAIL_SYSTEM},
                {"role": "user", "content": state.query},
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
            logger.info("Guardrail REJECTED query: %s", reason)
            return {
                "guardrail_rejected": True,
                "rejection_message": f"Query rejected: {reason}",
            }
        
        logger.info("Guardrail ACCEPTED query")
        return {"guardrail_rejected": False, "rejection_message": ""}

    except Exception as exc:
        logger.warning("Guardrail check failed: %s — allowing query", exc)
        return {"guardrail_rejected": False, "rejection_message": ""}
