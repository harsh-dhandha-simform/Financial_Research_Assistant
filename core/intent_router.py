"""
Intent Router — determines the goal of the user's input.

Replaces the old "two-tab" UI design with a single unified orchestrator.
Uses Groq's llama-3.3-70b-versatile (or compound-mini) to classify user input.

Intents:
  FULL_PIPELINE   : "Analyse Apple's latest 10-K"
  SINGLE_AGENT    : "Just run the news agent for AAPL"
  RAG_CHAT        : "What was their revenue last quarter?"
  INGEST_DOC      : "/ingest https://sec.gov/..." or file upload
  EXPORT          : "Download the memo as PDF"
  VIEW_SOURCES    : "Show me the sources for the last answer"
  OUT_OF_DOMAIN   : "Write me a poem"
  CHITCHAT        : "Hello", "Thanks"
"""

import json
import logging
from enum import Enum
from typing import Any

from langchain_openai import ChatOpenAI

from config import settings
from callbacks import get_langfuse_handler

logger = logging.getLogger(__name__)


class UserIntent(str, Enum):
    FULL_PIPELINE = "full_pipeline"
    SINGLE_AGENT = "single_agent"
    RAG_CHAT = "rag_chat"
    INGEST_DOC = "ingest_doc"
    EXPORT = "export"
    VIEW_SOURCES = "view_sources"
    OUT_OF_DOMAIN = "out_of_domain"
    CHITCHAT = "chitchat"


# We use a 70B model or similar for reliable structured routing
ROUTER_MODEL = "llama-3.3-70b-versatile"
# Fallback to faster model if needed
FALLBACK_MODEL = "compound-mini"

ROUTER_SYSTEM_PROMPT = """You are the central intent router for a Financial Research Assistant.
Your job is to read the user's input and classify their intent into exactly one of the allowed categories.

ALLOWED INTENTS:
1. "full_pipeline": User wants a complete research analysis (metrics, risks, news) on a company. (e.g., "Analyze Apple", "Run research on TSLA", "Give me a report on MSFT")
2. "single_agent": User explicitly asks to run just ONE agent. (e.g., "Run the news agent for AAPL", "Just get the metrics for Apple")
3. "rag_chat": User is asking a specific financial question that requires searching the database or documents. (e.g., "What was the revenue in Q3?", "What are the main risk factors?")
4. "ingest_doc": User is providing a URL to ingest, or asking to upload a document. (e.g., "/ingest http...", "Here is the PDF to read")
5. "export": User wants to export or download a report/memo as PDF or DOCX. (e.g., "Download as PDF", "Export memo")
6. "view_sources": User asks to see the sources, citations, or original document for an answer. (e.g., "Where did you get that?", "Show me the sources")
7. "out_of_domain": User asks for something completely unrelated to finance, investing, or the assistant's capabilities. (e.g., "Write a poem", "How do I bake a cake?")
8. "chitchat": General conversational pleasantries. (e.g., "Hello", "Thanks", "Ok")

Respond ONLY in valid JSON with the following schema:
{
  "reasoning": "brief explanation of why",
  "intent": "one of the allowed intents",
  "extracted_company": "company name if mentioned, else null",
  "extracted_ticker": "ticker if mentioned, else null",
  "target_agent": "metrics/risk/news if single_agent, else null"
}
"""

def route_intent(user_input: str, session_id: str = "") -> dict[str, Any]:
    """Classify the user's input into a specific intent."""
    logger.info("Routing intent for input: '%s'", user_input[:80])

    if not settings.groq_api_key:
        logger.warning("GROQ_API_KEY missing — defaulting to RAG_CHAT")
        return {"intent": UserIntent.RAG_CHAT.value, "reasoning": "No Groq key"}

    # Special case handling for explicit commands
    if user_input.lower().startswith("/ingest"):
        return {"intent": UserIntent.INGEST_DOC.value, "reasoning": "Explicit /ingest command"}

    llm = ChatOpenAI(
        base_url="https://api.groq.com/openai/v1",
        api_key=settings.groq_api_key,
        model=ROUTER_MODEL,
        temperature=0,
        response_format={"type": "json_object"},
    )

    handler = get_langfuse_handler(
        session_id=session_id,
        trace_name="intent_classification",
    )

    try:
        response = llm.invoke(
            [
                {"role": "system", "content": ROUTER_SYSTEM_PROMPT},
                {"role": "user", "content": user_input},
            ],
            config={"callbacks": [handler]},
        )
        
        result = json.loads(response.content)
        
        # Validate intent
        intent_str = result.get("intent", "").lower()
        try:
            intent = UserIntent(intent_str)
        except ValueError:
            intent = UserIntent.RAG_CHAT
            
        result["intent"] = intent.value
        
        logger.info(
            "Intent router classified as %s (company: %s)",
            result["intent"], result.get("extracted_company")
        )
        
        return result

    except Exception as exc:
        logger.warning("Intent routing failed: %s — falling back to RAG_CHAT", exc)
        return {
            "intent": UserIntent.RAG_CHAT.value,
            "reasoning": f"Routing error: {exc}"
        }
