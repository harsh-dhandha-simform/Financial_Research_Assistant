"""
query_rewriter — rewrites user queries for better retrieval.

Handles pronoun resolution and context expansion using chat history.
E.g. "their revenue" → "Apple Inc. revenue"

Used by: Chat Agent.
"""

import logging

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool

logger = logging.getLogger(__name__)

REWRITE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", (
        "You are a query rewriting assistant. Given a user question and "
        "chat history, rewrite the question as a standalone query that "
        "can be used for document retrieval. Resolve pronouns, expand "
        "context, and make the query self-contained. "
        "Return ONLY the rewritten query, nothing else."
    )),
    ("human", (
        "Chat History:\n{chat_history}\n\n"
        "User Question: {question}\n\n"
        "Rewritten standalone query:"
    )),
])


@tool
def query_rewriter(question: str, chat_history: str = "") -> str:
    """Rewrite a user question into a standalone retrieval query.

    Resolves pronouns and expands context from chat history.
    E.g., "What about margins?" with history about Apple 10-K
    → "What is Apple Inc. gross and operating margin?"

    Args:
        question: The user's current question.
        chat_history: Formatted previous conversation turns.

    Returns:
        A standalone rewritten query string.
    """
    if not chat_history:
        logger.info("query_rewriter: no history, returning as-is")
        return question

    from agents.base import get_llm
    llm = get_llm("news")  # Use a fast model for rewriting
    chain = REWRITE_PROMPT | llm

    result = chain.invoke({
        "question": question,
        "chat_history": chat_history,
    })

    rewritten = result.content.strip()
    logger.info(
        "query_rewriter: '%s' → '%s'",
        question[:50],
        rewritten[:50],
    )
    return rewritten
