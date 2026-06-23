"""
query_rewriter — adaptive query rewriting for better retrieval.

Rewrites user queries ONLY when needed:
  - Resolves pronouns/anaphoric references using chat history
  - Skips rewriting for long, self-contained queries (preserves exact
    terms which are critical for BM25 keyword matching)

E.g. "their revenue" → "Apple Inc. revenue"  (rewrite needed)
E.g. "As per the Indian Companies Act, 2013..." → unchanged  (self-contained)

Used by: Chat Agent.
"""

import logging
import re

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


# Pronoun / anaphoric tokens that signal dependency on prior context
_REFERENCE_TOKENS = frozenset({
    "it", "its", "they", "their", "them", "that", "this", "these", "those",
    "he", "she", "his", "her", "same", "above", "mentioned", "previous",
    "earlier", "prior", "aforementioned",
})

# Short reference phrases (2 tokens) indicating context dependency
_REFERENCE_PHRASES = (
    "the company", "the firm", "the stock", "the filing", "the document",
    "the report", "the act", "the same", "that company",
)

# Queries longer than this word count are treated as self-contained by default
_SELF_CONTAINED_MIN_WORDS = 10


def _needs_rewrite(question: str, chat_history: str) -> bool:
    """Decide whether a query needs rewriting before retrieval.

    Returns True ONLY when rewriting would improve retrieval quality:
      - Query has clear pronoun / anaphoric references to prior history
      - Query is a very short follow-up (< 5 words) with prior context

    Returns False (skip rewriting) when:
      - No chat history available — nothing to resolve
      - Query is long and specific (≥ 10 words) with no pronoun references
      - Query looks like a verbatim citation / completion request

    Args:
        question: The raw user question.
        chat_history: Formatted prior conversation.

    Returns:
        True if query should be rewritten, False to use original.
    """
    if not chat_history or not chat_history.strip():
        return False

    words = re.findall(r"\b\w+\b", question.lower())
    word_set = set(words)

    # Check individual pronoun tokens
    has_pronoun = bool(word_set & _REFERENCE_TOKENS)

    # Check multi-word reference phrases
    question_lower = question.lower()
    has_phrase_ref = any(phrase in question_lower for phrase in _REFERENCE_PHRASES)

    has_reference = has_pronoun or has_phrase_ref

    # Long, specific query with no references → already self-contained
    if len(words) >= _SELF_CONTAINED_MIN_WORDS and not has_reference:
        return False

    # Very short query with history → likely a follow-up, rewrite it
    if len(words) < 5 and chat_history.strip():
        return True

    return has_reference


@tool
def query_rewriter(question: str, chat_history: str = "") -> str:
    """Rewrite a user question into a standalone retrieval query.

    Resolves pronouns and expands context from chat history, but ONLY
    when the query actually needs it. Long, self-contained queries are
    returned unchanged to preserve exact terms for BM25 keyword matching.

    Args:
        question: The user's current question.
        chat_history: Formatted previous conversation turns.

    Returns:
        A standalone rewritten query string, or the original if no rewrite needed.
    """
    if not chat_history:
        logger.info("query_rewriter: no history, returning as-is")
        return question

    if not _needs_rewrite(question, chat_history):
        logger.info(
            "query_rewriter: self-contained query — no rewrite needed for '%s'",
            question[:60],
        )
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
