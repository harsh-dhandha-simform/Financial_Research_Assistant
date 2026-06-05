"""
sentiment_scorer — scores sentiment of individual news articles.

Uses Gemini Flash (same as News Agent) for fast, inexpensive
per-article sentiment scoring.

Used by: News Agent (max 5 calls per search).
"""

import logging

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from agents.base import get_llm

logger = logging.getLogger(__name__)


class SentimentResult(BaseModel):
    """Structured sentiment classification of a news article."""

    sentiment: str = Field(
        ...,
        description="Sentiment: positive, neutral, or negative",
    )
    score: float = Field(
        default=0.0,
        ge=-1.0,
        le=1.0,
        description="Sentiment score (-1 very negative to +1 very positive)",
    )
    reasoning: str = Field(
        default="",
        description="Brief reasoning for the sentiment classification",
    )


SENTIMENT_PROMPT = ChatPromptTemplate.from_messages([
    ("system", (
        "You are a financial sentiment analyst. Given a news headline and "
        "snippet about a company, classify the sentiment and provide a "
        "score from -1 (very negative) to +1 (very positive). "
        "Be objective — focus on facts, not speculation."
    )),
    ("human", "Headline: {headline}\n\nSnippet: {snippet}"),
])


@tool
def sentiment_scorer(headline: str, snippet: str) -> dict:
    """Score the sentiment of a single news article.

    Classifies news headline + snippet as positive, neutral, or negative
    with a numerical score and reasoning.

    Args:
        headline: News article headline.
        snippet: Brief summary or excerpt from the article.

    Returns:
        Dict with sentiment, score (-1 to +1), and reasoning.
    """
    llm = get_llm("news")
    structured_llm = llm.with_structured_output(SentimentResult)
    chain = SENTIMENT_PROMPT | structured_llm

    result = chain.invoke({"headline": headline, "snippet": snippet})

    logger.info(
        "sentiment_scorer: %s (%.2f) — %s",
        result.sentiment,
        result.score,
        headline[:60],
    )
    return result.model_dump()
