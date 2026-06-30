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
        "Be objective — focus on facts, not speculation.\n\n"
        "Respond with ONLY a JSON object in this exact format:\n"
        '{{"sentiment": "positive|neutral|negative", "score": 0.0, "reasoning": "brief reason"}}'
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
    try:
        from agents.base import get_llm, AGENT_MODELS, _create_llm, _get_structured_llm

        # Use dedicated lightweight sentiment model (Groq llama-3.1-8b-instant)
        # This model reliably supports json_mode and is much faster/cheaper than
        # the news agent's primary (Gemini). We never use the news agent's model
        # here because Gemini does not support json_mode via OpenAI-compat.
        sentiment_models = AGENT_MODELS.get("sentiment", [])
        structured_llm = None
        for model_config in sentiment_models:
            try:
                llm = _create_llm(model_config)
                structured_llm = _get_structured_llm(
                    llm, SentimentResult, model_config.structured_method
                )
                break
            except Exception:
                continue

        if structured_llm is None:
            raise RuntimeError("No sentiment model available")

        chain = SENTIMENT_PROMPT | structured_llm
        result = chain.invoke({"headline": headline, "snippet": snippet})

        logger.info(
            "sentiment_scorer: %s (%.2f) — %s",
            result.sentiment,
            result.score,
            headline[:60],
        )
        return result.model_dump()
    except Exception as exc:
        logger.warning("sentiment_scorer failed: %s — returning neutral", exc)
        return {
            "sentiment": "neutral",
            "score": 0.0,
            "reasoning": f"Classification failed: {str(exc)[:100]}",
        }
