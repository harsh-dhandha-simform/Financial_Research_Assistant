"""
risk_classifier — classifies individual risk paragraphs.

Uses the same model as the Risk Agent (Qwen-2.5-72B) with
structured output for independent risk classification.

Used by: Risk Agent.
"""

import logging

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from agents.base import get_llm

logger = logging.getLogger(__name__)


class RiskClassification(BaseModel):
    """Structured classification of a single risk."""

    category: str = Field(
        ...,
        description=(
            "Risk category: market, regulatory, operational, financial, "
            "competitive, technological, or other"
        ),
    )
    risk_level: str = Field(
        ...,
        description="Risk severity: low, medium, high, or critical",
    )
    confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Classification confidence (0-1)",
    )


CLASSIFY_PROMPT = ChatPromptTemplate.from_messages([
    ("system", (
        "You are a risk classification expert. Given a risk paragraph from a "
        "financial filing, classify it into a category and severity level. "
        "Categories: market, regulatory, operational, financial, competitive, "
        "technological, other. Levels: low, medium, high, critical."
    )),
    ("human", "Classify this risk:\n\n{risk_text}"),
])


@tool
def risk_classifier(risk_text: str) -> dict:
    """Classify a single risk paragraph into category and severity.

    Uses structured LLM output to categorise risks from financial filings
    into standard categories with severity levels.

    Args:
        risk_text: A single risk paragraph from a financial filing.

    Returns:
        Dict with category, risk_level, and confidence.
    """
    llm = get_llm("risk")
    structured_llm = llm.with_structured_output(RiskClassification)
    chain = CLASSIFY_PROMPT | structured_llm

    result = chain.invoke({"risk_text": risk_text})

    logger.info(
        "risk_classifier: category=%s, level=%s, confidence=%.2f",
        result.category,
        result.risk_level,
        result.confidence,
    )
    return result.model_dump()
