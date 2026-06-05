"""
hallucination_checker — verifies LLM answers are grounded in sources.

Checks if claims in the answer are supported by the source chunks.
If ungrounded, the chat graph retries from Query Rewriter (max 2 retries).

Used by: Chat Agent.
"""

import logging

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from agents.base import get_llm

logger = logging.getLogger(__name__)


class HallucinationCheck(BaseModel):
    """Result of a hallucination check."""

    is_grounded: bool = Field(
        ...,
        description="True if ALL claims in the answer are supported by sources",
    )
    confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Confidence in the grounding assessment (0-1)",
    )
    ungrounded_claims: list[str] = Field(
        default_factory=list,
        description="List of claims not supported by sources",
    )


CHECK_PROMPT = ChatPromptTemplate.from_messages([
    ("system", (
        "You are a factual grounding checker. Given an answer and source "
        "chunks, determine if every factual claim in the answer is supported "
        "by the source material. List any ungrounded claims."
    )),
    ("human", (
        "Answer to check:\n{answer}\n\n"
        "Source chunks:\n{source_chunks}\n\n"
        "Is the answer fully grounded in the sources?"
    )),
])


@tool
def hallucination_checker(answer: str, source_chunks: str) -> dict:
    """Check if an LLM answer is grounded in the source documents.

    Verifies that all factual claims in the answer are supported by
    the provided source chunks. Lists any ungrounded claims.

    Args:
        answer: The LLM-generated answer to verify.
        source_chunks: Concatenated source document chunks.

    Returns:
        Dict with is_grounded (bool), confidence (float),
        and ungrounded_claims (list of strings).
    """
    llm = get_llm("news")  # Use fast model for checking
    structured_llm = llm.with_structured_output(HallucinationCheck)
    chain = CHECK_PROMPT | structured_llm

    result = chain.invoke({
        "answer": answer,
        "source_chunks": source_chunks,
    })

    logger.info(
        "hallucination_checker: grounded=%s, confidence=%.2f, ungrounded=%d claims",
        result.is_grounded,
        result.confidence,
        len(result.ungrounded_claims),
    )
    return result.model_dump()
