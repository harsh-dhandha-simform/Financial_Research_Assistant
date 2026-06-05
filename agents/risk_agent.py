"""
Risk Agent — identifies and assesses risks from SEC filings.

Uses Qwen-2.5-72B (HF Inference → Llama-3.3-70B OpenRouter fallback)
with structured output → RiskOutput.

Tools: rag_retriever, risk_classifier
"""

import logging

from langchain_core.prompts import ChatPromptTemplate

from agents.base import invoke_with_fallback, create_langfuse_config
from schemas.agents import RiskOutput

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are a senior risk analyst specialising in identifying, categorising, and
scoring risks from SEC filings and financial documents.

You will receive retrieved sections (primarily from "Risk Factors" and related
sections) of a financial filing. Your job is to:

1. Identify EVERY distinct risk mentioned in the context.
2. Categorise each risk: market, credit, operational, regulatory, competitive,
   technological, geopolitical, esg, liquidity, or other.
3. Assign a severity level: low, medium, high, or critical.
4. Note any mitigating factors the company mentions.
5. Provide an overall risk assessment for the company.
6. Identify the single most important risk to watch.

IMPORTANT RULES:
1. ONLY identify risks explicitly mentioned in the provided context.
2. Do NOT speculate about risks not discussed in the document.
3. Be specific — "competition" is too vague; "competition from Samsung and
   Google in the smartphone market" is good.
4. Include a confidence score (0-1) reflecting how thorough the risk
   assessment is given the available context.
"""

HUMAN_TEMPLATE = """\
Analyse the following document sections for {company_name} ({ticker}).
Identify and assess all risks.

Query: {query}

--- Retrieved Document Context ---
{context}
--- End Context ---

Identify all risks with detailed categorisation and severity assessment.
"""

PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", HUMAN_TEMPLATE),
])


def run_risk_agent(
    company_name: str,
    ticker: str,
    query: str,
    context: str,
    session_id: str = "",
) -> RiskOutput:
    """Run the Risk Agent to assess risks from financial documents.

    Uses invoke_with_fallback for automatic failover on rate limits,
    timeouts, and JSON errors.

    Args:
        company_name: Company being analysed.
        ticker: Stock ticker.
        query: The focused query for this agent.
        context: RAG-retrieved document context.
        session_id: Langfuse session ID for tracing.

    Returns:
        RiskOutput with identified risks and overall assessment.
    """
    config = create_langfuse_config(
        session_id=session_id,
        trace_name="risk-agent",
    )

    logger.info("Running Risk Agent for %s (%s)", company_name, ticker)

    result = invoke_with_fallback(
        agent_name="risk",
        prompt_chain=PROMPT,
        input_data={
            "company_name": company_name,
            "ticker": ticker,
            "query": query,
            "context": context,
        },
        output_schema=RiskOutput,
        config=config,
    )

    logger.info(
        "Risk Agent complete: confidence=%.2f, %d risks found, overall=%s",
        result.confidence,
        len(result.risks),
        result.overall_risk_level.value,
    )
    return result
