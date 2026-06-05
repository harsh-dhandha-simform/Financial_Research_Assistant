"""
Risk Agent — identifies and assesses risks from SEC filings.

Uses Qwen-2.5-72B (HF Inference → Llama-3.3-70B OpenRouter fallback)
with ReAct tool calling. The LLM autonomously retrieves risk sections
and classifies each risk.

Tools: rag_retriever, risk_classifier
Output: RiskOutput (structured)
"""

import logging

from agents.base import run_tool_agent, create_langfuse_config
from tools.retriever import rag_retriever
from tools.risk import risk_classifier
from schemas.agents import RiskOutput

logger = logging.getLogger(__name__)

RISK_TOOLS = [rag_retriever, risk_classifier]

SYSTEM_PROMPT = """\
You are a senior risk analyst specialising in identifying, categorising, and
scoring risks from SEC filings and financial documents.

You have access to the following tools:
1. rag_retriever — retrieves relevant document sections from the filing.
   Use section_filter="risk_factors" to target risk sections specifically.
   Use top_k=12 for broad risk coverage (risk sections tend to be long).
2. risk_classifier — classifies a single risk paragraph into category and
   severity. Call this for each distinct risk you identify.

WORKFLOW:
1. Call rag_retriever with section_filter="risk_factors" and top_k=12
   to get the risk factors section.
2. Read through the retrieved content and identify each distinct risk.
3. For each key risk, call risk_classifier with the risk paragraph to get
   a structured classification (category + severity).
4. You may call rag_retriever again with different queries to find
   additional risks (operational, regulatory, etc.).

IMPORTANT RULES:
1. ONLY identify risks explicitly mentioned in the retrieved context.
2. Do NOT speculate about risks not discussed in the document.
3. Be specific — "competition" is too vague; "competition from Samsung and
   Google in the smartphone market" is good.
4. Include a confidence score (0-1) reflecting how thorough the risk
   assessment is given the available context.
5. Identify the single most important risk to watch.
"""


def run_risk_agent(
    company_name: str,
    ticker: str,
    query: str,
    session_id: str = "",
) -> RiskOutput:
    """Run the Risk Agent with autonomous tool calling.

    The agent decides what to retrieve and classifies risks independently.

    Args:
        company_name: Company being analysed.
        ticker: Stock ticker.
        query: The focused query for this agent.
        session_id: Langfuse session ID for tracing.

    Returns:
        RiskOutput with identified risks and overall assessment.
    """
    config = create_langfuse_config(
        session_id=session_id,
        trace_name="risk-agent",
    )

    human_prompt = (
        f"Analyse financial documents for {company_name} ({ticker}).\n"
        f"Identify and assess all risks.\n\n"
        f"Query: {query}\n\n"
        f"Use your tools to retrieve risk factor sections from the filing, "
        f"classify each risk, and provide a comprehensive risk assessment."
    )

    logger.info("Running Risk Agent for %s (%s)", company_name, ticker)

    result = run_tool_agent(
        agent_name="risk",
        system_prompt=SYSTEM_PROMPT,
        human_prompt=human_prompt,
        tools=RISK_TOOLS,
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
