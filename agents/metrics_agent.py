"""
Metrics Agent — extracts financial metrics from SEC filings.

Uses Llama-3.1-8B (HF Inference → Groq fallback) with ReAct tool calling.
The LLM autonomously calls rag_retriever to fetch relevant sections and
extract_financial_table to parse tabular data.

Tools: rag_retriever, extract_financial_table
Output: MetricsOutput (structured)
"""

import logging

from agents.base import run_tool_agent, create_langfuse_config
from tools.retriever import rag_retriever
from tools.financial import extract_financial_table
from schemas.agents import MetricsOutput

logger = logging.getLogger(__name__)

METRICS_TOOLS = [rag_retriever, extract_financial_table]

SYSTEM_PROMPT = """\
You are a senior financial analyst specialising in extracting key financial metrics
from SEC filings (10-K, 10-Q, 8-K, DEF 14A).

You have access to the following tools:
1. rag_retriever — retrieves relevant document sections from the filing.
   Use section_filter="mda" for financial metrics, or "" for broad search.
   Use top_k=6 for thorough coverage.
2. extract_financial_table — extracts structured metrics from text containing
   $ or % values. Call this on retrieved chunks that contain tabular data.

WORKFLOW:
1. First, call rag_retriever with the query and section_filter="mda" to get
   relevant financial sections.
2. If the retrieved text contains financial tables ($ or % values),
   call extract_financial_table on that text.
3. You may call rag_retriever again with different queries to find missing metrics.

IMPORTANT RULES:
1. ONLY extract metrics explicitly stated in the retrieved context.
2. Do NOT fabricate or estimate values not in the documents.
3. If a metric is not found after retrieval, leave the field as null.
4. For 8-K filings, focus on material events instead of periodic metrics.
5. For DEF 14A (proxy), extract executive compensation data.
6. Include a confidence score (0-1) reflecting how complete the extraction is.
"""


def run_metrics_agent(
    company_name: str,
    ticker: str,
    query: str,
    session_id: str = "",
) -> MetricsOutput:
    """Run the Metrics Agent with autonomous tool calling.

    The agent decides what to retrieve and how to extract metrics.
    No pre-fetched context needed.

    Args:
        company_name: Company being analysed.
        ticker: Stock ticker.
        query: The focused query for this agent.
        session_id: Langfuse session ID for tracing.

    Returns:
        MetricsOutput with extracted financial metrics.
    """
    config = create_langfuse_config(
        session_id=session_id,
        trace_name="metrics-agent",
    )

    human_prompt = (
        f"Analyse financial documents for {company_name} ({ticker}).\n"
        f"Extract all key financial metrics.\n\n"
        f"Query: {query}\n\n"
        f"Use your tools to retrieve relevant document sections, then extract "
        f"all financial metrics with precision."
    )

    logger.info("Running Metrics Agent for %s (%s)", company_name, ticker)

    result = run_tool_agent(
        agent_name="metrics",
        system_prompt=SYSTEM_PROMPT,
        human_prompt=human_prompt,
        tools=METRICS_TOOLS,
        output_schema=MetricsOutput,
        config=config,
    )

    logger.info(
        "Metrics Agent complete: confidence=%.2f, revenue=%s",
        result.confidence,
        result.revenue.value if result.revenue else "N/A",
    )
    return result
