"""
Metrics Agent — extracts financial metrics from SEC filings.

Uses llama-3.1-8b via OpenRouter with structured output → MetricsOutput.
Receives RAG context from the hybrid retriever (dense + BM25 + RRF).
"""

import logging

from langchain_core.prompts import ChatPromptTemplate

from agents.base import get_openrouter_llm, create_langfuse_config
from schemas.agents import MetricsOutput

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are a senior financial analyst specialising in extracting key financial metrics
from SEC filings (10-K, 10-Q, 8-K, DEF 14A).

You will receive retrieved sections from a financial filing. Extract ALL financial
metrics you can find with precision. For each metric, note:
- The exact value from the document
- The reporting period (FY2024, Q3 2024, etc.)
- Year-over-year change if available
- The document section where you found it

IMPORTANT RULES:
1. ONLY extract metrics explicitly stated in the provided context.
2. Do NOT fabricate or estimate values not in the document.
3. If a metric is not found, leave the field as null.
4. For 8-K filings, focus on material events instead of periodic metrics.
5. For DEF 14A (proxy), extract executive compensation data.
6. Include a confidence score (0-1) reflecting how complete the extraction is.
"""

HUMAN_TEMPLATE = """\
Analyse the following financial document sections for {company_name} ({ticker}).
Extract all key financial metrics.

Query: {query}

--- Retrieved Document Context ---
{context}
--- End Context ---

Extract all financial metrics with precision. Be thorough.
"""

# Build the prompt template
PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", HUMAN_TEMPLATE),
])


def run_metrics_agent(
    company_name: str,
    ticker: str,
    query: str,
    context: str,
    session_id: str = "",
) -> MetricsOutput:
    """Run the Metrics Agent to extract financial metrics.

    Args:
        company_name: Company being analysed.
        ticker: Stock ticker.
        query: The focused query for this agent.
        context: RAG-retrieved document context.
        session_id: Langfuse session ID for tracing.

    Returns:
        MetricsOutput with extracted financial metrics.
    """
    llm = get_openrouter_llm("metrics")
    structured_llm = llm.with_structured_output(MetricsOutput)

    chain = PROMPT | structured_llm

    config = create_langfuse_config(
        session_id=session_id,
        trace_name="metrics-agent",
    )

    logger.info("Running Metrics Agent for %s (%s)", company_name, ticker)

    result = chain.invoke(
        {
            "company_name": company_name,
            "ticker": ticker,
            "query": query,
            "context": context,
        },
        config=config,
    )

    logger.info(
        "Metrics Agent complete: confidence=%.2f, revenue=%s",
        result.confidence,
        result.revenue.value if result.revenue else "N/A",
    )
    return result
