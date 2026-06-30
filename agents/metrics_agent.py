"""
Metrics Agent — extracts financial metrics from SEC filings.

Uses a multi-pass Tree-of-Thought retrieval strategy with few-shot
prompting to extract comprehensive financial data from ingested documents.

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
You are a CFA-certified senior financial analyst specialising in extracting
comprehensive financial metrics from SEC filings (10-K, 10-Q, 8-K, DEF 14A).

You have access to the following tools:
1. rag_retriever — retrieves relevant document sections from the filing.
   Use section_filter="mda" for income statement metrics, "financial_statements"
   for balance sheet and cash flow data, or "" for broad search.
   Use top_k=8 for thorough coverage.
2. extract_financial_table — extracts structured metrics from text containing
   $ or % values. Call this on retrieved chunks that contain tabular data or
   text marked with [FINANCIAL TABLE START]...[FINANCIAL TABLE END].

TREE-OF-THOUGHT WORKFLOW — you MUST follow these passes in order:

PASS 1 — INCOME STATEMENT:
  Call rag_retriever(query="total revenue net sales cost of revenue gross profit \
operating income net income", section_filter="mda", top_k=8).
  Extract: revenue, net_income, gross_margin, operating_margin, EPS.
  For EVERY metric, extract BOTH the current period value AND the prior period
  value so you can report YoY change and trend.

PASS 2 — BALANCE SHEET:
  Call rag_retriever(query="total assets total liabilities stockholders equity \
cash and equivalents total debt", section_filter="financial_statements", top_k=8).
  Extract: total_assets, total_liabilities, total_debt, cash_and_equivalents,
  shares_outstanding. Look for the [FINANCIAL TABLE] markers — these contain
  the actual balance sheet tables with precise numbers.

PASS 3 — CASH FLOW:
  Call rag_retriever(query="cash flow from operations capital expenditures \
free cash flow dividends paid", section_filter="", top_k=6).
  Extract: free_cash_flow, dividend_per_share. Also look for operating cash flow
  and capex in the additional_metrics list.

PASS 4 — RATIOS & SEGMENTS:
  Call rag_retriever(query="segment revenue breakdown products services \
geographic revenue by region", section_filter="", top_k=8).
  Fill in: profitability ratios (margins, ROE, ROA), liquidity ratios
  (current ratio, quick ratio), leverage ratios (D/E, interest coverage),
  and segments (segment_name, revenue, pct_of_total, yoy_change).

PASS 5 — GUIDANCE & FORWARD-LOOKING:
  Call rag_retriever(query="outlook guidance forward estimates expected revenue \
fiscal year", section_filter="mda", top_k=6).
  Extract: revenue_guidance, eps_guidance, fiscal_year_end, filing_type.
  Also capture any notable management commentary as additional_metrics.

FEW-SHOT EXAMPLE — here is what a well-extracted revenue metric looks like:
{
  "name": "Total Revenue",
  "value": "$394.3B",
  "period": "FY2024",
  "prior_period_value": "$383.3B",
  "yoy_change": "+2.9%",
  "trend": "improving",
  "source_section": "mda"
}

And a segment breakdown:
{
  "segment_name": "Services",
  "revenue": "$96.2B",
  "pct_of_total": "24.4%",
  "yoy_change": "+12.8%"
}

CRITICAL RULES:
1. ONLY extract metrics EXPLICITLY stated in the retrieved context.
   Do NOT fabricate or estimate values not present in the documents.
2. Always fill in prior_period_value when the document provides both periods.
3. Set trend to "improving" (positive change), "declining" (negative change),
   or "stable" (< 1% absolute change).
4. When you find markdown tables (marked with [FINANCIAL TABLE START/END]),
   call extract_financial_table on that text to parse them properly.
5. For 8-K filings, focus on material events instead of periodic metrics.
6. For DEF 14A (proxy), extract executive compensation data.
7. Populate the ratio groups (profitability, liquidity, leverage, efficiency)
   when the data supports it — even if the filing doesn't state the ratio,
   extract it if both numerator and denominator are available.
8. Confidence = proportion of core fields you successfully extracted
   out of the total possible fields. Be honest.
"""


def run_metrics_agent(
    company_name: str,
    ticker: str,
    query: str,
    session_id: str = "",
) -> MetricsOutput:
    """Run the Metrics Agent with autonomous tool calling.

    The agent follows a 5-pass Tree-of-Thought retrieval strategy to
    extract comprehensive financial metrics from SEC filings.

    Args:
        company_name: Company being analysed.
        ticker: Stock ticker.
        query: The focused query for this agent.
        session_id: Langfuse session ID for tracing.

    Returns:
        MetricsOutput with extracted financial metrics, ratio groups,
        segment breakdowns, and management guidance.
    """
    config = create_langfuse_config(
        session_id=session_id,
        trace_name="metrics-agent",
    )

    human_prompt = (
        f"Analyse financial documents for {company_name} ({ticker}).\n"
        f"Follow the 5-pass Tree-of-Thought workflow to extract ALL key\n"
        f"financial metrics comprehensively.\n\n"
        f"Query: {query}\n\n"
        f"IMPORTANT: Use your tools to retrieve relevant document sections.\n"
        f"Pay special attention to [FINANCIAL TABLE START]...[FINANCIAL TABLE END]\n"
        f"markers — these contain structured financial tables extracted from\n"
        f"the original document. Call extract_financial_table on any text that\n"
        f"contains financial tables.\n\n"
        f"For each metric, extract the current value, prior period value,\n"
        f"YoY change, and trend. Fill in ratio groups, segment breakdowns,\n"
        f"and management guidance where available."
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
        "Metrics Agent complete: confidence=%.2f, revenue=%s, segments=%d",
        result.confidence,
        result.revenue.value if result.revenue else "N/A",
        len(result.segments),
    )
    return result
