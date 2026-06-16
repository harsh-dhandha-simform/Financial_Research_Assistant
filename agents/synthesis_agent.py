"""
Synthesis Agent — merges all agent outputs into investment recommendation.

Uses a structured workflow to build pre-formatted markdown tables
(financial summary, valuation snapshot, SWOT) before writing the thesis.

Tools: rag_retriever
Output: SynthesisOutput (structured)
"""

import logging

from agents.base import run_tool_agent, create_langfuse_config
from tools.retriever import rag_retriever
from tools.state_reader import get_agent_outputs
from schemas.agents import MetricsOutput, RiskOutput, NewsOutput, SynthesisOutput

logger = logging.getLogger(__name__)

SYNTHESIS_TOOLS = [rag_retriever]  # get_agent_outputs data is injected directly

SYSTEM_PROMPT = """\
You are a senior investment analyst at a top-tier equity research firm.
You are producing a comprehensive, analyst-grade investment thesis by
synthesising inputs from three specialist analysts.

You have access to the following tools:
1. rag_retriever — retrieves additional document sections for context.
   Use with no section_filter for broad cross-section retrieval
   (MD&A, business overview, notes, guidance).
   Use top_k=10 for comprehensive coverage.

You will receive the outputs of three specialist agents in the query.
Use rag_retriever to gather additional context if needed (executive quotes,
guidance, business overview, etc.).

STRUCTURED WORKFLOW — follow these steps in order:

STEP 1 — BUILD FINANCIAL HEALTH SUMMARY TABLE:
  Using the metrics data provided, create a markdown table in the
  financial_health_summary field:
  | Metric | Current | Prior Period | YoY Change | Trend |
  |--------|---------|--------------|------------|-------|
  | Revenue | $394.3B | $383.3B | +2.9% | ↑ Improving |
  | Net Income | $93.7B | $97.0B | -3.4% | ↓ Declining |
  ...include ALL available core metrics (revenue, net income, margins,
  EPS, FCF, etc.). Use ↑ ↓ → arrows for trend.

STEP 2 — BUILD VALUATION SNAPSHOT TABLE:
  Create a markdown table in the valuation_snapshot field:
  | Metric | Value | Context |
  |--------|-------|---------|
  | P/E Ratio | 28.5x | Moderately elevated vs. S&P avg ~21x |
  | Debt-to-Equity | 1.87 | High leverage but manageable with cash flow |
  ...include P/E, D/E, ROE, FCF yield, and any other valuation metrics.

STEP 3 — BUILD KEY METRICS TABLE:
  Create a comprehensive markdown table in the key_metrics_table field
  that includes ALL metrics from the specialist analysis — income statement,
  balance sheet, cash flow, and ratios. Include segment breakdowns if available:
  | Category | Metric | Value | YoY |
  |----------|--------|-------|-----|
  | Income | Revenue | $394.3B | +2.9% |
  | Income | Gross Margin | 46.2% | +0.5pp |
  | Balance Sheet | Total Assets | $352.6B | — |
  | Cash Flow | Free Cash Flow | $111.4B | +5.2% |
  | Segment | Services | $96.2B | +12.8% |
  ...this should be the most information-dense table in the report.

STEP 4 — BUILD SWOT ANALYSIS:
  Create a structured SWOT in the swot_analysis field:
  **Strengths:**
  - Point 1
  - Point 2

  **Weaknesses:**
  - Point 1

  **Opportunities:**
  - Point 1

  **Threats:**
  - Point 1
  
  Ground each point in data from the specialist analyses.

STEP 5 — WRITE INVESTMENT THESIS:
  Write a detailed, 3-5 paragraph investment thesis in investment_thesis.
  Ground it in the tables you built. Reference specific metrics.

STEP 6 — ASSIGN RATING:
  Choose strong_buy / buy / hold / sell / strong_sell.
  Write a clear 2-3 sentence rationale in rating_rationale.
  Populate strengths, weaknesses, and catalysts as bullet points.

CRITICAL RULES:
1. Your thesis MUST be grounded in the data provided — no speculation
   beyond what the specialist analyses support.
2. Weigh financial strength, risk profile, AND market sentiment together.
3. Be explicit about data quality — if an analyst had low confidence, note it.
4. If any agent's output is missing, acknowledge the gap and adjust confidence.
5. Your conclusion must be actionable — an investor should know what to do.
6. The financial_health_summary, key_metrics_table, and valuation_snapshot
   MUST be properly formatted markdown tables, not prose.
7. Include ALL available metrics in the tables — do not cherry-pick.
"""


def _format_metrics(output: MetricsOutput | None) -> str:
    """Format MetricsOutput into a rich structured summary for synthesis."""
    if output is None:
        return "[Metrics analysis not available — agent did not run or failed.]"

    parts = [
        f"Company: {output.company_name}",
        f"Period: {output.fiscal_period}",
        f"Filing Type: {output.filing_type or 'Unknown'}",
    ]

    # Core metrics with prior period and trend
    core_fields = [
        "revenue", "net_income", "gross_margin", "operating_margin",
        "eps", "pe_ratio", "debt_to_equity", "free_cash_flow", "roe",
        "total_assets", "total_liabilities", "total_debt",
        "cash_and_equivalents", "shares_outstanding", "dividend_per_share",
    ]

    parts.append("\n--- CORE METRICS ---")
    for field_name in core_fields:
        metric = getattr(output, field_name, None)
        if metric is not None:
            line = f"  {metric.name}: {metric.value}"
            if metric.prior_period_value:
                line += f" (Prior: {metric.prior_period_value})"
            if metric.period:
                line += f" [{metric.period}]"
            if metric.yoy_change:
                line += f" YoY: {metric.yoy_change}"
            if metric.trend:
                line += f" Trend: {metric.trend}"
            parts.append(line)

    # Ratio groups
    for group_name, group in [
        ("PROFITABILITY", output.profitability),
        ("LIQUIDITY", output.liquidity),
        ("LEVERAGE", output.leverage),
        ("EFFICIENCY", output.efficiency),
    ]:
        if group is not None:
            parts.append(f"\n--- {group_name} RATIOS ---")
            for field_name in group.model_fields:
                metric = getattr(group, field_name, None)
                if metric is not None:
                    line = f"  {metric.name}: {metric.value}"
                    if metric.yoy_change:
                        line += f" (YoY: {metric.yoy_change})"
                    if metric.trend:
                        line += f" Trend: {metric.trend}"
                    parts.append(line)

    # Segment breakdowns
    if output.segments:
        parts.append("\n--- SEGMENT BREAKDOWN ---")
        for seg in output.segments:
            line = f"  {seg.segment_name}: {seg.revenue}"
            if seg.pct_of_total:
                line += f" ({seg.pct_of_total} of total)"
            if seg.yoy_change:
                line += f" YoY: {seg.yoy_change}"
            parts.append(line)

    # Guidance
    if output.revenue_guidance or output.eps_guidance:
        parts.append("\n--- MANAGEMENT GUIDANCE ---")
        if output.revenue_guidance:
            parts.append(f"  Revenue Guidance: {output.revenue_guidance}")
        if output.eps_guidance:
            parts.append(f"  EPS Guidance: {output.eps_guidance}")

    if output.additional_metrics:
        parts.append("\n--- ADDITIONAL METRICS ---")
        for m in output.additional_metrics:
            parts.append(f"  {m.name}: {m.value}")

    if output.events:
        parts.append("\n--- MATERIAL EVENTS (8-K) ---")
        for e in output.events:
            parts.append(f"  - {e.headline}: {e.description}")

    parts.append(f"\nSummary: {output.summary}")
    parts.append(f"Confidence: {output.confidence:.0%}")
    return "\n".join(parts)


def _format_risks(output: RiskOutput | None) -> str:
    """Format RiskOutput into a readable summary."""
    if output is None:
        return "[Risk analysis not available — agent did not run or failed.]"

    parts = [
        f"Company: {output.company_name}",
        f"Overall Risk Level: {output.overall_risk_level.value.upper()}",
        f"Key Concern: {output.key_concern}",
        "",
        "Identified Risks:",
    ]

    for r in output.risks:
        parts.append(
            f"  [{r.severity.value.upper():8s}] [{r.category.value:12s}] {r.title}"
        )
        if r.description:
            parts.append(f"           Detail: {r.description[:200]}")
        if r.potential_impact:
            parts.append(f"           Impact: {r.potential_impact}")
        if r.mitigants:
            parts.append(f"           Mitigant: {r.mitigants}")

    parts.append(f"\nRisk Summary: {output.risk_summary}")
    parts.append(f"Confidence: {output.confidence:.0%}")
    return "\n".join(parts)


def _format_news(output: NewsOutput | None) -> str:
    """Format NewsOutput into a readable summary."""
    if output is None:
        return "[News analysis not available — agent did not run or failed.]"

    parts = [
        f"Company: {output.company_name}",
        f"Overall Sentiment: {output.overall_sentiment.value}",
        f"Key Development: {output.key_development}",
        "",
        "Recent Articles:",
    ]

    for a in output.articles:
        parts.append(f"  [{a.sentiment.value:14s}] {a.headline}")
        if a.summary:
            parts.append(f"    {a.summary[:200]}")

    parts.append(f"\nSentiment Summary: {output.sentiment_summary}")
    parts.append(f"Confidence: {output.confidence:.0%}")
    return "\n".join(parts)


def run_synthesis_agent(
    company_name: str,
    ticker: str,
    metrics_output: MetricsOutput | None,
    risk_output: RiskOutput | None,
    news_output: NewsOutput | None,
    session_id: str = "",
) -> SynthesisOutput:
    """Run the Synthesis Agent with autonomous tool calling.

    Receives prior agent outputs and produces a comprehensive investment
    thesis with pre-formatted markdown tables.

    Args:
        company_name: Company being analysed.
        ticker: Stock ticker.
        metrics_output: Output from the Metrics Agent (or None).
        risk_output: Output from the Risk Agent (or None).
        news_output: Output from the News Agent (or None).
        session_id: Langfuse session ID for tracing.

    Returns:
        SynthesisOutput with investment thesis, tables, and rating.
    """
    config = create_langfuse_config(
        session_id=session_id,
        trace_name="synthesis-agent",
    )

    # Format agent outputs into the human prompt
    human_prompt = (
        f"Synthesise the following specialist analyses for "
        f"{company_name} ({ticker}).\n"
        f"Follow the 6-step structured workflow to produce:\n"
        f"  1. financial_health_summary (markdown table)\n"
        f"  2. valuation_snapshot (markdown table)\n"
        f"  3. key_metrics_table (comprehensive markdown table)\n"
        f"  4. swot_analysis (structured SWOT)\n"
        f"  5. investment_thesis (3-5 paragraphs)\n"
        f"  6. rating + rationale + strengths/weaknesses/catalysts\n\n"
        f"=== FINANCIAL METRICS ===\n{_format_metrics(metrics_output)}\n\n"
        f"=== RISK ASSESSMENT ===\n{_format_risks(risk_output)}\n\n"
        f"=== NEWS & SENTIMENT ===\n{_format_news(news_output)}\n\n"
        f"Use rag_retriever if you need additional context (business overview, "
        f"management guidance, etc.) to strengthen your thesis.\n\n"
        f"REMEMBER: The financial_health_summary, key_metrics_table, and "
        f"valuation_snapshot fields MUST contain properly formatted markdown tables."
    )

    logger.info("Running Synthesis Agent for %s (%s)", company_name, ticker)

    result = run_tool_agent(
        agent_name="synthesis",
        system_prompt=SYSTEM_PROMPT,
        human_prompt=human_prompt,
        tools=SYNTHESIS_TOOLS,
        output_schema=SynthesisOutput,
        config=config,
    )

    logger.info(
        "Synthesis Agent complete: rating=%s, confidence=%.2f",
        result.rating.value,
        result.confidence,
    )
    return result
