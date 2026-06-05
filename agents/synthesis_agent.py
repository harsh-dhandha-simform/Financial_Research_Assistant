"""
Synthesis Agent — merges all agent outputs into investment recommendation.

Uses gpt-oss-120b (Cerebras → Qwen3-32B OpenRouter fallback) with
structured output → SynthesisOutput.

This agent receives the outputs of Metrics, Risk, and News agents and
produces a unified investment thesis with actionable recommendation.
"""

import logging

from langchain_core.prompts import ChatPromptTemplate

from agents.base import get_llm, create_langfuse_config
from schemas.agents import MetricsOutput, RiskOutput, NewsOutput, SynthesisOutput

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are a senior investment analyst at a top-tier research firm. You are
producing a comprehensive investment thesis by synthesising inputs from
three specialist analysts:

1. **Metrics Analyst** — extracted financial metrics from SEC filings
2. **Risk Analyst** — identified and categorised risks
3. **News Analyst** — analysed recent news and market sentiment

Your job is to:
1. Synthesise all three analyses into a coherent investment thesis (3-5 paragraphs).
2. Assign an investment rating: strong_buy, buy, hold, sell, or strong_sell.
3. Provide clear rationale for your rating.
4. List key strengths (bull case) and weaknesses (bear case).
5. Identify upcoming catalysts that could move the stock.

IMPORTANT RULES:
1. Your thesis must be grounded in the data provided — no speculation beyond
   what the specialist analyses support.
2. Weigh financial strength, risk profile, AND market sentiment together.
3. Be explicit about data quality — if an analyst had low confidence, note it.
4. If any agent's output is missing, acknowledge the gap and adjust confidence.
5. Your conclusion must be actionable — an investor should know what to do.
"""

HUMAN_TEMPLATE = """\
Synthesise the following specialist analyses for {company_name} ({ticker}).
Produce an investment thesis with a clear recommendation.

=== FINANCIAL METRICS ===
{metrics_summary}

=== RISK ASSESSMENT ===
{risk_summary}

=== NEWS & SENTIMENT ===
{news_summary}

Provide your investment thesis, rating, and actionable recommendation.
"""

PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", HUMAN_TEMPLATE),
])


def _format_metrics(output: MetricsOutput | None) -> str:
    """Format MetricsOutput into a readable summary for the synthesis prompt."""
    if output is None:
        return "[Metrics analysis not available — agent did not run or failed.]"

    parts = [f"Company: {output.company_name}", f"Period: {output.fiscal_period}"]

    for field_name in [
        "revenue", "net_income", "gross_margin", "operating_margin",
        "eps", "pe_ratio", "debt_to_equity", "free_cash_flow", "roe",
        "total_assets", "total_liabilities", "total_debt",
        "cash_and_equivalents", "shares_outstanding", "dividend_per_share",
    ]:
        metric = getattr(output, field_name, None)
        if metric is not None:
            line = f"  {metric.name}: {metric.value}"
            if metric.period:
                line += f" ({metric.period})"
            if metric.yoy_change:
                line += f" [YoY: {metric.yoy_change}]"
            parts.append(line)

    if output.additional_metrics:
        for m in output.additional_metrics:
            parts.append(f"  {m.name}: {m.value}")

    if output.events:
        parts.append("\nMaterial Events (8-K):")
        for e in output.events:
            parts.append(f"  - {e.headline}: {e.description}")

    parts.append(f"\nSummary: {output.summary}")
    parts.append(f"Confidence: {output.confidence:.0%}")

    return "\n".join(parts)


def _format_risks(output: RiskOutput | None) -> str:
    """Format RiskOutput into a readable summary for the synthesis prompt."""
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
        if r.potential_impact:
            parts.append(f"           Impact: {r.potential_impact}")
        if r.mitigants:
            parts.append(f"           Mitigant: {r.mitigants}")

    parts.append(f"\nRisk Summary: {output.risk_summary}")
    parts.append(f"Confidence: {output.confidence:.0%}")

    return "\n".join(parts)


def _format_news(output: NewsOutput | None) -> str:
    """Format NewsOutput into a readable summary for the synthesis prompt."""
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
        parts.append(
            f"  [{a.sentiment.value:14s}] {a.headline}"
        )
        if a.summary:
            parts.append(f"    {a.summary[:150]}...")

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
    """Run the Synthesis Agent to produce an investment thesis.

    Merges outputs from all specialist agents into a unified
    recommendation with clear rationale.

    Args:
        company_name: Company being analysed.
        ticker: Stock ticker.
        metrics_output: Output from the Metrics Agent (or None).
        risk_output: Output from the Risk Agent (or None).
        news_output: Output from the News Agent (or None).
        session_id: Langfuse session ID for tracing.

    Returns:
        SynthesisOutput with investment thesis and rating.
    """
    llm = get_llm("synthesis")
    structured_llm = llm.with_structured_output(SynthesisOutput)

    chain = PROMPT | structured_llm

    config = create_langfuse_config(
        session_id=session_id,
        trace_name="synthesis-agent",
    )

    logger.info("Running Synthesis Agent for %s (%s)", company_name, ticker)

    result = chain.invoke(
        {
            "company_name": company_name,
            "ticker": ticker,
            "metrics_summary": _format_metrics(metrics_output),
            "risk_summary": _format_risks(risk_output),
            "news_summary": _format_news(news_output),
        },
        config=config,
    )

    logger.info(
        "Synthesis Agent complete: rating=%s, confidence=%.2f",
        result.rating.value,
        result.confidence,
    )
    return result
