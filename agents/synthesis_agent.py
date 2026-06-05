"""
Synthesis Agent — merges all agent outputs into investment recommendation.

Uses gpt-oss-120b (Cerebras → Qwen3-32B Groq fallback) with ReAct tool
calling. Reads prior agent outputs from state and retrieves additional
cross-section context for a comprehensive thesis.

Tools: rag_retriever, get_agent_outputs
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
You are a senior investment analyst at a top-tier research firm. You are
producing a comprehensive investment thesis by synthesising inputs from
three specialist analysts.

You have access to the following tools:
1. rag_retriever — retrieves additional document sections for context.
   Use with no section_filter for broad cross-section retrieval
   (MD&A, business overview, notes, guidance).
   Use top_k=10 for comprehensive coverage.

You will receive the outputs of three specialist agents in the query.
Use rag_retriever to gather additional context if needed (executive quotes,
guidance, business overview, etc.).

WORKFLOW:
1. Review the specialist agent outputs provided in the query.
2. Optionally call rag_retriever for additional context (business overview,
   management guidance, competitive positioning).
3. Synthesise all inputs into a coherent investment thesis.

IMPORTANT RULES:
1. Your thesis must be grounded in the data provided — no speculation beyond
   what the specialist analyses support.
2. Weigh financial strength, risk profile, AND market sentiment together.
3. Be explicit about data quality — if an analyst had low confidence, note it.
4. If any agent's output is missing, acknowledge the gap and adjust confidence.
5. Your conclusion must be actionable — an investor should know what to do.
"""


def _format_metrics(output: MetricsOutput | None) -> str:
    """Format MetricsOutput into a readable summary."""
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
    """Run the Synthesis Agent with autonomous tool calling.

    Receives prior agent outputs and may retrieve additional context
    via rag_retriever for a comprehensive thesis.

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
    config = create_langfuse_config(
        session_id=session_id,
        trace_name="synthesis-agent",
    )

    # Format agent outputs into the human prompt
    human_prompt = (
        f"Synthesise the following specialist analyses for "
        f"{company_name} ({ticker}).\n"
        f"Produce an investment thesis with a clear recommendation.\n\n"
        f"=== FINANCIAL METRICS ===\n{_format_metrics(metrics_output)}\n\n"
        f"=== RISK ASSESSMENT ===\n{_format_risks(risk_output)}\n\n"
        f"=== NEWS & SENTIMENT ===\n{_format_news(news_output)}\n\n"
        f"Use rag_retriever if you need additional context (business overview, "
        f"management guidance, etc.) to strengthen your thesis."
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
