"""
Output Fork — generates InvestmentMemo + ResearchReport from SynthesisOutput.

Takes the SynthesisOutput (which merges Metrics, Risk, News analyses) and
produces two distinct documents in parallel:

  1. InvestmentMemo  — concise 1-page executive summary for quick decisions
  2. ResearchReport  — detailed multi-section analyst report for deep dives

Both use template-based generation (no LLM call) — the Synthesis Agent already
did the heavy analytical work. The output fork just structures and formats.

This is intentionally NOT an LLM call — it's a deterministic formatter.
Rationale:
  - SynthesisOutput already contains all the analysis
  - We don't want another LLM call adding latency and potential hallucination
  - Templates ensure consistent formatting every time
  - The user gets the report ~instantly after synthesis completes
"""

import logging
from datetime import datetime, timezone

from schemas.agents import (
    MetricsOutput,
    RiskOutput,
    NewsOutput,
    SynthesisOutput,
    InvestmentRating,
)
from schemas.reports import InvestmentMemo, ResearchReport, ReportSection

logger = logging.getLogger(__name__)


# ═════════════════════════════════════════════════════════════════════════════
# Investment Memo generator
# ═════════════════════════════════════════════════════════════════════════════


def _format_key_metrics(metrics: MetricsOutput | None) -> str:
    """Format metrics into bullet points for the memo."""
    if metrics is None:
        return "• Financial metrics not available (agent did not run or data unavailable)"

    lines = []
    metric_fields = [
        ("revenue", "Revenue"),
        ("net_income", "Net Income"),
        ("gross_margin", "Gross Margin"),
        ("operating_margin", "Operating Margin"),
        ("eps", "EPS (Diluted)"),
        ("pe_ratio", "P/E Ratio"),
        ("debt_to_equity", "Debt-to-Equity"),
        ("free_cash_flow", "Free Cash Flow"),
        ("roe", "Return on Equity"),
    ]

    for field_name, display_name in metric_fields:
        metric = getattr(metrics, field_name, None)
        if metric is not None:
            line = f"• {display_name}: {metric.value}"
            if metric.yoy_change:
                line += f" (YoY: {metric.yoy_change})"
            lines.append(line)

    if not lines:
        return "• No specific financial metrics extracted from filings"

    return "\n".join(lines)


def _format_key_risks(risk: RiskOutput | None) -> str:
    """Format top risks into bullet points for the memo."""
    if risk is None:
        return "• Risk assessment not available (agent did not run or data unavailable)"

    lines = []
    # Take top 5 risks sorted by severity
    severity_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
    sorted_risks = sorted(
        risk.risks,
        key=lambda r: severity_order.get(r.severity.value, 99),
    )

    for r in sorted_risks[:5]:
        lines.append(f"• [{r.severity.value.upper()}] {r.title}: {r.description[:120]}")

    if not lines:
        return "• No specific risks identified from filings"

    return "\n".join(lines)


def _format_recent_news(news: NewsOutput | None) -> str:
    """Format news summary for the memo."""
    if news is None:
        return "No recent news data available."

    parts = []
    if news.sentiment_summary:
        parts.append(news.sentiment_summary)

    if news.key_development:
        parts.append(f"\nKey development: {news.key_development}")

    if news.articles:
        parts.append(f"\n{len(news.articles)} articles analysed — "
                     f"overall sentiment: {news.overall_sentiment.value}")

    return "\n".join(parts) if parts else "No news data available."


def _rating_display(rating: InvestmentRating) -> str:
    """Convert rating enum to display string."""
    display = {
        InvestmentRating.STRONG_BUY: "STRONG BUY",
        InvestmentRating.BUY: "BUY",
        InvestmentRating.HOLD: "HOLD",
        InvestmentRating.SELL: "SELL",
        InvestmentRating.STRONG_SELL: "STRONG SELL",
    }
    return display.get(rating, rating.value.upper())


def generate_memo(
    synthesis: SynthesisOutput,
    metrics: MetricsOutput | None = None,
    risk: RiskOutput | None = None,
    news: NewsOutput | None = None,
) -> InvestmentMemo:
    """Generate a concise InvestmentMemo from SynthesisOutput.

    The memo is a deterministic format — no LLM call needed.
    SynthesisOutput already contains the analytical content.

    Args:
        synthesis: Output from the Synthesis Agent.
        metrics: Optional MetricsOutput for detailed metric formatting.
        risk: Optional RiskOutput for detailed risk formatting.
        news: Optional NewsOutput for news summary formatting.

    Returns:
        A populated InvestmentMemo.
    """
    rating_str = _rating_display(synthesis.rating)

    memo = InvestmentMemo(
        title=f"{synthesis.company_name} ({synthesis.ticker}) — {rating_str}",
        company_name=synthesis.company_name,
        ticker=synthesis.ticker,
        rating=synthesis.rating,
        rating_rationale=synthesis.rating_rationale,
        executive_summary=synthesis.investment_thesis,
        key_metrics=_format_key_metrics(metrics),
        key_risks=_format_key_risks(risk),
        recent_news=_format_recent_news(news),
        conclusion=(
            f"**Recommendation: {rating_str}**\n\n"
            f"{synthesis.rating_rationale}\n\n"
            f"Strengths: {', '.join(synthesis.strengths[:3])}\n"
            f"Key risks: {', '.join(synthesis.weaknesses[:3])}\n\n"
            f"Confidence: {synthesis.confidence:.0%}"
            + (f"\n\nNote: {synthesis.data_quality_note}" if synthesis.data_quality_note else "")
        ),
    )

    logger.info("Investment memo generated: %s", memo.title)
    return memo


# ═════════════════════════════════════════════════════════════════════════════
# Research Report generator
# ═════════════════════════════════════════════════════════════════════════════


def _build_financial_analysis(
    synthesis: SynthesisOutput,
    metrics: MetricsOutput | None,
) -> str:
    """Build the financial analysis section of the report."""
    parts = []

    if metrics and metrics.summary:
        parts.append(metrics.summary)
        parts.append("")

    if metrics:
        parts.append("### Key Financial Metrics\n")
        parts.append(_format_key_metrics(metrics))
        parts.append("")

        if metrics.events:
            parts.append("\n### Material Events (8-K)\n")
            for e in metrics.events:
                parts.append(f"- **{e.headline}** ({e.event_date}): {e.description}")
                if e.financial_impact:
                    parts.append(f"  - Financial impact: {e.financial_impact}")

        if metrics.executive_compensation:
            parts.append("\n### Executive Compensation (Proxy)\n")
            for ec in metrics.executive_compensation:
                parts.append(f"- **{ec.name}** ({ec.title}): {ec.total_compensation}")
    else:
        parts.append(
            "Financial metrics were not available for this analysis. "
            "This may be due to filing data not being ingested into the "
            "vector store, or the Metrics Agent encountering errors during retrieval."
        )

    return "\n".join(parts)


def _build_risk_assessment(
    synthesis: SynthesisOutput,
    risk: RiskOutput | None,
) -> str:
    """Build the risk assessment section of the report."""
    parts = []

    if risk:
        parts.append(
            f"**Overall Risk Level: {risk.overall_risk_level.value.upper()}**\n"
        )
        if risk.key_concern:
            parts.append(f"**Key Concern:** {risk.key_concern}\n")

        parts.append(risk.risk_summary)
        parts.append("\n### Identified Risks\n")

        for r in risk.risks:
            parts.append(
                f"#### [{r.severity.value.upper()}] {r.title}\n"
                f"**Category:** {r.category.value} | "
                f"**Likelihood:** {r.likelihood or 'Not assessed'}\n\n"
                f"{r.description}\n"
            )
            if r.potential_impact:
                parts.append(f"**Potential Impact:** {r.potential_impact}\n")
            if r.mitigants:
                parts.append(f"**Mitigating Factors:** {r.mitigants}\n")
    else:
        parts.append(
            "Risk assessment was not available for this analysis. "
            "The report's risk discussion is based on the Synthesis Agent's "
            "general knowledge rather than document-specific risk factors."
        )
        # Use synthesis weaknesses as proxy
        if synthesis.weaknesses:
            parts.append("\n### Potential Risk Areas (from synthesis)\n")
            for w in synthesis.weaknesses:
                parts.append(f"- {w}")

    return "\n".join(parts)


def _build_market_sentiment(
    synthesis: SynthesisOutput,
    news: NewsOutput | None,
) -> str:
    """Build the market sentiment section of the report."""
    parts = []

    if news:
        parts.append(
            f"**Overall Sentiment: {news.overall_sentiment.value.replace('_', ' ').title()}**\n"
        )

        if news.sentiment_summary:
            parts.append(news.sentiment_summary)

        if news.key_development:
            parts.append(f"\n**Key Development:** {news.key_development}\n")

        if news.articles:
            parts.append("\n### Recent Articles\n")
            for a in news.articles:
                source_str = f" ({a.source})" if a.source else ""
                date_str = f" — {a.published_date}" if a.published_date else ""
                sentiment = a.sentiment.value.replace("_", " ").title()
                parts.append(
                    f"- **{a.headline}**{source_str}{date_str}\n"
                    f"  Sentiment: {sentiment} | "
                    f"Relevance: {a.relevance:.0%}\n"
                    f"  {a.summary[:200]}\n"
                )
    else:
        parts.append(
            "News and sentiment data was not available for this analysis."
        )

    return "\n".join(parts)


def generate_report(
    synthesis: SynthesisOutput,
    metrics: MetricsOutput | None = None,
    risk: RiskOutput | None = None,
    news: NewsOutput | None = None,
) -> ResearchReport:
    """Generate a detailed ResearchReport from SynthesisOutput.

    The report is a deterministic format — no LLM call needed.
    SynthesisOutput contains the thesis; individual agent outputs
    provide the detailed section data.

    Args:
        synthesis: Output from the Synthesis Agent.
        metrics: Optional MetricsOutput for financial section.
        risk: Optional RiskOutput for risk section.
        news: Optional NewsOutput for sentiment section.

    Returns:
        A populated ResearchReport.
    """
    rating_str = _rating_display(synthesis.rating)

    # Build company overview from synthesis context
    company_overview = (
        f"{synthesis.company_name} ({synthesis.ticker}) is the subject of this "
        f"equity research analysis. The following report synthesises data from "
        f"SEC filings, financial metrics extraction, risk factor analysis, "
        f"and recent news sentiment to provide a comprehensive investment view."
    )

    report = ResearchReport(
        title=f"{synthesis.company_name} ({synthesis.ticker}) — Equity Research Report",
        company_name=synthesis.company_name,
        ticker=synthesis.ticker,
        rating=synthesis.rating,
        rating_rationale=synthesis.rating_rationale,
        executive_summary=(
            f"**Rating: {rating_str}** | "
            f"Confidence: {synthesis.confidence:.0%}\n\n"
            f"{synthesis.investment_thesis}"
        ),
        company_overview=company_overview,
        financial_analysis=_build_financial_analysis(synthesis, metrics),
        risk_assessment=_build_risk_assessment(synthesis, risk),
        market_sentiment=_build_market_sentiment(synthesis, news),
        investment_thesis=synthesis.investment_thesis,
        strengths=synthesis.strengths,
        weaknesses=synthesis.weaknesses,
        catalysts=synthesis.catalysts,
        conclusion=(
            f"## Investment Recommendation: {rating_str}\n\n"
            f"{synthesis.rating_rationale}\n\n"
            f"### Bull Case\n"
            + "\n".join(f"- {s}" for s in synthesis.strengths)
            + f"\n\n### Bear Case\n"
            + "\n".join(f"- {w}" for w in synthesis.weaknesses)
            + (f"\n\n### Upcoming Catalysts\n"
               + "\n".join(f"- {c}" for c in synthesis.catalysts)
               if synthesis.catalysts else "")
            + f"\n\n---\n*Analysis confidence: {synthesis.confidence:.0%}*"
            + (f"\n*{synthesis.data_quality_note}*" if synthesis.data_quality_note else "")
        ),
    )

    logger.info("Research report generated: %s", report.title)
    return report


# ═════════════════════════════════════════════════════════════════════════════
# Output Fork — generates both documents
# ═════════════════════════════════════════════════════════════════════════════


def generate_output_fork(
    synthesis: SynthesisOutput,
    metrics: MetricsOutput | None = None,
    risk: RiskOutput | None = None,
    news: NewsOutput | None = None,
) -> tuple[InvestmentMemo, ResearchReport]:
    """Generate both InvestmentMemo and ResearchReport from SynthesisOutput.

    This is the main entry point for the output fork. It generates both
    documents deterministically — no LLM call needed.

    Args:
        synthesis: Output from the Synthesis Agent.
        metrics: Optional MetricsOutput for detailed data.
        risk: Optional RiskOutput for risk details.
        news: Optional NewsOutput for news data.

    Returns:
        A tuple of (InvestmentMemo, ResearchReport).
    """
    logger.info("Generating output fork for %s (%s)...",
                synthesis.company_name, synthesis.ticker)

    memo = generate_memo(synthesis, metrics, risk, news)
    report = generate_report(synthesis, metrics, risk, news)

    logger.info(
        "Output fork complete: memo='%s', report='%s'",
        memo.title, report.title,
    )
    return memo, report
