"""
Pydantic structured-output models for each LLM agent.

Every agent uses .with_structured_output(Model) — no raw string parsing.

Agent → Model mapping:
  Metrics Agent  (llama-3.1-8b  via OpenRouter) → MetricsOutput
  Risk Agent     (qwen-2.5-72b  via OpenRouter) → RiskOutput
  News Agent     (gemini-flash  via OpenRouter)  → NewsOutput
  Synthesis Agent(gpt-4o)                        → SynthesisOutput
  Supervisor     (router)                        → SupervisorDecision
"""

from datetime import date
from enum import Enum

from pydantic import BaseModel, Field


# ═════════════════════════════════════════════════════════════════════════════
# Metrics Agent — financial metrics extraction
# ═════════════════════════════════════════════════════════════════════════════


class FinancialMetric(BaseModel):
    """A single extracted financial metric with sourcing."""

    name: str = Field(..., description="Metric name, e.g. 'Revenue', 'Net Income'")
    value: str = Field(..., description="Metric value, e.g. '$394.3B', '25.3%'")
    period: str = Field(
        default="", description="Reporting period, e.g. 'FY2024', 'Q3 2024'"
    )
    prior_period_value: str = Field(
        default="",
        description="Prior period value for YoY comparison, e.g. '$351.0B'",
    )
    yoy_change: str = Field(
        default="",
        description="Year-over-year change, e.g. '+8.2%', '-3.1%'",
    )
    trend: str = Field(
        default="",
        description="Trend direction: 'improving', 'stable', or 'declining'",
    )
    source_section: str = Field(
        default="",
        description="Document section where this metric was found",
    )


# ── Ratio groups — structured metric clusters ────────────────────────────────


class ProfitabilityRatios(BaseModel):
    """Profitability ratio cluster."""
    gross_margin: FinancialMetric | None = Field(default=None, description="Gross margin %")
    operating_margin: FinancialMetric | None = Field(default=None, description="Operating margin %")
    net_margin: FinancialMetric | None = Field(default=None, description="Net profit margin %")
    roe: FinancialMetric | None = Field(default=None, description="Return on equity %")
    roa: FinancialMetric | None = Field(default=None, description="Return on assets %")


class LiquidityRatios(BaseModel):
    """Liquidity ratio cluster."""
    current_ratio: FinancialMetric | None = Field(default=None, description="Current ratio")
    quick_ratio: FinancialMetric | None = Field(default=None, description="Quick ratio")
    cash_ratio: FinancialMetric | None = Field(default=None, description="Cash ratio")


class LeverageRatios(BaseModel):
    """Leverage ratio cluster."""
    debt_to_equity: FinancialMetric | None = Field(default=None, description="Debt-to-equity ratio")
    interest_coverage: FinancialMetric | None = Field(default=None, description="Interest coverage ratio")
    debt_to_ebitda: FinancialMetric | None = Field(default=None, description="Debt/EBITDA ratio")


class EfficiencyRatios(BaseModel):
    """Efficiency ratio cluster."""
    asset_turnover: FinancialMetric | None = Field(default=None, description="Asset turnover ratio")
    inventory_turnover: FinancialMetric | None = Field(default=None, description="Inventory turnover")
    receivables_turnover: FinancialMetric | None = Field(default=None, description="Receivables turnover")


class SegmentBreakdown(BaseModel):
    """Revenue breakdown by business segment."""
    segment_name: str = Field(..., description="Segment name, e.g. 'iPhone', 'Services'")
    revenue: str = Field(..., description="Segment revenue, e.g. '$200.6B'")
    pct_of_total: str = Field(default="", description="Percentage of total revenue, e.g. '52.2%'")
    yoy_change: str = Field(default="", description="YoY change, e.g. '+5.3%'")


class MetricsOutput(BaseModel):
    """Structured output from the Metrics Agent.

    Extracts key financial metrics from SEC filings and financial documents.
    """

    company_name: str = Field(..., description="Company name")
    ticker: str = Field(default="", description="Stock ticker")
    fiscal_period: str = Field(
        default="", description="Fiscal period analysed, e.g. 'FY2024'"
    )

    # ── Core financial metrics ───────────────────────────────────────────────
    revenue: FinancialMetric | None = Field(
        default=None, description="Total revenue / net sales"
    )
    net_income: FinancialMetric | None = Field(default=None, description="Net income")
    gross_margin: FinancialMetric | None = Field(
        default=None, description="Gross profit margin %"
    )
    operating_margin: FinancialMetric | None = Field(
        default=None, description="Operating margin %"
    )
    eps: FinancialMetric | None = Field(
        default=None, description="Earnings per share (diluted)"
    )
    pe_ratio: FinancialMetric | None = Field(
        default=None, description="Price-to-earnings ratio"
    )
    debt_to_equity: FinancialMetric | None = Field(
        default=None, description="Debt-to-equity ratio"
    )
    free_cash_flow: FinancialMetric | None = Field(
        default=None, description="Free cash flow"
    )
    roe: FinancialMetric | None = Field(
        default=None, description="Return on equity %"
    )

    # ── Balance sheet & liquidity (critical for full analysis) ──────────────
    total_assets: FinancialMetric | None = Field(
        default=None, description="Total assets"
    )
    total_liabilities: FinancialMetric | None = Field(
        default=None, description="Total liabilities"
    )
    total_debt: FinancialMetric | None = Field(
        default=None, description="Total debt (short-term + long-term)"
    )
    cash_and_equivalents: FinancialMetric | None = Field(
        default=None, description="Cash and cash equivalents"
    )
    shares_outstanding: FinancialMetric | None = Field(
        default=None, description="Shares outstanding (diluted)"
    )
    dividend_per_share: FinancialMetric | None = Field(
        default=None, description="Dividend per share (if applicable)"
    )

    # ── Additional metrics (catch-all) ───────────────────────────────
    additional_metrics: list[FinancialMetric] = Field(
        default_factory=list,
        description="Any other notable metrics not covered above",
    )

    # ── Ratio groups (structured clusters) ─────────────────────────────────
    profitability: ProfitabilityRatios | None = Field(
        default=None, description="Profitability ratios (margins, ROE, ROA)"
    )
    liquidity: LiquidityRatios | None = Field(
        default=None, description="Liquidity ratios (current, quick, cash)"
    )
    leverage: LeverageRatios | None = Field(
        default=None, description="Leverage ratios (D/E, interest coverage, Debt/EBITDA)"
    )
    efficiency: EfficiencyRatios | None = Field(
        default=None, description="Efficiency ratios (asset turnover, inventory, receivables)"
    )

    # ── Segment breakdowns ─────────────────────────────────────────────────
    segments: list[SegmentBreakdown] = Field(
        default_factory=list,
        description="Revenue breakdown by business segment",
    )

    # ── Guidance & forward-looking ─────────────────────────────────────────
    revenue_guidance: str = Field(
        default="", description="Management revenue guidance/outlook, e.g. '$400-410B'"
    )
    eps_guidance: str = Field(
        default="", description="Management EPS guidance, e.g. '$6.50-6.70'"
    )
    filing_type: str = Field(
        default="", description="Filing type detected: '10-K', '10-Q', '8-K', etc."
    )
    fiscal_year_end: str = Field(
        default="", description="Fiscal year end date, e.g. 'September 28, 2024'"
    )

    # ── 8-K material events (populated only for 8-K filings) ──────────────
    events: list["MaterialEvent"] = Field(
        default_factory=list,
        description="Material events from 8-K filings (empty for 10-K/10-Q)",
    )

    # ── Proxy executive compensation (populated only for DEF 14A) ────────
    executive_compensation: list["ExecutiveCompensation"] = Field(
        default_factory=list,
        description="Executive compensation data from proxy filings (empty for 10-K/10-Q)",
    )

    summary: str = Field(
        ..., description="3-5 sentence summary of the company's financial position"
    )
    confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Agent confidence in the extraction (0-1)",
    )


# ═════════════════════════════════════════════════════════════════════════════
# 8-K Material Events — event-driven filings
# ═════════════════════════════════════════════════════════════════════════════


class MaterialEvent(BaseModel):
    """A material event disclosed in an 8-K filing.

    8-K filings are fundamentally different from periodic filings —
    they report a single material event rather than periodic financials.
    This model captures the event cleanly without complicating the
    standard metrics flow.
    """

    event_type: str = Field(
        ...,
        description=(
            "Event category, e.g. 'acquisition', 'leadership_change', "
            "'earnings_release', 'bankruptcy', 'material_agreement', 'other'"
        ),
    )
    headline: str = Field(..., description="One-line summary of the event")
    description: str = Field(..., description="Detailed event description")
    event_date: str = Field(default="", description="Date of the event")
    financial_impact: str = Field(
        default="",
        description="Estimated financial impact, e.g. '$2.1B acquisition cost'",
    )


# ═════════════════════════════════════════════════════════════════════════════
# Proxy Executive Compensation — DEF 14A filings
# ═════════════════════════════════════════════════════════════════════════════


class ExecutiveCompensation(BaseModel):
    """Compensation data for a single executive from a Proxy statement.

    Keeps it simple — the Synthesis Agent uses this to add a compensation
    note to the report without overloading the main metrics pipeline.
    """

    name: str = Field(..., description="Executive name")
    title: str = Field(..., description="Executive title, e.g. 'CEO', 'CFO'")
    total_compensation: str = Field(
        ..., description="Total compensation, e.g. '$63.2M'"
    )
    base_salary: str = Field(default="", description="Base salary")
    stock_awards: str = Field(default="", description="Stock awards value")
    bonus: str = Field(default="", description="Cash bonus / incentive")


# ═════════════════════════════════════════════════════════════════════════════
# Risk Agent — risk assessment
# ═════════════════════════════════════════════════════════════════════════════


class RiskSeverity(str, Enum):
    """Risk severity classification."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class RiskCategory(str, Enum):
    """Standard risk categories for financial analysis."""

    MARKET = "market"
    CREDIT = "credit"
    OPERATIONAL = "operational"
    REGULATORY = "regulatory"
    COMPETITIVE = "competitive"
    TECHNOLOGICAL = "technological"
    GEOPOLITICAL = "geopolitical"
    ESG = "esg"
    LIQUIDITY = "liquidity"
    OTHER = "other"


class IdentifiedRisk(BaseModel):
    """A single risk identified from the document."""

    title: str = Field(..., description="Short risk title")
    description: str = Field(..., description="Detailed risk description")
    category: RiskCategory = Field(..., description="Risk category")
    severity: RiskSeverity = Field(..., description="Severity level")
    likelihood: str = Field(
        default="", description="Likelihood assessment, e.g. 'probable', 'possible'"
    )
    potential_impact: str = Field(
        default="", description="Potential financial/operational impact"
    )
    mitigants: str = Field(
        default="", description="Any mitigating factors mentioned in the filing"
    )
    source_section: str = Field(
        default="", description="Document section where this risk was found"
    )


class RiskOutput(BaseModel):
    """Structured output from the Risk Agent.

    Identifies, categorises, and scores risks from financial documents.
    """

    company_name: str = Field(..., description="Company name")
    risks: list[IdentifiedRisk] = Field(
        default_factory=list, description="All identified risks"
    )
    overall_risk_level: RiskSeverity = Field(
        ..., description="Overall risk assessment for the company"
    )
    risk_summary: str = Field(
        ..., description="2-3 sentence executive summary of the risk landscape"
    )
    key_concern: str = Field(
        default="",
        description="The single most important risk to watch",
    )
    confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Agent confidence in the assessment (0-1)",
    )


# ═════════════════════════════════════════════════════════════════════════════
# News Agent — recent news + sentiment
# ═════════════════════════════════════════════════════════════════════════════


class SentimentLabel(str, Enum):
    """News sentiment classification."""

    VERY_POSITIVE = "very_positive"
    POSITIVE = "positive"
    NEUTRAL = "neutral"
    NEGATIVE = "negative"
    VERY_NEGATIVE = "very_negative"


class NewsItem(BaseModel):
    """A single news article or event."""

    headline: str = Field(..., description="News headline")
    source: str = Field(default="", description="Publication / source name")
    url: str = Field(default="", description="Article URL")
    published_date: str = Field(default="", description="Publication date")
    summary: str = Field(..., description="Brief summary of the article")
    sentiment: SentimentLabel = Field(..., description="Sentiment of this article")
    relevance: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Relevance to the company (0-1)",
    )


class NewsOutput(BaseModel):
    """Structured output from the News Agent.

    Searches recent news via Tavily and analyses sentiment.
    """

    company_name: str = Field(..., description="Company name")
    search_query: str = Field(
        default="", description="The search query used to find news"
    )
    articles: list[NewsItem] = Field(
        default_factory=list, description="Retrieved news articles"
    )
    overall_sentiment: SentimentLabel = Field(
        ..., description="Aggregate sentiment across all articles"
    )
    sentiment_summary: str = Field(
        ...,
        description="2-3 sentence summary of the news landscape and market sentiment",
    )
    key_development: str = Field(
        default="",
        description="The single most impactful recent development",
    )
    confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Agent confidence in the analysis (0-1)",
    )


# ═════════════════════════════════════════════════════════════════════════════
# Synthesis Agent — combined analysis + recommendation
# ═════════════════════════════════════════════════════════════════════════════


class InvestmentRating(str, Enum):
    """Investment recommendation rating."""

    STRONG_BUY = "strong_buy"
    BUY = "buy"
    HOLD = "hold"
    SELL = "sell"
    STRONG_SELL = "strong_sell"


class SynthesisOutput(BaseModel):
    """Structured output from the Synthesis Agent.

    Merges outputs from Metrics, Risk, and News agents into a
    coherent investment thesis with an actionable recommendation.
    """

    company_name: str = Field(..., description="Company name")
    ticker: str = Field(default="", description="Stock ticker")

    # ── Investment thesis ────────────────────────────────────────────────────
    investment_thesis: str = Field(
        ..., description="Detailed investment thesis (3-5 paragraphs)"
    )
    rating: InvestmentRating = Field(..., description="Investment recommendation")
    rating_rationale: str = Field(
        ..., description="Why this rating was chosen (2-3 sentences)"
    )

    # ── Structured sections ──────────────────────────────────────────────────
    strengths: list[str] = Field(
        default_factory=list, description="Key strengths / bull case points"
    )
    weaknesses: list[str] = Field(
        default_factory=list, description="Key weaknesses / bear case points"
    )
    catalysts: list[str] = Field(
        default_factory=list,
        description="Upcoming catalysts that could move the stock",
    )

    # ── Pre-formatted summary tables (markdown) ──────────────────────────────
    financial_health_summary: str = Field(
        default="",
        description=(
            "Markdown table summarising key financial metrics: "
            "| Metric | Current | Prior | YoY Change | Trend |"
        ),
    )
    valuation_snapshot: str = Field(
        default="",
        description=(
            "Markdown table of valuation metrics: "
            "| Metric | Value | Interpretation |"
        ),
    )
    swot_analysis: str = Field(
        default="",
        description=(
            "Structured SWOT analysis in markdown format covering "
            "Strengths, Weaknesses, Opportunities, and Threats"
        ),
    )
    key_metrics_table: str = Field(
        default="",
        description=(
            "Comprehensive markdown table of ALL extracted financial metrics "
            "for easy scanning"
        ),
    )

    # ── Confidence ───────────────────────────────────────────────────────────
    data_quality_note: str = Field(
        default="",
        description="Note on data quality / completeness of the analysis",
    )
    confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Overall confidence in the synthesis (0-1)",
    )


# ═════════════════════════════════════════════════════════════════════════════
# Supervisor — agent routing decisions
# ═════════════════════════════════════════════════════════════════════════════


class AgentTask(BaseModel):
    """A single task assignment from the Supervisor to a specialist agent."""

    agent_name: str = Field(
        ...,
        description="Agent to invoke: 'metrics', 'risk', or 'news'",
    )
    query: str = Field(
        ...,
        description="Focused query/instruction tailored for this agent",
    )
    context_needed: bool = Field(
        default=True,
        description="Whether this agent needs RAG context from the retriever",
    )


class SupervisorDecision(BaseModel):
    """Structured output from the Supervisor node.

    The Supervisor analyses the user query and decides which specialist
    agents to invoke in parallel via LangGraph Send(). This model is
    used with .with_structured_output(SupervisorDecision).
    """

    company_name: str = Field(..., description="Company being analysed")
    ticker: str = Field(default="", description="Stock ticker if identified")
    original_query: str = Field(..., description="The user's original research query")
    tasks: list[AgentTask] = Field(
        ..., description="Agent tasks to execute in parallel via Send()"
    )
    reasoning: str = Field(
        ..., description="Why these agents were selected for this query"
    )
