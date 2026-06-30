"""
Pydantic schemas for all stages of the Financial Research Analyst pipeline.

Re-exports every model so consumers can do:
    from schemas import RawDocument, MetricsOutput, InvestmentMemo, ...
"""

# ── Ingestion ────────────────────────────────────────────────────────────────
from schemas.ingestion import (
    DocumentMetadata,
    DocumentPage,
    FilingType,
    IngestionSource,
    RawDocument,
)

# ── Chunking ─────────────────────────────────────────────────────────────────
from schemas.chunks import STANDARD_SECTIONS, ChildChunk, ParentChunk

# ── Retrieval ────────────────────────────────────────────────────────────────
from schemas.retrieval import RetrievedChunk, RetrievalResult

# ── Agent outputs ────────────────────────────────────────────────────────────
from schemas.agents import (
    AgentTask,
    EfficiencyRatios,
    ExecutiveCompensation,
    FinancialMetric,
    IdentifiedRisk,
    InvestmentRating,
    LeverageRatios,
    LiquidityRatios,
    MaterialEvent,
    MetricsOutput,
    NewsItem,
    NewsOutput,
    ProfitabilityRatios,
    RiskCategory,
    RiskOutput,
    RiskSeverity,
    SegmentBreakdown,
    SentimentLabel,
    SupervisorDecision,
    SynthesisOutput,
)

# ── HITL Feedback ────────────────────────────────────────────────────────────
from schemas.feedback import FeedbackAction, HumanFeedback

# ── Chat / Conversational RAG ────────────────────────────────────────────────
from schemas.chat import ChatMessage, ChatRequest, ChatResponse, MessageRole

# ── Report outputs ───────────────────────────────────────────────────────────
from schemas.reports import InvestmentMemo, ReportSection, ResearchReport

__all__ = [
    # Ingestion
    "IngestionSource",
    "FilingType",
    "DocumentMetadata",
    "DocumentPage",
    "RawDocument",
    # Chunking
    "STANDARD_SECTIONS",
    "ParentChunk",
    "ChildChunk",
    # Retrieval
    "RetrievedChunk",
    "RetrievalResult",
    # Agent outputs
    "FinancialMetric",
    "ProfitabilityRatios",
    "LiquidityRatios",
    "LeverageRatios",
    "EfficiencyRatios",
    "SegmentBreakdown",
    "MetricsOutput",
    "MaterialEvent",
    "ExecutiveCompensation",
    "RiskSeverity",
    "RiskCategory",
    "IdentifiedRisk",
    "RiskOutput",
    "SentimentLabel",
    "NewsItem",
    "NewsOutput",
    "InvestmentRating",
    "SynthesisOutput",
    # Supervisor
    "AgentTask",
    "SupervisorDecision",
    # HITL Feedback
    "FeedbackAction",
    "HumanFeedback",
    # Chat
    "MessageRole",
    "ChatMessage",
    "ChatRequest",
    "ChatResponse",
    # Reports
    "InvestmentMemo",
    "ReportSection",
    "ResearchReport",
]
