"""
LangGraph shared state for the financial research pipeline.

This defines the TypedDict that flows through the entire graph:
    Supervisor → (Metrics, Risk, News) → Synthesis → Output

Uses LangGraph's Annotated reducer pattern for merging parallel
agent outputs into the shared state.
"""

from typing import Annotated, Any

from langgraph.graph import MessagesState
from pydantic import BaseModel

from schemas.agents import (
    MetricsOutput,
    RiskOutput,
    NewsOutput,
    SynthesisOutput,
    SupervisorDecision,
)
from schemas.reports import InvestmentMemo, ResearchReport


def _replace(existing: Any, new: Any) -> Any:
    """Reducer: last write wins (replaces the value)."""
    return new


def _merge_list(existing: list, new: list) -> list:
    """Reducer: append new items to the existing list."""
    return existing + new


class ResearchState(BaseModel):
    """Shared state flowing through the LangGraph workflow.

    Fields use Annotated reducers so parallel agent nodes
    can write to the same state without conflicts.

    Flow:
        1. User sets: query, company_name, ticker
        2. Supervisor populates: supervisor_decision
        3. Retriever populates: context (RAG results)
        4. Agents populate: metrics_output, risk_output, news_output
        5. Synthesis populates: synthesis_output
        6. Output fork: memo, report
    """

    # ── User input ───────────────────────────────────────────────────────────
    query: str = ""
    company_name: str = ""
    ticker: str = ""
    session_id: str = ""

    # ── Supervisor decision ──────────────────────────────────────────────────
    supervisor_decision: SupervisorDecision | None = None

    # ── Retrieved context (from hybrid retriever) ────────────────────────────
    # Keyed by section filter used, e.g. {"mda": "...", "risk_factors": "..."}
    context: dict[str, str] = {}

    # ── Agent outputs (populated in parallel) ────────────────────────────────
    metrics_output: MetricsOutput | None = None
    risk_output: RiskOutput | None = None
    news_output: NewsOutput | None = None

    # ── Synthesis output ─────────────────────────────────────────────────────
    synthesis_output: SynthesisOutput | None = None

    # ── Final reports ────────────────────────────────────────────────────────
    memo: InvestmentMemo | None = None
    report: ResearchReport | None = None

    # ── Error tracking ───────────────────────────────────────────────────────
    errors: list[str] = []

    class Config:
        arbitrary_types_allowed = True
