"""
Agents router — test individual agents without the full pipeline.

POST /agents/metrics  → run Metrics Agent standalone
POST /agents/risk     → run Risk Agent standalone
POST /agents/news     → run News Agent standalone
POST /agents/synthesis → run Synthesis Agent with mock inputs
"""

import logging

from fastapi import APIRouter

from api.models import AgentTestRequest
from agents.metrics_agent import run_metrics_agent
from agents.risk_agent import run_risk_agent
from agents.news_agent import run_news_agent
from agents.synthesis_agent import run_synthesis_agent

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/agents", tags=["agents"])


@router.post("/metrics")
async def test_metrics_agent(req: AgentTestRequest):
    """Run the Metrics Agent standalone (for testing/debugging)."""
    try:
        result = run_metrics_agent(
            company_name=req.company_name,
            ticker=req.ticker,
            session_id=req.session_id or "test-metrics",
        )
        return {
            "agent": "metrics",
            "status": "success",
            "output": result.model_dump(mode="json"),
        }
    except Exception as exc:
        logger.error("Metrics agent test failed: %s", exc)
        return {"agent": "metrics", "status": "failed", "error": str(exc)}


@router.post("/risk")
async def test_risk_agent(req: AgentTestRequest):
    """Run the Risk Agent standalone (for testing/debugging)."""
    try:
        result = run_risk_agent(
            company_name=req.company_name,
            ticker=req.ticker,
            session_id=req.session_id or "test-risk",
        )
        return {
            "agent": "risk",
            "status": "success",
            "output": result.model_dump(mode="json"),
        }
    except Exception as exc:
        logger.error("Risk agent test failed: %s", exc)
        return {"agent": "risk", "status": "failed", "error": str(exc)}


@router.post("/news")
async def test_news_agent(req: AgentTestRequest):
    """Run the News Agent standalone (for testing/debugging)."""
    try:
        result = run_news_agent(
            company_name=req.company_name,
            ticker=req.ticker,
            session_id=req.session_id or "test-news",
        )
        return {
            "agent": "news",
            "status": "success",
            "output": result.model_dump(mode="json"),
        }
    except Exception as exc:
        logger.error("News agent test failed: %s", exc)
        return {"agent": "news", "status": "failed", "error": str(exc)}


@router.post("/synthesis")
async def test_synthesis_agent(req: AgentTestRequest):
    """Run the Synthesis Agent standalone (no prior agent outputs)."""
    try:
        result = run_synthesis_agent(
            company_name=req.company_name,
            ticker=req.ticker,
            metrics_output=None,
            risk_output=None,
            news_output=None,
            session_id=req.session_id or "test-synthesis",
        )
        return {
            "agent": "synthesis",
            "status": "success",
            "output": result.model_dump(mode="json"),
        }
    except Exception as exc:
        logger.error("Synthesis agent test failed: %s", exc)
        return {"agent": "synthesis", "status": "failed", "error": str(exc)}
