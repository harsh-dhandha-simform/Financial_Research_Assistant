"""
get_agent_outputs — pure state reader tool for Synthesis Agent.

Reads metrics, risk, and news outputs from the LangGraph state.
No LLM call — just makes state reads visible in Langfuse traces.

Used by: Synthesis Agent.
"""

import logging
from typing import Any

from langchain_core.tools import tool

logger = logging.getLogger(__name__)


@tool
def get_agent_outputs(state: dict) -> dict:
    """Read specialist agent outputs from the graph state.

    Makes state reads visible in Langfuse traces and tool execution logs.
    No LLM call — pure state inspection.

    Args:
        state: The current LangGraph AgentState dict.

    Returns:
        Dict with metrics, risks, and news_signals from prior agents.
    """
    outputs = {
        "metrics": state.get("metrics_output"),
        "risks": state.get("risk_output"),
        "news_signals": state.get("news_output"),
    }

    available = [k for k, v in outputs.items() if v is not None]
    logger.info("get_agent_outputs: available=%s", available)

    return outputs
