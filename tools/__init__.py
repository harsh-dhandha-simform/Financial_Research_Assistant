"""Tools — LangChain @tool functions for the agent pipeline."""

from tools.retriever import rag_retriever
from tools.financial import extract_financial_table
from tools.risk import risk_classifier
from tools.sentiment import sentiment_scorer
from tools.search import tavily_search
from tools.state_reader import get_agent_outputs

__all__ = [
    "rag_retriever",
    "extract_financial_table",
    "risk_classifier",
    "sentiment_scorer",
    "tavily_search",
    "get_agent_outputs",
]

# ── Tool bundles per agent ───────────────────────────────────────────────────
# Each agent only receives the tools it needs — no cross-agent access.

metrics_tools = [rag_retriever, extract_financial_table]
risk_tools = [rag_retriever, risk_classifier]
news_tools = [tavily_search, sentiment_scorer]
synthesis_tools = [rag_retriever, get_agent_outputs]
