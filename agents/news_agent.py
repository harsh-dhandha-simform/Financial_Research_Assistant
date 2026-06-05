"""
News Agent — searches recent news and analyses sentiment.

Uses Gemini-3-Flash-Preview (Google AI → Gemini-2.5-Flash fallback)
with ReAct tool calling. The LLM autonomously searches for news
and scores sentiment per article.

Tools: tavily_search, sentiment_scorer
Output: NewsOutput (structured)
"""

import logging

from agents.base import run_tool_agent, create_langfuse_config
from tools.search import tavily_search
from tools.sentiment import sentiment_scorer
from schemas.agents import NewsOutput

logger = logging.getLogger(__name__)

NEWS_TOOLS = [tavily_search, sentiment_scorer]

SYSTEM_PROMPT = """\
You are a financial news analyst. You search for and analyse recent news
about companies to assess market sentiment.

You have access to the following tools:
1. tavily_search — searches the web for recent news. Build your query as:
   "{company_name} {ticker} news {relevant_topic}"
   Use max_results=8 for comprehensive coverage.
2. sentiment_scorer — scores the sentiment of a single news article.
   Call with headline and snippet for each article. Max 5 calls to
   avoid token bloat.

WORKFLOW:
1. Call tavily_search with a well-crafted query combining the company name,
   ticker, and relevant financial topics.
2. For the top 5 most relevant articles, call sentiment_scorer with the
   headline and content snippet.
3. You may call tavily_search again with different queries to cover
   different angles (earnings, competition, regulatory, etc.).

IMPORTANT RULES:
1. ONLY analyse actual news articles from the search results.
2. Do NOT fabricate articles or URLs.
3. Be objective — let the facts drive sentiment, not assumptions.
4. Consider both direct company news and broader market/sector news.
5. Include a confidence score (0-1) reflecting how complete the news
   coverage is for an informed assessment.
"""


def run_news_agent(
    company_name: str,
    ticker: str,
    query: str,
    session_id: str = "",
) -> NewsOutput:
    """Run the News Agent with autonomous tool calling.

    The agent searches for news and scores sentiment independently.
    No RAG context needed — uses web search.

    Args:
        company_name: Company being analysed.
        ticker: Stock ticker.
        query: The focused query for this agent.
        session_id: Langfuse session ID for tracing.

    Returns:
        NewsOutput with articles, sentiment analysis, and key developments.
    """
    config = create_langfuse_config(
        session_id=session_id,
        trace_name="news-agent",
    )

    human_prompt = (
        f"Search for and analyse recent news about {company_name} ({ticker}).\n\n"
        f"Query: {query}\n\n"
        f"Use tavily_search to find recent articles, then sentiment_scorer to "
        f"classify each article's sentiment. Provide a comprehensive news "
        f"and sentiment analysis."
    )

    logger.info("Running News Agent for %s (%s)", company_name, ticker)

    result = run_tool_agent(
        agent_name="news",
        system_prompt=SYSTEM_PROMPT,
        human_prompt=human_prompt,
        tools=NEWS_TOOLS,
        output_schema=NewsOutput,
        config=config,
    )

    logger.info(
        "News Agent complete: confidence=%.2f, %d articles, sentiment=%s",
        result.confidence,
        len(result.articles),
        result.overall_sentiment.value,
    )
    return result
