"""
News Agent — searches recent news and analyses sentiment.

Uses Gemini-3-Flash-Preview (Google AI → Gemini-2.5-Flash fallback)
with Tavily web search for live news retrieval, then structured
output → NewsOutput.
"""

import logging

from langchain_core.prompts import ChatPromptTemplate
from langchain_community.tools.tavily_search import TavilySearchResults

from agents.base import get_llm, create_langfuse_config
from config import settings
from schemas.agents import NewsOutput

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are a financial news analyst. You will receive recent news articles
about a company retrieved from web search.

Your job is to:
1. Summarise each article with its headline, source, date, and sentiment.
2. Classify sentiment as: very_positive, positive, neutral, negative, or very_negative.
3. Rate relevance of each article to the company (0-1).
4. Provide an overall sentiment assessment across all articles.
5. Identify the single most impactful recent development.

IMPORTANT RULES:
1. ONLY analyse the news articles provided — do not fabricate articles.
2. Be objective — let the facts drive sentiment, not assumptions.
3. Consider both direct company news and broader market/sector news.
4. Include a confidence score (0-1) reflecting how complete the news
   coverage is for an informed assessment.
"""

HUMAN_TEMPLATE = """\
Analyse recent news for {company_name} ({ticker}).

Query: {query}

--- Retrieved News Articles ---
{news_context}
--- End Articles ---

Provide a comprehensive news sentiment analysis.
"""

PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", HUMAN_TEMPLATE),
])


def _search_news(company_name: str, ticker: str, query: str) -> str:
    """Search for recent news using Tavily.

    Returns formatted news text for the LLM prompt.
    """
    search = TavilySearchResults(
        tavily_api_key=settings.tavily_api_key,
        max_results=8,
        search_depth="advanced",
        include_answer=True,
    )

    search_query = f"{company_name} {ticker} {query} latest news financial"
    logger.info("Tavily search: '%s'", search_query)

    try:
        results = search.invoke(search_query)
    except Exception as exc:
        logger.warning("Tavily search failed: %s", exc)
        return f"[News search failed: {exc}. Analyse based on available context only.]"

    if not results:
        return "[No recent news articles found for this company.]"

    # Format results for the LLM
    formatted = []
    for i, result in enumerate(results, 1):
        url = result.get("url", "")
        content = result.get("content", "")
        formatted.append(f"Article {i}:\nURL: {url}\n{content}\n")

    news_text = "\n---\n".join(formatted)
    logger.info("Tavily returned %d articles", len(results))
    return news_text


def run_news_agent(
    company_name: str,
    ticker: str,
    query: str,
    session_id: str = "",
) -> NewsOutput:
    """Run the News Agent to search and analyse recent news.

    Unlike Metrics/Risk agents, the News Agent does NOT receive RAG context.
    It uses Tavily to search for live news, then analyses sentiment.

    Args:
        company_name: Company being analysed.
        ticker: Stock ticker.
        query: The focused query for this agent.
        session_id: Langfuse session ID for tracing.

    Returns:
        NewsOutput with articles, sentiment analysis, and key developments.
    """
    # Step 1: Search for recent news
    news_context = _search_news(company_name, ticker, query)

    # Step 2: Analyse with LLM
    llm = get_llm("news")
    structured_llm = llm.with_structured_output(NewsOutput)

    chain = PROMPT | structured_llm

    config = create_langfuse_config(
        session_id=session_id,
        trace_name="news-agent",
    )

    logger.info("Running News Agent for %s (%s)", company_name, ticker)

    result = chain.invoke(
        {
            "company_name": company_name,
            "ticker": ticker,
            "query": query,
            "news_context": news_context,
        },
        config=config,
    )

    logger.info(
        "News Agent complete: confidence=%.2f, %d articles, sentiment=%s",
        result.confidence,
        len(result.articles),
        result.overall_sentiment.value,
    )
    return result
