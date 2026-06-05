"""
tavily_search — web search tool with DuckDuckGo fallback.

Primary: Tavily API (advanced search, max 8 results)
Fallback: DuckDuckGo (if Tavily is rate limited)

Used by: News Agent, Chat Agent.
"""

import logging

from langchain_core.tools import tool

from config import settings

logger = logging.getLogger(__name__)


@tool
def tavily_search(query: str, max_results: int = 8) -> list[dict]:
    """Search the web for recent news and information.

    Uses Tavily API for high-quality search results with content extraction.
    Falls back to DuckDuckGo if Tavily is rate limited.

    Args:
        query: Search query, e.g. "Apple AAPL news 2024".
        max_results: Maximum number of results (default: 8).

    Returns:
        List of dicts with url, title, content, and score.
    """
    # Try Tavily first
    if settings.tavily_api_key:
        try:
            from langchain_community.tools.tavily_search import TavilySearchResults

            search = TavilySearchResults(
                tavily_api_key=settings.tavily_api_key,
                max_results=max_results,
                search_depth="basic",
            )
            results = search.invoke(query)
            logger.info("tavily_search: '%s' → %d results", query[:50], len(results))
            return results
        except Exception as exc:
            logger.warning("Tavily search failed: %s — falling back to DuckDuckGo", exc)

    # Fallback: DuckDuckGo
    try:
        from langchain_community.tools import DuckDuckGoSearchResults

        ddg = DuckDuckGoSearchResults(max_results=min(max_results, 5))
        results = ddg.invoke(query)

        # DuckDuckGo returns a string, parse it
        if isinstance(results, str):
            # Convert string results to list format
            logger.info("tavily_search (DuckDuckGo fallback): '%s'", query[:50])
            return [{"content": results, "url": "", "source_type": "duckduckgo"}]

        logger.info("tavily_search (DuckDuckGo fallback): '%s' → %d results", query[:50], len(results))
        return results
    except Exception as exc:
        logger.error("Both Tavily and DuckDuckGo failed: %s", exc)
        return [{"content": f"Web search unavailable: {exc}", "url": ""}]
