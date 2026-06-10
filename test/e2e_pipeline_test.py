"""
End-to-end test of the full research pipeline.

Tests the complete flow:
    User query → Supervisor → (Metrics + Risk + News) in parallel → Synthesis

Note: This runs against LIVE LLM APIs and Tavily search.
      No Qdrant collection exists yet, so RAG retrieval will return empty.
      The pipeline should handle this gracefully — News Agent will work
      via Tavily, while Metrics/Risk agents will report low confidence.

Usage:
    uv run python test/e2e_pipeline_test.py
"""

import json
import logging
import os
import sys
import time

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)-25s | %(levelname)-5s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("e2e_test")


def test_supervisor_only():
    """Test just the Supervisor node — cheapest LLM call."""
    from agents.base import invoke_with_fallback, create_langfuse_config, AGENT_MODELS
    from langchain_core.prompts import ChatPromptTemplate
    from schemas.agents import SupervisorDecision

    logger.info("=" * 60)
    logger.info("TEST 1: Supervisor routing decision")
    logger.info("=" * 60)

    prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "You are a financial research supervisor. Decide which agents to invoke: "
            "'metrics', 'risk', 'news'. Include a focused query for each. "
            "Set context_needed=True for metrics/risk, False for news."
        )),
        ("human", "Query: {query}\nCompany: {company_name}\nTicker: {ticker}"),
    ])

    config = create_langfuse_config(
        session_id="e2e-test",
        trace_name="test-supervisor",
    )

    start = time.time()
    try:
        decision = invoke_with_fallback(
            agent_name="supervisor",
            prompt_chain=prompt,
            input_data={
                "query": "Analyse Apple's financial health and recent news",
                "company_name": "Apple Inc.",
                "ticker": "AAPL",
            },
            output_schema=SupervisorDecision,
            config=config,
        )
        elapsed = time.time() - start

        logger.info("✅ Supervisor completed in %.1fs", elapsed)
        logger.info("   Company: %s (%s)", decision.company_name, decision.ticker)
        logger.info("   Reasoning: %s", decision.reasoning[:100])
        logger.info("   Agents selected: %s", [t.agent_name for t in decision.tasks])
        for t in decision.tasks:
            logger.info("     → %s: %s", t.agent_name, t.query[:80])
        return True
    except Exception as exc:
        elapsed = time.time() - start
        logger.error("❌ Supervisor failed in %.1fs: %s", elapsed, exc)
        return False


def test_news_agent_only():
    """Test just the News Agent — uses Tavily, no RAG needed."""
    from agents.news_agent import run_news_agent

    logger.info("=" * 60)
    logger.info("TEST 2: News Agent (Tavily search + sentiment)")
    logger.info("=" * 60)

    start = time.time()
    try:
        result = run_news_agent(
            company_name="Apple Inc.",
            ticker="AAPL",
            query="Apple latest news financial performance 2025",
            session_id="e2e-test",
        )
        elapsed = time.time() - start

        logger.info("✅ News Agent completed in %.1fs", elapsed)
        logger.info("   Articles found: %d", len(result.articles))
        logger.info("   Overall sentiment: %s", result.overall_sentiment.value)
        logger.info("   Key development: %s", result.key_development[:100] if result.key_development else "N/A")
        logger.info("   Confidence: %.0f%%", result.confidence * 100)
        for a in result.articles[:3]:
            logger.info("     → [%s] %s", a.sentiment.value, a.headline[:60])
        return True, result
    except Exception as exc:
        elapsed = time.time() - start
        logger.error("❌ News Agent failed in %.1fs: %s", elapsed, exc)
        return False, None


async def test_full_pipeline():
    """Test the complete pipeline: Supervisor → Agents (parallel) → Synthesis."""
    from graph.workflow import run_research

    logger.info("=" * 60)
    logger.info("TEST 3: Full Pipeline (Supervisor → Agents → Synthesis)")
    logger.info("=" * 60)
    logger.info("NOTE: No Qdrant data — Metrics/Risk will get empty retrieval.")
    logger.info("      News Agent should work via Tavily.")
    logger.info("")

    start = time.time()
    try:
        final_state = await run_research(
            query="Analyse Apple's financial health, key risks, and recent news",
            company_name="Apple Inc.",
            ticker="AAPL",
            session_id="e2e-test-full",
        )
        elapsed = time.time() - start

        logger.info("")
        logger.info("=" * 60)
        logger.info("✅ Full pipeline completed in %.1fs", elapsed)
        logger.info("=" * 60)

        # Supervisor
        if final_state.supervisor_decision:
            d = final_state.supervisor_decision
            logger.info("  Supervisor: %d agents selected", len(d.tasks))
        else:
            logger.info("  Supervisor: FAILED (used fallback routing)")

        # Metrics
        if final_state.metrics_output:
            m = final_state.metrics_output
            logger.info("  Metrics: ✅ (confidence=%.0f%%)", m.confidence * 100)
            if m.revenue:
                logger.info("    Revenue: %s", m.revenue.value)
        else:
            logger.info("  Metrics: ❌ (no output — expected without Qdrant data)")

        # Risk
        if final_state.risk_output:
            r = final_state.risk_output
            logger.info("  Risk: ✅ (%d risks, overall=%s, confidence=%.0f%%)",
                       len(r.risks), r.overall_risk_level.value, r.confidence * 100)
        else:
            logger.info("  Risk: ❌ (no output — expected without Qdrant data)")

        # News
        if final_state.news_output:
            n = final_state.news_output
            logger.info("  News: ✅ (%d articles, sentiment=%s, confidence=%.0f%%)",
                       len(n.articles), n.overall_sentiment.value, n.confidence * 100)
        else:
            logger.info("  News: ❌ (unexpected — Tavily should work)")

        # Synthesis
        if final_state.synthesis_output:
            s = final_state.synthesis_output
            logger.info("  Synthesis: ✅ (rating=%s, confidence=%.0f%%)",
                       s.rating.value, s.confidence * 100)
            logger.info("  Thesis (first 200 chars): %s", s.investment_thesis[:200])
            logger.info("  Strengths: %s", s.strengths[:3])
            logger.info("  Weaknesses: %s", s.weaknesses[:3])
        else:
            logger.info("  Synthesis: ❌ (failed)")

        # Errors
        if final_state.errors:
            logger.warning("  Errors: %s", final_state.errors)

        return True
    except Exception as exc:
        elapsed = time.time() - start
        logger.error("❌ Full pipeline failed in %.1fs: %s", elapsed, exc)
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    logger.info("🚀 Financial Research Pipeline — End-to-End Test")
    logger.info("")

    results = {}

    # Test 1: Supervisor only (cheapest)
    results["supervisor"] = test_supervisor_only()
    logger.info("")

    if not results["supervisor"]:
        logger.error("Supervisor failed — skipping remaining tests.")
        sys.exit(1)

    # Test 2: News Agent only (no RAG needed)
    news_ok, news_result = test_news_agent_only()
    results["news_agent"] = news_ok
    logger.info("")

    # Test 3: Full pipeline
    import asyncio
    results["full_pipeline"] = asyncio.run(test_full_pipeline())
    logger.info("")

    # Summary
    logger.info("=" * 60)
    logger.info("TEST SUMMARY")
    logger.info("=" * 60)
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        logger.info("  %s: %s", test_name, status)

    all_passed = all(results.values())
    logger.info("")
    logger.info("Overall: %s", "✅ ALL TESTS PASSED" if all_passed else "❌ SOME TESTS FAILED")
    sys.exit(0 if all_passed else 1)
