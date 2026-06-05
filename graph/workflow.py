"""
LangGraph workflow — Supervisor → parallel agents → Synthesis → END.

Each agent autonomously calls its own tools (ReAct pattern):
  - Metrics Agent: rag_retriever + extract_financial_table
  - Risk Agent:    rag_retriever + risk_classifier
  - News Agent:    tavily_search + sentiment_scorer
  - Synthesis:     rag_retriever (optional cross-section context)

No pre-fetched context — each agent decides what to retrieve.

Graph topology:
    START → supervisor → run_agents → synthesis → END
                          ├─ metrics_agent (tools: rag_retriever, extract_financial_table)
                          ├─ risk_agent    (tools: rag_retriever, risk_classifier)
                          └─ news_agent    (tools: tavily_search, sentiment_scorer)
"""

import logging

import asyncio

from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END

from agents.base import invoke_with_fallback, create_langfuse_config
from agents.metrics_agent import run_metrics_agent
from agents.risk_agent import run_risk_agent
from agents.news_agent import run_news_agent
from agents.synthesis_agent import run_synthesis_agent
from graph.state import ResearchState
from schemas.agents import SupervisorDecision

logger = logging.getLogger(__name__)


# ═════════════════════════════════════════════════════════════════════════════
# Node: Supervisor — decides which agents to run
# ═════════════════════════════════════════════════════════════════════════════

SUPERVISOR_SYSTEM = """\
You are a financial research supervisor. Given a user's research query
about a company, decide which specialist agents to invoke:

Available agents:
- "metrics" — extracts financial metrics from SEC filings (uses RAG tools)
- "risk" — identifies and assesses risks from filings (uses RAG tools)
- "news" — searches recent news and analyses sentiment (uses web search)

RULES:
1. For a comprehensive analysis, invoke ALL three agents.
2. If the query is specifically about metrics/financials, at minimum invoke "metrics".
3. If the query is about risks, at minimum invoke "risk".
4. If the query is about news/sentiment, at minimum invoke "news".
5. Always include a focused query tailored for each agent.
6. Set context_needed=True for metrics and risk (they use RAG). Set False for news.
"""

SUPERVISOR_HUMAN = """\
User query: {query}
Company: {company_name}
Ticker: {ticker}

Decide which agents to invoke and provide focused queries for each.
"""

SUPERVISOR_PROMPT = ChatPromptTemplate.from_messages([
    ("system", SUPERVISOR_SYSTEM),
    ("human", SUPERVISOR_HUMAN),
])


def supervisor_node(state: ResearchState) -> dict:
    """Supervisor node: analyses query and decides agent routing."""
    logger.info("Supervisor: analysing query for %s", state.company_name)

    config = create_langfuse_config(
        session_id=state.session_id,
        trace_name="supervisor",
    )

    try:
        decision = invoke_with_fallback(
            agent_name="supervisor",
            prompt_chain=SUPERVISOR_PROMPT,
            input_data={
                "query": state.query,
                "company_name": state.company_name,
                "ticker": state.ticker,
            },
            output_schema=SupervisorDecision,
            config=config,
        )
        logger.info(
            "Supervisor decided: %s (agents: %s)",
            decision.reasoning[:80],
            [t.agent_name for t in decision.tasks],
        )
        return {"supervisor_decision": decision}
    except Exception as exc:
        logger.error("Supervisor failed: %s — running all agents as fallback", exc)
        fallback = SupervisorDecision(
            company_name=state.company_name,
            ticker=state.ticker,
            original_query=state.query,
            tasks=[
                {"agent_name": "metrics", "query": state.query, "context_needed": True},
                {"agent_name": "risk", "query": state.query, "context_needed": True},
                {"agent_name": "news", "query": state.query, "context_needed": False},
            ],
            reasoning="Supervisor LLM failed — defaulting to all agents.",
        )
        return {
            "supervisor_decision": fallback,
            "errors": [f"Supervisor LLM failed: {exc}"],
        }


# ═════════════════════════════════════════════════════════════════════════════
# Node: Agent Runner — parallel ReAct agents via asyncio.gather
# ═════════════════════════════════════════════════════════════════════════════


async def _run_single_agent(
    agent_name: str,
    company_name: str,
    ticker: str,
    query: str,
    session_id: str,
) -> tuple[str, object | None, str | None]:
    """Run a single agent in a thread (blocking LLM calls → thread pool).

    Returns:
        (agent_name, result_or_None, error_or_None)
    """
    try:
        if agent_name == "metrics":
            result = await asyncio.to_thread(
                run_metrics_agent,
                company_name=company_name,
                ticker=ticker,
                query=query,
                session_id=session_id,
            )
            return ("metrics_output", result, None)

        elif agent_name == "risk":
            result = await asyncio.to_thread(
                run_risk_agent,
                company_name=company_name,
                ticker=ticker,
                query=query,
                session_id=session_id,
            )
            return ("risk_output", result, None)

        elif agent_name == "news":
            result = await asyncio.to_thread(
                run_news_agent,
                company_name=company_name,
                ticker=ticker,
                query=query,
                session_id=session_id,
            )
            return ("news_output", result, None)

        else:
            return (agent_name, None, f"Unknown agent: {agent_name}")

    except Exception as exc:
        error_msg = f"Agent '{agent_name}' failed: {exc}"
        logger.error(error_msg)
        return (agent_name, None, error_msg)


async def _run_agents_parallel(state: ResearchState) -> dict:
    """Run all specialist agents in parallel using asyncio.gather.

    Total time = max(metrics, risk, news) instead of sum.
    Each agent failure is captured independently — one failure
    doesn't block the others.
    """
    decision = state.supervisor_decision
    if decision is None:
        return {"errors": ["No supervisor decision — skipping agents"]}

    # Build coroutines for each agent task
    tasks = [
        _run_single_agent(
            agent_name=task.agent_name,
            company_name=state.company_name,
            ticker=state.ticker,
            query=task.query,
            session_id=state.session_id,
        )
        for task in decision.tasks
    ]

    logger.info(
        "Running %d agents in parallel: %s",
        len(tasks),
        [t.agent_name for t in decision.tasks],
    )

    # Run all agents concurrently
    results = await asyncio.gather(*tasks)

    # Merge results
    output: dict = {}
    errors: list[str] = []

    for key, result, error in results:
        if error:
            errors.append(error)
        elif result is not None:
            output[key] = result

    if errors:
        output["errors"] = errors

    logger.info(
        "Parallel agents complete: %d succeeded, %d failed",
        len([r for r in results if r[2] is None and r[1] is not None]),
        len(errors),
    )
    return output


def run_agents_node(state: ResearchState) -> dict:
    """Run specialist agents in parallel (asyncio.gather).

    Wraps the async parallel runner for LangGraph's sync node interface.
    Total time ≈ slowest agent instead of sum of all agents.
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        # Already in an async context (e.g., inside LangGraph async runner)
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as pool:
            future = pool.submit(asyncio.run, _run_agents_parallel(state))
            return future.result()
    else:
        return asyncio.run(_run_agents_parallel(state))


# ═════════════════════════════════════════════════════════════════════════════
# Node: Synthesis — merges all outputs into recommendation
# ═════════════════════════════════════════════════════════════════════════════


def synthesis_node(state: ResearchState) -> dict:
    """Run the Synthesis Agent with autonomous tool calling."""
    try:
        synthesis = run_synthesis_agent(
            company_name=state.company_name,
            ticker=state.ticker,
            metrics_output=state.metrics_output,
            risk_output=state.risk_output,
            news_output=state.news_output,
            session_id=state.session_id,
        )
        return {"synthesis_output": synthesis}
    except Exception as exc:
        logger.error("Synthesis Agent failed: %s", exc)
        return {"errors": [f"Synthesis failed: {exc}"]}


# ═════════════════════════════════════════════════════════════════════════════
# Graph construction
# ═════════════════════════════════════════════════════════════════════════════


def build_research_graph() -> StateGraph:
    """Build and compile the LangGraph research workflow.

    Graph: supervisor → run_agents (ReAct tool loops) → synthesis → END

    Returns:
        A compiled StateGraph ready for .invoke().
    """
    graph = StateGraph(ResearchState)

    # Add nodes — no retrieve_context_node, agents retrieve their own data
    graph.add_node("supervisor", supervisor_node)
    graph.add_node("run_agents", run_agents_node)
    graph.add_node("synthesis", synthesis_node)

    # Define edges
    graph.set_entry_point("supervisor")
    graph.add_edge("supervisor", "run_agents")
    graph.add_edge("run_agents", "synthesis")
    graph.add_edge("synthesis", END)

    compiled = graph.compile()
    logger.info("Research graph compiled: supervisor → agents (ReAct) → synthesis")
    return compiled


# ═════════════════════════════════════════════════════════════════════════════
# Public API
# ═════════════════════════════════════════════════════════════════════════════


def run_research(
    query: str,
    company_name: str,
    ticker: str = "",
    session_id: str = "",
) -> ResearchState:
    """Run the full research pipeline.

    Args:
        query: User's research query, e.g. "Analyse Apple's FY2024 10-K".
        company_name: Company name, e.g. "Apple Inc.".
        ticker: Stock ticker, e.g. "AAPL".
        session_id: Langfuse session ID for tracing.

    Returns:
        The final ResearchState with all agent outputs and synthesis.
    """
    graph = build_research_graph()

    initial_state = ResearchState(
        query=query,
        company_name=company_name,
        ticker=ticker,
        session_id=session_id or f"research-{company_name.lower().replace(' ', '-')}",
    )

    config = create_langfuse_config(
        session_id=initial_state.session_id,
        trace_name="research-pipeline",
    )

    logger.info(
        "Starting research pipeline: '%s' for %s (%s)",
        query, company_name, ticker,
    )

    result = graph.invoke(initial_state.model_dump(), config=config)
    final_state = ResearchState(**result)

    logger.info(
        "Research pipeline complete. Rating: %s, Errors: %d",
        final_state.synthesis_output.rating.value if final_state.synthesis_output else "N/A",
        len(final_state.errors),
    )
    return final_state
