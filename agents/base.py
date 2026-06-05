"""
LLM factory with per-agent routing, ReAct tool-calling loops,
automatic fallback chains, and robust error handling.

Architecture:
    Each agent gets an LLM with .bind_tools() → runs a ReAct loop
    (tool calls → execute → feed back) → final structured output.

    Agents autonomously decide WHAT to retrieve and WHEN to call tools.
    No pre-fetched context — the LLM drives the retrieval.

Routing table:
    ┌──────────────┬─────────────────────────────────────────────┬──────────────────────────────────────────┐
    │ Agent        │ Primary                                     │ Fallback                                 │
    ├──────────────┼─────────────────────────────────────────────┼──────────────────────────────────────────┤
    │ Metrics      │ Llama-3.1-8B (HF Inference / Scaleway)      │ Llama-3.1-8B (Groq)                      │
    │ Risk         │ Qwen-2.5-72B (HF Inference / Novita)        │ Llama-3.3-70B-Instruct:free (OpenRouter)  │
    │ News         │ gemini-3-flash-preview (Google AI)           │ gemini-2.5-flash (Google AI)              │
    │ Supervisor   │ gpt-oss-120b (OpenRouter)                   │ Qwen-2.5-72B (HF Inference)              │
    │ Synthesis    │ gpt-oss-120b (Cerebras)                     │ Qwen3-32B (Groq)                         │
    └──────────────┴─────────────────────────────────────────────┴──────────────────────────────────────────┘

Fallback triggers:
    • Provider rate limit (HTTP 429) → switch to fallback immediately
    • Request timeout (> 30 seconds) → switch to faster fallback model
    • Malformed JSON / Pydantic validation failure → retry on fallback
    • Full provider outage (connection error) → entire chain falls to next
"""

import logging
import time
from dataclasses import dataclass
from typing import Any, TypeVar

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, ValidationError

from callbacks import get_langfuse_handler
from config import settings

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

# ── Provider base URLs ───────────────────────────────────────────────────────
HF_INFERENCE_URL = "https://router.huggingface.co/v1"
OPENROUTER_URL = "https://openrouter.ai/api/v1"
GROQ_URL = "https://api.groq.com/openai/v1"
CEREBRAS_URL = "https://api.cerebras.ai/v1"
GOOGLE_GENAI_URL = "https://generativelanguage.googleapis.com/v1beta/openai"

# ── Timeouts & limits ────────────────────────────────────────────────────────
# Per-call timeout. If a single LLM call exceeds this, the entire model
# is considered failed and we fall through to the next in the chain.
REQUEST_TIMEOUT = 10    # seconds — triggers fallback if exceeded
MAX_TOOL_ITERATIONS = 5  # max ReAct loop iterations before forcing output


# ── Model definitions ────────────────────────────────────────────────────────


@dataclass
class ModelConfig:
    """Configuration for a single LLM model."""

    model_id: str
    provider: str        # "hf_inference", "openrouter", "groq", "cerebras", "google"
    max_tokens: int = 4096
    temperature: float = 0.0
    timeout: int = REQUEST_TIMEOUT


# Agent → (primary, fallback) model chains
AGENT_MODELS: dict[str, list[ModelConfig]] = {
    "metrics": [
        ModelConfig("meta-llama/Llama-3.1-8B-Instruct:scaleway", "hf_inference"),
        ModelConfig("llama-3.1-8b-instant", "groq"),
    ],
    "risk": [
        ModelConfig("Qwen/Qwen2.5-72B-Instruct:novita", "hf_inference"),
        ModelConfig("meta-llama/llama-3.3-70b-instruct:free", "openrouter"),
    ],
    "news": [
        ModelConfig("gemini-3-flash-preview", "google"),
        ModelConfig("gemini-2.5-flash", "google"),
    ],
    "supervisor": [
        ModelConfig("openai/gpt-oss-120b", "openrouter"),
        ModelConfig("Qwen/Qwen2.5-72B-Instruct:novita", "hf_inference"),
    ],
    "synthesis": [
        ModelConfig("gpt-oss-120b", "cerebras"),
        ModelConfig("qwen/qwen3-32b", "groq"),
    ],
}


def _get_api_key(provider: str) -> str:
    """Get the API key for a given provider."""
    key_map = {
        "hf_inference": settings.hf_token,
        "openrouter": settings.openrouter_api_key,
        "groq": settings.groq_api_key,
        "cerebras": settings.cerebras_api_key,
        "google": settings.google_api_key,
    }
    return key_map.get(provider, "")


def _get_base_url(provider: str) -> str:
    """Get the base URL for a given provider."""
    url_map = {
        "hf_inference": HF_INFERENCE_URL,
        "openrouter": OPENROUTER_URL,
        "groq": GROQ_URL,
        "cerebras": CEREBRAS_URL,
        "google": GOOGLE_GENAI_URL,
    }
    return url_map.get(provider, "")


def _is_rate_limit_error(exc: Exception) -> bool:
    """Check if an exception is a 429 rate limit error."""
    exc_str = str(exc).lower()
    return "429" in exc_str or "rate limit" in exc_str or "rate_limit" in exc_str


def _is_timeout_error(exc: Exception) -> bool:
    """Check if an exception is a timeout error."""
    exc_str = str(exc).lower()
    return "timeout" in exc_str or "timed out" in exc_str


def _is_json_error(exc: Exception) -> bool:
    """Check if an exception is a JSON parsing / Pydantic validation error."""
    return isinstance(exc, (ValidationError, ValueError)) or "json" in str(exc).lower()


def _classify_error(exc: Exception) -> str:
    """Classify an error for logging purposes."""
    if _is_rate_limit_error(exc):
        return "RATE_LIMIT_429"
    if _is_timeout_error(exc):
        return "TIMEOUT"
    if _is_json_error(exc):
        return "JSON_VALIDATION"
    return "PROVIDER_ERROR"


def _create_llm(config: ModelConfig) -> ChatOpenAI:
    """Create a ChatOpenAI instance from a ModelConfig."""
    api_key = _get_api_key(config.provider)
    base_url = _get_base_url(config.provider)

    if not api_key:
        raise ValueError(f"No API key for provider '{config.provider}'")

    return ChatOpenAI(
        model=config.model_id,
        openai_api_key=api_key,
        openai_api_base=base_url,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
        request_timeout=config.timeout,
    )


def get_llm(agent_name: str) -> ChatOpenAI:
    """Get the best available LLM for the given agent (key-check only, no network call).

    Args:
        agent_name: One of 'metrics', 'risk', 'news', 'supervisor', 'synthesis'.

    Returns:
        A ChatOpenAI instance configured for the correct provider.
    """
    models = AGENT_MODELS.get(agent_name)
    if not models:
        raise ValueError(f"Unknown agent: '{agent_name}'. Valid: {list(AGENT_MODELS)}")

    for config in models:
        try:
            llm = _create_llm(config)
            logger.info(
                "Agent '%s' → %s via %s",
                agent_name, config.model_id, config.provider,
            )
            return llm
        except ValueError as exc:
            logger.warning(
                "Agent '%s' skipping %s (%s): %s",
                agent_name, config.model_id, config.provider, exc,
            )
            continue

    raise RuntimeError(
        f"All LLM models failed for agent '{agent_name}'. "
        f"Check your API keys in .env."
    )


# ═════════════════════════════════════════════════════════════════════════════
# Tool execution helpers
# ═════════════════════════════════════════════════════════════════════════════


def _resolve_tool(tool_name: str, tools: list[BaseTool]) -> BaseTool | None:
    """Find a tool by name from the tool list."""
    for t in tools:
        if t.name == tool_name:
            return t
    return None


def _execute_tool_calls(
    ai_message: AIMessage,
    tools: list[BaseTool],
) -> list[ToolMessage]:
    """Execute all tool calls from an AI message and return ToolMessages."""
    tool_messages = []

    for tool_call in ai_message.tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]
        tool_id = tool_call["id"]

        tool = _resolve_tool(tool_name, tools)
        if tool is None:
            result = f"Error: Unknown tool '{tool_name}'"
            logger.warning("Unknown tool called: %s", tool_name)
        else:
            try:
                result = tool.invoke(tool_args)
                logger.info(
                    "Tool '%s' executed: args=%s → %s chars",
                    tool_name,
                    {k: str(v)[:50] for k, v in tool_args.items()},
                    len(str(result)),
                )
            except Exception as exc:
                result = f"Tool error: {exc}"
                logger.warning("Tool '%s' failed: %s", tool_name, exc)

        tool_messages.append(
            ToolMessage(content=str(result), tool_call_id=tool_id)
        )

    return tool_messages


# ═════════════════════════════════════════════════════════════════════════════
# ReAct tool-calling agent loop
# ═════════════════════════════════════════════════════════════════════════════


def run_tool_agent(
    agent_name: str,
    system_prompt: str,
    human_prompt: str,
    tools: list[BaseTool],
    output_schema: type[T],
    config: dict | None = None,
) -> T:
    """Run a ReAct agent loop with tool calling and structured output.

    This is the core agent execution function. It:
    1. Binds tools to the LLM
    2. Runs a ReAct loop (LLM calls tools → we execute → feed back)
    3. When the LLM stops calling tools, extracts structured output
    4. Falls back to the next model in the chain on any error

    The LLM autonomously decides WHAT to retrieve and WHEN to call tools.

    Args:
        agent_name: Agent name for model selection + fallback.
        system_prompt: System message for the agent.
        human_prompt: Human message (already formatted with variables).
        tools: List of LangChain tools to bind.
        output_schema: Pydantic model class for the final structured output.
        config: Optional LangChain config (Langfuse callbacks).

    Returns:
        An instance of output_schema.

    Raises:
        RuntimeError: If all models in the fallback chain fail.
    """
    models = AGENT_MODELS.get(agent_name)
    if not models:
        raise ValueError(f"Unknown agent: '{agent_name}'")

    last_error: Exception | None = None

    for i, model_config in enumerate(models):
        is_fallback = i > 0
        label = "fallback" if is_fallback else "primary"

        try:
            api_key = _get_api_key(model_config.provider)
            if not api_key:
                logger.warning(
                    "[%s] %s: No API key for %s — skipping",
                    agent_name, label, model_config.provider,
                )
                continue

            llm = _create_llm(model_config)

            logger.info(
                "[%s] %s: Starting ReAct loop with %s via %s (tools: %s)",
                agent_name, label, model_config.model_id, model_config.provider,
                [t.name for t in tools],
            )

            # ── ReAct tool-calling loop ─────────────────────────
            llm_with_tools = llm.bind_tools(tools)
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=human_prompt),
            ]

            start = time.monotonic()
            tool_call_count = 0

            for iteration in range(MAX_TOOL_ITERATIONS):
                response = llm_with_tools.invoke(messages, config=config)
                messages.append(response)

                if not response.tool_calls:
                    logger.info(
                        "[%s] %s: LLM done after %d tool calls (%d iterations)",
                        agent_name, label, tool_call_count, iteration + 1,
                    )
                    break

                # Execute tool calls and feed results back
                tool_messages = _execute_tool_calls(response, tools)
                messages.extend(tool_messages)
                tool_call_count += len(response.tool_calls)

                logger.info(
                    "[%s] %s: Iteration %d — %d tool calls executed",
                    agent_name, label, iteration + 1, len(response.tool_calls),
                )

            # ── Structured output from accumulated messages ─────────
            # The messages already contain all tool results. One call
            # with .with_structured_output() extracts the typed answer.
            structured_llm = llm.with_structured_output(output_schema)
            result = structured_llm.invoke(messages, config=config)
            elapsed = time.monotonic() - start

            logger.info(
                "[%s] %s: ✅ Complete (%s, %d tool calls, %.1fs)",
                agent_name, label, model_config.model_id, tool_call_count, elapsed,
            )
            return result

        except Exception as exc:
            error_type = _classify_error(exc)
            logger.warning(
                "[%s] %s: ❌ %s from %s: %s",
                agent_name, label, error_type,
                model_config.model_id, str(exc)[:200],
            )
            last_error = exc
            continue

    raise RuntimeError(
        f"All models failed for agent '{agent_name}'. Last error: {last_error}"
    )


# ═════════════════════════════════════════════════════════════════════════════
# Structured-only invocation (no tools) — for Supervisor
# ═════════════════════════════════════════════════════════════════════════════


def invoke_with_fallback(
    agent_name: str,
    prompt_chain: Any,
    input_data: dict,
    output_schema: type[T],
    config: dict | None = None,
) -> T:
    """Invoke a prompt chain with structured output and automatic fallback.

    For agents that don't use tools (e.g., Supervisor).

    Args:
        agent_name: Agent name for model selection.
        prompt_chain: A LangChain prompt template.
        input_data: Dict of prompt variables.
        output_schema: Pydantic model class for structured output.
        config: Optional LangChain config.

    Returns:
        An instance of output_schema.
    """
    models = AGENT_MODELS.get(agent_name)
    if not models:
        raise ValueError(f"Unknown agent: '{agent_name}'")

    last_error: Exception | None = None

    for i, model_config in enumerate(models):
        is_fallback = i > 0
        label = "fallback" if is_fallback else "primary"

        try:
            api_key = _get_api_key(model_config.provider)
            if not api_key:
                logger.warning(
                    "[%s] %s: No API key for %s — skipping",
                    agent_name, label, model_config.provider,
                )
                continue

            llm = _create_llm(model_config)
            structured_llm = llm.with_structured_output(output_schema)
            chain = prompt_chain | structured_llm

            start = time.monotonic()
            result = chain.invoke(input_data, config=config)
            elapsed = time.monotonic() - start

            logger.info(
                "[%s] %s: ✅ Success (%s, %.1fs)",
                agent_name, label, model_config.model_id, elapsed,
            )
            return result

        except Exception as exc:
            error_type = _classify_error(exc)
            logger.warning(
                "[%s] %s: ❌ %s from %s: %s",
                agent_name, label, error_type,
                model_config.model_id, str(exc)[:200],
            )
            last_error = exc
            continue

    raise RuntimeError(
        f"All models failed for agent '{agent_name}'. Last error: {last_error}"
    )


def create_langfuse_config(
    session_id: str = "",
    trace_name: str = "",
    **kwargs,
) -> dict:
    """Create a LangChain config dict with Langfuse tracing attached."""
    handler = get_langfuse_handler(
        session_id=session_id,
        trace_name=trace_name,
        **kwargs,
    )
    return {"callbacks": [handler]}
