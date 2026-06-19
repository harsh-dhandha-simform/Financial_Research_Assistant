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
    │ Risk         │ Qwen-2.5-72B (HF Inference / Novita)        │ Nemotron-3-Super-120B (NVIDIA)            │
    │ News         │ gemini-2.5-flash (Google AI)                 │ gemini-3.5-flash (Google AI)        │
    │ Supervisor   │ gpt-oss-120b (OpenRouter)                   │ Qwen-2.5-72B (HF Inference)              │
    │ Synthesis    │ gpt-oss-120b (Cerebras)                     │ Qwen3-32B (Groq)                         │
    └──────────────┴─────────────────────────────────────────────┴──────────────────────────────────────────┘

Fallback triggers:
    • Provider rate limit (HTTP 429) → switch to fallback immediately
    • Request timeout (> 30 seconds) → switch to faster fallback model
    • Malformed JSON / Pydantic validation failure → retry on fallback
    • Full provider outage (connection error) → entire chain falls to next

Structured output methods:
    • "json_schema"  — strict JSON schema enforcement (OpenRouter, Cerebras)
    • "json_mode"    — basic json_object mode (HF Inference, Groq, NVIDIA)
    • "function_calling" — uses tool/function call (Google/Gemini)
    Models that don't support json_schema use json_mode with explicit
    JSON instructions injected into the prompt.
"""

import logging
import time
from dataclasses import dataclass, field
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
NVIDIA_URL = "https://integrate.api.nvidia.com/v1"

# ── Timeouts & limits ────────────────────────────────────────────────────────
# Per-call timeout. Structured output with large models (gpt-oss-120b,
# Qwen 72B) can take 15-25s. 30s gives headroom before triggering fallback.
REQUEST_TIMEOUT = 30    # seconds — triggers fallback if exceeded
MAX_TOOL_ITERATIONS = 5  # max ReAct loop iterations before forcing output
MAX_TOOL_OUTPUT_CHARS = 3000  # truncate tool outputs to prevent TPM blowout


# ── Google API key rotation ───────────────────────────────────────────────
# Tracks the current Google key index so on 429 we rotate to the next one.
_google_key_index = 0


# ── Model definitions ────────────────────────────────────────────────────────


@dataclass
class ModelConfig:
    """Configuration for a single LLM model.

    structured_method controls how .with_structured_output() works:
      - "json_schema"      → strict schema enforcement (default)
      - "json_mode"        → response_format: {"type": "json_object"}
      - "function_calling" → uses tool/function calling to enforce schema
    """

    model_id: str
    provider: str        # "hf_inference", "openrouter", "groq", "cerebras", "google", "nvidia"
    max_tokens: int | None = None   # None = use model's own max output limit
    temperature: float = 0.0
    timeout: int = REQUEST_TIMEOUT
    structured_method: str = "json_schema"
    max_tool_output_chars: int = MAX_TOOL_OUTPUT_CHARS  # per-model override to prevent TPM blowout


# Agent → (primary, fallback) model chains
#
# Structured method notes:
#   - HF Inference (Novita/Scaleway): json_schema technically supported but
#     causes "add_generation_prompt" issues → use json_mode
#   - Groq: llama-3.1-8b-instant doesn't support json_schema → use json_mode
#   - NVIDIA: OpenAI-compat but json_mode is most reliable
#   - OpenRouter: supports json_schema via function_calling
#   - Cerebras: supports json_schema
#   - Google: use function_calling (most reliable for Gemini)
AGENT_MODELS: dict[str, list[ModelConfig]] = {
    "metrics": [
        ModelConfig(
            "gpt-oss-120b", "cerebras",
            structured_method="json_mode",
        ),
        ModelConfig(
            "llama-3.3-70b-versatile", "groq",
            max_tokens=8192,  # Keep low to avoid Groq TPM limits
            structured_method="json_mode",
            max_tool_output_chars=1500,  # Tighter limit for free-tier Groq
        ),
        ModelConfig(
            "qwen/qwen3-32b", "groq",
            structured_method="json_mode",
            max_tool_output_chars=8192,
        )
    ],
    "risk": [
        ModelConfig(
            "nvidia/nemotron-3-super-120b-a12b", "nvidia",
            structured_method="json_mode",
        ),
        ModelConfig(
            "gpt-oss-120b", "cerebras",
            structured_method="json_mode",
        ),
        ModelConfig(
            "llama-3.3-70b-versatile", "groq",
            max_tokens=11000,
            structured_method="json_mode",
            max_tool_output_chars=1500,
        ),
    ],
    "news": [
        # gemini-2.5-flash primary — Groq Llama fallback
        # 8192 tokens is generous for chat responses without risk of runaway output
        ModelConfig(
            "gemini-2.5-flash", "google",
            max_tokens=8192,
            structured_method="function_calling",
        ),
        ModelConfig(
            "llama-3.3-70b-versatile", "groq",
            max_tokens=8192,
            structured_method="json_mode",
            max_tool_output_chars=1000,  # Very tight — Groq 12k TPM limit
        ),
        ModelConfig(
            "groq/compound", "groq",
            max_tokens=8192,
            structured_method="json_mode",
        ),
    ],
    "supervisor": [
        ModelConfig("openai/gpt-oss-120b", "openrouter"),
        ModelConfig(
            "Qwen/Qwen2.5-72B-Instruct:featherless-ai", "hf_inference",
            structured_method="json_mode",
        ),
    ],
    "synthesis": [
        ModelConfig("gpt-oss-120b", "cerebras"),
        ModelConfig(
            "Qwen/Qwen2.5-72B-Instruct:featherless-ai", "hf_inference",
            structured_method="json_mode",
            timeout=120,
        ),
    ],
    # Dedicated lightweight model for per-article sentiment scoring
    # Groq llama-3.1-8b is fast and supports json_mode reliably
    "sentiment": [
        ModelConfig(
            "llama-3.1-8b-instant", "groq",
            max_tokens=512,
            structured_method="json_mode",
        ),
    ],
}


def _get_api_key(provider: str) -> str:
    """Get the API key for a given provider.

    For Google, uses the rotation index to pick from all available keys.
    """
    global _google_key_index
    if provider == "google":
        keys = settings.google_api_keys
        if not keys:
            return ""
        idx = _google_key_index % len(keys)
        return keys[idx]

    key_map = {
        "hf_inference": settings.hf_token,
        "openrouter": settings.openrouter_api_key,
        "groq": settings.groq_api_key,
        "cerebras": settings.cerebras_api_key,
        "nvidia": settings.nvidia_api_key,
    }
    return key_map.get(provider, "")


def _rotate_google_key():
    """Rotate to the next Google API key after a 429 rate limit."""
    global _google_key_index
    keys = settings.google_api_keys
    if len(keys) <= 1:
        return  # Nothing to rotate
    old_idx = _google_key_index % len(keys)
    _google_key_index += 1
    new_idx = _google_key_index % len(keys)
    logger.info(
        "Google API key rotated: key %d → key %d (of %d total)",
        old_idx + 1, new_idx + 1, len(keys),
    )


def _get_base_url(provider: str) -> str:
    """Get the base URL for a given provider."""
    url_map = {
        "hf_inference": HF_INFERENCE_URL,
        "openrouter": OPENROUTER_URL,
        "groq": GROQ_URL,
        "cerebras": CEREBRAS_URL,
        "google": GOOGLE_GENAI_URL,
        "nvidia": NVIDIA_URL,
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

    kwargs: dict = {
        "model": config.model_id,
        "openai_api_key": api_key,
        "openai_api_base": base_url,
        "temperature": config.temperature,
        "request_timeout": config.timeout,
    }
    # Only pass max_tokens when explicitly set — omitting it lets the model use its own max
    if config.max_tokens is not None:
        kwargs["max_tokens"] = config.max_tokens
    
    # Disable built-in retries for Google so our manual fallback/rotation takes over instantly
    if config.provider == "google":
        kwargs["max_retries"] = 0

    # For groq/compound, pass its required compound_custom config
    if config.model_id == "groq/compound":
        kwargs["model_kwargs"] = {
            "compound_custom": {"tools": {"enabled_tools": ["web_search", "wolfram_aplha", "visit_website"]}}
        }

    return ChatOpenAI(**kwargs)


def _get_structured_llm(
    llm: ChatOpenAI,
    output_schema: type[T],
    method: str,
) -> Any:
    """Create a structured output LLM using the correct method for the provider.

    Args:
        llm: The base ChatOpenAI instance.
        output_schema: The Pydantic model to enforce.
        method: One of "json_schema", "json_mode", "function_calling".

    Returns:
        An LLM wrapped with .with_structured_output().
    """
    if method == "json_mode":
        return llm.with_structured_output(output_schema, method="json_mode")
    elif method == "function_calling":
        return llm.with_structured_output(output_schema, method="function_calling")
    else:
        # Default: json_schema (strict)
        return llm.with_structured_output(output_schema)


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
    config: dict | None = None,
    max_tool_output_chars: int = MAX_TOOL_OUTPUT_CHARS,
) -> list[ToolMessage]:
    """Execute all tool calls from an AI message and return ToolMessages."""
    tool_messages = []

    for tool_call in ai_message.tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]
        tool_id = tool_call["id"]

        tool = _resolve_tool(tool_name, tools)
        if tool is None:
            result_str = f"Error: Unknown tool '{tool_name}'"
            logger.warning("Unknown tool called: %s", tool_name)
        else:
            try:
                result = tool.invoke(tool_args, config=config)
                result_str = str(result)
                # Truncate to prevent TPM blowout on free-tier providers
                if len(result_str) > max_tool_output_chars:
                    result_str = result_str[:max_tool_output_chars] + "\n... [truncated]"
                logger.info(
                    "Tool '%s' executed: args=%s → %s chars",
                    tool_name,
                    {k: str(v)[:50] for k, v in tool_args.items()},
                    len(result_str),
                )
            except Exception as exc:
                result_str = f"Tool error: {exc}"
                logger.warning("Tool '%s' failed: %s", tool_name, exc)

        tool_messages.append(
            ToolMessage(content=result_str, tool_call_id=tool_id)
        )

    return tool_messages


# ═════════════════════════════════════════════════════════════════════════════
# ReAct tool-calling agent loop
# ═════════════════════════════════════════════════════════════════════════════


def _prepare_messages_for_structured_output(
    messages: list,
    method: str,
    output_schema: type[BaseModel],
    max_total_chars: int = 12000,
) -> list:
    """Prepare message list for the structured output call.

    Fixes provider-specific issues:
    1. HF Inference / Groq: when last message is tool/assistant, injects a
       HumanMessage to fix turn ordering.
    2. json_mode providers need an explicit JSON instruction so the model
       outputs the schema (not another tool call).
    3. Truncates accumulated tool message content to avoid 413 / TPM errors.

    Args:
        messages: The accumulated conversation messages.
        method: The structured output method being used.
        output_schema: The Pydantic schema (used to build JSON instruction).
        max_total_chars: Hard cap on total chars of tool-output content.

    Returns:
        A copy of messages with any necessary fixups applied.
    """
    # ── 1. Truncate bulky tool outputs from the history ──────────────────
    # We summarise the tool messages to just the first N chars each so
    # the extraction call doesn't blow past free-tier TPM limits.
    msgs: list = []
    total_tool_chars = 0
    for msg in messages:
        if isinstance(msg, ToolMessage):
            content = str(msg.content)
            remaining = max(0, max_total_chars - total_tool_chars)
            if len(content) > remaining:
                content = content[:remaining] + "\n...[truncated for token limit]"
            total_tool_chars += len(content)
            msgs.append(ToolMessage(content=content, tool_call_id=msg.tool_call_id))
        else:
            msgs.append(msg)

    # ── 2. Build the extraction instruction ─────────────────────────────
    schema_hint = output_schema.model_json_schema()

    if method == "json_mode":
        # For json_mode providers (Groq, Cerebras, NVIDIA): inject an explicit
        # extraction prompt.  Critically, we forbid further tool calls to prevent
        # models like NVIDIA Nemotron from outputting a tool-call JSON.
        extraction_msg = HumanMessage(
            content=(
                "=== FINAL EXTRACTION PHASE ===\n"
                "You have finished gathering information. DO NOT call any tools.\n"
                "Produce ONLY a single JSON object matching the schema below.\n"
                "Do NOT wrap in markdown, do NOT add explanation, output JSON only.\n\n"
                f"Required JSON schema:\n```json\n{schema_hint}\n```\n\n"
                "Respond with only the JSON object:"
            )
        )
        msgs.append(extraction_msg)
    else:
        # function_calling / json_schema: just fix turn ordering if needed
        if msgs and isinstance(msgs[-1], (AIMessage, ToolMessage)):
            msgs.append(
                HumanMessage(
                    content=(
                        "Now produce your final structured analysis based on "
                        "everything above."
                    )
                )
            )

    return msgs


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

        max_attempts = len(settings.google_api_keys) if model_config.provider == "google" else 1

        for attempt in range(max_attempts):
            try:
                api_key = _get_api_key(model_config.provider)
                if not api_key:
                    logger.warning(
                        "[%s] %s: No API key for %s — skipping",
                        agent_name, label, model_config.provider,
                    )
                    break

                llm = _create_llm(model_config)

                logger.info(
                    "[%s] %s: Starting ReAct loop with %s via %s (attempt %d/%d, tools: %s)",
                    agent_name, label, model_config.model_id, model_config.provider,
                    attempt + 1, max_attempts, [t.name for t in tools],
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
                    tool_messages = _execute_tool_calls(
                        response, tools, config=config,
                        max_tool_output_chars=model_config.max_tool_output_chars,
                    )
                    messages.extend(tool_messages)
                    tool_call_count += len(response.tool_calls)

                    logger.info(
                        "[%s] %s: Iteration %d — %d tool calls executed",
                        agent_name, label, iteration + 1, len(response.tool_calls),
                    )

                # ── Structured output from accumulated messages ─────────
                # Prepare messages: fix turn ordering, inject JSON hints,
                # and truncate tool outputs to prevent TPM blowout.
                output_messages = _prepare_messages_for_structured_output(
                    messages,
                    model_config.structured_method,
                    output_schema,
                    max_total_chars=model_config.max_tool_output_chars * 4,
                )
                structured_llm = _get_structured_llm(
                    llm, output_schema, model_config.structured_method,
                )
                result = structured_llm.invoke(output_messages, config=config)
                elapsed = time.monotonic() - start

                logger.info(
                    "[%s] %s: ✅ Complete (%s, %d tool calls, %.1fs)",
                    agent_name, label, model_config.model_id, tool_call_count, elapsed,
                )
                return result

            except Exception as exc:
                error_type = _classify_error(exc)
                if model_config.provider == "google" and error_type == "RATE_LIMIT_429" and attempt < max_attempts - 1:
                    logger.warning(
                        "[%s] %s: ❌ Rate limit (429) on attempt %d/%d from %s. Rotating Google API key and retrying...",
                        agent_name, label, attempt + 1, max_attempts, model_config.model_id,
                    )
                    _rotate_google_key()
                    continue

                logger.warning(
                    "[%s] %s: ❌ %s from %s (attempt %d/%d): %s",
                    agent_name, label, error_type,
                    model_config.model_id, attempt + 1, max_attempts, str(exc)[:200],
                )
                last_error = exc
                break

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

        max_attempts = len(settings.google_api_keys) if model_config.provider == "google" else 1

        for attempt in range(max_attempts):
            try:
                api_key = _get_api_key(model_config.provider)
                if not api_key:
                    logger.warning(
                        "[%s] %s: No API key for %s — skipping",
                        agent_name, label, model_config.provider,
                    )
                    break

                llm = _create_llm(model_config)
                structured_llm = _get_structured_llm(
                    llm, output_schema, model_config.structured_method,
                )
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
                if model_config.provider == "google" and error_type == "RATE_LIMIT_429" and attempt < max_attempts - 1:
                    logger.warning(
                        "[%s] %s: ❌ Rate limit (429) on attempt %d/%d from %s. Rotating Google API key and retrying...",
                        agent_name, label, attempt + 1, max_attempts, model_config.model_id,
                    )
                    _rotate_google_key()
                    continue

                logger.warning(
                    "[%s] %s: ❌ %s from %s (attempt %d/%d): %s",
                    agent_name, label, error_type,
                    model_config.model_id, attempt + 1, max_attempts, str(exc)[:200],
                )
                last_error = exc
                break

    raise RuntimeError(
        f"All models failed for agent '{agent_name}'. Last error: {last_error}"
    )


def create_langfuse_config(
    session_id: str = "",
    trace_name: str = "",
    use_callbacks: bool = False,
    **kwargs,
) -> dict:
    """Create a LangChain config dict with Langfuse tracing attached.
    
    If use_callbacks is True, creates a fresh CallbackHandler.
    Otherwise, relies on parent callbacks for tracing to avoid UUID conflicts
    when running multiple parallel threads, but still propagates the session_id.
    """
    config = {
        "configurable": {
            "session_id": session_id,
        }
    }
    if use_callbacks:
        handler = get_langfuse_handler(
            session_id=session_id,
            trace_name=trace_name,
            **kwargs,
        )
        config["callbacks"] = [handler]
    return config